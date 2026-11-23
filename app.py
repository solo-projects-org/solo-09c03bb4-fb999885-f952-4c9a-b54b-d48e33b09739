"""Flask + SQLite service for phased A/B split verification.

The workflow is deliberately small:

* a round is configured once with 1..8 periods, each with a planned
  headcount and an independent group-B probability,
* one CSV per period (user id + actual group) is uploaded, in any order,
* complete consecutive prefixes are verified against the cumulative
  binomial distribution implied by the plan,
* every verification records the observed cumulative B count, an exact
  two-sided p-value and whether it crosses the Bonferroni threshold.

Run with ``gunicorn`` (see ``Dockerfile`` / ``compose.yaml``) or with
``flask --app app run`` for development.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
from typing import Optional, Sequence

from flask import (
    Flask,
    abort,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)

import stats

# ---------------------------------------------------------------------------
# Round constants
# ---------------------------------------------------------------------------

MAX_PERIODS = 8
TOTAL_USERS = 1000
MAX_CSV_BYTES = 2 * 1024 * 1024
MAX_USER_ID_LENGTH = 128
ALPHA = Decimal("0.05")

DEFAULT_DB = os.environ.get(
    "DIV_DB",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "split.sqlite3"),
)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "dev-only-secret-change-me"),
    MAX_CONTENT_LENGTH=MAX_CSV_BYTES + 64 * 1024,
    DB_PATH=DEFAULT_DB,
)
app.jinja_env.globals.update(
    MAX_PERIODS=MAX_PERIODS,
    TOTAL_USERS=TOTAL_USERS,
)

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS round_meta (
    id           INTEGER PRIMARY KEY CHECK (id = 1),
    periods      INTEGER NOT NULL CHECK (periods BETWEEN 1 AND 8),
    configured_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS plan (
    period        INTEGER PRIMARY KEY CHECK (period BETWEEN 1 AND 8),
    planned_users INTEGER NOT NULL CHECK (planned_users > 0),
    prob_b        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS uploads (
    period        INTEGER PRIMARY KEY CHECK (period BETWEEN 1 AND 8),
    checksum      TEXT    NOT NULL,
    row_count     INTEGER NOT NULL,
    b_count       INTEGER NOT NULL,
    accepted_at   TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);

CREATE TABLE IF NOT EXISTS assignments (
    user_id       TEXT    NOT NULL PRIMARY KEY,
    period        INTEGER NOT NULL CHECK (period BETWEEN 1 AND 8),
    grp           TEXT    NOT NULL CHECK (grp IN ('A','B'))
);

CREATE INDEX IF NOT EXISTS assignments_period_idx ON assignments(period);

CREATE TABLE IF NOT EXISTS results (
    period         INTEGER PRIMARY KEY CHECK (period BETWEEN 1 AND 8),
    verified_users INTEGER NOT NULL,
    observed_b     INTEGER NOT NULL,
    expected_b     TEXT    NOT NULL,
    variance       TEXT    NOT NULL,
    tail_low       TEXT    NOT NULL,
    tail_high      TEXT    NOT NULL,
    p_value        TEXT    NOT NULL,
    threshold      TEXT    NOT NULL,
    flagged        INTEGER NOT NULL CHECK (flagged IN (0, 1)),
    verified_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
"""


def connect(path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(
        path or current_app.config["DB_PATH"],
        timeout=30,
        isolation_level=None,  # explicit BEGIN IMMEDIATE transactions
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.OperationalError:
        # A read-only filesystem or another writer; the connection still works.
        pass
    return conn


_schema_ready_paths: set[str] = set()


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def get_db() -> sqlite3.Connection:
    """Per-request connection, with lazy idempotent schema creation."""
    if "db" not in g:
        db = connect()
        path = current_app.config["DB_PATH"]
        if path not in _schema_ready_paths:
            try:
                init_schema(db)
            except sqlite3.OperationalError:
                # Another worker may be creating the schema concurrently.
                time.sleep(0.1)
                init_schema(db)
            _schema_ready_paths.add(path)
        g.db = db
    return g.db


@app.teardown_appcontext
def close_db(exc: Optional[BaseException]) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def bootstrap(path: Optional[str] = None) -> None:
    """Create the database file and schema (used by CLI/entrypoint/tests)."""
    target = path or app.config["DB_PATH"]
    os.makedirs(os.path.dirname(os.path.abspath(target)) or ".", exist_ok=True)
    conn = connect(target)
    try:
        init_schema(conn)
    finally:
        conn.close()
    _schema_ready_paths.add(target)


@app.cli.command("init-db")
def init_db_command() -> None:
    bootstrap()
    print(f"initialised {app.config['DB_PATH']}")


# ---------------------------------------------------------------------------
# Small data-access helpers
# ---------------------------------------------------------------------------


def round_periods(db: sqlite3.Connection) -> Optional[int]:
    row = db.execute("SELECT periods FROM round_meta WHERE id = 1").fetchone()
    return int(row["periods"]) if row else None


def load_plan(db: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(db.execute("SELECT period, planned_users, prob_b FROM plan ORDER BY period"))


def plan_key(plan: Sequence[sqlite3.Row]) -> tuple[tuple[int, str], ...]:
    return tuple((int(row["planned_users"]), str(row["prob_b"])) for row in plan)


@lru_cache(maxsize=16)
def cached_distributions(key: tuple[tuple[int, str], ...]):
    return stats.all_prefix_distributions(
        [(n, Decimal(p)) for n, p in key]
    )


def plan_distributions(plan: Sequence[sqlite3.Row]):
    return cached_distributions(plan_key(plan))


def verification_threshold(periods: int) -> Decimal:
    with localcontext() as ctx:
        ctx.prec = stats.PRECISION
        return ALPHA / Decimal(periods)


def plan_is_locked(db: sqlite3.Connection) -> bool:
    row = db.execute(
        "SELECT (SELECT COUNT(*) FROM uploads) + (SELECT COUNT(*) FROM results) AS n"
    ).fetchone()
    return bool(row and row["n"])


def round_state(db: sqlite3.Connection) -> dict:
    """Everything the dashboard needs, with results joined to the plan."""
    periods = round_periods(db)
    if periods is None:
        return {
            "configured": False,
            "periods": 0,
            "plan": [],
            "uploads": {},
            "results": {},
            "pending": [],
            "threshold": None,
            "earliest_flag": None,
            "locked": False,
            "max_periods": MAX_PERIODS,
            "total_users": TOTAL_USERS,
        }

    plan = load_plan(db)
    uploads = {
        int(row["period"]): row
        for row in db.execute("SELECT period, row_count, b_count, accepted_at, checksum FROM uploads")
    }
    results = {
        int(row["period"]): row
        for row in db.execute(
            "SELECT period, verified_users, observed_b, expected_b, variance, "
            "tail_low, tail_high, p_value, threshold, flagged, verified_at "
            "FROM results ORDER BY period"
        )
    }
    verified_periods = sorted(results)
    pending = [
        period
        for period in sorted(uploads)
        if period not in results
    ]
    earliest_flag = next(
        (results[p]["period"] for p in verified_periods if results[p]["flagged"]), None
    )
    return {
        "configured": True,
        "periods": periods,
        "plan": plan,
        "uploads": uploads,
        "results": results,
        "pending": pending,
        "threshold": verification_threshold(periods),
        "earliest_flag": earliest_flag,
        "locked": plan_is_locked(db),
        "max_periods": MAX_PERIODS,
        "total_users": TOTAL_USERS,
    }


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

USER_HEADERS = {"user_id", "userid", "user", "用户编号", "用户id", "用户标识", "成员id"}
GROUP_HEADERS = {"group", "grp", "ab_group", "ab组", "组别", "分组", "实际组别"}


class CSVError(ValueError):
    """A user-facing CSV validation error."""


@dataclass(frozen=True)
class UploadRow:
    user_id: str
    group: str


@dataclass
class ParsedUpload:
    rows: tuple[UploadRow, ...]
    b_count: int
    checksum: str


def _normalise_header(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "_")


def parse_csv(raw: bytes) -> ParsedUpload:
    if not raw:
        raise CSVError("CSV 文件为空")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CSVError("CSV 必须是 UTF-8 编码") from exc
    if "\x00" in text:
        raise CSVError("CSV 含有二进制空字符")

    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as exc:
        raise CSVError("CSV 文件为空") from exc

    normalised = [_normalise_header(cell) for cell in header]
    user_index = next(
        (i for i, name in enumerate(normalised) if name in USER_HEADERS), None
    )
    group_index = next(
        (i for i, name in enumerate(normalised) if name in GROUP_HEADERS), None
    )
    if user_index is None or group_index is None:
        raise CSVError("表头必须同时包含用户编号列（user_id）和组别列（group）")
    if user_index == group_index:
        raise CSVError("用户编号列与组别列不能相同")

    rows: list[UploadRow] = []
    seen: set[str] = set()
    for line_no, record in enumerate(reader, start=2):
        if not any(cell.strip() for cell in record):
            continue
        if len(record) <= max(user_index, group_index):
            raise CSVError(f"第 {line_no} 行字段不足")
        user_id = record[user_index].strip()
        group = record[group_index].strip().upper()
        if not user_id:
            raise CSVError(f"第 {line_no} 行用户编号为空")
        if len(user_id) > MAX_USER_ID_LENGTH:
            raise CSVError(f"第 {line_no} 行用户编号超过 {MAX_USER_ID_LENGTH} 字符")
        if group not in ("A", "B"):
            raise CSVError(f"第 {line_no} 行组别必须为 A 或 B（实际为 {record[group_index]!r}）")
        if user_id in seen:
            raise CSVError(f"用户编号 {user_id!r} 在文件内重复")
        seen.add(user_id)
        rows.append(UploadRow(user_id=user_id, group=group))
        if len(rows) > TOTAL_USERS:
            raise CSVError(f"单次上传不能超过 {TOTAL_USERS} 行")

    if not rows:
        raise CSVError("CSV 没有数据行")

    canonical = "\n".join(
        f"{row.user_id}\t{row.group}" for row in sorted(rows, key=lambda r: (r.user_id, r.group))
    )
    checksum = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ParsedUpload(
        rows=tuple(rows),
        b_count=sum(1 for row in rows if row.group == "B"),
        checksum=checksum,
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def validate_upload_against_plan(
    parsed: ParsedUpload, planned_users: int
) -> None:
    if len(parsed.rows) != planned_users:
        raise CSVError(
            f"人数不符：该期计划 {planned_users} 人，CSV 实际 {len(parsed.rows)} 人"
        )


def recompute_results(db: sqlite3.Connection) -> list[int]:
    """Rebuild the verified contiguous prefix and return its periods.

    Must be called inside the caller's write transaction.  Results are
    deleted and re-inserted so a period never has more than one row.
    """
    plan = load_plan(db)
    periods = round_periods(db) or len(plan)
    if not plan:
        db.execute("DELETE FROM results")
        return []

    distributions = plan_distributions(plan)
    threshold = verification_threshold(periods)

    db.execute("DELETE FROM results")
    cum_users = 0
    cum_b = 0
    verified: list[int] = []
    for index, row in enumerate(plan):
        upload = db.execute(
            "SELECT row_count, b_count FROM uploads WHERE period = ?",
            (row["period"],),
        ).fetchone()
        if upload is None:
            break

        cum_users += int(upload["row_count"])
        cum_b += int(upload["b_count"])
        distribution = distributions[index]
        p_value = stats.two_sided_p_value(distribution, cum_b)
        tail_low, tail_high = stats.tail_probabilities(distribution, cum_b)
        expected_b = sum(
            Decimal(str(plan[i]["planned_users"])) * Decimal(str(plan[i]["prob_b"]))
            for i in range(index + 1)
        )
        variance = sum(
            Decimal(str(plan[i]["planned_users"]))
            * Decimal(str(plan[i]["prob_b"]))
            * (Decimal(1) - Decimal(str(plan[i]["prob_b"])))
            for i in range(index + 1)
        )
        flagged = 1 if p_value <= threshold else 0
        db.execute(
            "INSERT INTO results (period, verified_users, observed_b, expected_b, "
            "variance, tail_low, tail_high, p_value, threshold, flagged) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(row["period"]),
                cum_users,
                cum_b,
                str(expected_b),
                str(variance),
                str(tail_low),
                str(tail_high),
                str(p_value),
                str(threshold),
                flagged,
            ),
        )
        verified.append(int(row["period"]))
    return verified


@dataclass
class UploadOutcome:
    status: str  # "accepted" | "duplicate" | "conflict" | "error"
    message: str


def _rollback(db: sqlite3.Connection) -> None:
    try:
        db.execute("ROLLBACK")
    except sqlite3.OperationalError:
        pass


def store_upload(period: int, parsed: ParsedUpload) -> UploadOutcome:
    """Validate and store an upload atomically.

    Writers serialise on ``BEGIN IMMEDIATE``.  A repeated identical upload is
    acknowledged without inserting a second row/result; different content
    for a period that is already stored is rejected.  Any failure rolls back,
    leaving previously stored details and verification results untouched.
    """
    last_error: Optional[BaseException] = None
    for attempt in range(6):
        db = get_db()
        try:
            db.execute("BEGIN IMMEDIATE")
            plan_row = db.execute(
                "SELECT planned_users, prob_b FROM plan WHERE period = ?", (period,)
            ).fetchone()
            if plan_row is None:
                _rollback(db)
                return UploadOutcome("error", f"第 {period} 期不在计划中")

            expected_rows = int(plan_row["planned_users"])
            if len(parsed.rows) != expected_rows:
                _rollback(db)
                return UploadOutcome(
                    "error",
                    f"人数不符：该期计划 {expected_rows} 人，CSV 实际 {len(parsed.rows)} 人",
                )

            existing = db.execute(
                "SELECT checksum, row_count, b_count FROM uploads WHERE period = ?",
                (period,),
            ).fetchone()
            if existing is not None:
                _rollback(db)
                if existing["checksum"] == parsed.checksum:
                    return UploadOutcome(
                        "duplicate",
                        f"第 {period} 期相同明细已接收，本次未重复计入。",
                    )
                return UploadOutcome(
                    "conflict",
                    f"第 {period} 期已接收过不同内容的明细，拒收本次上传。",
                )

            # Every user counts once for the whole round.  The PRIMARY KEY on
            # assignments is the final guard; this pre-check gives a precise
            # message for sequential uploads.
            other_users = {
                row["user_id"]
                for row in db.execute(
                    "SELECT user_id FROM assignments WHERE period <> ?", (period,)
                )
            }
            conflicts = [row.user_id for row in parsed.rows if row.user_id in other_users]
            if conflicts:
                _rollback(db)
                sample = ", ".join(sorted(conflicts)[:5])
                more = " 等" if len(conflicts) > 5 else ""
                return UploadOutcome(
                    "error",
                    f"拒收：{len(conflicts)} 个用户编号已出现在其他期（{sample}{more}），每人全程只计一次。",
                )

            db.execute(
                "INSERT INTO uploads (period, checksum, row_count, b_count) "
                "VALUES (?, ?, ?, ?)",
                (period, parsed.checksum, len(parsed.rows), parsed.b_count),
            )
            db.executemany(
                "INSERT INTO assignments (user_id, period, grp) VALUES (?, ?, ?)",
                [(row.user_id, period, row.group) for row in parsed.rows],
            )
            verified = recompute_results(db)
            db.execute("COMMIT")
            suffix = (
                f"连续核验到第 {verified[-1]} 期。"
                if verified
                else "尚无连续到齐的期次，暂未核验。"
            )
            return UploadOutcome(
                "accepted",
                f"第 {period} 期已接收（{len(parsed.rows)} 人，B 组 {parsed.b_count} 人）。{suffix}",
            )
        except sqlite3.IntegrityError as exc:
            _rollback(db)
            last_error = exc
            # Most likely a concurrent insert of one of the same users, or the
            # same period; re-check once if this was a lock retry.
            if attempt < 2:
                continue
            return UploadOutcome(
                "error",
                "拒收：用户编号与已有期次冲突（每人全程只计一次），或该期已被并发上传占用。",
            )
        except sqlite3.OperationalError as exc:
            _rollback(db)
            last_error = exc
            if "locked" in str(exc).lower() or "busy" in str(exc).lower():
                time.sleep(0.05 * (attempt + 1) + random.random() * 0.02)
                continue
            raise
        except Exception:
            # Any non-SQL failure must not leave a half-written period behind.
            _rollback(db)
            raise

    return UploadOutcome(
        "error",
        f"数据库繁忙，请稍后重试（{last_error}）。",
    )


# ---------------------------------------------------------------------------
# Template helpers / routes
# ---------------------------------------------------------------------------


@app.template_filter("prob")
def prob_filter(value) -> str:
    return stats.format_probability(Decimal(str(value)))


@app.template_filter("expected")
def expected_filter(value) -> str:
    return stats.format_expected(Decimal(str(value)))


@app.template_filter("sci")
def sci_filter(value) -> str:
    return stats.format_probability(Decimal(str(value)), digits=4)


@app.get("/healthz")
def healthz():
    db = get_db()
    db.execute("SELECT 1")
    return jsonify({"ok": True})


@app.get("/")
def index():
    db = get_db()
    state = round_state(db)
    latest = None
    latest_stats = None
    if state["configured"] and state["results"]:
        verified = sorted(state["results"])
        latest_period = verified[-1]
        result = state["results"][latest_period]
        plan = state["plan"]
        distributions = plan_distributions(plan)
        distribution = distributions[latest_period - 1]
        latest_stats = stats.distribution_stats(
            distribution,
            Decimal(result["expected_b"]),
            Decimal(result["variance"]),
        )
        latest = result
    return render_template(
        "index.html",
        state=state,
        latest=latest,
        latest_stats=latest_stats,
    )


@app.route("/plan", methods=["GET", "POST"])
def plan():
    db = get_db()
    periods = round_periods(db)
    locked = plan_is_locked(db)
    existing = load_plan(db)

    if request.method == "POST":
        if locked:
            flash("本轮已有上传或核验结果，计划已锁定；如需换比例请先重置本轮数据。", "error")
            return redirect(url_for("plan"))
        try:
            try:
                periods_value = int(request.form.get("periods", ""))
            except ValueError as exc:
                raise ValueError("计划期数必须是整数") from exc
            if not 1 <= periods_value <= MAX_PERIODS:
                raise ValueError(f"计划期数必须为 1 到 {MAX_PERIODS}")
            counts: list[int] = []
            probabilities: list[Decimal] = []
            for period_no in range(1, periods_value + 1):
                count_text = request.form.get(f"count_{period_no}", "").strip()
                prob_text = request.form.get(f"prob_{period_no}", "").strip()
                if not count_text or not prob_text:
                    raise ValueError(f"第 {period_no} 期的人数或 B 组概率未填写")
                try:
                    count = int(count_text)
                except ValueError as exc:
                    raise ValueError(f"第 {period_no} 期人数必须是整数") from exc
                if count < 1:
                    raise ValueError(f"第 {period_no} 期人数必须为正整数")
                if count > TOTAL_USERS:
                    raise ValueError(f"第 {period_no} 期人数不能超过 {TOTAL_USERS}")
                try:
                    probability = Decimal(prob_text)
                except InvalidOperation as exc:
                    raise ValueError(f"第 {period_no} 期概率格式不正确") from exc
                if not probability.is_finite() or not (Decimal(0) < probability < Decimal(1)):
                    raise ValueError(f"第 {period_no} 期概率必须严格介于 0 与 1 之间")
                if abs(probability.as_tuple().exponent) > 12:
                    raise ValueError(f"第 {period_no} 期概率最多 12 位小数")
                counts.append(count)
                probabilities.append(probability)
            if sum(counts) != TOTAL_USERS:
                raise ValueError(
                    f"各期应到人数合计必须为 {TOTAL_USERS}，当前为 {sum(counts)}"
                )
            canonical = [stats.canonical_probability(p) for p in probabilities]
        except (ValueError, InvalidOperation) as exc:
            flash(str(exc), "error")
            return (
                render_template(
                    "plan.html",
                    periods=periods or 4,
                    existing=existing,
                    locked=locked,
                    form=request.form,
                    max_periods=MAX_PERIODS,
                    total_users=TOTAL_USERS,
                ),
                400,
            )

        try:
            db.execute("BEGIN IMMEDIATE")
            if plan_is_locked(db):
                _rollback(db)
                flash("计划已被其他操作锁定，请刷新后重试。", "error")
                return redirect(url_for("plan"))
            db.execute("DELETE FROM plan")
            db.executemany(
                "INSERT INTO plan (period, planned_users, prob_b) VALUES (?, ?, ?)",
                [
                    (index + 1, count, probability)
                    for index, (count, probability) in enumerate(zip(counts, canonical))
                ],
            )
            db.execute(
                "INSERT INTO round_meta (id, periods) VALUES (1, ?) "
                "ON CONFLICT(id) DO UPDATE SET periods = excluded.periods, "
                "configured_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (periods_value,),
            )
            db.execute("COMMIT")
        except Exception:
            _rollback(db)
            raise
        flash(f"已固定 {periods_value} 期计划，合计 {TOTAL_USERS} 人。", "success")
        return redirect(url_for("index"))

    return render_template(
        "plan.html",
        periods=periods or 4,
        existing=existing,
        locked=locked,
        form=None,
        max_periods=MAX_PERIODS,
        total_users=TOTAL_USERS,
    )


@app.post("/round/reset")
def reset_round():
    db = get_db()
    if request.form.get("confirm", "").strip() != "RESET":
        flash("重置确认口令不正确（请输入 RESET）。", "error")
        return redirect(url_for("plan"))
    try:
        db.execute("BEGIN IMMEDIATE")
        db.execute("DELETE FROM results")
        db.execute("DELETE FROM assignments")
        db.execute("DELETE FROM uploads")
        db.execute("COMMIT")
    except Exception:
        _rollback(db)
        raise
    flash("已清空本轮上传明细与核验结果，计划保留且可重新调整。", "success")
    return redirect(url_for("plan"))


@app.route("/period/<int:period_no>", methods=["GET", "POST"])
def period_detail(period_no: int):
    db = get_db()
    periods = round_periods(db)
    if periods is None:
        flash("请先固定本轮计划。", "error")
        return redirect(url_for("plan"))
    if not 1 <= period_no <= periods:
        abort(404)

    plan_row = db.execute(
        "SELECT period, planned_users, prob_b FROM plan WHERE period = ?",
        (period_no,),
    ).fetchone()
    if plan_row is None:
        abort(404)

    if request.method == "POST":
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            flash("请选择要上传的 CSV 文件。", "error")
            return redirect(url_for("period_detail", period_no=period_no))
        raw = upload.read(MAX_CSV_BYTES + 1)
        if len(raw) > MAX_CSV_BYTES:
            flash(f"CSV 超过 {MAX_CSV_BYTES // 1024} KiB 限制。", "error")
            return redirect(url_for("period_detail", period_no=period_no))
        try:
            parsed = parse_csv(raw)
            validate_upload_against_plan(parsed, int(plan_row["planned_users"]))
        except CSVError as exc:
            flash(f"上传失败：{exc}", "error")
            return redirect(url_for("period_detail", period_no=period_no))

        outcome = store_upload(period_no, parsed)
        flash(outcome.message, "success" if outcome.status in ("accepted", "duplicate") else "error")
        if outcome.status in ("error", "conflict"):
            return redirect(url_for("period_detail", period_no=period_no), code=303)
        return redirect(url_for("period_detail", period_no=period_no))

    state = round_state(db)
    upload_row = state["uploads"].get(period_no)
    result_row = state["results"].get(period_no)
    plan = state["plan"]
    distributions = plan_distributions(plan)
    distribution = distributions[period_no - 1]
    expected_b = sum(
        Decimal(str(plan[i]["planned_users"])) * Decimal(str(plan[i]["prob_b"]))
        for i in range(period_no)
    )
    variance = sum(
        Decimal(str(plan[i]["planned_users"]))
        * Decimal(str(plan[i]["prob_b"]))
        * (Decimal(1) - Decimal(str(plan[i]["prob_b"])))
        for i in range(period_no)
    )
    summary = stats.distribution_stats(distribution, expected_b, variance)

    quantiles = []
    for q in ("0.001", "0.005", "0.025", "0.05", "0.5", "0.95", "0.975", "0.995", "0.999"):
        quantiles.append(
            (q, stats.quantile(distribution, Decimal(q)))
        )

    # A compact histogram for display.  Chunk masses are summed with Decimal
    # and only the (bounded, well-conditioned) ratio to the peak becomes a
    # float, so extreme probabilities cannot underflow the chart.
    width = max(1, (len(distribution) + 19) // 20)
    chunks = []
    for start in range(0, len(distribution), width):
        chunk = distribution[start : start + width]
        chunks.append((start, start + len(chunk) - 1, sum(chunk)))
    peak_mass = max((mass for _, _, mass in chunks), default=Decimal(0))
    histogram = []
    for start, end, mass in chunks:
        ratio = mass / peak_mass if peak_mass > 0 else Decimal(0)
        histogram.append(
            {
                "start": start,
                "end": end,
                "mass": stats.format_probability(mass, digits=3),
                "height": float(ratio),
            }
        )

    return render_template(
        "period.html",
        state=state,
        period_no=period_no,
        plan_row=plan_row,
        upload=upload_row,
        result=result_row,
        summary=summary,
        quantiles=quantiles,
        histogram=histogram,
        distribution=distribution,
    )


@app.get("/api/period/<int:period_no>")
def period_api(period_no: int):
    """Machine-readable view for the data team."""
    db = get_db()
    state = round_state(db)
    if not state["configured"] or not 1 <= period_no <= state["periods"]:
        abort(404)
    plan = state["plan"]
    distributions = plan_distributions(plan)
    distribution = distributions[period_no - 1]
    result = state["results"].get(period_no)
    payload = {
        "period": period_no,
        "planned_users": int(plan[period_no - 1]["planned_users"]),
        "prob_b": plan[period_no - 1]["prob_b"],
        "threshold": str(state["threshold"]),
        "distribution": [str(value) for value in distribution],
    }
    if result is not None:
        payload.update(
            {
                "verified": True,
                "verified_users": int(result["verified_users"]),
                "observed_b": int(result["observed_b"]),
                "expected_b": result["expected_b"],
                "variance": result["variance"],
                "tail_low": result["tail_low"],
                "tail_high": result["tail_high"],
                "p_value": result["p_value"],
                "flagged": bool(result["flagged"]),
            }
        )
    else:
        payload["verified"] = False
    return jsonify(payload)


@app.errorhandler(413)
def too_large(_error):
    flash(f"上传内容超过 {MAX_CSV_BYTES // 1024} KiB 限制。", "error")
    # Never echo an arbitrary Referer header into a redirect; go to a fixed
    # page instead.
    return redirect(url_for("index"))


@app.errorhandler(404)
def not_found(_error):
    return render_template("404.html"), 404


if __name__ == "__main__":
    bootstrap()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=True)
