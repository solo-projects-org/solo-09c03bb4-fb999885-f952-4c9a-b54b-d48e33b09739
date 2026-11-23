"""Pytest suite for the split-verification service."""

from __future__ import annotations

import io
import sqlite3
import threading
from decimal import Decimal, localcontext

import pytest

import app as app_module
import stats


def exact_sum(values):
    with localcontext() as ctx:
        ctx.prec = 80
        return sum(values)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    db_path = str(tmp_path / "test.sqlite3")
    application = app_module.app
    application.config.update(
        TESTING=True,
        DB_PATH=db_path,
        SECRET_KEY="test-secret",
    )
    app_module.bootstrap(db_path)
    with application.test_client() as test_client:
        yield test_client
    app_module._schema_ready_paths.discard(db_path)


def make_csv(rows):
    out = io.StringIO()
    out.write("user_id,group\n")
    for user_id, group in rows:
        out.write(f"{user_id},{group}\n")
    return out.getvalue().encode("utf-8")


def upload(client, period, rows):
    return client.post(
        f"/period/{period}",
        data={"file": (io.BytesIO(make_csv(rows)), f"p{period}.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )


def configure(client, counts, probs):
    data = {"periods": str(len(counts))}
    for i, (count, prob) in enumerate(zip(counts, probs), start=1):
        data[f"count_{i}"] = str(count)
        data[f"prob_{i}"] = str(prob)
    return client.post("/plan", data=data, follow_redirects=True)


def rows_for(period, count, b_count, prefix="u"):
    rows = []
    for i in range(count):
        rows.append((f"{prefix}{period}-{i}", "B" if i < b_count else "A"))
    return rows


def fetch(application, sql, params=()):
    conn = sqlite3.connect(application.config["DB_PATH"])
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_binomial_pmf_sums_to_one():
    pmf = stats.binomial_pmf(40, Decimal("0.3"))
    assert len(pmf) == 41
    assert abs(exact_sum(pmf) - Decimal(1)) < Decimal("1e-50")
    with localcontext() as ctx:
        ctx.prec = 80
        exact_first = (Decimal(1) - Decimal("0.3")) ** 40
    assert abs(pmf[0] - exact_first) < Decimal("1e-60")


def test_convolution_matches_known_binomial_identity():
    # Binomial(10, p) * Binomial(10, p) == Binomial(20, p)
    p = Decimal("0.4")
    single = stats.binomial_pmf(10, p)
    doubled = stats.convolve(single, single)
    expected = stats.binomial_pmf(20, p)
    for a, b in zip(doubled, expected):
        assert abs(a - b) < Decimal("1e-40")


def test_two_sided_p_value_symmetric_and_capped():
    pmf = stats.binomial_pmf(20, Decimal("0.5"))
    # Centre of a symmetric distribution yields p == 1 (capped).
    assert stats.two_sided_p_value(pmf, 10) == Decimal(1)
    low = stats.two_sided_p_value(pmf, 2)
    high = stats.two_sided_p_value(pmf, 18)
    assert abs(low - high) < Decimal("1e-40")
    # Hand-check: 2 * P(X <= 2) for Binomial(20, .5).
    expected = 2 * sum(pmf[:3])
    assert abs(low - expected) < Decimal("1e-40")


def test_two_sided_p_value_includes_observed_value():
    # Asymmetric distribution: p = 2*min(P(X<=k), P(X>=k)), both inclusive.
    pmf = stats.binomial_pmf(10, Decimal("0.2"))
    k = 4
    expected = 2 * min(sum(pmf[: k + 1]), sum(pmf[k:]))
    assert stats.two_sided_p_value(pmf, k) == min(expected, Decimal(1))


def test_prefix_distributions_track_cumulative_moments():
    plan = [(100, Decimal("0.5")), (200, Decimal("0.3"))]
    dists = stats.all_prefix_distributions(plan)
    assert len(dists[0]) == 101
    assert len(dists[1]) == 301
    mean = exact_sum(i * p for i, p in enumerate(dists[1]))
    assert abs(mean - Decimal("110")) < Decimal("1e-30")


def test_format_tiny_probability():
    assert stats.format_probability(Decimal("1e-30")) == "1e-30"
    assert stats.format_probability(Decimal("0.0125")) == "0.0125"
    assert stats.format_probability(Decimal("1")) == "1"


# ---------------------------------------------------------------------------
# Plan configuration
# ---------------------------------------------------------------------------


def test_plan_requires_total_1000(client):
    resp = configure(client, [300, 300], ["0.5", "0.5"])
    assert resp.status_code == 400
    assert "1000" in resp.get_data(as_text=True)
    assert fetch(app_module.app, "SELECT * FROM plan") == []


def test_plan_rejects_more_than_8_periods(client):
    resp = configure(client, [125] * 9, ["0.5"] * 9)
    assert resp.status_code == 400


def test_plan_rejects_probability_out_of_range(client):
    resp = configure(client, [500, 500], ["0", "0.5"])
    assert resp.status_code == 400
    assert "概率" in resp.get_data(as_text=True)


def test_plan_is_stored_and_locked_after_upload(client):
    configure(client, [400, 600], ["0.5", "0.4"])
    upload(client, 1, rows_for(1, 400, 200))
    resp = client.get("/plan")
    text = resp.get_data(as_text=True)
    assert "计划已锁定" in text
    locked = configure(client, [500, 500], ["0.5", "0.5"])
    assert "锁定" in locked.get_data(as_text=True)
    plan = fetch(app_module.app, "SELECT * FROM plan ORDER BY period")
    assert [r["planned_users"] for r in plan] == [400, 600]


def test_reset_allows_reconfiguring_probabilities(client):
    configure(client, [500, 500], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 500, 250))
    client.post("/round/reset", data={"confirm": "RESET"}, follow_redirects=True)
    assert fetch(app_module.app, "SELECT * FROM uploads") == []
    assert fetch(app_module.app, "SELECT * FROM results") == []
    configure(client, [500, 500], ["0.6", "0.4"])
    probs = [r["prob_b"] for r in fetch(app_module.app, "SELECT prob_b FROM plan ORDER BY period")]
    assert probs == ["0.6", "0.4"]


def test_reset_requires_confirmation(client):
    configure(client, [500, 500], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 500, 250))
    client.post("/round/reset", data={"confirm": "nope"}, follow_redirects=True)
    assert len(fetch(app_module.app, "SELECT * FROM uploads")) == 1


def test_probability_change_changes_p_value_for_same_data(client):
    rows = [(f"u{i}", "B" if i < 150 else "A") for i in range(250)]
    configure(client, [250, 750], ["0.6", "0.4"])
    upload(client, 1, rows)
    p_06 = Decimal(client.get("/api/period/1").get_json()["p_value"])
    client.post("/round/reset", data={"confirm": "RESET"}, follow_redirects=True)
    configure(client, [250, 750], ["0.4", "0.6"])
    upload(client, 1, rows)
    p_04 = Decimal(client.get("/api/period/1").get_json()["p_value"])
    assert p_06 > Decimal("0.5")
    assert p_04 < Decimal("0.0001")


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------


def test_upload_rejects_wrong_headcount(client):
    configure(client, [400, 600], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 399, 200))
    assert fetch(app_module.app, "SELECT * FROM uploads") == []
    assert fetch(app_module.app, "SELECT * FROM results") == []


def test_upload_rejects_bad_group(client):
    configure(client, [10, 990], ["0.5", "0.5"])
    resp = upload(client, 1, [(f"u{i}", "C") for i in range(10)])
    assert "组别必须为 A 或 B" in resp.get_data(as_text=True)
    assert fetch(app_module.app, "SELECT * FROM uploads") == []


def test_upload_rejects_duplicate_ids_within_file(client):
    configure(client, [3, 997], ["0.5", "0.5"])
    resp = upload(client, 1, [("u1", "A"), ("u1", "B"), ("u2", "A")])
    assert "重复" in resp.get_data(as_text=True)


def test_upload_rejects_user_seen_in_other_period(client):
    configure(client, [3, 997], ["0.5", "0.5"])
    upload(client, 1, [("u1", "A"), ("u2", "A"), ("u3", "A")])
    resp = upload(client, 2, [("u2", "B")] + [(f"v{i}", "A") for i in range(996)])
    assert "只计一次" in resp.get_data(as_text=True)
    assert fetch(app_module.app, "SELECT period, COUNT(*) c FROM assignments GROUP BY period")[0]["c"] == 3


def test_upload_rejects_missing_columns(client):
    configure(client, [2, 998], ["0.5", "0.5"])
    raw = b"id,grp\nu1,A\nu2,B\n"
    resp = client.post(
        "/period/1",
        data={"file": (io.BytesIO(raw), "bad.csv")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "表头" in resp.get_data(as_text=True)


# ---------------------------------------------------------------------------
# Ordering / verification
# ---------------------------------------------------------------------------


def test_out_of_order_upload_waits_then_fills(client):
    configure(client, [300, 300, 400], ["0.5", "0.4", "0.6"])
    upload(client, 2, rows_for(2, 300, 120))
    results = fetch(app_module.app, "SELECT * FROM results")
    assert results == []

    upload(client, 1, rows_for(1, 300, 150))
    results = fetch(app_module.app, "SELECT * FROM results ORDER BY period")
    assert [r["period"] for r in results] == [1, 2]
    assert results[0]["observed_b"] == 150
    assert results[1]["observed_b"] == 270
    assert results[1]["verified_users"] == 600


def test_gap_prevents_verification_of_later_periods(client):
    configure(client, [250, 250, 250, 250], ["0.5"] * 4)
    upload(client, 1, rows_for(1, 250, 125))
    upload(client, 3, rows_for(3, 250, 125))
    upload(client, 4, rows_for(4, 250, 125))
    results = fetch(app_module.app, "SELECT period FROM results ORDER BY period")
    assert [r["period"] for r in results] == [1]
    upload(client, 2, rows_for(2, 250, 125))
    results = fetch(app_module.app, "SELECT period FROM results ORDER BY period")
    assert [r["period"] for r in results] == [1, 2, 3, 4]


def test_results_use_plan_probabilities_of_each_prefix(client):
    # Period 2 has a very different probability: the null distribution must
    # switch, not reuse period 1's probability.
    configure(client, [500, 500], ["0.5", "0.1"])
    upload(client, 1, rows_for(1, 500, 250))
    upload(client, 2, rows_for(2, 500, 300))
    result = fetch(app_module.app, "SELECT * FROM results WHERE period = 2")[0]
    assert abs(Decimal(result["expected_b"]) - Decimal("300")) < Decimal("1e-20")
    dist = stats.all_prefix_distributions(
        [(500, Decimal("0.5")), (500, Decimal("0.1"))]
    )[1]
    expected = stats.two_sided_p_value(dist, 550)
    assert Decimal(result["p_value"]) == expected


# ---------------------------------------------------------------------------
# Duplicate / conflict handling
# ---------------------------------------------------------------------------


def test_identical_reupload_is_accepted_once(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    rows = rows_for(1, 100, 50)
    first = upload(client, 1, rows)
    assert "已接收" in first.get_data(as_text=True)
    second = upload(client, 1, rows)
    text = second.get_data(as_text=True)
    assert "相同明细已接收" in text
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 1
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 100
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM results")[0]["c"] == 1


def test_changed_reupload_is_rejected_and_keeps_results(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 100, 50))
    before = fetch(app_module.app, "SELECT * FROM results")[0]
    changed = rows_for(1, 100, 50)
    changed[0] = (changed[0][0], "B")  # flip one A -> B
    resp = upload(client, 1, changed)
    assert "拒收" in resp.get_data(as_text=True)
    after = fetch(app_module.app, "SELECT * FROM results")[0]
    assert after["observed_b"] == before["observed_b"] == 50
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 1


def test_failed_upload_preserves_existing_results(client):
    configure(client, [200, 300, 500], ["0.5", "0.4", "0.6"])
    upload(client, 1, rows_for(1, 200, 100))
    before = fetch(app_module.app, "SELECT * FROM results")
    assert len(before) == 1

    # Wrong headcount for period 2 -> rejected, period 1 results untouched.
    upload(client, 2, rows_for(2, 299, 100))
    after = fetch(app_module.app, "SELECT * FROM results")
    assert len(after) == 1
    assert after[0]["observed_b"] == before[0]["observed_b"]
    assert fetch(app_module.app, "SELECT COUNT(*) c FROM uploads")[0]["c"] == 1


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------


def test_flag_threshold_is_bonferroni(client):
    configure(client, [250] * 4, ["0.5"] * 4)
    # 250 users each period; period 1 with all B is extreme.
    upload(client, 1, rows_for(1, 250, 250))
    result = fetch(app_module.app, "SELECT * FROM results WHERE period = 1")[0]
    assert Decimal(result["threshold"]) == Decimal("0.0125")
    assert result["flagged"] == 1
    assert Decimal(result["p_value"]) <= Decimal("0.0125")


def test_earliest_flag_displayed(client):
    configure(client, [500, 500], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 500, 250))  # normal
    upload(client, 2, rows_for(2, 500, 500))  # extreme
    text = client.get("/").get_data(as_text=True)
    assert "最早触发分流异常的期次" in text
    assert "第 2 期" in text


def test_first_period_flag(client):
    configure(client, [500, 500], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 500, 0))
    text = client.get("/").get_data(as_text=True)
    assert "第 1 期" in text


def test_no_flag_when_observed_is_typical(client):
    configure(client, [500, 500], ["0.5", "0.5"])
    upload(client, 1, rows_for(1, 500, 251))
    result = fetch(app_module.app, "SELECT * FROM results WHERE period = 1")[0]
    assert result["flagged"] == 0
    text = client.get("/").get_data(as_text=True)
    assert "没有期次的 p 值低于阈值" in text


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_identical_uploads_yield_one_result(tmp_path):
    db_path = str(tmp_path / "concurrent.sqlite3")
    application = app_module.app
    application.config.update(TESTING=True, DB_PATH=db_path)
    app_module.bootstrap(db_path)

    with application.test_client() as test_client:
        configure(test_client, [200, 800], ["0.5", "0.5"])

    rows = rows_for(1, 200, 100)
    payload = make_csv(rows)
    barrier = threading.Barrier(4)
    outcomes = []
    lock = threading.Lock()

    def worker():
        with application.test_client() as worker_client:
            barrier.wait()
            resp = worker_client.post(
                "/period/1",
                data={"file": (io.BytesIO(payload), "p1.csv")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )
            with lock:
                outcomes.append(resp.get_data(as_text=True))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(fetch(application, "SELECT * FROM uploads")) == 1
    assert len(fetch(application, "SELECT * FROM results")) == 1
    assert fetch(application, "SELECT COUNT(*) c FROM assignments")[0]["c"] == 200
    assert len(outcomes) == 4
    app_module._schema_ready_paths.discard(db_path)


def test_concurrent_conflicting_uploads_keep_one(tmp_path):
    db_path = str(tmp_path / "concurrent2.sqlite3")
    application = app_module.app
    application.config.update(TESTING=True, DB_PATH=db_path)
    app_module.bootstrap(db_path)

    with application.test_client() as test_client:
        configure(test_client, [50, 950], ["0.5", "0.5"])

    payload_a = make_csv(rows_for(1, 50, 10, prefix="a"))
    payload_b = make_csv(rows_for(1, 50, 40, prefix="b"))
    barrier = threading.Barrier(2)

    def worker(payload):
        with application.test_client() as worker_client:
            barrier.wait()
            worker_client.post(
                "/period/1",
                data={"file": (io.BytesIO(payload), "p1.csv")},
                content_type="multipart/form-data",
                follow_redirects=True,
            )

    threads = [threading.Thread(target=worker, args=(payload,)) for payload in (payload_a, payload_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    uploads = fetch(application, "SELECT * FROM uploads")
    results = fetch(application, "SELECT * FROM results")
    assert len(uploads) == 1
    assert len(results) == 1
    assert results[0]["observed_b"] in (10, 40)
    app_module._schema_ready_paths.discard(db_path)


# ---------------------------------------------------------------------------
# API + pages
# ---------------------------------------------------------------------------


def test_period_api_exposes_distribution(client):
    configure(client, [100, 900], ["0.3", "0.5"])
    upload(client, 1, rows_for(1, 100, 30))
    payload = client.get("/api/period/1").get_json()
    assert payload["verified"] is True
    assert payload["planned_users"] == 100
    assert payload["prob_b"] == "0.3"
    assert len(payload["distribution"]) == 101
    assert payload["observed_b"] == 30


def test_missing_period_404(client):
    configure(client, [100, 900], ["0.5", "0.5"])
    assert client.get("/period/3").status_code == 404


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True}
