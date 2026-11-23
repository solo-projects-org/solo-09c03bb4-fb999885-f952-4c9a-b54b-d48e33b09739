"""Statistical helpers for the split-verification service.

All inferential arithmetic is done with :class:`decimal.Decimal` so the
reported p-values do not depend on binary floating point; floats are only used
for the relative bar heights of the distribution chart.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from typing import Sequence

PRECISION = 80
ONE = Decimal(1)
ZERO = Decimal(0)


def canonical_probability(value: str | Decimal) -> str:
    """Return a canonical, human-readable decimal string for a probability."""
    dec = Decimal(str(value).strip())
    if not (ZERO < dec < ONE):
        raise ValueError("probability must be strictly between 0 and 1")
    text = format(dec.normalize(), "f")
    return text


def binomial_pmf(n: int, p: Decimal) -> tuple[Decimal, ...]:
    """Probability mass function of Binomial(n, p) as an exact-ish Decimal tuple."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if p < 0 or p > 1:
        raise ValueError("p must be in [0, 1]")
    if p == 0:
        return (ONE,) + (ZERO,) * n
    if p == 1:
        return (ZERO,) * n + (ONE,)

    with localcontext() as ctx:
        ctx.prec = PRECISION
        q = ONE - p
        ratio = p / q
        pmf = [ZERO] * (n + 1)
        pmf[0] = q**n
        for b in range(n):
            pmf[b + 1] = pmf[b] * Decimal(n - b) / Decimal(b + 1) * ratio
        total = sum(pmf)
        # Round-off from the recurrence is tiny, but normalising keeps the
        # distribution interpretable and stable for tail sums.
        return tuple(value / total for value in pmf)


def convolve(a: Sequence[Decimal], b: Sequence[Decimal]) -> tuple[Decimal, ...]:
    """Convolution of two discrete distributions."""
    if not a or not b:
        raise ValueError("distributions must be non-empty")
    with localcontext() as ctx:
        ctx.prec = PRECISION
        out = [ZERO] * (len(a) + len(b) - 1)
        for i, ai in enumerate(a):
            if ai == 0:
                continue
            for j, bj in enumerate(b):
                if bj == 0:
                    continue
                out[i + j] += ai * bj
        return tuple(out)


def all_prefix_distributions(
    plan: Sequence[tuple[int, Decimal]],
) -> tuple[tuple[Decimal, ...], ...]:
    """Cumulative-B null distributions for every prefix of the plan.

    ``plan`` is a sequence of ``(planned_users, prob_b)`` in period order.
    Element ``k`` of the result is the distribution of the cumulative number
    of B users over periods 1..k+1, i.e. a convolution of the corresponding
    independent binomials.
    """
    result: list[tuple[Decimal, ...]] = []
    dist: tuple[Decimal, ...] = (ONE,)
    for n, p in plan:
        dist = convolve(dist, binomial_pmf(n, p))
        result.append(dist)
    return tuple(result)


def two_sided_p_value(distribution: Sequence[Decimal], observed: int) -> Decimal:
    """Exact two-sided p-value for a discrete null distribution.

    The p-value is twice the smaller of ``P(X <= observed)`` and
    ``P(X >= observed)`` (both tails include the observed value), capped at 1.
    """
    if observed < 0 or observed >= len(distribution):
        # A cumulative count outside the support is still meaningful; clamp so
        # that callers get 0 / 1 rather than an indexing error.
        if observed < 0:
            return ONE
        return ZERO
    with localcontext() as ctx:
        ctx.prec = PRECISION
        lower = sum(distribution[: observed + 1])
        upper = sum(distribution[observed:])
        value = 2 * min(lower, upper)
        return min(value, ONE)


def tail_probabilities(
    distribution: Sequence[Decimal], observed: int
) -> tuple[Decimal, Decimal]:
    """Return ``(P(X <= observed), P(X >= observed))``."""
    if observed < 0:
        return ZERO, ONE
    if observed >= len(distribution):
        return ONE, ZERO
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return sum(distribution[: observed + 1]), sum(distribution[observed:])


def quantile(distribution: Sequence[Decimal], q: Decimal) -> int:
    """Smallest index whose cumulative probability is at least ``q``."""
    if not (ZERO <= q <= ONE):
        raise ValueError("q must be in [0, 1]")
    with localcontext() as ctx:
        ctx.prec = PRECISION
        cumulative = ZERO
        for index, probability in enumerate(distribution):
            cumulative += probability
            if cumulative >= q:
                return index
    return len(distribution) - 1


def distribution_stats(
    distribution: Sequence[Decimal], expected_b: Decimal, variance: Decimal
) -> dict[str, object]:
    """Summary statistics used by the distribution page."""
    total = Decimal(len(distribution) - 1)
    lower_tail = (ONE - Decimal("0.95")) / 2
    upper_tail = ONE - lower_tail
    lower_index = quantile(distribution, lower_tail)
    upper_index = quantile(distribution, upper_tail)
    mode_index = max(
        range(len(distribution)), key=lambda i: distribution[i]
    )
    # Interpolate the mean/sd rather than recomputing from the distribution;
    # both are exact functions of the plan.
    with localcontext() as ctx:
        ctx.prec = PRECISION
        sd = variance.sqrt()
    return {
        "support_max": int(total),
        "mean": expected_b,
        "variance": variance,
        "sd": sd,
        "ci_low": lower_index,
        "ci_high": upper_index,
        "mode": mode_index,
        "mode_probability": distribution[mode_index],
    }


def _significant(value: Decimal, digits: int = 6) -> str:
    """Format a Decimal compactly without losing tiny p-values."""
    if value == 0:
        return "0"
    if value == ONE:
        return "1"
    text = format(value, f".{digits}g")
    # Decimal renders very small values like ``1e-17`` / ``1.2e-7``; keep that
    # style but make exponent casing consistent.
    return text.replace("E", "e")


def _trim_mantissa(mantissa: str) -> str:
    if "." not in mantissa:
        return mantissa
    trimmed = mantissa.rstrip("0").rstrip(".")
    return trimmed or "0"


def format_probability(value: Decimal, digits: int = 6) -> str:
    """Human-friendly probability formatting (scientific for tiny values)."""
    if value == 0:
        return "0"
    if value == ONE:
        return "1"
    if value < Decimal("0.0001"):
        text = format(value, f".{digits - 1}e")
        mantissa, exponent = text.split("e")
        return f"{_trim_mantissa(mantissa)}e{int(exponent)}"
    text = _significant(value, digits)
    if Decimal(text) >= ONE:
        # Values just below 1 must not be rounded up to a misleading "1"; the
        # ellipsis marks that the true value is strictly less than the shown
        # digits would suggest.
        floor = ONE - Decimal(10) ** -(digits - 1)
        return f"{format(floor, 'f')}…"
    return text


def format_expected(value: Decimal) -> str:
    with localcontext() as ctx:
        ctx.prec = PRECISION
        return format(value, ".2f")
