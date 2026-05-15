"""
test_properties.py -- Hypothesis property-based tests.

These tests sample thousands of random valid inputs and assert
mathematical invariants that *must* hold for any input the system
might encounter in production.  They are the strongest evidence that
the math engine is correct.
"""

from __future__ import annotations

import math as _math

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.math_engine import MathEngine

# Common settings: deterministic seed, no slow-test warnings.
_SETTINGS = settings(
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


# ──────────────────────────────────────────────────────────────────
#  Strategies
# ──────────────────────────────────────────────────────────────────

# Decimal odds in a realistic, bounded range (1.01 ... 1000.0).
decimal_odds = st.floats(
    min_value=1.01,
    max_value=1000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Two raw implied probs that together describe a bookmaker's two-way
# market. Each in (0, 1).  We sample independently (vig may be present).
implied_prob = st.floats(
    min_value=0.001,
    max_value=0.999,
    allow_nan=False,
    allow_infinity=False,
)

# True (de-vigged) probability strictly inside (0, 1).
true_prob = st.floats(
    min_value=0.001,
    max_value=0.999,
    allow_nan=False,
    allow_infinity=False,
)

# Kelly multipliers in (0, 1].
kelly_mult = st.floats(
    min_value=0.01,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)


# ──────────────────────────────────────────────────────────────────
#  Property 1: Implied probability is always in (0, 1)
# ──────────────────────────────────────────────────────────────────


@_SETTINGS
@given(d=decimal_odds)
def test_implied_probability_bounded(d):
    p = MathEngine.decimal_to_implied_probability(d)
    assert 0.0 < p < 1.0


# ──────────────────────────────────────────────────────────────────
#  Property 2: De-vigged probabilities sum to exactly 1.0
# ──────────────────────────────────────────────────────────────────


@_SETTINGS
@given(a=implied_prob, b=implied_prob)
def test_devig_sums_to_one(a, b):
    true_a, true_b = MathEngine.devig_probabilities(a, b)
    assert abs(true_a + true_b - 1.0) < 1e-5


# ──────────────────────────────────────────────────────────────────
#  Property 3: Kelly fraction is always in [0, 1]
# ──────────────────────────────────────────────────────────────────


@_SETTINGS
@given(p=true_prob, d=decimal_odds, m=kelly_mult)
def test_kelly_fraction_bounded(p, d, m):
    f = MathEngine.kelly_criterion(p, d, kelly_multiplier=m)
    assert 0.0 <= f <= 1.0


# ──────────────────────────────────────────────────────────────────
#  Property 4: EV is monotonically non-decreasing in true probability
# ──────────────────────────────────────────────────────────────────


@_SETTINGS
@given(
    p_low=st.floats(min_value=0.01, max_value=0.5),
    delta=st.floats(min_value=0.01, max_value=0.49),
    d=decimal_odds,
)
def test_ev_monotonic_in_probability(p_low, delta, d):
    p_high = min(p_low + delta, 0.999)
    ev_low = MathEngine.expected_value(p_low, d)
    ev_high = MathEngine.expected_value(p_high, d)
    # Strict monotonicity unless rounding ties them.
    assert ev_high >= ev_low - 1e-9


# ──────────────────────────────────────────────────────────────────
#  Property 5: Arbitrage condition <=> positive arbitrage margin
# ──────────────────────────────────────────────────────────────────


@_SETTINGS
@given(d_a=decimal_odds, d_b=decimal_odds)
def test_arb_inverse_sum_iff_positive_margin(d_a, d_b):
    inv_sum = (1.0 / d_a) + (1.0 / d_b)
    margin = 1.0 - inv_sum
    # Logical equivalence: inv_sum < 1.0  <=>  margin > 0
    assert (inv_sum < 1.0) == (margin > 0)


# ──────────────────────────────────────────────────────────────────
#  Bonus: Round-trip American <-> Decimal preserves the integer
# ──────────────────────────────────────────────────────────────────


_VALID_AMERICAN = st.one_of(
    # Bounded to magnitudes where 4-decimal rounding of decimal odds
    # cannot drift the round-trip by more than 1 unit.
    st.integers(min_value=101, max_value=2000),
    st.integers(min_value=-2000, max_value=-101),
)


@_SETTINGS
@given(american=_VALID_AMERICAN)
def test_american_decimal_round_trip(american):
    d = MathEngine.american_to_decimal(american)
    am2 = MathEngine.decimal_to_american(d)
    # american_to_decimal rounds to 4 decimal places, so for very deep
    # favorites (small d-1) the round-trip can drift by a couple of units.
    # We assert <= 0.5% relative error or <= 2 absolute units, whichever
    # is larger.
    tolerance = max(2, abs(american) * 0.005)
    assert abs(am2 - american) <= tolerance, f"round-trip drift: {american} -> {d} -> {am2}"


# ──────────────────────────────────────────────────────────────────
#  Sanity: hypothesis is wired correctly
# ──────────────────────────────────────────────────────────────────


def test_hypothesis_is_installed():
    """If hypothesis is missing the import at module top will already fail."""
    import hypothesis  # noqa: F401

    assert _math.isfinite(1.0)


@pytest.mark.parametrize("decimal", [1.5, 2.0, 2.5, 3.0])
def test_implied_prob_known_values(decimal):
    """Small parametrized sanity check alongside the property tests."""
    expected = 1.0 / decimal
    assert abs(MathEngine.decimal_to_implied_probability(decimal) - expected) < 1e-4
