"""
test_math_engine.py -- Comprehensive tests for the quantitative math engine.

Covers: odds conversion, vig/de-vig, expected value, Kelly criterion,
and arbitrage math. Minimum 20 tests as specified.
"""

from __future__ import annotations

import pytest

from src.math_engine import MathEngine


# ══════════════════════════════════════════════
#  1. Odds Conversion (7 tests)
# ══════════════════════════════════════════════


class TestAmericanToDecimal:
    """American odds → Decimal odds conversion."""

    def test_positive_odds(self, math):
        """Positive American odds: +150 → 2.50."""
        assert math.american_to_decimal(+150) == 2.5

    def test_negative_odds(self, math):
        """Negative American odds: -200 → 1.50."""
        assert math.american_to_decimal(-200) == 1.5

    def test_plus_100(self, math):
        """Edge case: +100 → 2.00 (even money)."""
        assert math.american_to_decimal(+100) == 2.0

    def test_minus_100(self, math):
        """Edge case: -100 → 2.00 (even money)."""
        assert math.american_to_decimal(-100) == 2.0

    def test_zero_raises(self, math):
        """Odds of 0 are undefined in American format."""
        with pytest.raises(ValueError, match="undefined"):
            math.american_to_decimal(0)

    def test_heavy_favorite(self, math):
        """-1000 → 1.10 (very heavy favorite)."""
        assert math.american_to_decimal(-1000) == 1.1

    def test_long_underdog(self, math):
        """+1000 → 11.00 (extreme long shot)."""
        assert math.american_to_decimal(+1000) == 11.0


class TestDecimalRoundTrip:
    """Verify that converting American → Decimal → Implied → Decimal
    produces a consistent round-trip."""

    @pytest.mark.parametrize("american", [+150, -200, +100, -100, +300, -300])
    def test_round_trip(self, math, american):
        """Convert to decimal, to implied, back to decimal — should match."""
        dec = math.american_to_decimal(american)
        imp = math.decimal_to_implied_probability(dec)
        dec_back = math.implied_probability_to_decimal(imp)
        assert abs(dec - dec_back) < 0.01, (
            f"Round-trip failed for {american}: {dec} → {imp} → {dec_back}"
        )


class TestImpliedProbability:
    """Implied probability from decimal / American odds."""

    def test_even_money(self, math):
        """Decimal 2.00 → 50% implied probability."""
        assert math.decimal_to_implied_probability(2.0) == 0.5

    def test_heavy_favorite(self, math):
        """Decimal 1.25 → 80% implied probability."""
        assert math.decimal_to_implied_probability(1.25) == 0.8

    def test_invalid_decimal_raises(self, math):
        """Decimal ≤ 1.0 is not a valid price."""
        with pytest.raises(ValueError):
            math.decimal_to_implied_probability(0.5)

    def test_invalid_prob_to_decimal(self, math):
        """Probability outside (0, 1) should raise."""
        with pytest.raises(ValueError):
            math.implied_probability_to_decimal(1.5)
        with pytest.raises(ValueError):
            math.implied_probability_to_decimal(0.0)


# ══════════════════════════════════════════════
#  2. Vig Calculation & De-Vigging (5 tests)
# ══════════════════════════════════════════════


class TestVigAndDevig:
    """Vig calculation and multiplicative normalization."""

    def test_standard_vig(self, math):
        """-110/-110 line has ~4.76% vig."""
        imp_a = math.decimal_to_implied_probability(
            math.american_to_decimal(-110)
        )
        imp_b = math.decimal_to_implied_probability(
            math.american_to_decimal(-110)
        )
        vig = math.calculate_vig(imp_a, imp_b)
        assert 0.045 < vig < 0.05, f"Expected ~4.76% vig, got {vig}"

    def test_devig_sums_to_one(self, math):
        """De-vigged probabilities must sum to exactly 1.0."""
        imp_a = math.decimal_to_implied_probability(
            math.american_to_decimal(-150)
        )
        imp_b = math.decimal_to_implied_probability(
            math.american_to_decimal(+130)
        )
        true_a, true_b = math.devig_probabilities(imp_a, imp_b)
        assert abs(true_a + true_b - 1.0) < 1e-6

    def test_devig_less_than_raw(self, math):
        """De-vigged probs should be less than raw implied probs
        (because vig has been removed)."""
        imp_a = 0.5263  # -110
        imp_b = 0.5263  # -110
        true_a, true_b = math.devig_probabilities(imp_a, imp_b)
        assert true_a < imp_a, "De-vigged should be < raw (vig removed)"
        assert true_b < imp_b

    def test_pinnacle_lower_margin(self, math):
        """Pinnacle has lower vig than soft books.
        Pinnacle -155/+135 should have lower vig than DK -170/+140."""
        pin_a = math.decimal_to_implied_probability(
            math.american_to_decimal(-155)
        )
        pin_b = math.decimal_to_implied_probability(
            math.american_to_decimal(+135)
        )
        dk_a = math.decimal_to_implied_probability(
            math.american_to_decimal(-170)
        )
        dk_b = math.decimal_to_implied_probability(
            math.american_to_decimal(+140)
        )
        pin_vig = math.calculate_vig(pin_a, pin_b)
        dk_vig = math.calculate_vig(dk_a, dk_b)
        assert pin_vig < dk_vig, (
            f"Pinnacle vig ({pin_vig}) should be < DK vig ({dk_vig})"
        )

    def test_devig_zero_prob_raises(self, math):
        """De-vig with zero probability should raise ValueError."""
        with pytest.raises(ValueError):
            math.devig_probabilities(0.0, 0.5)


# ══════════════════════════════════════════════
#  3. Expected Value (4 tests)
# ══════════════════════════════════════════════


class TestExpectedValue:
    """EV% calculation: true_prob × decimal_odds − 1."""

    def test_positive_ev(self, math):
        """When true_prob > implied_prob → positive EV."""
        ev = math.expected_value(0.55, 2.10)
        assert ev > 0, f"Expected positive EV, got {ev}"
        assert abs(ev - 0.155) < 0.001  # 15.5%

    def test_negative_ev(self, math):
        """When true_prob < implied_prob → negative EV."""
        ev = math.expected_value(0.40, 2.10)
        assert ev < 0, f"Expected negative EV, got {ev}"

    def test_zero_ev_at_breakeven(self, math):
        """When true_prob == implied_prob → zero EV."""
        # Decimal 2.0 implies 50% probability
        ev = math.expected_value(0.5, 2.0)
        assert abs(ev) < 1e-6, f"Expected ~0 EV at breakeven, got {ev}"

    def test_ev_scales_with_prob(self, math):
        """Higher true probability at same odds → higher EV."""
        ev_low = math.expected_value(0.50, 2.10)
        ev_high = math.expected_value(0.60, 2.10)
        assert ev_high > ev_low


# ══════════════════════════════════════════════
#  4. Kelly Criterion (8 tests)
# ══════════════════════════════════════════════


class TestKellyCriterion:
    """Optimal bet sizing via Kelly formula."""

    def test_full_kelly_formula(self, math, sample_kelly_inputs):
        """Verify full Kelly matches the known formula result."""
        inp = sample_kelly_inputs
        frac = math.kelly_criterion(
            inp["true_probability"], inp["decimal_odds"],
            kelly_multiplier=1.0,
        )
        assert abs(frac - inp["full_kelly_expected"]) < 0.001

    def test_half_kelly_exact_ratio(self, math, sample_kelly_inputs):
        """Half Kelly = exactly 50% of full Kelly."""
        inp = sample_kelly_inputs
        full = math.kelly_criterion(
            inp["true_probability"], inp["decimal_odds"],
            kelly_multiplier=1.0,
        )
        half = math.kelly_criterion(
            inp["true_probability"], inp["decimal_odds"],
            kelly_multiplier=0.5,
        )
        assert abs(half - full * 0.5) < 0.001

    def test_quarter_kelly_exact_ratio(self, math, sample_kelly_inputs):
        """Quarter Kelly = exactly 25% of full Kelly."""
        inp = sample_kelly_inputs
        full = math.kelly_criterion(
            inp["true_probability"], inp["decimal_odds"],
            kelly_multiplier=1.0,
        )
        quarter = math.kelly_criterion(
            inp["true_probability"], inp["decimal_odds"],
            kelly_multiplier=0.25,
        )
        assert abs(quarter - full * 0.25) < 0.001

    def test_negative_edge_returns_zero(self, math):
        """Negative edge → Kelly returns 0 (never bet negative EV)."""
        frac = math.kelly_criterion(0.30, 2.10, kelly_multiplier=1.0)
        assert frac == 0.0

    def test_kelly_never_exceeds_one(self, math):
        """Kelly fraction should never exceed 1.0 (100% of bankroll)."""
        # Even with extreme edge (99% prob, 10x odds)
        frac = math.kelly_criterion(0.99, 10.0, kelly_multiplier=1.0)
        assert frac <= 1.0

    def test_kelly_bet_size_dollars(self, math):
        """Dollar bet size = bankroll × Kelly fraction."""
        bankroll = 10000
        bet = math.kelly_bet_size(bankroll, 0.55, 2.10, kelly_multiplier=0.25)
        frac = math.kelly_criterion(0.55, 2.10, kelly_multiplier=0.25)
        assert abs(bet - bankroll * frac) < 0.01

    def test_kelly_invalid_prob_raises(self, math):
        """Probability outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError):
            math.kelly_criterion(0.0, 2.10)
        with pytest.raises(ValueError):
            math.kelly_criterion(1.0, 2.10)

    def test_kelly_invalid_multiplier_raises(self, math):
        """Kelly multiplier outside (0, 1] should raise ValueError."""
        with pytest.raises(ValueError):
            math.kelly_criterion(0.55, 2.10, kelly_multiplier=0.0)
        with pytest.raises(ValueError):
            math.kelly_criterion(0.55, 2.10, kelly_multiplier=1.5)


# ══════════════════════════════════════════════
#  5. Arbitrage Math (3 tests)
# ══════════════════════════════════════════════


class TestArbitrageMath:
    """Mathematical foundation for arbitrage detection."""

    def test_arb_exists_when_inverse_sum_below_one(self, math):
        """Arb exists when 1/dec_a + 1/dec_b < 1.0."""
        # Book A: +140 (2.40), Book B: +120 (2.20)
        dec_a = math.american_to_decimal(+140)
        dec_b = math.american_to_decimal(+120)
        inv_sum = (1.0 / dec_a) + (1.0 / dec_b)
        assert inv_sum < 1.0, f"Should be arb: inv_sum={inv_sum}"

    def test_no_arb_when_inverse_sum_above_one(self, math):
        """No arb when 1/dec_a + 1/dec_b >= 1.0."""
        # -110/-110 → each is ~1.909 → inv_sum ≈ 1.048 > 1.0
        dec_a = math.american_to_decimal(-110)
        dec_b = math.american_to_decimal(-110)
        inv_sum = (1.0 / dec_a) + (1.0 / dec_b)
        assert inv_sum > 1.0, f"Should NOT be arb: inv_sum={inv_sum}"

    def test_arb_profit_percentage(self, math):
        """Verify arb profit margin calculation.
        Profit % = 1 - (1/dec_a + 1/dec_b)."""
        dec_a = math.american_to_decimal(+140)  # 2.40
        dec_b = math.american_to_decimal(+120)  # 2.20
        inv_sum = (1.0 / dec_a) + (1.0 / dec_b)
        profit_pct = 1.0 - inv_sum
        assert profit_pct > 0.10, f"Expected >10% margin, got {profit_pct}"
