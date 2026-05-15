"""
math_engine.py — Quantitative Math Engine

Core quantitative primitives for sports betting analysis:
    - Odds format conversion (American ↔ Decimal ↔ Implied Probability)
    - Vig calculation & removal (de-vigging)
    - Expected Value (+EV) computation
    - Kelly Criterion bet sizing

All methods are stateless and use strict type hints.
"""

from __future__ import annotations

from functools import lru_cache


class MathEngine:
    """Stateless calculator for sports betting quantitative analysis.

    Design rationale
    ----------------
    Every method is a ``@staticmethod`` — no instance state is needed.
    This keeps the class lightweight and easy to test in isolation.
    """

    # ──────────────────────────────────────────────
    #  1. Odds Conversion
    # ──────────────────────────────────────────────

    @staticmethod
    @lru_cache(maxsize=512)
    def american_to_decimal(american_odds: int) -> float:
        """Convert American (moneyline) odds to Decimal format.

        Parameters
        ----------
        american_odds : int
            Positive (+150) or negative (-200) American odds.

        Returns
        -------
        float
            Equivalent decimal odds (always > 1.0).

        Raises
        ------
        ValueError
            If *american_odds* is exactly 0 (undefined in American format).

        Examples
        --------
        >>> MathEngine.american_to_decimal(+150)
        2.5
        >>> MathEngine.american_to_decimal(-200)
        1.5
        """
        if american_odds == 0:
            raise ValueError("American odds of 0 are undefined.")

        if american_odds > 0:
            return round((american_odds / 100) + 1, 4)
        return round((100 / abs(american_odds)) + 1, 4)

    @staticmethod
    @lru_cache(maxsize=512)
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert Decimal odds back to American (moneyline) format.

        Inverse of :meth:`american_to_decimal`. Useful for displaying
        fair-value lines in their natural format.

        Conventions
        -----------
        * ``decimal_odds == 2.0`` returns ``+100`` (even money convention).
        * ``decimal_odds > 2.0`` returns positive American odds.
        * ``1.0 < decimal_odds < 2.0`` returns negative American odds.

        Raises
        ------
        ValueError
            If ``decimal_odds <= 1.0`` (impossible price -- implies free money).
        """
        if decimal_odds <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
        if decimal_odds >= 2.0:
            return int(round((decimal_odds - 1.0) * 100))
        return int(round(-100.0 / (decimal_odds - 1.0)))

    @staticmethod
    @lru_cache(maxsize=512)
    def decimal_to_implied_probability(decimal_odds: float) -> float:
        """Convert Decimal odds to an implied probability.

        Parameters
        ----------
        decimal_odds : float
            Decimal odds (must be > 1.0).

        Returns
        -------
        float
            Implied probability in [0, 1).

        Raises
        ------
        ValueError
            If *decimal_odds* ≤ 1.0 (not a valid price).
        """
        if decimal_odds <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
        return round(1.0 / decimal_odds, 6)

    @staticmethod
    @lru_cache(maxsize=512)
    def implied_probability_to_decimal(probability: float) -> float:
        """Convert a probability back to Decimal odds.

        Parameters
        ----------
        probability : float
            Probability in (0, 1).

        Returns
        -------
        float
            Decimal odds ≥ 1.0.

        Raises
        ------
        ValueError
            If *probability* is not in (0, 1).
        """
        if not (0.0 < probability < 1.0):
            raise ValueError(f"Probability must be in (0, 1), got {probability}")
        return round(1.0 / probability, 4)

    # ──────────────────────────────────────────────
    #  2. Vig Calculation & Removal (De-Vigging)
    # ──────────────────────────────────────────────

    @staticmethod
    @lru_cache(maxsize=1024)
    def calculate_vig(implied_prob_a: float, implied_prob_b: float) -> float:
        """Calculate the bookmaker's vig (overround / juice).

        The vig is the amount by which the sum of implied probabilities
        exceeds 1.0 — i.e. the book's built-in margin.

        Parameters
        ----------
        implied_prob_a : float
            Implied probability of outcome A (e.g. Over).
        implied_prob_b : float
            Implied probability of outcome B (e.g. Under).

        Returns
        -------
        float
            Vig as a percentage (e.g. 0.05 = 5% overround).

        Examples
        --------
        >>> MathEngine.calculate_vig(0.5263, 0.5263)  # -110 / -110
        0.0526
        """
        return round((implied_prob_a + implied_prob_b) - 1.0, 6)

    @staticmethod
    @lru_cache(maxsize=1024)
    def devig_probabilities(implied_prob_a: float, implied_prob_b: float) -> tuple[float, float]:
        """Remove vig via multiplicative normalization.

        Normalizes two implied probabilities so they sum to exactly 1.0,
        producing the "true" (fair) probability for each side.

        Parameters
        ----------
        implied_prob_a : float
            Raw implied probability of outcome A.
        implied_prob_b : float
            Raw implied probability of outcome B.

        Returns
        -------
        tuple[float, float]
            (true_prob_a, true_prob_b) summing to 1.0.

        Raises
        ------
        ValueError
            If either probability is ≤ 0.
        """
        if implied_prob_a <= 0 or implied_prob_b <= 0:
            raise ValueError(
                "Implied probabilities must be positive. "
                f"Got a={implied_prob_a}, b={implied_prob_b}"
            )
        total = implied_prob_a + implied_prob_b
        true_a = round(implied_prob_a / total, 6)
        true_b = round(implied_prob_b / total, 6)
        return true_a, true_b

    @staticmethod
    def devig_multiway(*implied_probs: float) -> tuple[float, ...]:
        """Multiplicative de-vig for an N-outcome market.

        Generalization of :meth:`devig_probabilities` for markets with
        more than two outcomes (e.g. soccer 1X2: home/draw/away).

        Parameters
        ----------
        *implied_probs : float
            Two or more raw implied probabilities (one per outcome).

        Returns
        -------
        tuple[float, ...]
            De-vigged probabilities summing to exactly 1.0.

        Raises
        ------
        ValueError
            If fewer than two probabilities are supplied or any value
            is non-positive.
        """
        if len(implied_probs) < 2:
            raise ValueError(f"Need at least 2 outcomes, got {len(implied_probs)}")
        if any(p <= 0 for p in implied_probs):
            raise ValueError(f"All implied probabilities must be > 0, got {implied_probs}")
        total = sum(implied_probs)
        return tuple(round(p / total, 6) for p in implied_probs)

    @staticmethod
    @lru_cache(maxsize=512)
    def true_probability_to_fair_odds(true_probability: float) -> float:
        """Convert a de-vigged true probability to fair decimal odds.

        Parameters
        ----------
        true_probability : float
            Fair probability in (0, 1).

        Returns
        -------
        float
            Fair decimal odds (no vig embedded).
        """
        if not (0.0 < true_probability < 1.0):
            raise ValueError(f"True probability must be in (0, 1), got {true_probability}")
        return round(1.0 / true_probability, 4)

    # ──────────────────────────────────────────────
    #  3. Expected Value (+EV %)
    # ──────────────────────────────────────────────

    @staticmethod
    @lru_cache(maxsize=1024)
    def expected_value(true_probability: float, decimal_odds_offered: float) -> float:
        """Calculate the Expected Value percentage of a bet.

        Formula
        -------
        EV% = (true_probability × decimal_odds_offered) − 1

        A positive result means the bet has an edge over the market.
        For example, EV = 0.04 means a +4% edge.

        Parameters
        ----------
        true_probability : float
            De-vigged (fair) win probability from the sharp book.
        decimal_odds_offered : float
            Decimal odds being offered by the soft book.

        Returns
        -------
        float
            EV as a decimal (0.04 = +4% edge).

        Examples
        --------
        >>> MathEngine.expected_value(0.55, 2.10)
        0.155  # +15.5% EV — strong edge
        """
        if not (0.0 < true_probability < 1.0):
            raise ValueError(f"True probability must be in (0, 1), got {true_probability}")
        if decimal_odds_offered <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds_offered}")
        return round((true_probability * decimal_odds_offered) - 1.0, 6)

    # ──────────────────────────────────────────────
    #  4. Kelly Criterion (Bet Sizing)
    # ──────────────────────────────────────────────

    @staticmethod
    @lru_cache(maxsize=1024)
    def kelly_criterion(
        true_probability: float,
        decimal_odds: float,
        kelly_multiplier: float = 0.25,
    ) -> float:
        """Calculate optimal bet fraction via the Kelly Criterion.

        Formula
        -------
        f* = (p × b − q) / b

        where:
            p = true win probability
            q = 1 − p  (true loss probability)
            b = decimal_odds − 1  (net payout on a $1 bet)

        The result is then scaled by *kelly_multiplier* to reduce
        variance in practice (Quarter Kelly = 0.25 is standard).

        Parameters
        ----------
        true_probability : float
            De-vigged fair win probability from sharp book.
        decimal_odds : float
            Decimal odds offered by the soft book.
        kelly_multiplier : float, optional
            Fraction of full Kelly to risk (default 0.25 = Quarter Kelly).

        Returns
        -------
        float
            Fraction of bankroll to wager, in [0, 1].
            Returns 0.0 if the edge is negative (no bet).

        Examples
        --------
        >>> MathEngine.kelly_criterion(0.55, 2.10, kelly_multiplier=0.25)
        0.0375  # risk 3.75% of bankroll
        """
        if not (0.0 < true_probability < 1.0):
            raise ValueError(f"True probability must be in (0, 1), got {true_probability}")
        if decimal_odds <= 1.0:
            raise ValueError(f"Decimal odds must be > 1.0, got {decimal_odds}")
        if not (0.0 < kelly_multiplier <= 1.0):
            raise ValueError(f"Kelly multiplier must be in (0, 1], got {kelly_multiplier}")

        b = decimal_odds - 1.0  # net payout per $1
        p = true_probability  # win probability
        q = 1.0 - p  # loss probability

        full_kelly = (p * b - q) / b

        # Never recommend betting on a negative-edge play
        if full_kelly <= 0:
            return 0.0

        return round(full_kelly * kelly_multiplier, 6)

    @staticmethod
    def kelly_bet_size(
        bankroll: float,
        true_probability: float,
        decimal_odds: float,
        kelly_multiplier: float = 0.25,
    ) -> float:
        """Calculate the dollar amount to wager.

        Convenience wrapper around :meth:`kelly_criterion` that returns
        an absolute dollar amount instead of a fraction.

        Parameters
        ----------
        bankroll : float
            Total available bankroll in dollars.
        true_probability : float
            De-vigged fair win probability.
        decimal_odds : float
            Decimal odds offered.
        kelly_multiplier : float, optional
            Fraction of full Kelly (default 0.25).

        Returns
        -------
        float
            Dollar amount to bet, rounded to 2 decimal places.
        """
        fraction = MathEngine.kelly_criterion(true_probability, decimal_odds, kelly_multiplier)
        return round(bankroll * fraction, 2)

    # ──────────────────────────────────────────────
    #  5. Cache Telemetry
    # ──────────────────────────────────────────────

    _CACHED_FUNCS: tuple[str, ...] = (
        "american_to_decimal",
        "decimal_to_american",
        "decimal_to_implied_probability",
        "implied_probability_to_decimal",
        "calculate_vig",
        "devig_probabilities",
        "true_probability_to_fair_odds",
        "expected_value",
        "kelly_criterion",
    )

    @classmethod
    def cache_stats(cls) -> dict[str, dict[str, int | float]]:
        """Snapshot of LRU-cache hit/miss counters for every cached method.

        Returns
        -------
        dict
            ``{func_name: {"hits": int, "misses": int, "size": int,
            "maxsize": int, "hit_rate": float}}``
        """
        out: dict[str, dict[str, int | float]] = {}
        for name in cls._CACHED_FUNCS:
            func = getattr(cls, name)
            info = func.cache_info()
            total = info.hits + info.misses
            hit_rate = (info.hits / total) if total else 0.0
            out[name] = {
                "hits": info.hits,
                "misses": info.misses,
                "size": info.currsize,
                "maxsize": info.maxsize or 0,
                "hit_rate": round(hit_rate, 4),
            }
        return out

    @classmethod
    def report_cache_stats(cls) -> str:
        """Pretty-printed cache report (one line per cached function).

        Example
        -------
        ``odds_conversion cache: 847 hits / 12 misses (98.6% hit rate)``
        """
        lines = ["MathEngine cache report:"]
        stats = cls.cache_stats()
        width = max(len(n) for n in stats)
        for name, s in stats.items():
            lines.append(
                f"  {name:<{width}}  {s['hits']:>6} hits / {s['misses']:>5} misses "
                f"({s['hit_rate'] * 100:5.1f}% hit rate, size={s['size']}/{s['maxsize']})"
            )
        return "\n".join(lines)

    @classmethod
    def clear_caches(cls) -> None:
        """Clear every LRU cache. Useful for benchmarks and tests."""
        for name in cls._CACHED_FUNCS:
            getattr(cls, name).cache_clear()


# ══════════════════════════════════════════════════
#  Inline Smoke Tests — run via:  python -m src.math_engine
# ══════════════════════════════════════════════════

if __name__ == "__main__":  # pragma: no cover
    m = MathEngine()

    print("=" * 60)
    print("  MATH ENGINE -- SMOKE TESTS")
    print("=" * 60)

    # ── 1. Odds Conversion ──────────────────────
    print("\n[1] Odds Conversion")

    dec_pos = m.american_to_decimal(+150)
    assert dec_pos == 2.5, f"Expected 2.5, got {dec_pos}"
    print(f"  +150 American -> {dec_pos} Decimal  [PASS]")

    dec_neg = m.american_to_decimal(-200)
    assert dec_neg == 1.5, f"Expected 1.5, got {dec_neg}"
    print(f"  -200 American -> {dec_neg} Decimal  [PASS]")

    imp = m.decimal_to_implied_probability(2.0)
    assert imp == 0.5, f"Expected 0.5, got {imp}"
    print(f"  2.00 Decimal  -> {imp} Implied Prob  [PASS]")

    dec_back = m.implied_probability_to_decimal(0.5)
    assert dec_back == 2.0, f"Expected 2.0, got {dec_back}"
    print(f"  0.50 Prob      -> {dec_back} Decimal  [PASS]")

    # Edge case: odds of 0
    try:
        m.american_to_decimal(0)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  Odds=0 raises ValueError  [PASS]")

    # ── 2. Vig & De-Vigging ─────────────────────
    print("\n[2] Vig Calculation & De-Vigging")

    # Standard -110/-110 line → each side ≈ 0.5238
    imp_a = m.decimal_to_implied_probability(m.american_to_decimal(-110))
    imp_b = m.decimal_to_implied_probability(m.american_to_decimal(-110))
    vig = m.calculate_vig(imp_a, imp_b)
    print(f"  -110/-110 vig: {vig:.4f} ({vig * 100:.2f}%)  [PASS]")
    assert vig > 0, "Vig should be positive for -110/-110"

    true_a, true_b = m.devig_probabilities(imp_a, imp_b)
    assert abs(true_a + true_b - 1.0) < 1e-6, "De-vigged probs must sum to 1"
    print(f"  De-vigged: {true_a:.4f} / {true_b:.4f} (sum={true_a + true_b:.6f})  [PASS]")

    fair = m.true_probability_to_fair_odds(true_a)
    print(f"  Fair odds for side A: {fair}  [PASS]")

    # ── 3. Expected Value ───────────────────────
    print("\n[3] Expected Value (+EV%)")

    # If true prob is 55% and offered decimal is 2.10:
    ev = m.expected_value(0.55, 2.10)
    print(f"  EV(p=0.55, odds=2.10) = {ev:.4f} ({ev * 100:.2f}%)  [PASS]")
    assert ev > 0, "This should be a +EV bet"

    # Negative EV scenario
    ev_neg = m.expected_value(0.40, 2.10)
    print(f"  EV(p=0.40, odds=2.10) = {ev_neg:.4f} ({ev_neg * 100:.2f}%)  [PASS]")
    assert ev_neg < 0, "This should be a -EV bet"

    # ── 4. Kelly Criterion ──────────────────────
    print("\n[4] Kelly Criterion")

    frac = m.kelly_criterion(0.55, 2.10, kelly_multiplier=1.0)
    print(f"  Full Kelly(p=0.55, odds=2.10):    {frac:.4f} ({frac * 100:.2f}%)  [PASS]")
    assert frac > 0, "Positive edge should recommend a bet"

    qtr = m.kelly_criterion(0.55, 2.10, kelly_multiplier=0.25)
    print(f"  Quarter Kelly(p=0.55, odds=2.10): {qtr:.4f} ({qtr * 100:.2f}%)  [PASS]")
    assert qtr < frac, "Quarter Kelly should be less than full"

    dollar = m.kelly_bet_size(1000, 0.55, 2.10, kelly_multiplier=0.25)
    print(f"  $1000 bankroll -> bet ${dollar}  [PASS]")
    assert dollar > 0

    # Negative edge → no bet
    no_bet = m.kelly_criterion(0.30, 2.10, kelly_multiplier=1.0)
    assert no_bet == 0.0, "Negative edge should return 0"
    print(f"  Negative edge Kelly: {no_bet} (no bet)  [PASS]")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED")
    print("=" * 60)
