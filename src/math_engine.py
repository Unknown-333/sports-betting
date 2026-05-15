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
            raise ValueError(
                f"Decimal odds must be > 1.0, got {decimal_odds}"
            )
        return round(1.0 / decimal_odds, 6)

    @staticmethod
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
            raise ValueError(
                f"Probability must be in (0, 1), got {probability}"
            )
        return round(1.0 / probability, 4)

    # ──────────────────────────────────────────────
    #  2. Vig Calculation & Removal (De-Vigging)
    # ──────────────────────────────────────────────

    @staticmethod
    def calculate_vig(
        implied_prob_a: float, implied_prob_b: float
    ) -> float:
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
    def devig_probabilities(
        implied_prob_a: float, implied_prob_b: float
    ) -> tuple[float, float]:
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
            raise ValueError(
                f"True probability must be in (0, 1), got {true_probability}"
            )
        return round(1.0 / true_probability, 4)

    # ──────────────────────────────────────────────
    #  3. Expected Value (+EV %)
    # ──────────────────────────────────────────────

    @staticmethod
    def expected_value(
        true_probability: float, decimal_odds_offered: float
    ) -> float:
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
            raise ValueError(
                f"True probability must be in (0, 1), got {true_probability}"
            )
        if decimal_odds_offered <= 1.0:
            raise ValueError(
                f"Decimal odds must be > 1.0, got {decimal_odds_offered}"
            )
        return round(
            (true_probability * decimal_odds_offered) - 1.0, 6
        )
