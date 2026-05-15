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
