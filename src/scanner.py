"""
scanner.py -- Arbitrage & +EV Alpha Generation Engine

Consumes normalized odds data from data_ingestion and applies
the math_engine to detect two types of market edges:

1. **Arbitrage**: Guaranteed profit by hedging across books.
   Condition: (1/best_dec_A) + (1/best_dec_B) < 1.0

2. **+EV Bets**: Soft-book odds that exceed Pinnacle fair value.
   Condition: EV% > configurable threshold (default 1.5%)

Outputs clean Pandas DataFrames ready for the Streamlit dashboard.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.math_engine import MathEngine

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

SHARP_BOOK = "pinnacle"
SOFT_BOOKS = {"draftkings", "fanduel", "betmgm"}
EV_THRESHOLD = 0.015  # 1.5% minimum edge to flag


class Scanner:
    """Scans odds data for arbitrage opportunities and +EV bets.

    Parameters
    ----------
    math : MathEngine
        Instance of the quantitative math engine.
    bankroll : float
        Total bankroll in dollars for Kelly sizing.
    kelly_multiplier : float
        Fractional Kelly multiplier (0.25 = Quarter Kelly).
    """

    def __init__(
        self,
        math: MathEngine | None = None,
        bankroll: float = 1000.0,
        kelly_multiplier: float = 0.25,
    ) -> None:
        self.math = math or MathEngine()
        self.bankroll = bankroll
        self.kelly_multiplier = kelly_multiplier

    # ──────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────

    @staticmethod
    def _extract_outcomes(
        event: dict[str, Any], market_key: str
    ) -> dict[str, list[tuple[str, str, int]]]:
        """Extract all bookmaker prices for each outcome in an event.

        Returns
        -------
        dict[str, list[tuple[book_key, outcome_name, american_odds]]]
            Keyed by outcome name (e.g. "Lakers", "LeBron - Over").
        """
        outcomes: dict[str, list[tuple[str, str, int]]] = {}

        for bookmaker in event.get("bookmakers", []):
            bk_key = bookmaker["key"]
            for mkt in bookmaker.get("markets", []):
                if mkt["key"] != market_key:
                    continue
                for outcome in mkt.get("outcomes", []):
                    name = outcome["name"]
                    price = outcome["price"]
                    if name not in outcomes:
                        outcomes[name] = []
                    outcomes[name].append((bk_key, name, price))

        return outcomes

    # ──────────────────────────────────────────────
    #  1. Arbitrage Detection
    # ──────────────────────────────────────────────

    def _find_best_odds(
        self, prices: list[tuple[str, str, int]]
    ) -> tuple[str, int, float]:
        """Find the best (highest) odds among all bookmakers.

        Returns (book_key, american_odds, decimal_odds).
        """
        best_book, best_american, best_decimal = "", 0, 0.0
        for bk_key, _, american in prices:
            dec = self.math.american_to_decimal(american)
            if dec > best_decimal:
                best_book = bk_key
                best_american = american
                best_decimal = dec
        return best_book, best_american, best_decimal

    def scan_arbitrage(
        self,
        events: list[dict[str, Any]],
        market_key: str = "h2h",
    ) -> pd.DataFrame:
        """Scan events for cross-book arbitrage opportunities.

        For each event, finds the best odds for each side across
        all books. If the inverse sum < 1.0, it's a guaranteed arb.

        Parameters
        ----------
        events : list[dict]
            Raw event data from OddsAPIClient.fetch_odds().
        market_key : str
            Market to scan ('h2h', 'player_points', etc.).

        Returns
        -------
        pd.DataFrame
            Columns: Matchup, Side_1, Book_1, Odds_1, Side_2, Book_2,
                     Odds_2, Margin_Pct, Stake_1, Stake_2
        """
        arbs: list[dict[str, Any]] = []

        for event in events:
            matchup = f"{event['home_team']} vs {event['away_team']}"
            outcomes = self._extract_outcomes(event, market_key)
            outcome_names = list(outcomes.keys())

            # Arb requires exactly 2 sides for a two-way market
            # For props, we pair Over/Under for each player
            pairs = self._build_outcome_pairs(outcome_names)

            for side_a, side_b in pairs:
                if side_a not in outcomes or side_b not in outcomes:
                    continue

                bk_a, am_a, dec_a = self._find_best_odds(outcomes[side_a])
                bk_b, am_b, dec_b = self._find_best_odds(outcomes[side_b])

                if dec_a <= 1.0 or dec_b <= 1.0:
                    continue

                inv_sum = (1.0 / dec_a) + (1.0 / dec_b)

                if inv_sum < 1.0:
                    margin = round((1.0 - inv_sum) * 100, 2)
                    # Calculate hedge stakes for $100 total
                    total_stake = 100.0
                    stake_a = round(total_stake * (1.0 / dec_a) / inv_sum, 2)
                    stake_b = round(total_stake - stake_a, 2)

                    label = matchup if market_key == "h2h" else f"{matchup} | {side_a.split(' - ')[0]}"

                    arbs.append({
                        "Matchup": label,
                        "Side_1": side_a,
                        "Book_1": bk_a.title(),
                        "Odds_1": am_a,
                        "Side_2": side_b,
                        "Book_2": bk_b.title(),
                        "Odds_2": am_b,
                        "Margin_%": margin,
                        "Stake_1": f"${stake_a:.2f}",
                        "Stake_2": f"${stake_b:.2f}",
                    })

                    logger.info(
                        "ARB FOUND: %s | %s@%s(%+d) vs %s@%s(%+d) | margin=%.2f%%",
                        label, side_a, bk_a, am_a, side_b, bk_b, am_b, margin,
                    )

        df = pd.DataFrame(arbs)
        if df.empty:
            logger.info("No arbitrage opportunities found in %d events", len(events))
        else:
            logger.info("Found %d arbitrage opportunities", len(df))
        return df

    @staticmethod
    def _build_outcome_pairs(
        names: list[str],
    ) -> list[tuple[str, str]]:
        """Pair outcomes for arb checking.

        For h2h: pairs all combinations (usually 2 teams).
        For props: pairs "Player - Over" with "Player - Under".
        """
        pairs: list[tuple[str, str]] = []

        # Check for Over/Under pattern (props)
        overs = [n for n in names if "Over" in n]
        unders = [n for n in names if "Under" in n]

        if overs and unders:
            for over in overs:
                player = over.replace(" - Over", "")
                matching_under = f"{player} - Under"
                if matching_under in unders:
                    pairs.append((over, matching_under))
        elif len(names) == 2:
            # Standard h2h: just pair the two sides
            pairs.append((names[0], names[1]))
        elif len(names) > 2:
            # Multi-way (e.g. soccer draw): pair all combos
            for i, a in enumerate(names):
                for b in names[i + 1:]:
                    pairs.append((a, b))

        return pairs
