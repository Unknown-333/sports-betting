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
