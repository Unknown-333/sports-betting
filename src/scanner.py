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
        if bankroll <= 0:
            raise ValueError(f"Bankroll must be positive, got {bankroll}")
        if not (0 < kelly_multiplier <= 1.0):
            raise ValueError(f"Kelly multiplier must be in (0, 1], got {kelly_multiplier}")
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

    def _find_best_odds(self, prices: list[tuple[str, str, int]]) -> tuple[str, int, float]:
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
            try:
                matchup = f"{event['home_team']} vs {event['away_team']}"
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping malformed event: %s", exc)
                continue
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
                    if margin <= 0.0:
                        # Margin rounded to zero -> not a real arb edge.
                        continue
                    # Calculate hedge stakes for $100 total
                    total_stake = 100.0
                    stake_a = round(total_stake * (1.0 / dec_a) / inv_sum, 2)
                    stake_b = round(total_stake - stake_a, 2)

                    label = (
                        matchup if market_key == "h2h" else f"{matchup} | {side_a.split(' - ')[0]}"
                    )

                    arbs.append(
                        {
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
                        }
                    )

                    logger.info(
                        "ARB FOUND: %s | %s@%s(%+d) vs %s@%s(%+d) | margin=%.2f%%",
                        label,
                        side_a,
                        bk_a,
                        am_a,
                        side_b,
                        bk_b,
                        am_b,
                        margin,
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
                for b in names[i + 1 :]:
                    pairs.append((a, b))

        return pairs

    # ──────────────────────────────────────────────
    #  2. +EV Detection (Sharp vs Soft)
    # ──────────────────────────────────────────────

    def _get_pinnacle_prices(self, event: dict[str, Any], market_key: str) -> dict[str, int]:
        """Extract Pinnacle's odds for each outcome.

        Returns dict[outcome_name, american_odds].
        Returns empty dict if Pinnacle is missing from the event.
        """
        for bookmaker in event.get("bookmakers", []):
            if bookmaker["key"] != SHARP_BOOK:
                continue
            for mkt in bookmaker.get("markets", []):
                if mkt["key"] != market_key:
                    continue
                return {o["name"]: o["price"] for o in mkt.get("outcomes", [])}
        return {}

    def scan_ev(
        self,
        events: list[dict[str, Any]],
        market_key: str = "h2h",
        ev_threshold: float = EV_THRESHOLD,
    ) -> pd.DataFrame:
        """Scan for +EV bets by comparing soft books to Pinnacle fair odds.

        Parameters
        ----------
        events : list[dict]
            Raw event data from OddsAPIClient.
        market_key : str
            Market key to scan.
        ev_threshold : float
            Minimum EV% to flag a bet (default 1.5%).

        Returns
        -------
        pd.DataFrame
            Columns: Matchup, Outcome, Bookmaker, Offered_Odds,
                     Pinnacle_Fair, EV_Pct, Kelly_Bet
        """
        ev_bets: list[dict[str, Any]] = []

        for event in events:
            try:
                matchup = f"{event['home_team']} vs {event['away_team']}"
            except (KeyError, TypeError) as exc:
                logger.warning("Skipping malformed event in EV scan: %s", exc)
                continue
            pin_prices = self._get_pinnacle_prices(event, market_key)

            if not pin_prices:
                logger.debug("No Pinnacle data for %s -- skipping", matchup)
                continue

            # De-vig Pinnacle to get true probabilities.
            # For 3-way markets (e.g. soccer 1X2) we de-vig all outcomes
            # together using the multiplicative N-way normalization;
            # for 2-way markets we use the pairwise routine.
            pin_names = list(pin_prices.keys())

            true_probs: dict[str, float] = {}
            if len(pin_names) >= 3:
                imps = [
                    self.math.decimal_to_implied_probability(
                        self.math.american_to_decimal(pin_prices[n])
                    )
                    for n in pin_names
                ]
                fair = self.math.devig_multiway(*imps)
                for name, p in zip(pin_names, fair):
                    true_probs[name] = p
            else:
                pairs = self._build_outcome_pairs(pin_names)
                for side_a, side_b in pairs:
                    if side_a not in pin_prices or side_b not in pin_prices:
                        continue
                    dec_a = self.math.american_to_decimal(pin_prices[side_a])
                    dec_b = self.math.american_to_decimal(pin_prices[side_b])
                    imp_a = self.math.decimal_to_implied_probability(dec_a)
                    imp_b = self.math.decimal_to_implied_probability(dec_b)
                    true_a, true_b = self.math.devig_probabilities(imp_a, imp_b)
                    true_probs[side_a] = true_a
                    true_probs[side_b] = true_b

            # Now compare each soft book's odds to true probs
            for bookmaker in event.get("bookmakers", []):
                bk_key = bookmaker["key"]
                if bk_key not in SOFT_BOOKS:
                    continue

                for mkt in bookmaker.get("markets", []):
                    if mkt["key"] != market_key:
                        continue
                    for outcome in mkt.get("outcomes", []):
                        name = outcome["name"]
                        if name not in true_probs:
                            continue

                        offered_american = outcome["price"]
                        offered_dec = self.math.american_to_decimal(offered_american)
                        true_prob = true_probs[name]

                        ev = self.math.expected_value(true_prob, offered_dec)

                        if ev >= ev_threshold:
                            fair_dec = self.math.true_probability_to_fair_odds(true_prob)
                            kelly_bet = self.math.kelly_bet_size(
                                self.bankroll,
                                true_prob,
                                offered_dec,
                                self.kelly_multiplier,
                            )

                            label = (
                                matchup
                                if market_key == "h2h"
                                else f"{matchup} | {name.split(' - ')[0]}"
                            )

                            ev_bets.append(
                                {
                                    "Matchup": label,
                                    "Outcome": name,
                                    "Bookmaker": bk_key.title(),
                                    "Offered_Odds": offered_american,
                                    "Pinnacle_Fair": f"{fair_dec:.2f}",
                                    "EV_%": round(ev * 100, 2),
                                    "Kelly_Bet": f"${kelly_bet:.2f}",
                                }
                            )

                            logger.info(
                                "+EV FOUND: %s | %s@%s(%+d) | EV=%.2f%% | Kelly=$%.2f",
                                label,
                                name,
                                bk_key,
                                offered_american,
                                ev * 100,
                                kelly_bet,
                            )

        df = pd.DataFrame(ev_bets)
        if df.empty:
            logger.info("No +EV bets found in %d events", len(events))
        else:
            logger.info("Found %d +EV bets", len(df))
        return df


# ──────────────────────────────────────────────
#  Integration test -- run via:  python -m src.scanner
# ──────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    import asyncio

    from src.data_ingestion import OddsAPIClient

    async def _test() -> None:
        client = OddsAPIClient()  # mock mode
        scanner = Scanner(bankroll=1000, kelly_multiplier=0.25)

        for market in ["h2h", "player_points"]:
            print(f"\n{'='*60}")
            print(f"  SCANNING: basketball_nba / {market}")
            print(f"{'='*60}")

            events = await client.fetch_odds("basketball_nba", market)
            print(f"  Events fetched: {len(events)}")

            arb_df = scanner.scan_arbitrage(events, market)
            print(f"\n  Arbitrage opportunities: {len(arb_df)}")
            if not arb_df.empty:
                print(arb_df.to_string(index=False))

            ev_df = scanner.scan_ev(events, market)
            print(f"\n  +EV bets found: {len(ev_df)}")
            if not ev_df.empty:
                print(ev_df.to_string(index=False))

        print(f"\n{'='*60}")
        print("  SCANNER INTEGRATION TEST COMPLETE")
        print(f"{'='*60}")

    asyncio.run(_test())
