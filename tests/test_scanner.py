"""
test_scanner.py -- Tests for the arbitrage & +EV scanner engine.

Covers: arb detection, +EV detection, threshold filtering, output schema,
false positive prevention, and edge cases.
"""

from __future__ import annotations

import pytest
import pandas as pd

from src.math_engine import MathEngine
from src.scanner import Scanner, EV_THRESHOLD, SHARP_BOOK, SOFT_BOOKS


# ══════════════════════════════════════════════
#  Helper: build a minimal event for testing
# ══════════════════════════════════════════════


def _make_event(
    home: str,
    away: str,
    books: dict[str, tuple[int, int]],
    market: str = "h2h",
) -> dict:
    """Build a minimal event dict.

    Parameters
    ----------
    books : dict[book_key, (home_odds, away_odds)]
    """
    bookmakers = []
    for bk_key, (home_odds, away_odds) in books.items():
        bookmakers.append({
            "key": bk_key,
            "title": bk_key.title(),
            "markets": [{
                "key": market,
                "outcomes": [
                    {"name": home, "price": home_odds},
                    {"name": away, "price": away_odds},
                ],
            }],
        })
    return {
        "id": f"test_{home}_{away}",
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T00:00:00Z",
        "home_team": home,
        "away_team": away,
        "bookmakers": bookmakers,
    }


# ══════════════════════════════════════════════
#  1. Arbitrage Detection (5 tests)
# ══════════════════════════════════════════════


class TestArbDetection:
    """Cross-book arbitrage opportunity detection."""

    def test_arb_detected_with_wide_spread(self):
        """Known arb scenario: best odds sum inverse < 1.0."""
        event = _make_event(
            "Team A", "Team B",
            {
                "draftkings": (+140, -200),  # DK has Team A at +140
                "fanduel": (-180, +120),      # FD has Team B at +120
                "pinnacle": (-150, +130),
            },
        )
        scanner = Scanner()
        df = scanner.scan_arbitrage([event])
        assert not df.empty, "Should detect arb with +140 / +120"
        assert df.iloc[0]["Margin_%"] > 0

    def test_no_arb_in_tight_market(self):
        """No arb when all books are efficient (-110/-110)."""
        event = _make_event(
            "Team A", "Team B",
            {
                "draftkings": (-110, -110),
                "fanduel": (-110, -110),
                "pinnacle": (-110, -110),
                "betmgm": (-110, -110),
            },
        )
        scanner = Scanner()
        df = scanner.scan_arbitrage([event])
        assert df.empty, "Should NOT detect arb in efficient market"

    def test_arb_margin_calculation(self):
        """Verify arb margin matches manual calculation."""
        # +200 (3.00) and +200 (3.00) → inv_sum = 0.333 + 0.333 = 0.667
        # Margin = (1 - 0.667) * 100 = 33.33%
        event = _make_event(
            "Team A", "Team B",
            {
                "draftkings": (+200, -300),
                "fanduel": (-300, +200),
                "pinnacle": (-150, +130),
            },
        )
        scanner = Scanner()
        df = scanner.scan_arbitrage([event])
        assert not df.empty
        margin = df.iloc[0]["Margin_%"]
        assert 33.0 < margin < 34.0, f"Expected ~33.3% margin, got {margin}"

    def test_arb_has_stake_columns(self):
        """Arb results must include hedge stake allocations."""
        event = _make_event(
            "Team A", "Team B",
            {
                "draftkings": (+140, -200),
                "fanduel": (-180, +120),
                "pinnacle": (-150, +130),
            },
        )
        scanner = Scanner()
        df = scanner.scan_arbitrage([event])
        assert "Stake_1" in df.columns
        assert "Stake_2" in df.columns

    def test_arb_from_fixture(self, sample_arb_opportunity):
        """Arb detected using the conftest fixture."""
        scanner = Scanner()
        df = scanner.scan_arbitrage([sample_arb_opportunity])
        assert not df.empty, "Should detect arb from fixture"


# ══════════════════════════════════════════════
#  2. +EV Detection (5 tests)
# ══════════════════════════════════════════════


class TestEVDetection:
    """+EV bet detection using Pinnacle as sharp reference."""

    def test_ev_detected_when_soft_exceeds_fair(self):
        """Soft book offering better odds than Pinnacle fair value → +EV."""
        event = _make_event(
            "Team A", "Team B",
            {
                "pinnacle": (-155, +135),
                "draftkings": (-130, +155),   # DK offering +155 vs Pinnacle +135
            },
        )
        scanner = Scanner()
        df = scanner.scan_ev([event])
        assert not df.empty, "Should flag +EV when soft > fair"

    def test_no_ev_when_odds_match_pinnacle(self, efficient_market):
        """No +EV when all books match Pinnacle (efficient market)."""
        scanner = Scanner()
        df = scanner.scan_ev([efficient_market])
        assert df.empty, "Efficient market should produce zero +EV bets"

    def test_ev_threshold_filtering(self):
        """Only bets above the EV threshold should be flagged."""
        event = _make_event(
            "Team A", "Team B",
            {
                "pinnacle": (-150, +130),
                "draftkings": (-148, +132),   # Very slight edge
            },
        )
        scanner = Scanner()
        # With a high threshold, marginal edges are filtered out
        df = scanner.scan_ev([event], ev_threshold=0.10)  # 10% min
        assert df.empty, "Marginal edge should be filtered by high threshold"

    def test_ev_output_schema(self):
        """Every +EV opportunity must have all required columns."""
        event = _make_event(
            "Team A", "Team B",
            {
                "pinnacle": (-155, +135),
                "draftkings": (-120, +165),
            },
        )
        scanner = Scanner()
        df = scanner.scan_ev([event])
        if not df.empty:
            required = {"Matchup", "Outcome", "Bookmaker", "Offered_Odds",
                        "Pinnacle_Fair", "EV_%", "Kelly_Bet"}
            assert required.issubset(set(df.columns)), (
                f"Missing columns: {required - set(df.columns)}"
            )

    def test_ev_from_fixture(self, sample_ev_opportunity):
        """+EV detected using the conftest fixture."""
        scanner = Scanner()
        df = scanner.scan_ev([sample_ev_opportunity])
        assert not df.empty, "Should find +EV from fixture"


# ══════════════════════════════════════════════
#  3. Scanner Configuration (3 tests)
# ══════════════════════════════════════════════


class TestScannerConfig:
    """Scanner initialization and configuration."""

    def test_default_bankroll(self):
        scanner = Scanner()
        assert scanner.bankroll == 1000.0

    def test_custom_bankroll(self):
        scanner = Scanner(bankroll=5000)
        assert scanner.bankroll == 5000

    def test_kelly_multiplier_applied(self):
        """Kelly sizing uses configured multiplier."""
        event = _make_event(
            "Team A", "Team B",
            {
                "pinnacle": (-155, +135),
                "draftkings": (-120, +180),
            },
        )
        scanner_quarter = Scanner(bankroll=1000, kelly_multiplier=0.25)
        scanner_half = Scanner(bankroll=1000, kelly_multiplier=0.50)

        df_q = scanner_quarter.scan_ev([event])
        df_h = scanner_half.scan_ev([event])

        if not df_q.empty and not df_h.empty:
            # Half Kelly bets should be ~2x quarter Kelly bets
            bet_q = float(df_q.iloc[0]["Kelly_Bet"].replace("$", ""))
            bet_h = float(df_h.iloc[0]["Kelly_Bet"].replace("$", ""))
            assert bet_h > bet_q, "Half Kelly should bet more than quarter"


# ══════════════════════════════════════════════
#  4. Edge Cases & False Positive Prevention (4 tests)
# ══════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases and false positive prevention."""

    def test_no_pinnacle_skips_ev(self):
        """If Pinnacle is missing from event, skip EV scan (no reference)."""
        event = _make_event(
            "Team A", "Team B",
            {
                "draftkings": (-150, +130),
                "fanduel": (-155, +135),
            },
        )
        scanner = Scanner()
        df = scanner.scan_ev([event])
        assert df.empty, "No Pinnacle data should produce zero EV results"

    def test_empty_events_returns_empty_df(self):
        """Empty event list returns empty DataFrame."""
        scanner = Scanner()
        arb_df = scanner.scan_arbitrage([])
        ev_df = scanner.scan_ev([])
        assert arb_df.empty
        assert ev_df.empty

    def test_outcome_pairing_h2h(self):
        """h2h markets should pair the two teams correctly."""
        pairs = Scanner._build_outcome_pairs(["Lakers", "Celtics"])
        assert len(pairs) == 1
        assert ("Lakers", "Celtics") in pairs

    def test_outcome_pairing_props(self):
        """Props should pair Over/Under for same player."""
        names = [
            "LeBron James - Over",
            "LeBron James - Under",
            "Kevin Durant - Over",
            "Kevin Durant - Under",
        ]
        pairs = Scanner._build_outcome_pairs(names)
        assert len(pairs) == 2
        assert ("LeBron James - Over", "LeBron James - Under") in pairs
        assert ("Kevin Durant - Over", "Kevin Durant - Under") in pairs


# ══════════════════════════════════════════════
#  5. Robustness & Input Validation (5 tests)
# ══════════════════════════════════════════════


class TestRobustness:
    """Input validation and graceful degradation."""

    def test_negative_bankroll_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Scanner(bankroll=-100)

    def test_zero_bankroll_raises(self):
        with pytest.raises(ValueError, match="positive"):
            Scanner(bankroll=0)

    def test_invalid_kelly_multiplier_raises(self):
        with pytest.raises(ValueError, match="Kelly"):
            Scanner(kelly_multiplier=1.5)

    def test_malformed_event_skipped_in_arb(self):
        """Events missing home_team/away_team should be skipped, not crash."""
        good = _make_event("A", "B", {"pinnacle": (-110, -110)})
        bad = {"id": "broken", "bookmakers": []}  # missing teams
        scanner = Scanner()
        df = scanner.scan_arbitrage([bad, good])
        assert df is not None  # should complete without error

    def test_malformed_event_skipped_in_ev(self):
        """Malformed events in EV scan should be skipped gracefully."""
        good = _make_event("A", "B", {"pinnacle": (-155, +135), "draftkings": (-120, +165)})
        bad = {"id": "broken"}  # missing everything
        scanner = Scanner()
        df = scanner.scan_ev([bad, good])
        assert df is not None
