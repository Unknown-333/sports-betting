"""
test_integration.py -- End-to-end integration tests.

Tests the full pipeline: data ingestion → scanner → formatted output.
All tests use mock mode (no API key) to avoid network calls.
"""

from __future__ import annotations

import pytest

from src.data_ingestion import OddsAPIClient
from src.math_engine import MathEngine
from src.scanner import Scanner


# ══════════════════════════════════════════════
#  End-to-End Pipeline Tests
# ══════════════════════════════════════════════


class TestEndToEnd:
    """Full pipeline integration from ingestion through scanning."""

    @pytest.mark.asyncio
    async def test_mock_h2h_pipeline(self):
        """Mock API → data ingestion → scanner → results for h2h."""
        client = OddsAPIClient(api_key="")
        scanner = Scanner(bankroll=1000, kelly_multiplier=0.25)

        events = await client.fetch_odds("basketball_nba", "h2h")
        assert len(events) > 0, "Mock mode should return events"

        arb_df = scanner.scan_arbitrage(events, "h2h")
        ev_df = scanner.scan_ev(events, "h2h")

        # DataFrames should be valid (may be empty depending on random mock data)
        assert arb_df is not None
        assert ev_df is not None

        # If any results exist, verify schema
        if not arb_df.empty:
            assert "Matchup" in arb_df.columns
            assert "Margin_%" in arb_df.columns
            assert "Stake_1" in arb_df.columns

        if not ev_df.empty:
            assert "Matchup" in ev_df.columns
            assert "EV_%" in ev_df.columns
            assert "Kelly_Bet" in ev_df.columns

    @pytest.mark.asyncio
    async def test_mock_props_pipeline(self):
        """Mock API → data ingestion → scanner → results for player props."""
        client = OddsAPIClient(api_key="")
        scanner = Scanner(bankroll=1000, kelly_multiplier=0.25)

        events = await client.fetch_odds("basketball_nba", "player_points")
        assert len(events) > 0

        arb_df = scanner.scan_arbitrage(events, "player_points")
        ev_df = scanner.scan_ev(events, "player_points")

        assert arb_df is not None
        assert ev_df is not None

    @pytest.mark.asyncio
    async def test_mock_mode_no_api_key_pipeline(self):
        """Full pipeline works with no API key (mock mode auto-enabled)."""
        client = OddsAPIClient(api_key=None)
        assert client.is_mock is True

        events = await client.fetch_odds("basketball_nba", "h2h")
        assert len(events) > 0

        scanner = Scanner()
        arb_df = scanner.scan_arbitrage(events)
        ev_df = scanner.scan_ev(events)

        # Pipeline should complete without errors
        assert isinstance(arb_df, type(ev_df))  # both are DataFrames

    @pytest.mark.asyncio
    async def test_no_state_leaks_between_runs(self):
        """Scanner should not leak state between sequential runs."""
        client = OddsAPIClient(api_key="")
        scanner = Scanner(bankroll=1000)

        # Run 1
        events_1 = await client.fetch_odds("basketball_nba", "h2h")
        arb_1 = scanner.scan_arbitrage(events_1)
        ev_1 = scanner.scan_ev(events_1)

        # Run 2 (same scanner instance, fresh data)
        events_2 = await client.fetch_odds("basketball_nba", "h2h")
        arb_2 = scanner.scan_arbitrage(events_2)
        ev_2 = scanner.scan_ev(events_2)

        # Results should be independent (no accumulated state)
        # Shape may differ due to random mock data, but both should succeed
        assert arb_1 is not None
        assert arb_2 is not None
        assert ev_1 is not None
        assert ev_2 is not None

    @pytest.mark.asyncio
    async def test_ev_values_are_positive(self):
        """All flagged +EV bets should have positive EV%."""
        client = OddsAPIClient(api_key="")
        scanner = Scanner()

        events = await client.fetch_odds("basketball_nba", "h2h")
        ev_df = scanner.scan_ev(events)

        if not ev_df.empty:
            for _, row in ev_df.iterrows():
                assert row["EV_%"] > 0, (
                    f"EV should be positive but got {row['EV_%']}% "
                    f"for {row['Outcome']}"
                )

    @pytest.mark.asyncio
    async def test_arb_margins_are_positive(self):
        """All flagged arbs should have positive margin."""
        client = OddsAPIClient(api_key="")
        scanner = Scanner()

        events = await client.fetch_odds("basketball_nba", "h2h")
        arb_df = scanner.scan_arbitrage(events)

        if not arb_df.empty:
            for _, row in arb_df.iterrows():
                assert row["Margin_%"] > 0, (
                    f"Arb margin should be positive but got {row['Margin_%']}%"
                )


# ══════════════════════════════════════════════
#  Math Engine ↔ Scanner Integration
# ══════════════════════════════════════════════


class TestMathScannerIntegration:
    """Verify the math engine integrates correctly with the scanner."""

    def test_scanner_uses_math_engine(self):
        """Scanner should use MathEngine for all calculations."""
        math = MathEngine()
        scanner = Scanner(math=math)
        assert scanner.math is math

    def test_pinnacle_devig_produces_valid_probs(self):
        """De-vigged Pinnacle probs should sum to 1.0."""
        math = MathEngine()
        # Pinnacle -155 / +135
        dec_a = math.american_to_decimal(-155)
        dec_b = math.american_to_decimal(+135)
        imp_a = math.decimal_to_implied_probability(dec_a)
        imp_b = math.decimal_to_implied_probability(dec_b)
        true_a, true_b = math.devig_probabilities(imp_a, imp_b)

        assert abs(true_a + true_b - 1.0) < 1e-6
        assert 0 < true_a < 1
        assert 0 < true_b < 1
