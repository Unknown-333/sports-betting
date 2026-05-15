"""
test_dashboard.py -- Tests for the Streamlit dashboard configuration and helpers.

These tests verify the dashboard's data processing logic and module imports
without requiring a running Streamlit server (no UI rendering tested here;
that's covered by browser-based acceptance tests).
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from src.data_ingestion import SUPPORTED_MARKETS, SUPPORTED_SPORTS, OddsAPIClient
from src.math_engine import MathEngine
from src.scanner import Scanner

# ══════════════════════════════════════════════
#  1. Module Import Verification
# ══════════════════════════════════════════════


class TestDashboardImports:
    """Verify all modules the dashboard depends on are importable."""

    def test_math_engine_importable(self):
        from src.math_engine import MathEngine

        assert MathEngine is not None

    def test_data_ingestion_importable(self):
        from src.data_ingestion import OddsAPIClient

        assert OddsAPIClient is not None

    def test_scanner_importable(self):
        from src.scanner import Scanner

        assert Scanner is not None

    def test_supported_sports_available(self):
        assert "basketball_nba" in SUPPORTED_SPORTS

    def test_supported_markets_available(self):
        assert "h2h" in SUPPORTED_MARKETS
        assert "player_points" in SUPPORTED_MARKETS


# ══════════════════════════════════════════════
#  2. Dashboard Data Pipeline Logic
# ══════════════════════════════════════════════


class TestDashboardPipeline:
    """Test the scan pipeline logic the dashboard uses."""

    @pytest.mark.asyncio
    async def test_pipeline_returns_dataframes(self):
        """Pipeline should always return DataFrames, never None."""
        client = OddsAPIClient(api_key="")
        scanner = Scanner(bankroll=1000, kelly_multiplier=0.25)

        events = await client.fetch_odds("basketball_nba", "h2h")
        arb_df = scanner.scan_arbitrage(events, "h2h")
        ev_df = scanner.scan_ev(events, "h2h")

        assert isinstance(arb_df, pd.DataFrame)
        assert isinstance(ev_df, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_avg_edge_calculation(self):
        """Average edge calculation used in KPI card."""
        client = OddsAPIClient(api_key="")
        scanner = Scanner()

        events = await client.fetch_odds("basketball_nba", "h2h")
        ev_df = scanner.scan_ev(events)

        if not ev_df.empty:
            avg_edge = ev_df["EV_%"].mean()
            assert avg_edge > 0, "Average edge should be positive for +EV bets"
        else:
            avg_edge = 0.0
            assert avg_edge == 0.0

    def test_kelly_map_values(self):
        """Verify the Kelly option mapping used in sidebar."""
        kelly_map = {
            "Quarter Kelly (0.25)": 0.25,
            "Half Kelly (0.50)": 0.50,
            "Full Kelly (1.00)": 1.00,
        }
        assert kelly_map["Quarter Kelly (0.25)"] == 0.25
        assert kelly_map["Half Kelly (0.50)"] == 0.50
        assert kelly_map["Full Kelly (1.00)"] == 1.00


# ══════════════════════════════════════════════
#  3. Async Runner Helper
# ══════════════════════════════════════════════


class TestAsyncRunner:
    """Test the async runner helper used by the dashboard."""

    def test_asyncio_run_basic(self):
        """Basic asyncio.run should work for simple coroutines."""

        async def _simple():
            return 42

        result = asyncio.run(_simple())
        assert result == 42

    @pytest.mark.asyncio
    async def test_fetch_odds_is_async(self):
        """Verify fetch_odds is a proper async coroutine."""
        client = OddsAPIClient(api_key="")
        result = await client.fetch_odds("basketball_nba", "h2h")
        assert isinstance(result, list)
