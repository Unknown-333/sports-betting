"""Smoke + structural tests for the Plotly chart builders."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src import charts

# ──────────────────────────────────────────────────────────────────
#  Line-movement chart
# ──────────────────────────────────────────────────────────────────


class TestLineMovementChart:
    def test_empty_history_returns_placeholder(self):
        fig = charts.line_movement_chart(pd.DataFrame(), team="Lakers")
        assert isinstance(fig, go.Figure)
        assert "No line history" in fig.layout.title.text

    def test_one_trace_per_book(self):
        history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
                "book": ["draftkings", "draftkings", "fanduel", "fanduel"],
                "american_odds": [-110, -120, -105, -115],
            }
        )
        fig = charts.line_movement_chart(history, team="Lakers")
        assert len(fig.data) == 2
        assert {tr.name for tr in fig.data} == {"Draftkings", "Fanduel"}

    def test_uses_timestamp_dt_column_when_available(self):
        history = pd.DataFrame(
            {
                "timestamp": ["a", "b"],
                "timestamp_dt": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "book": ["draftkings", "draftkings"],
                "american_odds": [-110, -120],
            }
        )
        fig = charts.line_movement_chart(history, team="X")
        assert len(fig.data) == 1


# ──────────────────────────────────────────────────────────────────
#  CLV histogram
# ──────────────────────────────────────────────────────────────────


class TestClvHistogram:
    def test_empty_returns_placeholder(self):
        fig = charts.clv_histogram([])
        assert "No settled bets" in fig.layout.title.text

    def test_populated_has_one_histogram_trace(self):
        fig = charts.clv_histogram([0.01, -0.02, 0.03, 0.0])
        assert len(fig.data) == 1
        assert fig.data[0].type == "histogram"


# ──────────────────────────────────────────────────────────────────
#  Bankroll fan chart
# ──────────────────────────────────────────────────────────────────


class TestBankrollFanChart:
    def test_rejects_1d_input(self):
        with pytest.raises(ValueError):
            charts.bankroll_fan_chart(np.array([1.0, 2.0, 3.0]), starting=1000)

    def test_returns_band_plus_median(self):
        rng = np.random.default_rng(0)
        daily = 1000 * np.exp(rng.normal(0, 0.05, size=(50, 30)).cumsum(axis=1))
        fig = charts.bankroll_fan_chart(daily, starting=1000.0)
        # 0 = filled band, 1 = median line.
        assert len(fig.data) == 2
        assert fig.data[1].name == "Median"


# ──────────────────────────────────────────────────────────────────
#  Opportunity heatmap
# ──────────────────────────────────────────────────────────────────


class TestOpportunityHeatmap:
    def test_empty(self):
        fig = charts.opportunity_heatmap(pd.DataFrame())
        assert "No +EV" in fig.layout.title.text

    def test_with_sport_column(self):
        df = pd.DataFrame(
            {
                "Bookmaker": ["DraftKings", "DraftKings", "FanDuel"],
                "Sport": ["NBA", "NFL", "NBA"],
            }
        )
        fig = charts.opportunity_heatmap(df)
        assert len(fig.data) == 1
        assert fig.data[0].type == "heatmap"

    def test_without_sport_column_falls_back(self):
        df = pd.DataFrame({"Bookmaker": ["DraftKings", "FanDuel"]})
        fig = charts.opportunity_heatmap(df)
        assert len(fig.data) == 1


# ──────────────────────────────────────────────────────────────────
#  EV distribution
# ──────────────────────────────────────────────────────────────────


class TestEvDistribution:
    def test_empty(self):
        fig = charts.ev_distribution(pd.DataFrame())
        assert "No +EV bets" in fig.layout.title.text

    def test_populated(self):
        df = pd.DataFrame({"EV_%": [1.6, 2.4, 3.1, 4.8, 7.0]})
        fig = charts.ev_distribution(df, threshold_pct=1.5)
        assert len(fig.data) == 1
        assert fig.data[0].type == "histogram"
