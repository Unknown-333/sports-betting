"""
charts.py -- Plotly chart builders for the Streamlit dashboard.

All functions here return ``plotly.graph_objects.Figure`` instances so
the dashboard layer can simply call ``st.plotly_chart(fig)``.  Keeping
chart construction out of ``app.py`` makes it unit-testable without
spinning up Streamlit.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ──────────────────────────────────────────────────────────────────
#  1. Line-movement chart
# ──────────────────────────────────────────────────────────────────


def line_movement_chart(history: pd.DataFrame, team: str) -> go.Figure:
    """One line per book; X = timestamp, Y = American odds."""
    fig = go.Figure()
    if history is None or history.empty:
        fig.update_layout(title=f"No line history for {team}")
        return fig

    ts_col = "timestamp_dt" if "timestamp_dt" in history.columns else "timestamp"
    for book, sub in history.groupby("book"):
        fig.add_trace(
            go.Scatter(
                x=sub[ts_col],
                y=sub["american_odds"],
                mode="lines+markers",
                name=str(book).title(),
                hovertemplate="%{y:+.0f} American<br>%{x}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"Line movement: {team}",
        xaxis_title="Time",
        yaxis_title="American Odds",
        legend_title="Book",
        hovermode="x unified",
        template="plotly_dark",
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  2. CLV distribution
# ──────────────────────────────────────────────────────────────────


def clv_histogram(clv_values: Sequence[float]) -> go.Figure:
    fig = go.Figure()
    if not len(clv_values):
        fig.update_layout(title="No settled bets yet")
        return fig
    fig.add_trace(
        go.Histogram(
            x=list(clv_values),
            nbinsx=30,
            marker_color="#00ff88",
        )
    )
    fig.add_vline(x=0.0, line_dash="dash", line_color="white", annotation_text="Break-even")
    fig.update_layout(
        title="CLV distribution",
        xaxis_title="CLV (decimal fraction)",
        yaxis_title="Bets",
        template="plotly_dark",
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  3. Bankroll-simulator fan chart
# ──────────────────────────────────────────────────────────────────


def bankroll_fan_chart(daily: np.ndarray, starting: float) -> go.Figure:
    """Median path with shaded 10/90 percentile band."""
    if daily.ndim != 2:
        raise ValueError("`daily` must be a 2-D array")
    days = np.arange(daily.shape[1])
    p10 = np.percentile(daily, 10, axis=0)
    p50 = np.percentile(daily, 50, axis=0)
    p90 = np.percentile(daily, 90, axis=0)

    fig = go.Figure()
    # Shaded 10-90 band
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([days, days[::-1]]),
            y=np.concatenate([p90, p10[::-1]]),
            fill="toself",
            fillcolor="rgba(0,255,136,0.18)",
            line=dict(color="rgba(0,0,0,0)"),
            name="10-90 percentile",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=days,
            y=p50,
            mode="lines",
            name="Median",
            line=dict(color="#00ff88", width=2),
        )
    )
    fig.add_hline(
        y=starting, line_dash="dot", line_color="white", annotation_text="Starting bankroll"
    )
    fig.update_layout(
        title="Bankroll simulation (10,000 paths)",
        xaxis_title="Day",
        yaxis_title="Bankroll ($)",
        template="plotly_dark",
        hovermode="x unified",
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  4. Opportunity heatmap (book x sport)
# ──────────────────────────────────────────────────────────────────


def opportunity_heatmap(ev_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if ev_df is None or ev_df.empty:
        fig.update_layout(title="No +EV opportunities to plot")
        return fig
    if "Sport" not in ev_df.columns:
        # Fall back to a single 'All' column.
        ev_df = ev_df.copy()
        ev_df["Sport"] = "All"
    pivot = ev_df.groupby(["Bookmaker", "Sport"]).size().unstack(fill_value=0)
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Greens",
            colorbar=dict(title="Count"),
        )
    )
    fig.update_layout(
        title="Opportunity heatmap: book x sport",
        xaxis_title="Sport",
        yaxis_title="Book",
        template="plotly_dark",
    )
    return fig


# ──────────────────────────────────────────────────────────────────
#  5. EV distribution histogram
# ──────────────────────────────────────────────────────────────────


def ev_distribution(ev_df: pd.DataFrame, threshold_pct: float = 1.5) -> go.Figure:
    fig = go.Figure()
    if ev_df is None or ev_df.empty:
        fig.update_layout(title="No +EV bets to plot")
        return fig
    fig.add_trace(
        go.Histogram(
            x=ev_df["EV_%"],
            nbinsx=20,
            marker_color="#00ff88",
        )
    )
    fig.add_vline(
        x=threshold_pct,
        line_dash="dash",
        line_color="white",
        annotation_text=f"{threshold_pct}% threshold",
    )
    fig.update_layout(
        title="EV % distribution",
        xaxis_title="EV (%)",
        yaxis_title="Bets",
        template="plotly_dark",
    )
    return fig
