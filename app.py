"""
app.py -- Streamlit Dashboard for Sports Betting Arbitrage Scanner

Live dashboard that combines the data ingestion, math engine, and
scanner modules into an interactive UI.

Run via:  streamlit run app.py
"""

from __future__ import annotations

import asyncio
import logging

import streamlit as st

from src.data_ingestion import OddsAPIClient, SUPPORTED_SPORTS, SUPPORTED_MARKETS
from src.math_engine import MathEngine
from src.scanner import Scanner

# ──────────────────────────────────────────────
#  Page Config
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="Sports Betting Arb & +EV Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
#  Sidebar Controls
# ──────────────────────────────────────────────

st.sidebar.title("Scanner Configuration")
st.sidebar.markdown("---")

api_key = st.sidebar.text_input(
    "Odds API Key",
    type="password",
    help="Leave blank to use mock data for development.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Risk Management")

bankroll = st.sidebar.number_input(
    "Total Bankroll ($)",
    min_value=100,
    max_value=1_000_000,
    value=1_000,
    step=100,
)

kelly_option = st.sidebar.selectbox(
    "Kelly Multiplier",
    options=["Quarter Kelly (0.25)", "Half Kelly (0.50)", "Full Kelly (1.00)"],
    index=0,
)
kelly_map = {
    "Quarter Kelly (0.25)": 0.25,
    "Half Kelly (0.50)": 0.50,
    "Full Kelly (1.00)": 1.00,
}
kelly_multiplier = kelly_map[kelly_option]

st.sidebar.markdown("---")
st.sidebar.subheader("Market Selection")

sport = st.sidebar.selectbox(
    "Sport",
    options=list(SUPPORTED_SPORTS.keys()),
    format_func=lambda x: SUPPORTED_SPORTS[x],
)

market = st.sidebar.selectbox(
    "Market",
    options=list(SUPPORTED_MARKETS.keys()),
    format_func=lambda x: SUPPORTED_MARKETS[x],
)

scan_button = st.sidebar.button(
    "Scan for Edges",
    type="primary",
    use_container_width=True,
)

# Show mode indicator
if not api_key.strip():
    st.sidebar.warning("Running in **MOCK MODE** (synthetic data)")
else:
    st.sidebar.success("Live API mode")
