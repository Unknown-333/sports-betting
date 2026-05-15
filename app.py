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

# ──────────────────────────────────────────────
#  Main Header
# ──────────────────────────────────────────────

st.title("Sports Betting Arbitrage & +EV Scanner")
st.caption(
    "Market microstructure analysis across DraftKings, FanDuel, BetMGM "
    "| Sharp book: Pinnacle | Powered by The Odds API"
)
st.markdown("---")


# ──────────────────────────────────────────────
#  Async Runner Helper
# ──────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine from synchronous Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ──────────────────────────────────────────────
#  Scan Pipeline
# ──────────────────────────────────────────────

if scan_button:
    with st.spinner("Fetching odds data and scanning for edges..."):
        # Initialize modules
        client = OddsAPIClient(api_key=api_key if api_key.strip() else None)
        math = MathEngine()
        scanner = Scanner(
            math=math,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
        )

        # Fetch odds
        events = run_async(client.fetch_odds(sport, market))

        # Scan for edges
        arb_df = scanner.scan_arbitrage(events, market)
        ev_df = scanner.scan_ev(events, market)

    # Store results in session state
    st.session_state["events"] = events
    st.session_state["arb_df"] = arb_df
    st.session_state["ev_df"] = ev_df
    st.session_state["scanned"] = True


# ──────────────────────────────────────────────
#  Display Results
# ──────────────────────────────────────────────

if st.session_state.get("scanned"):
    events = st.session_state["events"]
    arb_df = st.session_state["arb_df"]
    ev_df = st.session_state["ev_df"]

    # ── KPI Cards ───────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Active Games", len(events))
    with col2:
        st.metric("Arbs Found", len(arb_df))
    with col3:
        st.metric("+EV Bets Found", len(ev_df))
    with col4:
        avg_edge = (
            f"{ev_df['EV_%'].mean():.2f}%"
            if not ev_df.empty
            else "0.00%"
        )
        st.metric("Avg Edge", avg_edge)

    st.markdown("---")

    # ── Arbitrage Table ─────────────────────────
    st.subheader("Arbitrage Opportunities")

    if arb_df.empty:
        st.info(
            "No arbitrage opportunities found. "
            "These are rare in efficient markets — try Player Props."
        )
    else:
        st.success(
            f"Found **{len(arb_df)} arbs** -- guaranteed profit "
            "if executed simultaneously!"
        )
        st.dataframe(
            arb_df.style.format({"Margin_%": "{:.2f}%"}).background_gradient(
                subset=["Margin_%"], cmap="Greens"
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # ── +EV Bets Table ──────────────────────────
    st.subheader("+EV Bets (Positive Expected Value)")

    if ev_df.empty:
        st.info(
            "No +EV bets found above the 1.5% threshold. "
            "Try scanning different sports or markets."
        )
    else:
        st.success(
            f"Found **{len(ev_df)} value bets** with edge over fair value!"
        )
        st.dataframe(
            ev_df.style.format({"EV_%": "+{:.2f}%"}).background_gradient(
                subset=["EV_%"], cmap="Greens"
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── Footer info ─────────────────────────────
    st.markdown("---")
    st.caption(
        f"Scan config: {SUPPORTED_SPORTS.get(sport, sport)} | "
        f"{SUPPORTED_MARKETS.get(market, market)} | "
        f"Bankroll: ${bankroll:,.0f} | Kelly: {kelly_multiplier}x"
    )

else:
    # Landing state before first scan
    st.markdown(
        """
        ### How to use this scanner

        1. **Configure** your settings in the sidebar
        2. Click **"Scan for Edges"** to analyze current odds
        3. Review arbitrage opportunities and +EV bets below

        > **Tip:** Leave the API key blank to use mock data for testing.
        > Player Props markets typically have more edge than Moneylines.
        """
    )
