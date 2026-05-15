"""
app.py -- Sports Betting Statistical Arbitrage & +EV Scanner Dashboard.

Five-tab Streamlit dashboard:
  1. Live Scanner       -- arb + EV tables (with confidence badges)
  2. Line Movement      -- line history charts from SQLite
  3. CLV Tracker        -- bet log + closing-line value stats
  4. Bankroll Sim       -- Monte-Carlo simulator with Plotly fan chart
  5. Settings           -- everything tweakable in one place
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pandas as pd
import streamlit as st

from src import charts
from src.clv_tracker import CLVTracker
from src.confidence import annotate_dataframe
from src.data_ingestion import (
    SUPPORTED_MARKETS,
    SUPPORTED_SPORTS,
    OddsAPIClient,
)
from src.line_tracker import LineTracker
from src.math_engine import MathEngine
from src.notifier import TelegramNotifier
from src.scanner import Scanner
from src.simulator import simulate_bankroll
from src.steam_detector import SteamDetector

# ════════════════════════════════════════════════════════════════════
#  Page configuration
# ════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="EV Scanner | Live",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════


def run_async(coro: Any) -> Any:
    """Run an async coroutine from a sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@st.cache_resource
def get_line_tracker() -> LineTracker:
    return LineTracker()


@st.cache_resource
def get_clv_tracker() -> CLVTracker:
    return CLVTracker()


@st.cache_resource
def get_notifier() -> TelegramNotifier:
    return TelegramNotifier()


# ════════════════════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("EV Scanner")
    st.caption("Pinnacle de-vig | Kelly sizing")
    st.markdown("---")

    api_key = st.text_input("Odds API Key", type="password", help="Leave blank for mock mode.")
    bankroll = st.number_input(
        "Bankroll ($)", min_value=100, max_value=1_000_000, value=1_000, step=100
    )
    kelly_label = st.selectbox(
        "Kelly Multiplier",
        ["Quarter Kelly (0.25)", "Half Kelly (0.50)", "Full Kelly (1.00)"],
        index=0,
    )
    kelly_multiplier = {
        "Quarter Kelly (0.25)": 0.25,
        "Half Kelly (0.50)": 0.50,
        "Full Kelly (1.00)": 1.00,
    }[kelly_label]

    st.markdown("---")
    sports = st.multiselect(
        "Sports",
        options=list(SUPPORTED_SPORTS.keys()),
        default=["basketball_nba"],
        format_func=lambda s: SUPPORTED_SPORTS[s],
    )
    market = st.selectbox(
        "Market",
        list(SUPPORTED_MARKETS.keys()),
        format_func=lambda m: SUPPORTED_MARKETS[m],
    )
    ev_threshold = st.slider("EV threshold (%)", 0.0, 10.0, 1.5, 0.1) / 100.0

    st.markdown("---")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_interval = st.selectbox(
        "Interval",
        [30, 60, 120],
        index=1,
        format_func=lambda x: f"{x}s",
        disabled=not auto_refresh,
    )
    scan_button = st.button("Scan now", type="primary", use_container_width=True)

    st.markdown("---")
    if not api_key.strip():
        st.warning("MOCK MODE")
    else:
        st.success("Live API")


# ════════════════════════════════════════════════════════════════════
#  Scan pipeline
# ════════════════════════════════════════════════════════════════════


def scan(sports: list[str], market: str, threshold: float) -> dict[str, Any]:
    client = OddsAPIClient(api_key=api_key if api_key.strip() else None)
    scanner = Scanner(bankroll=bankroll, kelly_multiplier=kelly_multiplier)
    tracker = get_line_tracker()
    detector = SteamDetector(tracker)

    t0 = time.perf_counter()
    all_arbs: list[pd.DataFrame] = []
    all_evs: list[pd.DataFrame] = []
    all_signals = []
    total_events = 0

    for sport in sports:
        events = run_async(client.fetch_odds(sport, market))
        total_events += len(events)
        # Persist snapshots for the line-movement tab.
        try:
            tracker.insert_snapshot(events, market_key=market, sport=sport)
        except Exception:
            pass
        arb = scanner.scan_arbitrage(events, market)
        ev = scanner.scan_ev(events, market, ev_threshold=threshold)
        if not arb.empty:
            arb["Sport"] = SUPPORTED_SPORTS.get(sport, sport)
            all_arbs.append(arb)
        if not ev.empty:
            ev["Sport"] = SUPPORTED_SPORTS.get(sport, sport)
            all_evs.append(ev)
        all_signals.extend(detector.scan_events(events, market))

    arb_df = pd.concat(all_arbs, ignore_index=True) if all_arbs else pd.DataFrame()
    ev_df = pd.concat(all_evs, ignore_index=True) if all_evs else pd.DataFrame()
    if not ev_df.empty:
        steam_outcomes = {s.team for s in all_signals}
        ev_df = annotate_dataframe(ev_df, steam_outcomes=steam_outcomes)

    # Optional Telegram alerts.
    notifier = get_notifier()
    if notifier.enabled:
        for _, row in ev_df.iterrows():
            if row.get("EV_%", 0) >= 5.0:
                notifier.send(notifier.format_ev_alert(row.to_dict()))
        for _, row in arb_df.iterrows():
            notifier.send(notifier.format_arb_alert(row.to_dict()))
        for sig in all_signals:
            notifier.send(notifier.format_steam_alert(sig))

    return {
        "events": total_events,
        "arb": arb_df,
        "ev": ev_df,
        "signals": all_signals,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    }


if scan_button or (
    auto_refresh and time.time() - st.session_state.get("last_scan_ts", 0) > refresh_interval
):
    if not sports:
        st.error("Select at least one sport in the sidebar.")
    else:
        with st.spinner("Scanning..."):
            st.session_state["scan"] = scan(sports, market, ev_threshold)
            st.session_state["last_scan_ts"] = time.time()


# ════════════════════════════════════════════════════════════════════
#  Tabs
# ════════════════════════════════════════════════════════════════════

tab_scan, tab_lines, tab_clv, tab_sim, tab_settings = st.tabs(
    [
        "Live Scanner",
        "Line Movement",
        "CLV Tracker",
        "Bankroll Sim",
        "Settings",
    ]
)


# ── Tab 1: Live Scanner ────────────────────────────────────────────

with tab_scan:
    st.subheader("Live Scanner")
    scan_data = st.session_state.get("scan")
    if scan_data is None:
        st.info("Click *Scan now* in the sidebar to fetch odds.")
    else:
        ev_df = scan_data["ev"]
        arb_df = scan_data["arb"]
        signals = scan_data["signals"]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", scan_data["events"])
        c2.metric("Arbs", len(arb_df))
        c3.metric("+EV bets", len(ev_df))
        avg_ev = f"{ev_df['EV_%'].mean():.2f}%" if not ev_df.empty else "0.00%"
        c4.metric("Avg edge", avg_ev)

        if signals:
            st.warning(
                f"{len(signals)} steam move(s) detected: "
                + ", ".join(
                    f"{s.team} ({s.from_american:+d}->" f"{s.to_american:+d})" for s in signals[:5]
                )
            )

        st.markdown("**Arbitrage opportunities**")
        if arb_df.empty:
            st.info("No arbs in current scan.")
        else:
            st.dataframe(arb_df, hide_index=True, use_container_width=True)

        st.markdown("**+EV bets (sorted by confidence)**")
        if ev_df.empty:
            st.info("No +EV bets above threshold.")
        else:
            st.dataframe(ev_df, hide_index=True, use_container_width=True)
            st.plotly_chart(
                charts.ev_distribution(ev_df, threshold_pct=ev_threshold * 100),
                use_container_width=True,
            )
            st.plotly_chart(
                charts.opportunity_heatmap(ev_df),
                use_container_width=True,
            )


# ── Tab 2: Line Movement ──────────────────────────────────────────

with tab_lines:
    st.subheader("Line Movement")
    tracker = get_line_tracker()
    rows = tracker.row_count()
    st.caption(f"{rows} snapshots stored at {tracker.db_path}")
    if rows == 0:
        st.info("Run a scan to populate line history.")
    else:
        with tracker._connect() as conn:  # type: ignore[attr-defined]
            df_choices = pd.read_sql_query(
                "SELECT DISTINCT market_key, team FROM line_history "
                "ORDER BY market_key, team LIMIT 200",
                conn,
            )
        if df_choices.empty:
            st.info("No history rows yet.")
        else:
            choice = st.selectbox(
                "Outcome",
                options=df_choices.index,
                format_func=lambda i: (
                    f"{df_choices.loc[i, 'team']} " f"({df_choices.loc[i, 'market_key']})"
                ),
            )
            mkt = str(df_choices.loc[choice, "market_key"])
            team = str(df_choices.loc[choice, "team"])
            hours = st.slider("Window (hours)", 1, 168, 24)
            history = tracker.get_line_history(mkt, team, hours=hours)
            st.plotly_chart(
                charts.line_movement_chart(history, team),
                use_container_width=True,
            )


# ── Tab 3: CLV Tracker ────────────────────────────────────────────

with tab_clv:
    st.subheader("CLV Tracker")
    clv = get_clv_tracker()
    with st.expander("Log a bet", expanded=False):
        c1, c2, c3 = st.columns(3)
        team_in = c1.text_input("Team / outcome", "Lakers")
        book_in = c2.text_input("Book", "DraftKings")
        am_in = c3.number_input("American odds", value=150, step=5)
        sport_in = st.selectbox(
            "Sport", list(SUPPORTED_SPORTS.keys()), format_func=lambda s: SUPPORTED_SPORTS[s]
        )
        market_in = st.selectbox(
            "Market", list(SUPPORTED_MARKETS.keys()), format_func=lambda m: SUPPORTED_MARKETS[m]
        )
        if st.button("Log bet"):
            new_id = clv.log_bet(market_in, team_in, book_in, int(am_in), sport=sport_in)
            st.success(f"Logged bet id={new_id}")

    with st.expander("Settle a bet", expanded=False):
        sid = st.number_input("Bet id", min_value=1, step=1)
        closing = st.number_input("Closing American odds", value=130, step=5)
        if st.button("Settle"):
            try:
                c = clv.settle_bet(int(sid), int(closing))
                st.success(f"Settled. CLV = {c * 100:+.2f}%")
            except ValueError as exc:
                st.error(str(exc))

    stats = clv.aggregate_stats()
    c1, c2, c3 = st.columns(3)
    c1.metric("Settled", stats["n_settled"])
    c2.metric("Mean CLV", f"{stats['mean_clv'] * 100:+.2f}%")
    c3.metric("Win-rate vs close", f"{stats['win_rate_vs_close'] * 100:.1f}%")

    settled = clv.settled_bets()
    if not settled.empty:
        st.plotly_chart(
            charts.clv_histogram(settled["clv"].tolist()),
            use_container_width=True,
        )
        st.dataframe(settled, hide_index=True, use_container_width=True)


# ── Tab 4: Bankroll Simulator ─────────────────────────────────────

with tab_sim:
    st.subheader("Bankroll Simulator")
    c1, c2, c3 = st.columns(3)
    sb = c1.number_input("Starting bankroll", 100, 1_000_000, 1_000, 100)
    ev = c2.slider("Avg EV (%)", 0.0, 15.0, 3.0, 0.1) / 100.0
    odds = c3.slider("Avg decimal odds", 1.5, 5.0, 2.0, 0.1)
    c4, c5, c6 = st.columns(3)
    bpd = c4.slider("Bets per day", 1, 50, 5)
    days = c5.slider("Days", 7, 730, 90)
    nsim = c6.select_slider("Simulations", options=[1_000, 5_000, 10_000, 25_000], value=10_000)
    km = st.slider("Kelly multiplier", 0.05, 1.0, 0.25, 0.05)

    if st.button("Run simulation", type="primary"):
        with st.spinner("Simulating..."):
            try:
                result = simulate_bankroll(
                    starting_bankroll=sb,
                    avg_ev_pct=ev,
                    avg_decimal_odds=odds,
                    kelly_multiplier=km,
                    bets_per_day=bpd,
                    days=days,
                    n_simulations=nsim,
                    seed=42,
                )
                st.session_state["sim_result"] = result
            except ValueError as exc:
                st.error(str(exc))

    sim = st.session_state.get("sim_result")
    if sim is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Median final", f"${sim.median_final:,.0f}")
        c2.metric("p10 / p90", f"${sim.p10_final:,.0f} / ${sim.p90_final:,.0f}")
        c3.metric("Prob of ruin", f"{sim.prob_of_ruin * 100:.1f}%")
        c4.metric(
            "Days to 2x (median)",
            f"{sim.median_days_to_double:.0f}" if sim.median_days_to_double is not None else "N/A",
        )
        st.plotly_chart(
            charts.bankroll_fan_chart(sim.daily_bankroll, sb),
            use_container_width=True,
        )


# ── Tab 5: Settings ───────────────────────────────────────────────

with tab_settings:
    st.subheader("Settings")
    st.markdown("All sidebar controls live here for reference.")
    st.json(
        {
            "API key configured": bool(api_key.strip()),
            "Bankroll": bankroll,
            "Kelly": kelly_multiplier,
            "Sports": sports,
            "Market": market,
            "EV threshold (%)": round(ev_threshold * 100, 2),
            "Auto-refresh": auto_refresh,
            "Refresh interval (s)": refresh_interval,
        }
    )
    st.markdown("**Cache telemetry**")
    cache_info = OddsAPIClient.cache_stats()
    st.json(cache_info)
    st.text(MathEngine.report_cache_stats())


# ════════════════════════════════════════════════════════════════════
#  Footer
# ════════════════════════════════════════════════════════════════════

st.markdown("---")
scan_data = st.session_state.get("scan")
ds_cache = OddsAPIClient.cache_stats()
last_age = ds_cache.get("last_refresh_age_s")
footer = (
    f"Cache: {ds_cache['entries']} entries | " f"Hit rate: {ds_cache['hit_rate'] * 100:.0f}% | "
)
if last_age is not None:
    footer += f"Last refresh: {int(last_age)}s ago | "
if scan_data is not None:
    footer += f"Scan time: {scan_data['elapsed_s']}s | "
footer += "Built on The Odds API | Pinnacle de-vig | Not financial advice"
st.caption(footer)

if auto_refresh and st.session_state.get("scan") is not None:
    time.sleep(1)
    st.rerun()
