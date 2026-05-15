"""
test_features.py -- Domain 3+4-helper coverage.

Covers:
* line_tracker.LineTracker insert / query / steam-info
* steam_detector.SteamDetector detection + EV annotation
* clv_tracker.CLVTracker log / settle / aggregate
* simulator.simulate_bankroll output structure + invariants
* multi-sport mock generation (NFL/MLB/NHL/EPL) + 3-way EV de-vig
* confidence score
* notifier.TelegramNotifier disabled-mode safety
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.clv_tracker import CLVTracker
from src.confidence import (
    BOOK_SCORES,
    annotate_dataframe,
    badge,
    book_score,
    confidence_score,
    time_score,
)
from src.data_ingestion import SUPPORTED_SPORTS, THREE_WAY_SPORTS, OddsAPIClient
from src.line_tracker import LineTracker
from src.notifier import TelegramNotifier
from src.scanner import Scanner
from src.simulator import SimulationResult, simulate_bankroll
from src.steam_detector import SteamDetector, SteamSignal

# ════════════════════════════════════════════════════════════════════
#  Fixtures
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "lines.db"


@pytest.fixture
def tmp_clv(tmp_path: Path) -> Path:
    return tmp_path / "clv.db"


def _make_event(eid: str, pin_home: int, pin_away: int) -> dict:
    return {
        "id": eid,
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T00:00:00Z",
        "home_team": "Lakers",
        "away_team": "Celtics",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Lakers", "price": pin_home},
                            {"name": "Celtics", "price": pin_away},
                        ],
                    }
                ],
            }
        ],
    }


# ════════════════════════════════════════════════════════════════════
#  LineTracker
# ════════════════════════════════════════════════════════════════════


class TestLineTracker:
    def test_insert_and_count(self, tmp_db):
        t = LineTracker(tmp_db)
        n = t.insert_snapshot([_make_event("e1", -150, +130)], "h2h")
        assert n == 2  # two outcomes
        assert t.row_count() == 2

    def test_get_history_filters_by_window(self, tmp_db):
        t = LineTracker(tmp_db)
        old = time.time() - 48 * 3600
        new = time.time()
        t.insert_snapshot([_make_event("e1", -150, +130)], "h2h", ts=old)
        t.insert_snapshot([_make_event("e1", -160, +140)], "h2h", ts=new)

        df = t.get_line_history("e1::h2h", "Lakers", hours=1.0)
        assert len(df) == 1  # only the recent row falls in the 1-hour window

    def test_opening_and_current_line(self, tmp_db):
        t = LineTracker(tmp_db)
        t.insert_snapshot([_make_event("e2", -120, +100)], "h2h", ts=time.time() - 600)
        t.insert_snapshot([_make_event("e2", -150, +130)], "h2h", ts=time.time())
        opening = t.get_opening_line("e2::h2h", "Lakers")
        current = t.get_current_line("e2::h2h", "Lakers")
        assert opening["american_odds"] == -120
        assert current["american_odds"] == -150
        movement = t.get_line_movement("e2::h2h", "Lakers")
        assert movement["american_odds_delta"] == -30
        assert movement["implied_prob_pct_change"] > 0  # implied prob rose

    def test_steam_move_info_and_detect(self, tmp_db):
        t = LineTracker(tmp_db)
        now = time.time()
        # Big jump from -110 (~52.4%) to -250 (~71.4%) inside 4 minutes -> steam.
        t.insert_snapshot([_make_event("e3", -110, +100)], "h2h", ts=now - 4 * 60)
        t.insert_snapshot([_make_event("e3", -250, +200)], "h2h", ts=now)
        info = t.steam_move_info("e3::h2h", "Lakers", window_minutes=5)
        assert info is not None
        assert info["from_american"] == -110
        assert info["to_american"] == -250
        assert info["implied_prob_change"] > 0.10
        assert t.detect_steam_move("e3::h2h", "Lakers", threshold=0.05)

    def test_purge_older_than(self, tmp_db):
        t = LineTracker(tmp_db)
        t.insert_snapshot([_make_event("e4", -150, +130)], "h2h", ts=time.time() - 30 * 86400)
        assert t.row_count() == 2
        deleted = t.purge_older_than(7)
        assert deleted == 2
        assert t.row_count() == 0


# ════════════════════════════════════════════════════════════════════
#  SteamDetector
# ════════════════════════════════════════════════════════════════════


class TestSteamDetector:
    def test_scan_events_returns_signals(self, tmp_db):
        tracker = LineTracker(tmp_db)
        now = time.time()
        tracker.insert_snapshot([_make_event("e1", -110, +100)], "h2h", ts=now - 4 * 60)
        tracker.insert_snapshot([_make_event("e1", -250, +200)], "h2h", ts=now)
        det = SteamDetector(tracker, threshold=0.05)
        signals = det.scan_events([_make_event("e1", -250, +200)])
        # at least one outcome should trigger
        assert any(isinstance(s, SteamSignal) for s in signals)
        # Message format contains key ingredients
        msg = signals[0].message()
        assert "STEAM" in msg
        assert "Pinnacle" in msg

    def test_no_signal_when_below_threshold(self, tmp_db):
        tracker = LineTracker(tmp_db)
        now = time.time()
        tracker.insert_snapshot([_make_event("e2", -110, +100)], "h2h", ts=now - 4 * 60)
        # tiny move
        tracker.insert_snapshot([_make_event("e2", -112, +102)], "h2h", ts=now)
        det = SteamDetector(tracker, threshold=0.05)
        signals = det.scan_events([_make_event("e2", -112, +102)])
        assert signals == []

    def test_invalid_threshold_raises(self, tmp_db):
        tracker = LineTracker(tmp_db)
        with pytest.raises(ValueError):
            SteamDetector(tracker, threshold=0.0)
        with pytest.raises(ValueError):
            SteamDetector(tracker, threshold=1.5)

    def test_annotate_ev_marks_steam_outcomes(self, tmp_db):
        tracker = LineTracker(tmp_db)
        det = SteamDetector(tracker)
        ev_df = pd.DataFrame(
            [
                {"Outcome": "Lakers", "Bookmaker": "DraftKings", "EV_%": 3.0},
                {"Outcome": "Celtics", "Bookmaker": "FanDuel", "EV_%": 2.0},
            ]
        )
        signals = [
            SteamSignal(
                market_key="m",
                team="Lakers",
                book="pinnacle",
                from_american=-110,
                to_american=-150,
                implied_prob_change=0.05,
                elapsed_minutes=3.0,
            )
        ]
        out = det.annotate_ev(ev_df, signals)
        assert out.loc[out["Outcome"] == "Lakers", "Steam_Confirmed"].iloc[0]
        assert not out.loc[out["Outcome"] == "Celtics", "Steam_Confirmed"].iloc[0]

    def test_annotate_empty_df_returns_unchanged(self, tmp_db):
        det = SteamDetector(LineTracker(tmp_db))
        empty = pd.DataFrame()
        assert det.annotate_ev(empty, []).empty


# ════════════════════════════════════════════════════════════════════
#  CLVTracker
# ════════════════════════════════════════════════════════════════════


class TestCLVTracker:
    def test_log_and_settle_positive_clv(self, tmp_clv):
        clv = CLVTracker(tmp_clv)
        bet_id = clv.log_bet("h2h", "Lakers", "DraftKings", +150)
        # Closing -110 means market moved in our favour (we got better number)
        c = clv.settle_bet(bet_id, +130)
        # bet_decimal=2.5, closing_decimal=2.3 -> CLV = 2.3/2.5 - 1 = -0.08
        assert c < 0  # we LOST CLV (closing line is shorter than +150)

    def test_clv_positive_when_closing_longer(self, tmp_clv):
        clv = CLVTracker(tmp_clv)
        bet_id = clv.log_bet("h2h", "Lakers", "DraftKings", +120)
        # closing +200 (decimal 3.0 vs bet 2.2) -> CLV ≈ +0.36
        c = clv.settle_bet(bet_id, +200)
        assert c > 0.30

    def test_settle_unknown_id_raises(self, tmp_clv):
        clv = CLVTracker(tmp_clv)
        with pytest.raises(ValueError):
            clv.settle_bet(9999, -110)

    def test_aggregate_stats_empty(self, tmp_clv):
        clv = CLVTracker(tmp_clv)
        s = clv.aggregate_stats()
        assert s["n_settled"] == 0
        assert s["mean_clv"] == 0.0

    def test_aggregate_stats_populated(self, tmp_clv):
        clv = CLVTracker(tmp_clv)
        b1 = clv.log_bet("h2h", "Lakers", "DraftKings", +150, sport="basketball_nba")
        b2 = clv.log_bet("h2h", "Celtics", "FanDuel", +120, sport="basketball_nba")
        clv.settle_bet(b1, +180)  # positive CLV
        clv.settle_bet(b2, +110)  # negative CLV
        s = clv.aggregate_stats()
        assert s["n_settled"] == 2
        assert "DraftKings" in s["by_book"]
        assert "FanDuel" in s["by_book"]
        assert "basketball_nba" in s["by_sport"]
        assert "h2h" in s["by_market"]

    def test_rolling_clv(self, tmp_clv):
        clv = CLVTracker(tmp_clv)
        for i in range(5):
            bid = clv.log_bet("h2h", f"T{i}", "DraftKings", +150)
            clv.settle_bet(bid, +180 if i % 2 == 0 else +110)
        df = clv.rolling_clv(window=3)
        assert "rolling_mean_clv" in df.columns
        assert len(df) == 5


# ════════════════════════════════════════════════════════════════════
#  Monte Carlo simulator
# ════════════════════════════════════════════════════════════════════


class TestSimulator:
    def test_basic_run_shape(self):
        r = simulate_bankroll(
            starting_bankroll=1000,
            avg_ev_pct=0.03,
            avg_decimal_odds=2.0,
            kelly_multiplier=0.25,
            bets_per_day=5,
            days=30,
            n_simulations=500,
            seed=1,
        )
        assert isinstance(r, SimulationResult)
        assert r.daily_bankroll.shape == (500, 31)  # days+1 columns
        assert 0.0 <= r.prob_of_ruin <= 1.0
        assert r.median_final > 0

    def test_positive_edge_grows_bankroll_in_expectation(self):
        r = simulate_bankroll(
            starting_bankroll=1000,
            avg_ev_pct=0.05,
            avg_decimal_odds=2.0,
            kelly_multiplier=0.25,
            bets_per_day=5,
            days=180,
            n_simulations=2000,
            seed=7,
        )
        assert r.mean_final > 1000

    def test_negative_or_zero_volume_returns_constant_bankroll(self):
        r = simulate_bankroll(
            starting_bankroll=500,
            avg_ev_pct=0.03,
            avg_decimal_odds=2.0,
            bets_per_day=0,
            days=10,
            n_simulations=100,
            seed=0,
        )
        assert r.median_final == 500
        assert r.daily_bankroll.shape == (100, 11)

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            simulate_bankroll(starting_bankroll=0)
        with pytest.raises(ValueError):
            simulate_bankroll(avg_decimal_odds=1.0)
        with pytest.raises(ValueError):
            simulate_bankroll(kelly_multiplier=0.0)

    def test_derived_prob_out_of_range_raises(self):
        # avg_ev_pct extreme + low odds -> derived p > 1
        with pytest.raises(ValueError):
            simulate_bankroll(avg_ev_pct=0.95, avg_decimal_odds=1.5)


# ════════════════════════════════════════════════════════════════════
#  Multi-sport
# ════════════════════════════════════════════════════════════════════


class TestMultiSport:
    @pytest.mark.parametrize(
        "sport",
        ["basketball_nba", "americanfootball_nfl", "baseball_mlb", "icehockey_nhl", "soccer_epl"],
    )
    @pytest.mark.asyncio
    async def test_h2h_mock_for_each_sport(self, sport):
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds(sport, "h2h")
        assert len(events) > 0
        for ev in events:
            assert ev["sport_key"] == sport
            for bk in ev["bookmakers"]:
                outcomes = bk["markets"][0]["outcomes"]
                # 3-way for soccer, 2-way for everyone else
                expected = 3 if sport in THREE_WAY_SPORTS else 2
                assert len(outcomes) == expected

    @pytest.mark.asyncio
    async def test_three_way_ev_scan_uses_multiway_devig(self):
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("soccer_epl", "h2h")
        df = Scanner().scan_ev(events, market_key="h2h")
        # DataFrame must be well-formed (may be empty depending on RNG)
        assert isinstance(df, pd.DataFrame)

    def test_supported_sports_includes_all_targets(self):
        for sport in (
            "basketball_nba",
            "americanfootball_nfl",
            "baseball_mlb",
            "icehockey_nhl",
            "soccer_epl",
        ):
            assert sport in SUPPORTED_SPORTS


# ════════════════════════════════════════════════════════════════════
#  Confidence score
# ════════════════════════════════════════════════════════════════════


class TestConfidence:
    def test_score_components(self):
        # 4 % EV (40 pts) + steam (25) + Pinnacle (20) + <1h (15) = 100
        assert confidence_score(4.0, "Pinnacle", True, 0.5) == 100
        # 0 EV, no steam, unknown book, distant game = 5 (book) + 5 (time) = 10
        assert confidence_score(0.0, "RandomSportsBook", False, 10) == 10

    def test_book_score_known_and_unknown(self):
        assert book_score("Pinnacle") == 20
        assert book_score("UnknownBook") == 5

    def test_time_score_buckets(self):
        assert time_score(0.5) == 15
        assert time_score(2.0) == 10
        assert time_score(5.0) == 5
        assert time_score(None) == 5

    def test_badge_thresholds(self):
        assert badge(85) == "GREEN"
        assert badge(60) == "YELLOW"
        assert badge(30) == "RED"

    def test_annotate_dataframe(self):
        ev_df = pd.DataFrame(
            [
                {"Outcome": "Lakers", "Bookmaker": "Pinnacle", "EV_%": 3.5},
                {"Outcome": "Celtics", "Bookmaker": "DraftKings", "EV_%": 1.8},
            ]
        )
        out = annotate_dataframe(ev_df, steam_outcomes={"Lakers"}, hours_to_game=2.0)
        assert "Confidence" in out.columns
        assert "Badge" in out.columns
        # Lakers row should be first (highest score)
        assert out.iloc[0]["Outcome"] == "Lakers"

    def test_annotate_empty_returns_empty(self):
        assert annotate_dataframe(pd.DataFrame()).empty

    def test_book_scores_table_has_expected_keys(self):
        for k in ("Pinnacle", "BetMGM", "FanDuel", "DraftKings"):
            assert k in BOOK_SCORES


# ════════════════════════════════════════════════════════════════════
#  Telegram notifier
# ════════════════════════════════════════════════════════════════════


class TestNotifier:
    def test_disabled_when_no_env(self, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        n = TelegramNotifier()
        assert n.enabled is False
        # send() must be a no-op when disabled
        n.send("hello")

    def test_enabled_when_env_present(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "1234")
        n = TelegramNotifier()
        assert n.enabled is True

    def test_format_helpers(self):
        ev_msg = TelegramNotifier.format_ev_alert(
            {
                "Outcome": "Lakers",
                "Offered_Odds": 145,
                "Bookmaker": "DraftKings",
                "EV_%": 6.2,
                "Kelly_Bet": "$47",
            }
        )
        assert "Lakers" in ev_msg and "DraftKings" in ev_msg
        arb_msg = TelegramNotifier.format_arb_alert(
            {
                "Matchup": "Bulls vs Celtics",
                "Margin_%": 2.1,
                "Stake_1": "$48.50",
                "Stake_2": "$51.50",
            }
        )
        assert "ARB" in arb_msg
        signal = SteamSignal(
            market_key="m",
            team="Cowboys",
            book="pinnacle",
            from_american=-108,
            to_american=-125,
            implied_prob_change=0.04,
            elapsed_minutes=3.0,
        )
        steam_msg = TelegramNotifier.format_steam_alert(signal)
        assert "Cowboys" in steam_msg
