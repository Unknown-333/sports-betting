"""
steam_detector.py -- Real-time steam-move detection on top of LineTracker.

A "steam move" is a fast, large move on the sharp book (Pinnacle) that
typically signals coordinated sharp action.  The implementation polls
:class:`src.line_tracker.LineTracker` for every (market, team) pair in
the current scan and returns those whose Pinnacle line moved more than
``threshold`` (default 3 %) in the last ``window_minutes`` (default 5).

The output is consumed by the dashboard sidebar and by the +EV scanner
to flag opportunities as ``STEAM CONFIRMED``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.line_tracker import LineTracker

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SteamSignal:
    """One detected steam move."""

    market_key: str
    team: str
    book: str
    from_american: int
    to_american: int
    implied_prob_change: float
    elapsed_minutes: float

    def message(self) -> str:
        sign = "+" if self.implied_prob_change >= 0 else ""
        return (
            f"STEAM: {self.team} moved from {self.from_american:+d} "
            f"to {self.to_american:+d} on {self.book.title()} "
            f"({sign}{self.implied_prob_change * 100:.2f}% prob, "
            f"last {self.elapsed_minutes:.1f} min)"
        )


class SteamDetector:
    """Convenience wrapper around :class:`LineTracker.steam_move_info`."""

    def __init__(
        self,
        tracker: LineTracker,
        threshold: float = 0.03,
        window_minutes: float = 5.0,
        sharp_book: str = "pinnacle",
    ) -> None:
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be in (0, 1)")
        self.tracker = tracker
        self.threshold = threshold
        self.window_minutes = window_minutes
        self.sharp_book = sharp_book

    # ───────────────────────────────────────────
    #  Detection
    # ───────────────────────────────────────────

    def scan_events(
        self,
        events: list[dict[str, Any]],
        market_key: str = "h2h",
    ) -> list[SteamSignal]:
        """Return one :class:`SteamSignal` per (event, outcome) qualifying."""
        signals: list[SteamSignal] = []
        for event in events:
            mkey_id = f"{event.get('id', '')}::{market_key}"
            outcomes_seen: set[str] = set()
            for bk in event.get("bookmakers", []):
                if bk.get("key") != self.sharp_book:
                    continue
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != market_key:
                        continue
                    for o in mkt.get("outcomes", []):
                        team = o.get("name")
                        if team is None or team in outcomes_seen:
                            continue
                        outcomes_seen.add(team)
                        info = self.tracker.steam_move_info(
                            market_key=mkey_id,
                            team=team,
                            window_minutes=self.window_minutes,
                            book=self.sharp_book,
                        )
                        if info is None:
                            continue
                        if abs(info["implied_prob_change"]) < self.threshold:
                            continue
                        signals.append(
                            SteamSignal(
                                market_key=mkey_id,
                                team=team,
                                book=self.sharp_book,
                                from_american=int(info["from_american"]),
                                to_american=int(info["to_american"]),
                                implied_prob_change=float(info["implied_prob_change"]),
                                elapsed_minutes=float(info["elapsed_minutes"]),
                            )
                        )
        for s in signals:
            logger.warning(s.message())
        return signals

    # ───────────────────────────────────────────
    #  EV-overlay helper
    # ───────────────────────────────────────────

    def annotate_ev(
        self,
        ev_df: "pd.DataFrame",  # noqa: F821 - imported lazily
        signals: list[SteamSignal],
    ) -> "pd.DataFrame":  # noqa: F821
        """Add a ``Steam_Confirmed`` boolean column to an EV DataFrame.

        A row is flagged when the same outcome name appears in both the
        EV scan and a steam signal.
        """
        if ev_df is None or ev_df.empty:
            return ev_df
        steam_outcomes = {s.team for s in signals}
        out = ev_df.copy()
        out["Steam_Confirmed"] = out["Outcome"].isin(steam_outcomes)
        return out
