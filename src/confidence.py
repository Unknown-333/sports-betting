"""
confidence.py -- Composite 0-100 confidence score for +EV opportunities.

Components
----------
* EV size           -- up to 40 pts  (4 % EV = max)
* Steam confirmed   -- 25 pts        (binary, sharp money agrees)
* Book quality      -- up to 20 pts  (Pinnacle 20, BetMGM 15, FanDuel 10, DK 8)
* Time-to-game      -- up to 15 pts  (<1 h = 15, <3 h = 10, otherwise 5)

The score is rendered as a colour-coded badge in the dashboard:
    >= 80 -> green   (strong)
    50-79 -> yellow  (moderate)
    < 50  -> red     (weak)
"""

from __future__ import annotations


import pandas as pd

BOOK_SCORES: dict[str, int] = {
    "Pinnacle": 20,
    "Betmgm": 15,
    "BetMGM": 15,
    "Fanduel": 10,
    "FanDuel": 10,
    "Draftkings": 8,
    "DraftKings": 8,
}


def book_score(book: str) -> int:
    return BOOK_SCORES.get(book, 5)


def time_score(hours_to_game: float | None) -> int:
    if hours_to_game is None:
        return 5
    if hours_to_game < 1:
        return 15
    if hours_to_game < 3:
        return 10
    return 5


def confidence_score(
    ev_pct: float,  # in percent (e.g. 3.5)
    book: str,
    steam_confirmed: bool = False,
    hours_to_game: float | None = None,
) -> int:
    """Return an integer score in [0, 100]."""
    ev_component = min(ev_pct * 10.0, 40.0)  # 4 % EV -> 40 pts
    steam_component = 25.0 if steam_confirmed else 0.0
    book_component = float(book_score(book))
    time_component = float(time_score(hours_to_game))
    raw = ev_component + steam_component + book_component + time_component
    return int(round(max(0.0, min(100.0, raw))))


def badge(score: int) -> str:
    if score >= 80:
        return "GREEN"
    if score >= 50:
        return "YELLOW"
    return "RED"


def annotate_dataframe(
    ev_df: pd.DataFrame,
    steam_outcomes: set[str] | None = None,
    hours_to_game: float | None = None,
) -> pd.DataFrame:
    """Add ``Confidence`` and ``Badge`` columns to an EV DataFrame.

    Sorted in descending confidence order.
    """
    if ev_df is None or ev_df.empty:
        return ev_df
    steam_outcomes = steam_outcomes or set()
    out = ev_df.copy()
    if "Steam_Confirmed" not in out.columns:
        out["Steam_Confirmed"] = out["Outcome"].isin(steam_outcomes)
    out["Confidence"] = [
        confidence_score(
            ev_pct=float(row["EV_%"]),
            book=str(row["Bookmaker"]),
            steam_confirmed=bool(row["Steam_Confirmed"]),
            hours_to_game=hours_to_game,
        )
        for _, row in out.iterrows()
    ]
    out["Badge"] = out["Confidence"].apply(badge)
    return out.sort_values("Confidence", ascending=False).reset_index(drop=True)
