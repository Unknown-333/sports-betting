"""
clv_tracker.py -- Closing-Line Value (CLV) bet log.

Lets the user record hypothetical bets and, once the closing odds are
known, computes the CLV (closing-decimal / bet-decimal - 1).  Positive
CLV is the most reliable evidence of a sustainable edge.

Schema
------
``bets``
    id            INTEGER PK AUTOINCREMENT
    market_key    TEXT
    team          TEXT
    book          TEXT
    sport         TEXT
    bet_american  INTEGER
    bet_decimal   REAL
    bet_ts        REAL          -- when the bet was logged
    closing_american INTEGER    -- nullable until settled
    closing_decimal  REAL       -- nullable until settled
    clv           REAL          -- nullable until settled
    settled_ts    REAL          -- nullable until settled
"""

from __future__ import annotations

import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.math_engine import MathEngine

logger = logging.getLogger(__name__)


DEFAULT_CLV_DB = Path(__file__).resolve().parent.parent / "data" / "clv.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_key TEXT NOT NULL,
    team TEXT NOT NULL,
    book TEXT NOT NULL,
    sport TEXT NOT NULL,
    bet_american INTEGER NOT NULL,
    bet_decimal REAL NOT NULL,
    bet_ts REAL NOT NULL,
    closing_american INTEGER,
    closing_decimal REAL,
    clv REAL,
    settled_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_bets_market ON bets(market_key, team);
CREATE INDEX IF NOT EXISTS idx_bets_book ON bets(book);
"""


class CLVTracker:
    """SQLite-backed bet log with CLV settlement."""

    def __init__(self, db_path: str | Path = DEFAULT_CLV_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ───────────────────────────────────────────
    #  Logging
    # ───────────────────────────────────────────

    def log_bet(
        self,
        market_key: str,
        team: str,
        book: str,
        american_odds: int,
        sport: str = "basketball_nba",
        ts: float | None = None,
    ) -> int:
        """Record a hypothetical bet.  Returns the new row id."""
        ts = ts if ts is not None else time.time()
        bet_decimal = MathEngine.american_to_decimal(int(american_odds))
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO bets (market_key, team, book, sport, "
                "bet_american, bet_decimal, bet_ts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (market_key, team, book, sport, int(american_odds), bet_decimal, ts),
            )
            return int(cur.lastrowid or 0)

    def settle_bet(
        self,
        bet_id: int,
        closing_american: int,
        ts: float | None = None,
    ) -> float:
        """Attach a closing line to *bet_id* and compute CLV.

        Returns the CLV value as a decimal fraction
        (e.g. ``0.025`` = +2.5 % CLV).
        """
        ts = ts if ts is not None else time.time()
        closing_decimal = MathEngine.american_to_decimal(int(closing_american))
        with self._connect() as conn:
            row = conn.execute("SELECT bet_decimal FROM bets WHERE id = ?", (bet_id,)).fetchone()
            if row is None:
                raise ValueError(f"No bet with id={bet_id}")
            bet_decimal = float(row[0])
            clv = round(closing_decimal / bet_decimal - 1.0, 6)
            conn.execute(
                "UPDATE bets SET closing_american = ?, closing_decimal = ?, "
                "clv = ?, settled_ts = ? WHERE id = ?",
                (int(closing_american), closing_decimal, clv, ts, bet_id),
            )
        return clv

    # ───────────────────────────────────────────
    #  Reads
    # ───────────────────────────────────────────

    def all_bets(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query("SELECT * FROM bets ORDER BY bet_ts DESC", conn)

    def settled_bets(self) -> pd.DataFrame:
        with self._connect() as conn:
            return pd.read_sql_query(
                "SELECT * FROM bets WHERE clv IS NOT NULL ORDER BY settled_ts DESC",
                conn,
            )

    # ───────────────────────────────────────────
    #  Aggregate stats
    # ───────────────────────────────────────────

    def aggregate_stats(self) -> dict[str, Any]:
        """Mean CLV overall + breakdowns by book, sport, market_key."""
        df = self.settled_bets()
        if df.empty:
            return {
                "n_settled": 0,
                "mean_clv": 0.0,
                "median_clv": 0.0,
                "win_rate_vs_close": 0.0,
                "by_book": {},
                "by_sport": {},
                "by_market": {},
            }
        return {
            "n_settled": int(len(df)),
            "mean_clv": float(round(df["clv"].mean(), 6)),
            "median_clv": float(round(df["clv"].median(), 6)),
            "win_rate_vs_close": float(round((df["clv"] > 0).mean(), 4)),
            "by_book": df.groupby("book")["clv"].mean().round(6).to_dict(),
            "by_sport": df.groupby("sport")["clv"].mean().round(6).to_dict(),
            "by_market": df.groupby("market_key")["clv"].mean().round(6).to_dict(),
        }

    def rolling_clv(self, window: int = 20) -> pd.DataFrame:
        df = self.settled_bets().sort_values("settled_ts")
        if df.empty:
            return df
        df["rolling_mean_clv"] = df["clv"].rolling(window, min_periods=1).mean()
        return df
