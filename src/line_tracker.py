"""
line_tracker.py -- SQLite-backed line-movement history.

Records every odds snapshot fetched from the live API (or generated in
mock mode) and exposes query helpers for the line-movement dashboard
tab and the steam-move detector.

Schema
------
``line_history``
    id            INTEGER PK AUTOINCREMENT
    timestamp     REAL    -- epoch seconds
    sport         TEXT
    market_key    TEXT    -- 'h2h', 'player_points', ...
    team          TEXT    -- outcome name
    book          TEXT    -- bookmaker key
    american_odds INTEGER
    decimal_odds  REAL
    implied_prob  REAL
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


DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "line_history.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS line_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    sport TEXT NOT NULL,
    market_key TEXT NOT NULL,
    team TEXT NOT NULL,
    book TEXT NOT NULL,
    american_odds INTEGER NOT NULL,
    decimal_odds REAL NOT NULL,
    implied_prob REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_line_lookup
    ON line_history(market_key, team, book, timestamp);
CREATE INDEX IF NOT EXISTS idx_line_recent
    ON line_history(timestamp);
"""


class LineTracker:
    """Append-only SQLite store for odds snapshots."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ───────────────────────────────────────────
    #  Connection helpers
    # ───────────────────────────────────────────

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # ───────────────────────────────────────────
    #  Writes
    # ───────────────────────────────────────────

    def insert_snapshot(
        self,
        events: list[dict[str, Any]],
        market_key: str = "h2h",
        sport: str | None = None,
        ts: float | None = None,
    ) -> int:
        """Insert one row per (event, book, outcome) found in *events*.

        Returns the number of rows inserted.
        """
        ts = ts if ts is not None else time.time()
        rows: list[tuple[Any, ...]] = []

        for event in events:
            sport_key = sport or event.get("sport_key", "unknown")
            mkey_id = f"{event.get('id', '')}::{market_key}"
            for bk in event.get("bookmakers", []):
                book = bk.get("key", "")
                for mkt in bk.get("markets", []):
                    if mkt.get("key") != market_key:
                        continue
                    for o in mkt.get("outcomes", []):
                        try:
                            am = int(o["price"])
                            dec = MathEngine.american_to_decimal(am)
                            ip = MathEngine.decimal_to_implied_probability(dec)
                        except (KeyError, ValueError):
                            continue
                        rows.append((ts, sport_key, mkey_id, o["name"], book, am, dec, ip))

        if not rows:
            return 0

        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO line_history "
                "(timestamp, sport, market_key, team, book, "
                " american_odds, decimal_odds, implied_prob) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        logger.info("Inserted %d line-history rows", len(rows))
        return len(rows)

    # ───────────────────────────────────────────
    #  Queries
    # ───────────────────────────────────────────

    def get_line_history(
        self,
        market_key: str,
        team: str,
        hours: float = 24.0,
        book: str | None = None,
    ) -> pd.DataFrame:
        """All snapshots for (market_key, team) in the last *hours* hours."""
        cutoff = time.time() - hours * 3600.0
        sql = (
            "SELECT timestamp, book, american_odds, decimal_odds, implied_prob "
            "FROM line_history "
            "WHERE market_key = ? AND team = ? AND timestamp >= ?"
        )
        params: list[Any] = [market_key, team, cutoff]
        if book is not None:
            sql += " AND book = ?"
            params.append(book)
        sql += " ORDER BY timestamp ASC"

        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if not df.empty:
            df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        return df

    def get_opening_line(
        self,
        market_key: str,
        team: str,
        book: str | None = None,
    ) -> dict[str, Any] | None:
        """First (oldest) recorded snapshot for the outcome."""
        sql = (
            "SELECT timestamp, book, american_odds, decimal_odds, implied_prob "
            "FROM line_history WHERE market_key = ? AND team = ?"
        )
        params: list[Any] = [market_key, team]
        if book is not None:
            sql += " AND book = ?"
            params.append(book)
        sql += " ORDER BY timestamp ASC LIMIT 1"
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "timestamp": row[0],
            "book": row[1],
            "american_odds": row[2],
            "decimal_odds": row[3],
            "implied_prob": row[4],
        }

    def get_current_line(
        self,
        market_key: str,
        team: str,
        book: str | None = None,
    ) -> dict[str, Any] | None:
        """Most recent snapshot for the outcome."""
        sql = (
            "SELECT timestamp, book, american_odds, decimal_odds, implied_prob "
            "FROM line_history WHERE market_key = ? AND team = ?"
        )
        params: list[Any] = [market_key, team]
        if book is not None:
            sql += " AND book = ?"
            params.append(book)
        sql += " ORDER BY timestamp DESC LIMIT 1"
        with self._connect() as conn:
            cur = conn.execute(sql, params)
            row = cur.fetchone()
        if row is None:
            return None
        return {
            "timestamp": row[0],
            "book": row[1],
            "american_odds": row[2],
            "decimal_odds": row[3],
            "implied_prob": row[4],
        }

    def get_line_movement(
        self,
        market_key: str,
        team: str,
        book: str | None = None,
    ) -> dict[str, Any] | None:
        """Open vs current odds + percent change in implied probability."""
        opening = self.get_opening_line(market_key, team, book)
        current = self.get_current_line(market_key, team, book)
        if opening is None or current is None:
            return None
        if opening["timestamp"] == current["timestamp"]:
            pct_change = 0.0
        else:
            pct_change = (current["implied_prob"] - opening["implied_prob"]) / opening[
                "implied_prob"
            ]
        return {
            "opening": opening,
            "current": current,
            "implied_prob_pct_change": round(pct_change, 6),
            "american_odds_delta": current["american_odds"] - opening["american_odds"],
        }

    # ───────────────────────────────────────────
    #  Steam-move detection
    # ───────────────────────────────────────────

    def detect_steam_move(
        self,
        market_key: str,
        team: str,
        threshold: float = 0.03,
        window_minutes: float = 5.0,
        book: str = "pinnacle",
    ) -> bool:
        """True iff the implied probability moved >= *threshold* on
        *book* inside the last *window_minutes*."""
        info = self.steam_move_info(market_key, team, window_minutes, book)
        return info is not None and abs(info["implied_prob_change"]) >= threshold

    def steam_move_info(
        self,
        market_key: str,
        team: str,
        window_minutes: float = 5.0,
        book: str = "pinnacle",
    ) -> dict[str, Any] | None:
        """Snapshot delta for *book* across the trailing *window_minutes*."""
        cutoff = time.time() - window_minutes * 60.0
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT timestamp, american_odds, implied_prob "
                "FROM line_history "
                "WHERE market_key = ? AND team = ? AND book = ? "
                "  AND timestamp >= ? "
                "ORDER BY timestamp ASC",
                (market_key, team, book, cutoff),
            )
            rows = cur.fetchall()
        if len(rows) < 2:
            return None
        first_ts, first_am, first_ip = rows[0]
        last_ts, last_am, last_ip = rows[-1]
        return {
            "book": book,
            "team": team,
            "from_american": first_am,
            "to_american": last_am,
            "from_ts": first_ts,
            "to_ts": last_ts,
            "implied_prob_change": round(last_ip - first_ip, 6),
            "elapsed_minutes": round((last_ts - first_ts) / 60.0, 2),
        }

    # ───────────────────────────────────────────
    #  Maintenance
    # ───────────────────────────────────────────

    def row_count(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM line_history").fetchone()[0]

    def purge_older_than(self, days: float) -> int:
        cutoff = time.time() - days * 86400.0
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM line_history WHERE timestamp < ?", (cutoff,))
            return cur.rowcount
