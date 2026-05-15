"""
data_ingestion.py -- Async Odds Data Ingestion Module

Fetches live odds from The Odds API using non-blocking aiohttp.
If no API key is configured, returns realistic synthetic data
so the dashboard and scanner can be developed offline.

Supported markets:
    - h2h       (Moneyline)
    - player_points, player_rebounds  (Player Props)

Sharp book: Pinnacle ("pinnacle")
Soft books: DraftKings ("draftkings"), FanDuel ("fanduel")
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from datetime import datetime, timezone
from typing import Any

import aiohttp
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

BASE_URL = "https://api.the-odds-api.com/v4"

SUPPORTED_SPORTS = {
    "basketball_nba": "Basketball - NBA",
    "soccer_epl": "Soccer - EPL",
}

SUPPORTED_MARKETS = {
    "h2h": "Moneyline",
    "player_points": "Player Points",
    "player_rebounds": "Player Rebounds",
}

BOOKMAKERS = ["draftkings", "fanduel", "betmgm", "pinnacle"]


class OddsAPIClient:
    """Async client for The Odds API with automatic mock fallback.

    Parameters
    ----------
    api_key : str or None
        The Odds API key. If None or empty, the client operates
        in **mock mode** and returns synthetic odds data.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("ODDS_API_KEY", "")
        self.is_mock = not bool(self.api_key.strip())

        if self.is_mock:
            logger.warning(
                "No ODDS_API_KEY found -- running in MOCK MODE "
                "(synthetic data will be returned)"
            )
        else:
            logger.info("OddsAPIClient initialized with live API key")
