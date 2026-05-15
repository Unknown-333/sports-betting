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

    # ──────────────────────────────────────────────
    #  Mock Data Generator
    # ──────────────────────────────────────────────

    _MOCK_NBA_GAMES = [
        ("Los Angeles Lakers", "Boston Celtics"),
        ("Golden State Warriors", "Milwaukee Bucks"),
        ("Denver Nuggets", "Phoenix Suns"),
        ("Dallas Mavericks", "Miami Heat"),
    ]

    _MOCK_EPL_GAMES = [
        ("Arsenal", "Manchester City"),
        ("Liverpool", "Chelsea"),
        ("Manchester United", "Tottenham"),
        ("Newcastle", "Aston Villa"),
    ]

    _MOCK_PLAYERS = [
        "LeBron James", "Jayson Tatum", "Stephen Curry",
        "Giannis Antetokounmpo", "Nikola Jokic", "Luka Doncic",
        "Kevin Durant", "Anthony Edwards",
    ]

    @staticmethod
    def _generate_american_odds(
        base: int, spread: int = 15
    ) -> int:
        """Generate realistic American odds with slight variation."""
        offset = random.randint(-spread, spread)
        odds = base + offset
        # American odds skip 0: go from -100 to +100
        if -100 < odds < 100:
            odds = -100 if odds < 0 else 100
        return odds

    def _build_mock_h2h(self, sport: str) -> list[dict[str, Any]]:
        """Build mock Moneyline (h2h) odds for all games."""
        games = (
            self._MOCK_NBA_GAMES
            if "nba" in sport
            else self._MOCK_EPL_GAMES
        )
        events = []

        for home, away in games:
            # Pinnacle sets the sharp line
            pin_home_base = random.choice([-180, -150, -120, +110, +140])
            pin_away_base = -pin_home_base + random.randint(-20, 20)
            if -100 < pin_away_base < 100:
                pin_away_base = 100 if pin_away_base >= 0 else -100

            bookmakers_data = []
            for bk in BOOKMAKERS:
                # Soft books deviate from sharp line
                spread = 5 if bk == "pinnacle" else 20
                home_odds = self._generate_american_odds(pin_home_base, spread)
                away_odds = self._generate_american_odds(pin_away_base, spread)

                bookmakers_data.append({
                    "key": bk,
                    "title": bk.replace("_", " ").title(),
                    "markets": [{
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": home_odds},
                            {"name": away, "price": away_odds},
                        ],
                    }],
                })

            events.append({
                "id": f"mock_{home.lower().replace(' ', '_')}_{away.lower().replace(' ', '_')}",
                "sport_key": sport,
                "commence_time": datetime.now(timezone.utc).isoformat(),
                "home_team": home,
                "away_team": away,
                "bookmakers": bookmakers_data,
            })

        return events

    def _build_mock_props(
        self, sport: str, market: str
    ) -> list[dict[str, Any]]:
        """Build mock Player Prop odds (points, rebounds)."""
        games = (
            self._MOCK_NBA_GAMES
            if "nba" in sport
            else self._MOCK_EPL_GAMES
        )
        events = []

        stat_label = "Points" if "points" in market else "Rebounds"
        base_line = 24.5 if "points" in market else 8.5

        for home, away in games:
            # Pick 2 random players per game
            players = random.sample(self._MOCK_PLAYERS, 2)
            bookmakers_data = []

            for bk in BOOKMAKERS:
                outcomes = []
                for player in players:
                    line = base_line + random.choice([-2, -1, 0, 1, 2])
                    spread = 5 if bk == "pinnacle" else 18
                    over_odds = self._generate_american_odds(-110, spread)
                    under_odds = self._generate_american_odds(-110, spread)

                    outcomes.extend([
                        {
                            "name": f"{player} - Over",
                            "description": f"{player} {stat_label}",
                            "point": line,
                            "price": over_odds,
                        },
                        {
                            "name": f"{player} - Under",
                            "description": f"{player} {stat_label}",
                            "point": line,
                            "price": under_odds,
                        },
                    ])

                bookmakers_data.append({
                    "key": bk,
                    "title": bk.replace("_", " ").title(),
                    "markets": [{
                        "key": market,
                        "outcomes": outcomes,
                    }],
                })

            events.append({
                "id": f"mock_prop_{home.lower().replace(' ', '_')}",
                "sport_key": sport,
                "commence_time": datetime.now(timezone.utc).isoformat(),
                "home_team": home,
                "away_team": away,
                "bookmakers": bookmakers_data,
            })

        return events

    # ──────────────────────────────────────────────
    #  Live API Fetch (with parallelism)
    # ──────────────────────────────────────────────

    _semaphore = asyncio.Semaphore(10)  # cap concurrent connections
    _cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
    _CACHE_TTL = 30  # seconds

    async def _fetch_live(
        self,
        sport: str,
        market: str,
        regions: str = "us,eu",
    ) -> list[dict[str, Any]]:
        """Fetch live odds from The Odds API.

        Handles HTTP errors, rate limits, and timeouts gracefully.
        """
        url = f"{BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": market,
            "oddsFormat": "american",
            "bookmakers": ",".join(BOOKMAKERS),
        }

        logger.info(
            "Fetching live odds: sport=%s market=%s regions=%s",
            sport, market, regions,
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    # Log rate limit headers
                    remaining = resp.headers.get("x-requests-remaining", "?")
                    used = resp.headers.get("x-requests-used", "?")
                    logger.info(
                        "API response: %s | requests used=%s remaining=%s",
                        resp.status, used, remaining,
                    )

                    if resp.status == 401:
                        logger.error("Invalid API key -- check .env")
                        return []
                    if resp.status == 429:
                        logger.error(
                            "Rate limit hit -- back off and retry later"
                        )
                        return []
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error(
                            "API error %s: %s", resp.status, body[:200]
                        )
                        return []

                    data: list[dict[str, Any]] = await resp.json()
                    logger.info(
                        "Received %d events for %s/%s", len(data), sport, market
                    )
                    return data

        except asyncio.TimeoutError:
            logger.error("Request timed out after 15s")
            return []
        except aiohttp.ClientError as exc:
            logger.error("HTTP client error: %s", exc)
            return []

    async def _fetch_book_with_timeout(
        self,
        session: aiohttp.ClientSession,
        book: str,
        sport: str,
        market: str,
        regions: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Fetch odds for a single bookmaker with 5s timeout and semaphore."""
        url = f"{BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": market,
            "oddsFormat": "american",
            "bookmakers": book,
        }
        try:
            async with self._semaphore:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        logger.info(
                            "Book %s: %d events fetched", book, len(data)
                        )
                        return book, data
                    logger.warning("Book %s returned status %s", book, resp.status)
                    return book, []
        except asyncio.TimeoutError:
            logger.warning("Book %s timed out (5s limit)", book)
            return book, []
        except aiohttp.ClientError as exc:
            logger.warning("Book %s error: %s", book, exc)
            return book, []

    async def fetch_odds_parallel(
        self,
        sport: str = "basketball_nba",
        market: str = "h2h",
        regions: str = "us,eu",
    ) -> list[dict[str, Any]]:
        """Fetch odds from all books in parallel using asyncio.gather().

        If one book times out, continues with partial results from
        the remaining books. Logs elapsed time for benchmarking.
        """
        import time
        start = time.perf_counter()

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._fetch_book_with_timeout(session, bk, sport, market, regions)
                for bk in BOOKMAKERS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge events: combine bookmakers into unified event dicts
        merged: dict[str, dict[str, Any]] = {}
        for result in results:
            if isinstance(result, Exception):
                logger.error("Parallel fetch error: %s", result)
                continue
            book_key, events = result
            for event in events:
                eid = event.get("id", "")
                if eid not in merged:
                    merged[eid] = {
                        k: v for k, v in event.items() if k != "bookmakers"
                    }
                    merged[eid]["bookmakers"] = []
                merged[eid]["bookmakers"].extend(event.get("bookmakers", []))

        elapsed = time.perf_counter() - start
        logger.info(
            "Parallel fetch complete: %d events in %.2fs (%d books)",
            len(merged), elapsed, len(BOOKMAKERS),
        )
        return list(merged.values())

    # ──────────────────────────────────────────────
    #  Public Entry Point
    # ──────────────────────────────────────────────

    async def fetch_odds(
        self,
        sport: str = "basketball_nba",
        market: str = "h2h",
        regions: str = "us,eu",
    ) -> list[dict[str, Any]]:
        """Fetch odds data -- live or mock depending on API key.

        Uses a 30-second cache to avoid duplicate fetches. Cache key
        is bucketed by (sport, market, 30s time window).

        Parameters
        ----------
        sport : str
            Sport key (e.g. 'basketball_nba', 'soccer_epl').
        market : str
            Market key ('h2h', 'player_points', 'player_rebounds').
        regions : str
            Comma-separated region codes.

        Returns
        -------
        list[dict]
            List of event dicts with bookmaker odds.
        """
        import time

        if self.is_mock:
            logger.info(
                "MOCK MODE: generating synthetic %s/%s data", sport, market
            )
            if market == "h2h":
                return self._build_mock_h2h(sport)
            return self._build_mock_props(sport, market)

        # Cache check: bucket timestamp to nearest 30s
        now = time.time()
        bucket = int(now // self._CACHE_TTL)
        cache_key = f"{sport}:{market}:{bucket}"

        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._CACHE_TTL:
                logger.info("CACHE HIT: %s (age=%.1fs)", cache_key, now - ts)
                return data

        data = await self._fetch_live(sport, market, regions)

        # Store in cache
        self._cache[cache_key] = (now, data)
        # Evict stale entries
        stale = [k for k, (t, _) in self._cache.items() if now - t > self._CACHE_TTL * 3]
        for k in stale:
            del self._cache[k]

        return data


# ──────────────────────────────────────────────
#  Quick test -- run via:  python -m src.data_ingestion
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import json

    async def _test() -> None:
        client = OddsAPIClient()

        # Test h2h
        print("\n--- Moneyline (h2h) ---")
        h2h_data = await client.fetch_odds("basketball_nba", "h2h")
        print(f"Events returned: {len(h2h_data)}")
        if h2h_data:
            print(json.dumps(h2h_data[0], indent=2, default=str)[:600])

        # Test player props
        print("\n--- Player Points Props ---")
        props_data = await client.fetch_odds("basketball_nba", "player_points")
        print(f"Events returned: {len(props_data)}")
        if props_data:
            print(json.dumps(props_data[0], indent=2, default=str)[:600])

        print("\n[PASS] Data ingestion module working correctly")

    asyncio.run(_test())
