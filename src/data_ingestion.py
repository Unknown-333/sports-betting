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
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp
from cachetools import TTLCache
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
    "americanfootball_nfl": "Football - NFL",
    "baseball_mlb": "Baseball - MLB",
    "icehockey_nhl": "Hockey - NHL",
    "soccer_epl": "Soccer - EPL",
}

# Markets that contain a draw outcome (3-way) for de-vig branching.
THREE_WAY_SPORTS = frozenset({"soccer_epl"})

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
                "No ODDS_API_KEY found -- running in MOCK MODE " "(synthetic data will be returned)"
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

    _MOCK_NFL_GAMES = [
        ("Kansas City Chiefs", "San Francisco 49ers"),
        ("Dallas Cowboys", "Philadelphia Eagles"),
        ("Buffalo Bills", "Cincinnati Bengals"),
    ]

    _MOCK_MLB_GAMES = [
        ("New York Yankees", "Los Angeles Dodgers"),
        ("Houston Astros", "Atlanta Braves"),
        ("Boston Red Sox", "Toronto Blue Jays"),
    ]

    _MOCK_NHL_GAMES = [
        ("Vegas Golden Knights", "Edmonton Oilers"),
        ("Boston Bruins", "Toronto Maple Leafs"),
        ("Colorado Avalanche", "Dallas Stars"),
    ]

    _MOCK_EPL_GAMES = [
        ("Arsenal", "Manchester City"),
        ("Liverpool", "Chelsea"),
        ("Manchester United", "Tottenham"),
        ("Newcastle", "Aston Villa"),
    ]

    _SPORT_GAME_MAP: dict[str, list[tuple[str, str]]] = {
        "basketball_nba": _MOCK_NBA_GAMES,
        "americanfootball_nfl": _MOCK_NFL_GAMES,
        "baseball_mlb": _MOCK_MLB_GAMES,
        "icehockey_nhl": _MOCK_NHL_GAMES,
        "soccer_epl": _MOCK_EPL_GAMES,
    }

    _MOCK_PLAYERS = [
        "LeBron James",
        "Jayson Tatum",
        "Stephen Curry",
        "Giannis Antetokounmpo",
        "Nikola Jokic",
        "Luka Doncic",
        "Kevin Durant",
        "Anthony Edwards",
    ]

    @staticmethod
    def _generate_american_odds(base: int, spread: int = 15) -> int:
        """Generate realistic American odds with slight variation."""
        offset = random.randint(-spread, spread)
        odds = base + offset
        # American odds skip 0: go from -100 to +100
        if -100 < odds < 100:
            odds = -100 if odds < 0 else 100
        return odds

    def _games_for_sport(self, sport: str) -> list[tuple[str, str]]:
        return self._SPORT_GAME_MAP.get(sport, self._MOCK_NBA_GAMES)

    def _build_mock_h2h(self, sport: str) -> list[dict[str, Any]]:
        """Build mock Moneyline (h2h) odds for all games."""
        games = self._games_for_sport(sport)
        is_three_way = sport in THREE_WAY_SPORTS
        events = []

        for home, away in games:
            # Pinnacle sets the sharp line
            pin_home_base = random.choice([-180, -150, -120, +110, +140])
            pin_away_base = -pin_home_base + random.randint(-20, 20)
            if -100 < pin_away_base < 100:
                pin_away_base = 100 if pin_away_base >= 0 else -100
            pin_draw_base = random.choice([+220, +250, +280, +310])

            bookmakers_data = []
            for bk in BOOKMAKERS:
                # Soft books deviate from sharp line
                spread = 5 if bk == "pinnacle" else 20
                home_odds = self._generate_american_odds(pin_home_base, spread)
                away_odds = self._generate_american_odds(pin_away_base, spread)
                outcomes = [
                    {"name": home, "price": home_odds},
                    {"name": away, "price": away_odds},
                ]
                if is_three_way:
                    draw_odds = self._generate_american_odds(pin_draw_base, spread)
                    outcomes.insert(1, {"name": "Draw", "price": draw_odds})

                bookmakers_data.append(
                    {
                        "key": bk,
                        "title": bk.replace("_", " ").title(),
                        "markets": [
                            {
                                "key": "h2h",
                                "outcomes": outcomes,
                            }
                        ],
                    }
                )

            events.append(
                {
                    "id": f"mock_{home.lower().replace(' ', '_')}_{away.lower().replace(' ', '_')}",
                    "sport_key": sport,
                    "commence_time": datetime.now(timezone.utc).isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "bookmakers": bookmakers_data,
                }
            )

        return events

    def _build_mock_props(self, sport: str, market: str) -> list[dict[str, Any]]:
        """Build mock Player Prop odds (points, rebounds)."""
        games = self._games_for_sport(sport)
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

                    outcomes.extend(
                        [
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
                        ]
                    )

                bookmakers_data.append(
                    {
                        "key": bk,
                        "title": bk.replace("_", " ").title(),
                        "markets": [
                            {
                                "key": market,
                                "outcomes": outcomes,
                            }
                        ],
                    }
                )

            events.append(
                {
                    "id": f"mock_prop_{home.lower().replace(' ', '_')}",
                    "sport_key": sport,
                    "commence_time": datetime.now(timezone.utc).isoformat(),
                    "home_team": home,
                    "away_team": away,
                    "bookmakers": bookmakers_data,
                }
            )

        return events

    # ──────────────────────────────────────────────
    #  Live API Fetch (with parallelism)
    # ──────────────────────────────────────────────

    _semaphore = asyncio.Semaphore(10)  # cap concurrent connections
    _CACHE_TTL = 30  # seconds
    _CACHE_MAXSIZE = 500
    # Thread-safe TTL cache.  Wrapping ``time.time`` in a lambda gives
    # us late binding so test code that monkey-patches ``time.time``
    # still controls cache expiry.
    _cache: TTLCache[str, list[dict[str, Any]]] = TTLCache(
        maxsize=_CACHE_MAXSIZE,
        ttl=_CACHE_TTL,
        timer=lambda: time.time(),
    )
    # Internal hit/miss counters for dashboard telemetry.
    _cache_hits: int = 0
    _cache_misses: int = 0
    _last_fetch_ts: float = 0.0

    # Retry policy for transient HTTP failures (429 / 503).
    _RETRY_STATUSES = (429, 503)
    _RETRY_BACKOFFS = (1.0, 2.0, 4.0)  # seconds

    @classmethod
    def cache_stats(cls) -> dict[str, Any]:
        """Snapshot of the TTL cache for dashboard display.

        Returns ``{entries, hits, misses, hit_rate, last_refresh_age_s}``.
        """
        total = cls._cache_hits + cls._cache_misses
        hit_rate = (cls._cache_hits / total) if total else 0.0
        age = time.time() - cls._last_fetch_ts if cls._last_fetch_ts else None
        return {
            "entries": len(cls._cache),
            "maxsize": cls._CACHE_MAXSIZE,
            "ttl": cls._CACHE_TTL,
            "hits": cls._cache_hits,
            "misses": cls._cache_misses,
            "hit_rate": round(hit_rate, 4),
            "last_refresh_age_s": round(age, 1) if age is not None else None,
        }

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
            sport,
            market,
            regions,
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
                        resp.status,
                        used,
                        remaining,
                    )

                    if resp.status == 401:
                        logger.error("Invalid API key -- check .env")
                        return []
                    if resp.status == 429:
                        logger.error("Rate limit hit -- back off and retry later")
                        return []
                    if resp.status != 200:
                        body = await resp.text()
                        logger.error("API error %s: %s", resp.status, body[:200])
                        return []

                    data: list[dict[str, Any]] = await resp.json()
                    logger.info("Received %d events for %s/%s", len(data), sport, market)
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
        """Fetch one bookmaker's odds with a 5 s wall-clock cap and retry.

        Wraps the inner GET in :func:`asyncio.wait_for` so a hung book
        cannot stall the gather.  Transient ``429`` / ``503`` responses
        trigger an exponential back-off retry (1 s -> 2 s -> 4 s).
        """
        url = f"{BASE_URL}/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": market,
            "oddsFormat": "american",
            "bookmakers": book,
        }

        async def _do_request() -> tuple[int, list[dict[str, Any]]]:
            async with self._semaphore:
                async with session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status == 200:
                        payload = await resp.json()
                        return 200, payload
                    return resp.status, []

        start = time.perf_counter()
        attempts = 1 + len(self._RETRY_BACKOFFS)
        for attempt in range(attempts):
            try:
                status, data = await asyncio.wait_for(_do_request(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Book %s timed out (5s limit)", book)
                return book, []
            except aiohttp.ClientError as exc:
                logger.warning("Book %s error: %s", book, exc)
                return book, []

            if status == 200:
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                logger.info(
                    "%s: %dms (%d events)",
                    book.title(),
                    int(elapsed_ms),
                    len(data),
                )
                return book, data

            # Retry on transient errors only.
            if status in self._RETRY_STATUSES and attempt < len(self._RETRY_BACKOFFS):
                delay = self._RETRY_BACKOFFS[attempt]
                logger.warning(
                    "Book %s status %s -- retrying in %.1fs (attempt %d/%d)",
                    book,
                    status,
                    delay,
                    attempt + 1,
                    attempts,
                )
                await asyncio.sleep(delay)
                continue

            logger.warning("Book %s returned status %s -- giving up", book, status)
            return book, []

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
        start = time.perf_counter()

        # TCPConnector pool: limits total sockets, caches DNS for 5 min
        # so repeated polls don't re-resolve the API host every refresh.
        connector = aiohttp.TCPConnector(limit=20, ttl_dns_cache=300)
        async with aiohttp.ClientSession(connector=connector) as session:
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
                    merged[eid] = {k: v for k, v in event.items() if k != "bookmakers"}
                    merged[eid]["bookmakers"] = []
                merged[eid]["bookmakers"].extend(event.get("bookmakers", []))

        elapsed = time.perf_counter() - start
        logger.info(
            "Parallel fetch complete: %d events in %.2fs (%d books)",
            len(merged),
            elapsed,
            len(BOOKMAKERS),
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
        if self.is_mock:
            logger.info("MOCK MODE: generating synthetic %s/%s data", sport, market)
            if market == "h2h":
                return self._build_mock_h2h(sport)
            return self._build_mock_props(sport, market)

        # Cache check: bucket timestamp to nearest TTL window.
        now = time.time()
        bucket = int(now // self._CACHE_TTL)
        cache_key = f"{sport}:{market}:{bucket}"

        try:
            cached = self._cache[cache_key]
        except KeyError:
            cached = None

        if cached is not None:
            type(self)._cache_hits += 1
            logger.info("CACHE HIT: %s", cache_key)
            return cached

        type(self)._cache_misses += 1
        data = await self._fetch_live(sport, market, regions)
        self._cache[cache_key] = data
        type(self)._last_fetch_ts = now
        return data


# ──────────────────────────────────────────────
#  Quick test -- run via:  python -m src.data_ingestion
# ──────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
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
