"""
test_performance.py -- Tests for Domain 2 (performance) additions.

Covers:
* numeric correctness of the vectorized math primitives in
  ``src.vectorized``;
* equivalence between the vectorized primitives and the scalar
  (LRU-cached) ``MathEngine`` methods on the same inputs;
* the new cache-telemetry helpers (``MathEngine.cache_stats``,
  ``report_cache_stats``, ``clear_caches``);
* the data-ingestion ``cache_stats()`` snapshot;
* the retry/back-off path in ``OddsAPIClient._fetch_book_with_timeout``.
"""

from __future__ import annotations

import asyncio
import re

import numpy as np
import pytest
from aioresponses import aioresponses

from src import vectorized as vec
from src.data_ingestion import OddsAPIClient
from src.math_engine import MathEngine

# ════════════════════════════════════════════════════════════════════
#  Vectorized primitives
# ════════════════════════════════════════════════════════════════════


class TestVectorizedConversions:
    @pytest.mark.parametrize(
        "american,expected_decimal",
        [(150, 2.50), (-200, 1.50), (100, 2.0), (-100, 2.0), (300, 4.0)],
    )
    def test_scalar_value_in_array(self, american, expected_decimal):
        result = vec.american_to_decimal(np.array([american]))
        assert abs(result[0] - expected_decimal) < 1e-6

    def test_bulk_matches_scalar(self):
        """Vectorized output must agree with scalar LRU output."""
        rng = np.random.default_rng(0)
        american = rng.integers(low=-500, high=500, size=200)
        american = np.where(american == 0, 100, american)
        bulk = vec.american_to_decimal(american)
        scalar = np.array([MathEngine.american_to_decimal(int(a)) for a in american])
        np.testing.assert_allclose(bulk, scalar, atol=1e-3)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            vec.american_to_decimal(np.array([0, 100]))

    def test_decimal_to_implied_probability(self):
        decs = np.array([1.5, 2.0, 4.0])
        probs = vec.decimal_to_implied_probability(decs)
        np.testing.assert_allclose(probs, [1 / 1.5, 0.5, 0.25])

    def test_decimal_to_implied_probability_invalid(self):
        with pytest.raises(ValueError):
            vec.decimal_to_implied_probability(np.array([1.0, 1.5]))


class TestVectorizedEVAndKelly:
    def test_expected_value_bulk(self):
        p = np.array([0.5, 0.55, 0.4])
        d = np.array([2.0, 2.1, 2.5])
        ev = vec.expected_value(p, d)
        np.testing.assert_allclose(ev, p * d - 1.0)

    def test_kelly_negative_clamped_to_zero(self):
        # p=0.3, d=2.0 -> negative full Kelly -> clamp to 0
        f = vec.kelly_fraction(np.array([0.3, 0.55]), np.array([2.0, 2.0]), kelly_multiplier=1.0)
        assert f[0] == 0.0
        assert f[1] == pytest.approx(0.10, abs=1e-6)

    def test_kelly_zero_payout_safe(self):
        """``decimal_odds == 1.0`` (zero net payout) returns 0, not NaN."""
        f = vec.kelly_fraction(
            np.array([0.55]),
            np.array([1.0]),
            kelly_multiplier=1.0,
        )
        assert f[0] == 0.0


class TestArbVectorized:
    def test_arb_inverse_sum_shape(self):
        m = np.array([[2.4, 2.2], [1.91, 1.91]])
        s = vec.arb_inverse_sum(m)
        assert s.shape == (2,)
        # Row 0 is an arb (1/2.4 + 1/2.2 < 1), row 1 is not.
        assert s[0] < 1.0
        assert s[1] > 1.0

    def test_arb_mask(self):
        m = np.array([[2.4, 2.2], [1.91, 1.91]])
        mask = vec.arb_mask(m)
        assert mask.tolist() == [True, False]

    def test_arb_inverse_sum_wrong_dims(self):
        with pytest.raises(ValueError):
            vec.arb_inverse_sum(np.array([1.5, 2.0]))


class TestDevigRows:
    def test_devig_rows_sum_to_one(self):
        m = np.array([[0.55, 0.50], [0.40, 0.40, 0.30]], dtype=object)
        # Heterogeneous shapes aren't allowed -- use a true 2-D array:
        m = np.array([[0.55, 0.50], [0.30, 0.75]])
        out = vec.devig_rows(m)
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], atol=1e-9)

    def test_devig_rows_invalid(self):
        with pytest.raises(ValueError):
            vec.devig_rows(np.array([[0.5, 0.0]]))


# ════════════════════════════════════════════════════════════════════
#  MathEngine cache telemetry
# ════════════════════════════════════════════════════════════════════


class TestMathEngineCacheTelemetry:
    def setup_method(self):
        MathEngine.clear_caches()

    def test_clear_caches_resets_counters(self):
        MathEngine.american_to_decimal(150)
        MathEngine.american_to_decimal(150)
        stats_before = MathEngine.cache_stats()
        assert stats_before["american_to_decimal"]["hits"] >= 1

        MathEngine.clear_caches()
        stats_after = MathEngine.cache_stats()
        for s in stats_after.values():
            assert s["hits"] == 0
            assert s["misses"] == 0
            assert s["size"] == 0

    def test_cache_stats_keys(self):
        stats = MathEngine.cache_stats()
        for fname in (
            "american_to_decimal",
            "decimal_to_implied_probability",
            "expected_value",
            "kelly_criterion",
            "devig_probabilities",
        ):
            assert fname in stats
            assert {"hits", "misses", "size", "maxsize", "hit_rate"} <= set(stats[fname].keys())

    def test_hit_rate_after_warm_calls(self):
        for _ in range(10):
            MathEngine.american_to_decimal(150)
        stats = MathEngine.cache_stats()["american_to_decimal"]
        assert stats["hits"] == 9  # first call is the miss
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.9

    def test_report_cache_stats_text(self):
        MathEngine.american_to_decimal(150)
        report = MathEngine.report_cache_stats()
        assert "MathEngine cache report" in report
        assert "american_to_decimal" in report
        assert "hit rate" in report


# ════════════════════════════════════════════════════════════════════
#  OddsAPIClient cache telemetry + retry behaviour
# ════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _reset_cache_counters():
    OddsAPIClient._cache.clear()
    OddsAPIClient._cache_hits = 0
    OddsAPIClient._cache_misses = 0
    OddsAPIClient._last_fetch_ts = 0.0
    yield
    OddsAPIClient._cache.clear()
    OddsAPIClient._cache_hits = 0
    OddsAPIClient._cache_misses = 0
    OddsAPIClient._last_fetch_ts = 0.0


class TestOddsAPICacheStats:
    @pytest.mark.asyncio
    async def test_cache_stats_after_hit_and_miss(self, monkeypatch):
        client = OddsAPIClient(api_key="valid")

        async def fake_fetch(sport, market, regions="us,eu"):
            return [
                {
                    "id": "x",
                    "sport_key": sport,
                    "home_team": "A",
                    "away_team": "B",
                    "bookmakers": [],
                }
            ]

        monkeypatch.setattr(client, "_fetch_live", fake_fetch)

        await client.fetch_odds("basketball_nba", "h2h")
        await client.fetch_odds("basketball_nba", "h2h")  # cache hit

        stats = client.cache_stats()
        assert stats["entries"] >= 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["ttl"] == 30
        assert stats["maxsize"] == 500
        assert stats["last_refresh_age_s"] is not None


class TestOddsAPIRetry:
    @pytest.mark.asyncio
    async def test_retry_succeeds_after_transient_503(self, monkeypatch):
        """A 503 followed by a 200 must surface the 200 payload."""

        # Skip the back-off sleep so the test is fast.
        async def no_sleep(_):
            return None

        monkeypatch.setattr("src.data_ingestion.asyncio.sleep", no_sleep)

        import aiohttp

        client = OddsAPIClient(api_key="valid")
        good_payload = [
            {
                "id": "y",
                "sport_key": "basketball_nba",
                "home_team": "A",
                "away_team": "B",
                "bookmakers": [],
            }
        ]

        with aioresponses() as mocked:
            mocked.get(re.compile(r".*/odds.*"), status=503)
            mocked.get(re.compile(r".*/odds.*"), payload=good_payload)
            async with aiohttp.ClientSession() as session:
                book, data = await client._fetch_book_with_timeout(
                    session,
                    "draftkings",
                    "basketball_nba",
                    "h2h",
                    "us",
                )
        assert book == "draftkings"
        assert data == good_payload

    @pytest.mark.asyncio
    async def test_retry_gives_up_after_repeated_429(self, monkeypatch):
        async def no_sleep(_):
            return None

        monkeypatch.setattr("src.data_ingestion.asyncio.sleep", no_sleep)

        import aiohttp

        client = OddsAPIClient(api_key="valid")

        with aioresponses() as mocked:
            for _ in range(5):
                mocked.get(re.compile(r".*/odds.*"), status=429)
            async with aiohttp.ClientSession() as session:
                book, data = await client._fetch_book_with_timeout(
                    session,
                    "fanduel",
                    "basketball_nba",
                    "h2h",
                    "us",
                )
        assert data == []


class TestOddsAPIParallelWithPool:
    @pytest.mark.asyncio
    async def test_parallel_fetch_uses_connection_pool(self, monkeypatch):
        """Smoke test: fetch_odds_parallel completes with mocked book responses."""

        async def fake_book(self_, session, book, sport, market, regions):
            return book, [
                {
                    "id": f"e_{book}",
                    "sport_key": sport,
                    "home_team": "A",
                    "away_team": "B",
                    "bookmakers": [
                        {
                            "key": book,
                            "title": book.title(),
                            "markets": [
                                {
                                    "key": "h2h",
                                    "outcomes": [
                                        {"name": "A", "price": -110},
                                        {"name": "B", "price": -110},
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]

        monkeypatch.setattr(
            OddsAPIClient,
            "_fetch_book_with_timeout",
            fake_book,
        )
        client = OddsAPIClient(api_key="valid")
        result = await client.fetch_odds_parallel("basketball_nba", "h2h")
        assert len(result) >= 1
