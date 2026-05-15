"""
test_data_ingestion.py -- Tests for the async odds data ingestion module.

All tests mock HTTP requests -- no real API calls are ever made.
Tests cover: mock mode, API parsing, error handling, schema validation.
"""

from __future__ import annotations

import asyncio

import pytest

from src.data_ingestion import (
    BASE_URL,
    BOOKMAKERS,
    OddsAPIClient,
    SUPPORTED_MARKETS,
    SUPPORTED_SPORTS,
)


# ══════════════════════════════════════════════
#  1. Mock Mode Tests
# ══════════════════════════════════════════════


class TestMockMode:
    """When no API key is provided, client should auto-switch to mock mode."""

    def test_no_key_enables_mock(self):
        """Empty API key triggers mock mode."""
        client = OddsAPIClient(api_key="")
        assert client.is_mock is True

    def test_none_key_enables_mock(self):
        """None API key triggers mock mode."""
        client = OddsAPIClient(api_key=None)
        assert client.is_mock is True

    def test_valid_key_disables_mock(self):
        """Valid API key disables mock mode."""
        client = OddsAPIClient(api_key="test_key_12345")
        assert client.is_mock is False

    @pytest.mark.asyncio
    async def test_mock_h2h_returns_events(self):
        """Mock mode returns realistic h2h events."""
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("basketball_nba", "h2h")
        assert len(events) > 0
        assert isinstance(events, list)

    @pytest.mark.asyncio
    async def test_mock_props_returns_events(self):
        """Mock mode returns realistic player prop events."""
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("basketball_nba", "player_points")
        assert len(events) > 0


# ══════════════════════════════════════════════
#  2. Schema Validation Tests
# ══════════════════════════════════════════════


class TestSchemaValidation:
    """Mock and live data must follow the same schema."""

    @pytest.mark.asyncio
    async def test_mock_event_has_required_fields(self):
        """Every mock event must have: id, sport_key, home_team, away_team, bookmakers."""
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("basketball_nba", "h2h")
        required_fields = {"id", "sport_key", "commence_time",
                           "home_team", "away_team", "bookmakers"}
        for event in events:
            assert required_fields.issubset(event.keys()), (
                f"Missing fields in event: {required_fields - event.keys()}"
            )

    @pytest.mark.asyncio
    async def test_mock_bookmaker_has_required_fields(self):
        """Every bookmaker entry must have: key, title, markets."""
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("basketball_nba", "h2h")
        for event in events:
            for bk in event["bookmakers"]:
                assert "key" in bk
                assert "title" in bk
                assert "markets" in bk
                assert len(bk["markets"]) > 0

    @pytest.mark.asyncio
    async def test_mock_outcome_has_price(self):
        """Every outcome must have a 'price' (American odds) field."""
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("basketball_nba", "h2h")
        for event in events:
            for bk in event["bookmakers"]:
                for mkt in bk["markets"]:
                    for outcome in mkt["outcomes"]:
                        assert "price" in outcome
                        assert isinstance(outcome["price"], int)

    @pytest.mark.asyncio
    async def test_mock_props_have_point_field(self):
        """Player prop outcomes must have 'point' (the line)."""
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("basketball_nba", "player_points")
        for event in events:
            for bk in event["bookmakers"]:
                for mkt in bk["markets"]:
                    for outcome in mkt["outcomes"]:
                        assert "point" in outcome, (
                            f"Prop outcome missing 'point': {outcome}"
                        )

    @pytest.mark.asyncio
    async def test_mock_all_bookmakers_present(self):
        """Mock data should include all configured bookmakers."""
        client = OddsAPIClient(api_key="")
        events = await client.fetch_odds("basketball_nba", "h2h")
        for event in events:
            bk_keys = {bk["key"] for bk in event["bookmakers"]}
            for required_bk in BOOKMAKERS:
                assert required_bk in bk_keys, (
                    f"Bookmaker {required_bk} missing from event"
                )


# ══════════════════════════════════════════════
#  3. Error Handling Tests (Mocked HTTP)
# ══════════════════════════════════════════════


class TestErrorHandling:
    """Test API error scenarios with mocked HTTP responses."""

    @pytest.mark.asyncio
    async def test_api_401_returns_empty(self):
        """Invalid API key (401) returns empty list gracefully."""
        from aioresponses import aioresponses

        client = OddsAPIClient(api_key="invalid_key")
        url_pattern = f"{BASE_URL}/sports/basketball_nba/odds"

        with aioresponses() as mocked:
            mocked.get(url_pattern, status=401)
            result = await client._fetch_live("basketball_nba", "h2h")
            assert result == []

    @pytest.mark.asyncio
    async def test_api_429_rate_limit_returns_empty(self):
        """Rate limit (429) returns empty list gracefully."""
        from aioresponses import aioresponses

        client = OddsAPIClient(api_key="valid_key")
        url_pattern = f"{BASE_URL}/sports/basketball_nba/odds"

        with aioresponses() as mocked:
            mocked.get(url_pattern, status=429)
            result = await client._fetch_live("basketball_nba", "h2h")
            assert result == []

    @pytest.mark.asyncio
    async def test_timeout_returns_empty(self):
        """Request timeout returns empty list, not crash."""
        from aioresponses import aioresponses

        client = OddsAPIClient(api_key="valid_key")
        url_pattern = f"{BASE_URL}/sports/basketball_nba/odds"

        with aioresponses() as mocked:
            mocked.get(url_pattern, exception=asyncio.TimeoutError())
            result = await client._fetch_live("basketball_nba", "h2h")
            assert result == []

    @pytest.mark.asyncio
    async def test_api_200_parses_json(self):
        """Successful 200 response parses JSON payload."""
        import re
        from aioresponses import aioresponses

        client = OddsAPIClient(api_key="valid_key")
        pattern = re.compile(r"^https://api\.the-odds-api\.com/v4/sports/.+/odds.*$")
        mock_payload = [{"id": "test", "sport_key": "basketball_nba",
                         "home_team": "A", "away_team": "B", "bookmakers": []}]

        with aioresponses() as mocked:
            mocked.get(pattern, payload=mock_payload,
                       headers={"x-requests-remaining": "99",
                                "x-requests-used": "1"})
            result = await client._fetch_live("basketball_nba", "h2h")
            assert len(result) == 1
            assert result[0]["id"] == "test"

    @pytest.mark.asyncio
    async def test_mock_mode_format_matches_live(self):
        """Mock data should have the same structure as live data would."""
        client = OddsAPIClient(api_key="")
        mock_events = await client.fetch_odds("basketball_nba", "h2h")

        # Verify structure matches what the scanner expects
        for event in mock_events:
            assert "home_team" in event
            assert "away_team" in event
            assert "bookmakers" in event
            for bk in event["bookmakers"]:
                assert "key" in bk
                assert "markets" in bk


# ══════════════════════════════════════════════
#  4. Constants & Configuration
# ══════════════════════════════════════════════


class TestConstants:
    """Verify module constants are correctly defined."""

    def test_supported_sports_not_empty(self):
        assert len(SUPPORTED_SPORTS) > 0

    def test_supported_markets_not_empty(self):
        assert len(SUPPORTED_MARKETS) > 0

    def test_bookmakers_include_pinnacle(self):
        assert "pinnacle" in BOOKMAKERS

    def test_bookmakers_include_soft_books(self):
        assert "draftkings" in BOOKMAKERS
        assert "fanduel" in BOOKMAKERS
        assert "betmgm" in BOOKMAKERS

    def test_base_url_format(self):
        assert BASE_URL.startswith("https://")
        assert "odds-api" in BASE_URL
