"""
conftest.py -- Shared pytest fixtures for the sports betting test suite.

All fixtures are session-scoped where data is static, and
function-scoped where fresh state is needed per test.
"""

from __future__ import annotations

import pytest

from src.math_engine import MathEngine


# ──────────────────────────────────────────────
#  Math Engine Instance
# ──────────────────────────────────────────────

@pytest.fixture
def math():
    """Fresh MathEngine instance."""
    return MathEngine()


# ──────────────────────────────────────────────
#  Moneyline Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_moneyline_odds():
    """Realistic two-outcome moneyline market: Lakers vs Celtics.

    Pinnacle is the sharp book (tightest margin).
    DraftKings/FanDuel/BetMGM are soft books (wider margin, more deviation).
    """
    return {
        "id": "fixture_lakers_celtics",
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T00:00:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": -155},
                        {"name": "Boston Celtics", "price": +135},
                    ],
                }],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": -150},
                        {"name": "Boston Celtics", "price": +145},
                    ],
                }],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": -160},
                        {"name": "Boston Celtics", "price": +150},
                    ],
                }],
            },
            {
                "key": "betmgm",
                "title": "BetMGM",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": -145},
                        {"name": "Boston Celtics", "price": +130},
                    ],
                }],
            },
        ],
    }


@pytest.fixture
def sample_pinnacle_odds():
    """Sharp book (Pinnacle) odds for Lakers vs Celtics only."""
    return {
        "key": "pinnacle",
        "title": "Pinnacle",
        "markets": [{
            "key": "h2h",
            "outcomes": [
                {"name": "Los Angeles Lakers", "price": -155},
                {"name": "Boston Celtics", "price": +135},
            ],
        }],
    }


# ──────────────────────────────────────────────
#  Player Prop Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def sample_player_prop():
    """Player Points over/under prop with odds from 3 books + Pinnacle."""
    return {
        "id": "fixture_prop_lebron",
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T00:00:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {
                            "name": "LeBron James - Over",
                            "description": "LeBron James Points",
                            "point": 25.5,
                            "price": -115,
                        },
                        {
                            "name": "LeBron James - Under",
                            "description": "LeBron James Points",
                            "point": 25.5,
                            "price": -105,
                        },
                    ],
                }],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {
                            "name": "LeBron James - Over",
                            "description": "LeBron James Points",
                            "point": 25.5,
                            "price": -105,
                        },
                        {
                            "name": "LeBron James - Under",
                            "description": "LeBron James Points",
                            "point": 25.5,
                            "price": -115,
                        },
                    ],
                }],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [{
                    "key": "player_points",
                    "outcomes": [
                        {
                            "name": "LeBron James - Over",
                            "description": "LeBron James Points",
                            "point": 25.5,
                            "price": +100,
                        },
                        {
                            "name": "LeBron James - Under",
                            "description": "LeBron James Points",
                            "point": 25.5,
                            "price": -120,
                        },
                    ],
                }],
            },
        ],
    }


# ──────────────────────────────────────────────
#  Pre-built Arbitrage Scenario
# ──────────────────────────────────────────────

@pytest.fixture
def sample_arb_opportunity():
    """Guaranteed-profit arb: best odds across books sum to < 1.0.

    Book A has Lakers +140 (decimal 2.40)
    Book B has Celtics +120 (decimal 2.20)
    Inverse sum: 1/2.40 + 1/2.20 = 0.4167 + 0.4545 = 0.8712 < 1.0
    Arb margin: 1 - 0.8712 = 12.88% guaranteed profit
    """
    return {
        "id": "fixture_arb",
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T00:00:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": +140},
                        {"name": "Boston Celtics", "price": -200},
                    ],
                }],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": -180},
                        {"name": "Boston Celtics", "price": +120},
                    ],
                }],
            },
        ],
    }


# ──────────────────────────────────────────────
#  Pre-built +EV Scenario
# ──────────────────────────────────────────────

@pytest.fixture
def sample_ev_opportunity():
    """DraftKings offering Celtics +155 while Pinnacle fair value is +140.

    Pinnacle: Lakers -155 / Celtics +135 → de-vig gives true probs.
    DraftKings Celtics +155 (decimal 2.55) should have positive EV
    vs Pinnacle fair value.
    """
    return {
        "id": "fixture_ev",
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T00:00:00Z",
        "home_team": "Los Angeles Lakers",
        "away_team": "Boston Celtics",
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": -155},
                        {"name": "Boston Celtics", "price": +135},
                    ],
                }],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Los Angeles Lakers", "price": -170},
                        {"name": "Boston Celtics", "price": +155},
                    ],
                }],
            },
        ],
    }


# ──────────────────────────────────────────────
#  Mock API Response
# ──────────────────────────────────────────────

@pytest.fixture
def mock_odds_api_response(sample_moneyline_odds):
    """Full realistic API response matching The Odds API v4 schema."""
    return [sample_moneyline_odds]


# ──────────────────────────────────────────────
#  Kelly Criterion Inputs
# ──────────────────────────────────────────────

@pytest.fixture
def sample_kelly_inputs():
    """Known inputs for Kelly Criterion verification.

    true_prob=0.55, decimal_odds=2.10, bankroll=10000
    Full Kelly: f* = (0.55*1.10 - 0.45) / 1.10 = 0.1409...
    """
    return {
        "true_probability": 0.55,
        "decimal_odds": 2.10,
        "bankroll": 10000,
        "full_kelly_expected": 0.140909,
        "half_kelly_expected": 0.070454,
        "quarter_kelly_expected": 0.035227,
    }


# ──────────────────────────────────────────────
#  Scanner Fixtures
# ──────────────────────────────────────────────

@pytest.fixture
def efficient_market():
    """Efficient market: all books offer identical odds → zero edge."""
    return {
        "id": "fixture_efficient",
        "sport_key": "basketball_nba",
        "commence_time": "2026-05-15T00:00:00Z",
        "home_team": "Team A",
        "away_team": "Team B",
        "bookmakers": [
            {
                "key": bk,
                "title": bk.title(),
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": "Team A", "price": -110},
                        {"name": "Team B", "price": -110},
                    ],
                }],
            }
            for bk in ["pinnacle", "draftkings", "fanduel", "betmgm"]
        ],
    }
