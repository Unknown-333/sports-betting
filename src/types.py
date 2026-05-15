"""
types.py -- Shared type definitions for the sports betting system.

Provides TypedDict definitions that mirror The Odds API v4 JSON schema,
giving IDE autocompletion and static type checking across all modules.
"""

from __future__ import annotations

from typing import TypedDict


class Outcome(TypedDict, total=False):
    """A single betting outcome from a bookmaker."""
    name: str
    price: int            # American odds
    point: float          # Line (player props only)
    description: str      # e.g. "LeBron James Points"


class Market(TypedDict):
    """A market within a bookmaker's offering."""
    key: str              # "h2h", "player_points", etc.
    outcomes: list[Outcome]


class Bookmaker(TypedDict):
    """A bookmaker's data for a single event."""
    key: str              # "draftkings", "pinnacle", etc.
    title: str            # Display name
    markets: list[Market]


class Event(TypedDict):
    """A single sporting event with odds from multiple bookmakers."""
    id: str
    sport_key: str
    commence_time: str
    home_team: str
    away_team: str
    bookmakers: list[Bookmaker]


class ArbOpportunity(TypedDict):
    """A detected arbitrage opportunity."""
    Matchup: str
    Side_1: str
    Book_1: str
    Odds_1: int
    Side_2: str
    Book_2: str
    Odds_2: int
    Margin_pct: float     # Column name is "Margin_%"
    Stake_1: str
    Stake_2: str


class EVBet(TypedDict):
    """A detected positive expected value bet."""
    Matchup: str
    Outcome: str
    Bookmaker: str
    Offered_Odds: int
    Pinnacle_Fair: str
    EV_pct: float         # Column name is "EV_%"
    Kelly_Bet: str
