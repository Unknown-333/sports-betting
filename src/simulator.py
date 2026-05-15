"""
simulator.py -- Monte-Carlo bankroll simulator.

Vectorized simulation of *n* parallel betting paths with constant
edge.  All ``n_simulations`` paths are realised in one call to
``np.random.random`` -- there are no Python loops over individual bets.

Use case
--------
The dashboard's "Bankroll Sim" tab lets the user vary edge, odds,
Kelly multiplier, daily volume and horizon, then displays the
distribution of final bankrolls and the probability of ruin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.math_engine import MathEngine


@dataclass(frozen=True)
class SimulationResult:
    """Aggregate metrics + raw daily-bankroll matrix for plotting."""

    median_final: float
    p10_final: float
    p90_final: float
    mean_final: float
    prob_of_ruin: float  # P(bankroll <= 10% of starting)
    median_days_to_double: float | None
    daily_bankroll: NDArray[np.float64]  # shape (n_simulations, days+1)
    inputs: dict[str, Any]


def simulate_bankroll(
    starting_bankroll: float = 1000.0,
    avg_ev_pct: float = 0.03,
    avg_decimal_odds: float = 2.0,
    kelly_multiplier: float = 0.25,
    bets_per_day: int = 5,
    days: int = 90,
    n_simulations: int = 10_000,
    seed: int | None = None,
    ruin_threshold: float = 0.10,
) -> SimulationResult:
    """Run a fully-vectorized bankroll Monte Carlo.

    Parameters
    ----------
    starting_bankroll : float
        Initial bankroll for each path.
    avg_ev_pct : float
        Expected edge per bet, expressed as a decimal (``0.03`` = 3 %).
    avg_decimal_odds : float
        Average decimal odds taken on each bet.
    kelly_multiplier : float
        Fractional Kelly multiplier to apply on top of full Kelly.
    bets_per_day : int
        Number of bets each simulated path places per day.
    days : int
        Simulation horizon, in days.
    n_simulations : int
        Number of independent paths.
    seed : int, optional
        ``numpy`` PRNG seed for reproducibility.
    ruin_threshold : float
        Bankroll fraction (of starting) below which the path is "ruined".
    """
    if starting_bankroll <= 0:
        raise ValueError("starting_bankroll must be positive")
    if avg_decimal_odds <= 1.0:
        raise ValueError("avg_decimal_odds must be > 1.0")
    if not (0.0 < kelly_multiplier <= 1.0):
        raise ValueError("kelly_multiplier must be in (0, 1]")
    if bets_per_day < 0 or days < 0 or n_simulations <= 0:
        raise ValueError("bets_per_day, days, n_simulations must be >= 0/positive")

    # ── Derive true win probability from EV definition ──────
    #   EV = p * d - 1   =>   p = (EV + 1) / d
    true_prob = (avg_ev_pct + 1.0) / avg_decimal_odds
    if not (0.0 < true_prob < 1.0):
        raise ValueError(
            f"derived win probability out of range: {true_prob:.4f} "
            "(check avg_ev_pct and avg_decimal_odds)"
        )

    # Kelly bet fraction (constant across the run).
    f = MathEngine.kelly_criterion(
        true_prob,
        avg_decimal_odds,
        kelly_multiplier=kelly_multiplier,
    )
    payout_win = avg_decimal_odds - 1.0  # net profit per $1 wagered
    total_bets = bets_per_day * days

    rng = np.random.default_rng(seed)

    # No bets -> bankroll constant.
    if total_bets == 0:
        daily = np.full((n_simulations, days + 1), starting_bankroll)
        return _summarise(daily, starting_bankroll, ruin_threshold, inputs=_inputs(locals()))

    # outcomes[i, t] = 1 if path i won bet t else 0
    outcomes = (rng.random((n_simulations, total_bets)) < true_prob).astype(np.int8)
    # per-bet multiplier on the bankroll: 1 + f * payout_win on win,
    #                                     1 - f             on loss.
    multipliers = np.where(
        outcomes == 1,
        1.0 + f * payout_win,
        1.0 - f,
    )
    # Cumulative product gives bankroll after each bet, starting at 1.
    bet_cum = np.cumprod(multipliers, axis=1) * starting_bankroll
    # Prepend the starting bankroll column.
    bet_cum = np.concatenate(
        [np.full((n_simulations, 1), starting_bankroll), bet_cum],
        axis=1,
    )
    # Down-sample to one bankroll per day.
    indices = np.arange(0, total_bets + 1, bets_per_day)
    daily = bet_cum[:, indices]

    return _summarise(
        daily,
        starting_bankroll,
        ruin_threshold,
        inputs=_inputs(locals()),
    )


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────


def _inputs(local_vars: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "starting_bankroll",
        "avg_ev_pct",
        "avg_decimal_odds",
        "kelly_multiplier",
        "bets_per_day",
        "days",
        "n_simulations",
        "ruin_threshold",
        "true_prob",
    )
    return {k: local_vars[k] for k in keep if k in local_vars}


def _summarise(
    daily: NDArray[np.float64],
    starting: float,
    ruin_threshold: float,
    inputs: dict[str, Any],
) -> SimulationResult:
    final = daily[:, -1]
    median_final = float(np.median(final))
    p10 = float(np.percentile(final, 10))
    p90 = float(np.percentile(final, 90))
    mean = float(final.mean())
    ruin_level = starting * ruin_threshold
    # A path is "ruined" if it ever drops to/below the threshold.
    ever_ruined = (daily <= ruin_level).any(axis=1)
    prob_ruin = float(ever_ruined.mean())

    # Days-to-double, median across paths that made it.
    target = starting * 2.0
    crossed = daily >= target
    any_double = crossed.any(axis=1)
    if any_double.any():
        first_day = np.argmax(crossed[any_double], axis=1).astype(float)
        median_days_to_double: float | None = float(np.median(first_day))
    else:
        median_days_to_double = None

    return SimulationResult(
        median_final=round(median_final, 2),
        p10_final=round(p10, 2),
        p90_final=round(p90, 2),
        mean_final=round(mean, 2),
        prob_of_ruin=round(prob_ruin, 4),
        median_days_to_double=median_days_to_double,
        daily_bankroll=daily,
        inputs=inputs,
    )
