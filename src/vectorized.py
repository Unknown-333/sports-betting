"""
vectorized.py -- Vectorized (numpy) implementations of the hot-path math.

The scalar implementations in :class:`src.math_engine.MathEngine` are
LRU-cached and fast enough for live scans (a few hundred outcomes per
refresh), but bulk historical analysis -- backtests, the Monte-Carlo
simulator, the benchmark harness -- can scan tens of thousands of
markets at once.  For those workloads we drop into numpy.

These helpers operate on 1-D ``np.ndarray[float | int]`` inputs and
return ``np.ndarray`` outputs.  They contain **zero Python-level loops**.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArr = NDArray[np.float64]


# ──────────────────────────────────────────────────────────────────
#  Conversions
# ──────────────────────────────────────────────────────────────────


def american_to_decimal(american: NDArray[np.int64] | NDArray[np.float64]) -> FloatArr:
    """Vectorized American → Decimal conversion.

    Equivalent to::

        np.where(odds > 0, odds/100 + 1, 100/(-odds) + 1)
    """
    arr = np.asarray(american, dtype=np.float64)
    if np.any(arr == 0):
        raise ValueError("American odds of 0 are undefined.")
    pos = arr / 100.0 + 1.0
    neg = 100.0 / np.abs(arr) + 1.0
    return np.where(arr > 0, pos, neg)


def decimal_to_implied_probability(decimal_odds: FloatArr) -> FloatArr:
    """Vectorized Decimal → Implied probability."""
    arr = np.asarray(decimal_odds, dtype=np.float64)
    if np.any(arr <= 1.0):
        raise ValueError("Decimal odds must all be > 1.0")
    return 1.0 / arr


# ──────────────────────────────────────────────────────────────────
#  Expected value & Kelly (bulk)
# ──────────────────────────────────────────────────────────────────


def expected_value(true_probs: FloatArr, decimal_odds: FloatArr) -> FloatArr:
    """Bulk EV: ``(true_prob * decimal_odds) - 1`` element-wise."""
    p = np.asarray(true_probs, dtype=np.float64)
    d = np.asarray(decimal_odds, dtype=np.float64)
    return p * d - 1.0


def kelly_fraction(
    true_probs: FloatArr,
    decimal_odds: FloatArr,
    kelly_multiplier: float = 0.25,
) -> FloatArr:
    """Bulk Kelly fraction with negative-edge clamp to zero.

    ``f* = (p * b - q) / b``  with  ``b = decimal_odds - 1``.
    """
    p = np.asarray(true_probs, dtype=np.float64)
    d = np.asarray(decimal_odds, dtype=np.float64)
    b = d - 1.0
    # Suppress div-by-zero for d == 1.0 (handled by clamp + mask).
    with np.errstate(divide="ignore", invalid="ignore"):
        full = np.where(b > 0, (p * b - (1.0 - p)) / b, 0.0)
    return np.maximum(full, 0.0) * kelly_multiplier


# ──────────────────────────────────────────────────────────────────
#  Arbitrage detection (matrix form)
# ──────────────────────────────────────────────────────────────────


def arb_inverse_sum(decimal_odds_matrix: FloatArr) -> FloatArr:
    """Sum of inverse decimal odds, one row per market.

    Parameters
    ----------
    decimal_odds_matrix : np.ndarray, shape (n_markets, n_outcomes)
        Each row holds the *best* decimal odds across books for one
        market's outcomes.

    Returns
    -------
    np.ndarray, shape (n_markets,)
        ``inv_sum[i] = sum_j 1 / dec[i, j]``.  Arb exists when ``< 1.0``.
    """
    arr = np.asarray(decimal_odds_matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D matrix, got shape {arr.shape}")
    return np.sum(1.0 / arr, axis=1)


def arb_mask(decimal_odds_matrix: FloatArr) -> NDArray[np.bool_]:
    """Boolean mask: ``True`` for rows where an arb exists (inv_sum < 1)."""
    return arb_inverse_sum(decimal_odds_matrix) < 1.0


# ──────────────────────────────────────────────────────────────────
#  Multiplicative de-vig (row-wise)
# ──────────────────────────────────────────────────────────────────


def devig_rows(implied_probs_matrix: FloatArr) -> FloatArr:
    """Row-wise multiplicative de-vig.

    Each row is normalized to sum to 1.0.  Works for any number of
    outcomes per row (2 for moneyline, 3 for soccer, etc.).
    """
    arr = np.asarray(implied_probs_matrix, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError("All implied probabilities must be > 0.")
    totals = arr.sum(axis=1, keepdims=True)
    return arr / totals
