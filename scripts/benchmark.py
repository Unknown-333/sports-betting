"""
benchmark.py -- Performance harness for the sports-betting scanner.

Runs the live-mode-equivalent scan path 100 times against synthetic
data of varying sizes, then reports latency percentiles, peak memory,
cache hit rates, and the speed-up of the vectorized math primitives
versus the scalar (LRU-cached) ones.

Run with::

    python -m scripts.benchmark

Outputs a JSON report at ``outputs/benchmark_results.json``.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from src import vectorized as vec
from src.data_ingestion import OddsAPIClient
from src.math_engine import MathEngine
from src.scanner import Scanner


N_RUNS = 100
N_LARGE_MARKETS = 10_000
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────


def _percentiles(samples: list[float]) -> dict[str, float]:
    samples_sorted = sorted(samples)
    n = len(samples_sorted)
    if n == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}

    def pct(p: float) -> float:
        idx = max(0, min(n - 1, int(round(p * (n - 1)))))
        return samples_sorted[idx]

    return {
        "mean": round(statistics.mean(samples_sorted) * 1000, 3),     # ms
        "p50": round(pct(0.50) * 1000, 3),
        "p95": round(pct(0.95) * 1000, 3),
        "p99": round(pct(0.99) * 1000, 3),
        "max": round(samples_sorted[-1] * 1000, 3),
        "min": round(samples_sorted[0] * 1000, 3),
    }


# ──────────────────────────────────────────────────────────────────
#  Benchmark 1: scanner end-to-end (mock data)
# ──────────────────────────────────────────────────────────────────


async def bench_scanner() -> dict[str, Any]:
    client = OddsAPIClient(api_key="")  # mock mode
    scanner = Scanner(bankroll=1000, kelly_multiplier=0.25)
    MathEngine.clear_caches()

    # Warm up so the first run doesn't dominate.
    events = await client.fetch_odds("basketball_nba", "h2h")
    scanner.scan_arbitrage(events)
    scanner.scan_ev(events)

    samples: list[float] = []
    for _ in range(N_RUNS):
        events = await client.fetch_odds("basketball_nba", "h2h")
        t0 = time.perf_counter()
        scanner.scan_arbitrage(events)
        scanner.scan_ev(events)
        samples.append(time.perf_counter() - t0)

    return {
        "n_runs": N_RUNS,
        "n_events_per_run": len(events),
        "latency_ms": _percentiles(samples),
    }


# ──────────────────────────────────────────────────────────────────
#  Benchmark 2: scalar vs vectorized hot-path math (10k markets)
# ──────────────────────────────────────────────────────────────────


def bench_math_primitives() -> dict[str, Any]:
    rng = np.random.default_rng(42)

    # Realistic synthetic odds: mix of favorites and dogs.
    american = rng.integers(low=-500, high=500, size=N_LARGE_MARKETS)
    american = np.where(american == 0, 100, american)

    # ── Scalar path (LRU cached) ─────────────────────────────
    MathEngine.clear_caches()
    t0 = time.perf_counter()
    decimals_scalar = [MathEngine.american_to_decimal(int(a)) for a in american]
    probs_scalar = [
        MathEngine.decimal_to_implied_probability(d) for d in decimals_scalar
    ]
    evs_scalar = [
        MathEngine.expected_value(min(max(p, 0.01), 0.99), d)
        for p, d in zip(probs_scalar, decimals_scalar)
    ]
    scalar_ms = (time.perf_counter() - t0) * 1000.0
    scalar_cache = MathEngine.cache_stats()

    # ── Vectorized path (numpy) ──────────────────────────────
    t0 = time.perf_counter()
    decimals_vec = vec.american_to_decimal(american)
    probs_vec = vec.decimal_to_implied_probability(decimals_vec)
    probs_clamped = np.clip(probs_vec, 0.01, 0.99)
    _ = vec.expected_value(probs_clamped, decimals_vec)
    vector_ms = (time.perf_counter() - t0) * 1000.0

    # Sanity: both paths must agree (within 4-decimal quantization).
    np.testing.assert_allclose(
        np.asarray(decimals_scalar), decimals_vec, atol=1e-3,
    )

    return {
        "n_markets": N_LARGE_MARKETS,
        "scalar_ms": round(scalar_ms, 3),
        "vectorized_ms": round(vector_ms, 3),
        "speedup_x": round(scalar_ms / max(vector_ms, 1e-6), 2),
        "scalar_cache_after_run": scalar_cache,
    }


# ──────────────────────────────────────────────────────────────────
#  Benchmark 3: arbitrage matrix scan (vectorized)
# ──────────────────────────────────────────────────────────────────


def bench_arb_matrix() -> dict[str, Any]:
    rng = np.random.default_rng(7)
    n_markets = N_LARGE_MARKETS
    # Two-outcome decimal odds in [1.5, 5.0]
    matrix = rng.uniform(1.5, 5.0, size=(n_markets, 2))
    t0 = time.perf_counter()
    mask = vec.arb_mask(matrix)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "n_markets": n_markets,
        "vectorized_ms": round(elapsed_ms, 3),
        "n_arbs_found": int(mask.sum()),
    }


# ──────────────────────────────────────────────────────────────────
#  Benchmark 4: cache effectiveness on repeated scans
# ──────────────────────────────────────────────────────────────────


async def bench_cache_effectiveness() -> dict[str, Any]:
    """Repeat the same scan many times -- cache hit rate should saturate."""
    MathEngine.clear_caches()
    client = OddsAPIClient(api_key="")
    scanner = Scanner()
    events = await client.fetch_odds("basketball_nba", "h2h")
    for _ in range(50):
        scanner.scan_ev(events)
    return MathEngine.cache_stats()


# ──────────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────────


async def main() -> dict[str, Any]:
    tracemalloc.start()
    snap_before = tracemalloc.take_snapshot()

    scanner_results = await bench_scanner()
    math_results = bench_math_primitives()
    arb_results = bench_arb_matrix()
    cache_results = await bench_cache_effectiveness()

    snap_after = tracemalloc.take_snapshot()
    diff = snap_after.compare_to(snap_before, "filename")
    peak_kb = sum(stat.size for stat in diff) / 1024.0
    tracemalloc.stop()

    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scanner": scanner_results,
        "math_primitives": math_results,
        "arb_matrix": arb_results,
        "cache_effectiveness_after_50_scans": cache_results,
        "memory_delta_kb": round(peak_kb, 1),
    }

    out_path = OUT_DIR / "benchmark_results.json"
    out_path.write_text(json.dumps(report, indent=2))

    # ── Console summary ─────────────────────────────────────
    print("=" * 64)
    print("  SCANNER LATENCY (100 runs, mock data)")
    print("=" * 64)
    lat = scanner_results["latency_ms"]
    print(f"  events / run : {scanner_results['n_events_per_run']}")
    print(f"  mean         : {lat['mean']} ms")
    print(f"  p50 / p95    : {lat['p50']} / {lat['p95']} ms")
    print(f"  p99 / max    : {lat['p99']} / {lat['max']} ms")

    print("\n" + "=" * 64)
    print(f"  MATH PRIMITIVES ({N_LARGE_MARKETS:,} markets)")
    print("=" * 64)
    print(f"  scalar (lru) : {math_results['scalar_ms']} ms")
    print(f"  vectorized   : {math_results['vectorized_ms']} ms")
    print(f"  speed-up     : {math_results['speedup_x']}x")

    print("\n" + "=" * 64)
    print(f"  ARB MATRIX ({N_LARGE_MARKETS:,} markets, vectorized)")
    print("=" * 64)
    print(f"  elapsed      : {arb_results['vectorized_ms']} ms")
    print(f"  arbs found   : {arb_results['n_arbs_found']}")

    print("\n" + "=" * 64)
    print("  MATH-ENGINE CACHE (after 50 repeat scans)")
    print("=" * 64)
    print(MathEngine.report_cache_stats())

    print(f"\n  Memory delta : {report['memory_delta_kb']} KB")
    print(f"  Report saved : {out_path.relative_to(ROOT)}")

    return report


if __name__ == "__main__":
    asyncio.run(main())
