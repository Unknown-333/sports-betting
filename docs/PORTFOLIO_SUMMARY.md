# Sports Betting Statistical Arbitrage & +EV Scanner

**One-page project summary**

## At a glance

| | |
|---|---|
| **Domain** | Quantitative finance / market microstructure (applied to fragmented sportsbook order books) |
| **Tech stack** | Python 3.11 · asyncio · aiohttp · numpy · pandas · Streamlit · Plotly · SQLite · pytest · Docker |
| **Status** | 257 tests, 95% line coverage, ~2 ms scanner latency, ~91× vectorized speedup |
| **Repo** | `app.py` · `src/` (12 modules) · `tests/` · `notebooks/` · `scripts/benchmark.py` |

## Problem

US-regulated sportsbooks are inefficient retail markets. Soft books (DraftKings, FanDuel, BetMGM) over-round (~5-8% hold) and price for marketing rather than information efficiency. Pinnacle accepts sharp action and runs ~2-3% hold, making it the closest available analogue to a true probability reference. The challenge: ingest live odds across multiple venues, build a fair-mid model, and surface mispricings (positive expected value, cross-book arbitrage) fast enough to act on them — while sizing positions in a risk-aware way.

## Methodology

1. **Async ingestion** — non-blocking aiohttp client, semaphore-throttled, with a 30-second TTL cache and exponential-backoff retry on 429 / 503.
2. **De-vig math** — multiplicative normalization on Pinnacle's implied probabilities to recover fair odds.
3. **Two scanners** — `scan_arbitrage` (inverse-decimal-sum < 1) and `scan_ev` (soft-book offered odds vs Pinnacle fair, threshold-gated).
4. **Risk sizing** — Kelly criterion with quarter / half / full multipliers and configurable bankroll.
5. **Persistence + analytics** — SQLite line history, steam-move detection, closing-line-value tracker, Monte-Carlo bankroll simulator.
6. **UX** — five-tab Streamlit dashboard with Plotly charts, dark theme, optional Telegram alerts.

## Technical achievements

- **Performance.** Pure-numpy vectorized math layer delivers **~91× speedup** over scalar equivalents on 10 k-odds batches; full scanner pass on mock data runs in **~2 ms**; **~98% TTL cache hit-rate** under load.
- **Correctness.** **257 tests / 95% coverage**, including hypothesis-based property tests (round-trip american↔decimal, vig invariants), async client tests with `aioresponses`, and end-to-end integration tests.
- **Production hygiene.** GitHub Actions CI (black + isort + flake8 + mypy + pytest with coverage gate), pre-commit hooks, Dockerfile + docker-compose, reproducible benchmark script.
- **Modularity.** 12-module `src/` package with strict separation: ingestion, math, vectorized math, scanner, line tracker, steam detector, CLV tracker, simulator, confidence scorer, notifier, charts, types.

## Skills demonstrated

- Async I/O at scale (aiohttp + semaphores + TCP connection pooling)
- Vectorized numerical computing (numpy broadcasting, zero Python loops in the hot path)
- LRU + TTL caching as first-class performance tools, with telemetry
- Test discipline: unit, property-based, async, integration; coverage gating in CI
- Quant-style market microstructure modeling (vig, fair-mid, statistical arbitrage)
- Risk management (Kelly sizing, Monte-Carlo bankroll simulation, risk-of-ruin)
- Production UX engineering (Streamlit + Plotly, dark theme, real-time refresh)
- Containerization & CI (Docker, compose, GitHub Actions)
