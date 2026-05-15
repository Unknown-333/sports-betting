# Sports Betting Statistical Arbitrage & +EV Scanner

A production-grade market microstructure system that identifies pricing inefficiencies across fragmented sportsbook order books (DraftKings, FanDuel, BetMGM). Uses Pinnacle as the sharp reference to derive true probabilities via multiplicative de-vigging.

---

## Core Features

| Module | What It Does |
|---|---|
| **De-Vig Engine** | Strips bookmaker margin from Pinnacle (sharp book) to derive true probabilities via multiplicative normalization |
| **+EV Scanner** | Flags soft-book odds exceeding Pinnacle fair value above a configurable threshold (default 1.5%) |
| **Arbitrage Detector** | Finds guaranteed-profit hedges across books when inverse-sum of best decimal odds < 1.0 |
| **Kelly Criterion** | Optimal bet sizing with full, half, and quarter Kelly multipliers |
| **Live Dashboard** | Real-time Streamlit UI with KPI cards, gradient-highlighted tables, auto-refresh, and scan timing |
| **Async Ingestion** | Non-blocking aiohttp client with parallel per-book fetching, semaphore throttling, and 30s TTL cache |
| **Mock Mode** | Synthetic data generators for offline development -- no API key required |

## Target Markets

- **Moneylines (h2h)** -- two-way markets, highly efficient, thin edges
- **Player Props (points, rebounds)** -- where the modern edge lives due to less efficient pricing

## Architecture

```
                      +------------------+
                      |  The Odds API    |
                      |  (or Mock Mode)  |
                      +--------+---------+
                               |
                    async fetch | aiohttp + semaphore(10)
                               v
                 +-------------+--------------+
                 |   data_ingestion.py         |
                 |   OddsAPIClient             |
                 |   - fetch_odds()            |
                 |   - fetch_odds_parallel()   |
                 |   - 30s TTL cache           |
                 +-------------+--------------+
                               |
                      raw events (JSON)
                               |
            +------------------+------------------+
            |                                     |
            v                                     v
  +---------+----------+            +-------------+-----------+
  |  scan_arbitrage()  |            |      scan_ev()          |
  |  Cross-book hedge  |            |  Pinnacle de-vig vs     |
  |  inv_sum < 1.0     |            |  soft-book odds         |
  +--------------------+            +-------------------------+
            |                                     |
            +------------------+------------------+
                               |
                        pandas DataFrames
                               |
                               v
                 +-------------+--------------+
                 |         app.py              |
                 |   Streamlit Dashboard       |
                 |   - KPI cards               |
                 |   - Arb table (green grad)  |
                 |   - +EV table (green grad)  |
                 |   - Auto-refresh timer      |
                 +----------------------------+
```

## Quantitative Methods

### Odds Conversion
```
American -> Decimal:  +150 -> (150/100) + 1 = 2.50
                      -200 -> (100/200) + 1 = 1.50
Decimal -> Implied:   1 / decimal_odds
```

### De-Vigging (Multiplicative)
```
Raw implied probs:    P_a + P_b = 1 + vig (e.g. 1.048)
True probs:           P_a_true = P_a / (P_a + P_b)
                      P_b_true = P_b / (P_a + P_b)
Verify:               P_a_true + P_b_true = 1.0
```

### Expected Value
```
EV = (true_prob * decimal_odds) - 1
Flag if EV > 1.5% (configurable threshold)
```

### Kelly Criterion
```
f* = (p * (d - 1) - (1 - p)) / (d - 1)
where p = true probability, d = decimal odds
Fractional Kelly: f_bet = f* * multiplier (0.25 = quarter Kelly)
Bet size ($): bankroll * f_bet
```

### Arbitrage Condition
```
inv_sum = (1 / best_dec_A) + (1 / best_dec_B)
If inv_sum < 1.0 -> guaranteed profit
Margin% = (1 - inv_sum) * 100
Stake allocation: stake_i = total * (1/dec_i) / inv_sum
```

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url> && cd sports-betting
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure (optional -- leave blank for mock mode)
cp .env.example .env
# Edit .env to add your Odds API key

# 3. Launch dashboard
streamlit run app.py

# 4. Run tests
python -m pytest tests/ -v --cov=src

# 5. Run individual modules
python -m src.math_engine       # Math engine self-test
python -m src.data_ingestion    # Data ingestion self-test
python -m src.scanner           # Full scanner integration test
```

## Project Structure

```
sports-betting/
|-- notebooks/
|   +-- math_engine_demo.ipynb     # Interactive formula walkthrough
|-- src/
|   |-- __init__.py
|   |-- math_engine.py             # Odds conversion, de-vig, EV, Kelly (LRU cached)
|   |-- data_ingestion.py          # Async API client + mock generators + TTL cache
|   +-- scanner.py                 # Arb detection & +EV alpha generation
|-- tests/
|   |-- conftest.py                # 10 shared pytest fixtures
|   |-- test_math_engine.py        # 27 tests: conversion, vig, EV, Kelly, arb math
|   |-- test_data_ingestion.py     # 19 tests: mock mode, schema, HTTP errors
|   |-- test_scanner.py            # 17 tests: arb/EV detection, edge cases
|   |-- test_integration.py        # 8 tests: end-to-end pipeline
|   +-- test_dashboard.py          # 10 tests: imports, pipeline, async runner
|-- app.py                         # Streamlit dashboard with auto-refresh
|-- requirements.txt               # Pinned dependencies
|-- setup.cfg                      # Pytest configuration
|-- Makefile                       # Test/coverage/lint shortcuts
|-- .env.example                   # Environment variable template
+-- README.md
```

## Test Suite

```
92 passed in 0.84s

Coverage:
  src/data_ingestion.py    79%
  src/scanner.py           79%
  src/math_engine.py       51%
  TOTAL                    70%
```

Run commands:
```bash
python -m pytest tests/ -v                          # Full suite
python -m pytest tests/ --ignore=tests/test_integration.py  # Fast (skip integration)
python -m pytest tests/ --cov=src --cov-report=html         # HTML coverage report
```

## Performance

| Optimization | Impact |
|---|---|
| `@lru_cache(512)` on odds conversion | O(1) repeated lookups during scanning |
| `asyncio.gather()` parallel fetch | All 4 books fetched simultaneously |
| `asyncio.Semaphore(10)` | Connection pooling, prevents thundering herd |
| 30s TTL odds cache | Deduplicates rapid re-scans |
| Auto-refresh (30/60/120s) | Continuous monitoring without manual clicks |
| Scan timing in footer | Benchmarking visibility |

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ODDS_API_KEY` | (empty) | The Odds API key. Leave blank for mock mode |
| Bankroll | $1,000 | Total bankroll for Kelly sizing |
| Kelly Multiplier | 0.25 (Quarter) | Risk management: 0.25 / 0.50 / 1.00 |
| EV Threshold | 1.5% | Minimum edge to flag a +EV bet |
| Sport | basketball_nba | Sport to scan |
| Market | h2h | Market type: h2h, player_points, player_rebounds |

## Data Source

[The Odds API](https://the-odds-api.com/) -- real-time odds from 70+ bookmakers worldwide.

## License

MIT
