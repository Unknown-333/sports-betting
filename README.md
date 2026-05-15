# Sports Betting Statistical Arbitrage & +EV Scanner

> A production-grade market-microstructure engine that treats sportsbooks as fragmented order books, strips bookmaker vig from Pinnacle (the sharpest reference market), and flags pricing inefficiencies across DraftKings / FanDuel / BetMGM in real time.

---

## Thesis

US-regulated sportsbooks are inefficient retail markets. Soft books (DraftKings, FanDuel, BetMGM) optimize for marketing and recreational hold rather than information-efficient pricing. Pinnacle, by contrast, accepts sharp action, has the lowest hold (~2-3%), and is widely treated by quants as the closest available proxy to a "true probability" reference. Reverse-engineering Pinnacle's vig yields a fair-odds estimate that, when compared to the soft books, surfaces two distinct edges:

1. **Positive expected value (+EV)** — soft odds priced _above_ Pinnacle fair value, which generate long-run profit when sized via Kelly.
2. **Cross-book arbitrage** — situations where the inverse decimal odds across books sum to less than 1, locking in risk-free profit.

This project operationalizes both edges into a single async pipeline, validated end-to-end with 257 tests at 95% coverage.

## Results

| Metric                   | Value                                                | How measured                         |
| ------------------------ | ---------------------------------------------------- | ------------------------------------ |
| Test suite               | **257 tests, 95% line coverage**                     | `pytest --cov=src`                   |
| Test runtime             | **~4.2 s**                                           | full suite, in-memory mocks          |
| Scanner latency (mock)   | **~2 ms / scan**                                     | `scripts/benchmark.py`, 100-run mean |
| Math-engine speedup      | **~91× vectorized vs scalar**                        | numpy batch over 10 k odds           |
| Cache hit rate           | **~98%**                                             | 30-s TTLCache, repeated scans        |
| Modules at 100% coverage | `math_engine`, `confidence`, `vectorized`, `charts`  | coverage report                      |
| Supported sports         | 5 (NBA, NFL, MLB, NHL, EPL)                          | `SUPPORTED_SPORTS`                   |
| Supported markets        | h2h, spreads, totals, player_points, player_rebounds | `SUPPORTED_MARKETS`                  |

## Architecture

```
                          ┌────────────────────┐
                          │   The Odds API     │
                          │   or Mock Mode     │
                          └──────────┬─────────┘
                                     │ async fetch
                                     ▼
   ┌──────────────────────────────────────────────────────────┐
   │  data_ingestion.OddsAPIClient                            │
   │   • aiohttp + Semaphore(10) + TCPConnector(limit=20)     │
   │   • TTLCache(maxsize=500, ttl=30 s)                      │
   │   • Retry on 429/503 with backoff [1, 2, 4] s            │
   └──────────────────────────────┬───────────────────────────┘
                                  │ list[event]
       ┌──────────────────────────┼─────────────────────────────┐
       ▼                          ▼                             ▼
 ┌─────────────┐         ┌────────────────┐            ┌────────────────┐
 │ math_engine │  ◀────▶ │    scanner     │ ────────▶  │  line_tracker  │
 │ (LRU cache) │         │ scan_arbitrage │            │  (SQLite)      │
 │  vectorized │         │   scan_ev      │            └────────┬───────┘
 └─────────────┘         └────────┬───────┘                     │
                                  │ DataFrame                   ▼
                                  │                    ┌────────────────┐
                                  ▼                    │ steam_detector │
                         ┌────────────────┐            └────────────────┘
                         │  confidence    │
                         │  (0-100 score) │
                         └────────┬───────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
       ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
       │ Streamlit UI │   │ TelegramBot  │   │ CLV tracker (DB) │
       │  (5 tabs)    │   │  (alerts)    │   │ + simulator      │
       └──────────────┘   └──────────────┘   └──────────────────┘
```

## The math

**American → decimal**

$$d = \begin{cases} 1 + \dfrac{a}{100}, & a > 0 \\\\ 1 + \dfrac{100}{|a|}, & a < 0 \end{cases}$$

**Decimal → implied probability** (with vig)

$$p_{\text{implied}} = \frac{1}{d}$$

**Vig (overround)** for an n-way market

$$v = \sum_{i=1}^{n} p_{\text{implied},i} - 1$$

**Multiplicative de-vig** (true probability)

$$p_{\text{true},i} = \frac{p_{\text{implied},i}}{\sum_{j=1}^{n} p_{\text{implied},j}}$$

**Pinnacle fair odds**

$$d_{\text{fair}} = \frac{1}{p_{\text{true}}}$$

**Expected value**

$$\text{EV} = p_{\text{true}} \cdot (d_{\text{offered}} - 1) - (1 - p_{\text{true}})$$

**Kelly criterion** (with multiplier $k \in (0, 1]$)

$$f^{*} = k \cdot \frac{p_{\text{true}} \cdot d - 1}{d - 1}$$

**Arbitrage condition**

$$\sum_{i=1}^{n} \frac{1}{d_i^{\text{best}}} < 1$$

with stake split

$$s_i = B \cdot \frac{1 / d_i^{\text{best}}}{\sum_j 1 / d_j^{\text{best}}}$$

## Performance

```
$ python scripts/benchmark.py
scanner.scan_ev   mean = 1.99 ms / call (n=100)
vectorized speedup vs scalar (10 000 odds) = 91.4×
TTL cache hit rate (10 repeats) = 98.0%
peak memory delta during scan = 1.3 MB
```

## Project layout

```
.
├── app.py                    # Streamlit dashboard (5 tabs)
├── src/
│   ├── data_ingestion.py     # Async aiohttp client + TTL cache
│   ├── math_engine.py        # LRU-cached scalar math (vig, devig, Kelly, EV)
│   ├── vectorized.py         # Numpy batch versions
│   ├── scanner.py            # scan_arbitrage + scan_ev
│   ├── line_tracker.py       # SQLite line-history snapshots
│   ├── steam_detector.py     # Sharp-money move detection
│   ├── clv_tracker.py        # Closing-line-value bet log
│   ├── simulator.py          # Monte-Carlo bankroll simulator
│   ├── confidence.py         # 0-100 EV-confidence scoring
│   ├── notifier.py           # Telegram alerts (background thread)
│   ├── charts.py             # Plotly figure builders
│   └── types.py              # TypedDict odds schema
├── tests/                    # 257 tests, 95% coverage
├── scripts/benchmark.py      # Reproducible perf harness
├── outputs/                  # benchmark_results.json + demo CSVs
└── notebooks/math_engine_demo.ipynb
```

## Mapping to finance

| This project                | Quant-finance analogue                        |
| --------------------------- | --------------------------------------------- |
| Pinnacle "true probability" | mid-price of the most-liquid venue            |
| Soft-book offered odds      | quotes on a less-liquid venue                 |
| Bookmaker vig               | bid-ask spread                                |
| De-vig step                 | building a fair-mid model                     |
| +EV scan                    | statistical-arbitrage signal                  |
| Cross-book arb              | latency / venue arbitrage                     |
| Kelly sizing                | optimal-fraction position sizing              |
| Steam-move detection        | order-flow imbalance / informed-trader signal |
| Closing-line value          | post-trade alpha attribution                  |
| Line-history SQLite         | tick-store / TAQ database                     |
| Bankroll Monte Carlo        | risk-of-ruin / VaR backtest                   |

## Installation

```powershell
# Windows / PowerShell
git clone <this-repo>
cd "sports betting"
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt

# (optional) live API key + Telegram alerts
copy .env.example .env
# then edit .env

# run dashboard
streamlit run app.py

# run tests + coverage
pytest --cov=src --cov-report=term-missing
```

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

### Docker

```bash
docker compose up        # builds image, runs dashboard on http://localhost:8501
```

## Environment variables

| Variable             | Required | Purpose                                          |
| -------------------- | -------- | ------------------------------------------------ |
| `THE_ODDS_API_KEY`   | optional | Live odds. Blank = mock mode (good for dev)      |
| `TELEGRAM_BOT_TOKEN` | optional | Push alerts to Telegram                          |
| `TELEGRAM_CHAT_ID`   | optional | Telegram chat to alert                           |
| `LINE_HISTORY_DB`    | optional | SQLite path (defaults to `data/line_history.db`) |
| `CLV_DB`             | optional | SQLite path (defaults to `data/clv.db`)          |

## References

- Levitt, S. D. (2004). _Why are gambling markets organized so differently from financial markets?_ Economic Journal 114(495).
- Thaler, R. H., & Ziemba, W. T. (1988). _Anomalies: Parimutuel betting markets — racetracks and lotteries._ Journal of Economic Perspectives 2(2).
- Vaughan Williams, L. (1999). _Information efficiency in betting markets: a survey._ Bulletin of Economic Research 51(1).
- Kelly, J. L. (1956). _A new interpretation of information rate._ Bell System Technical Journal 35(4).

## Disclaimer

This software is a research and engineering portfolio piece. It is **not** financial advice and is **not** a betting service. Sports gambling is illegal in many jurisdictions; comply with all applicable laws.
