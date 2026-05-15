# LinkedIn post draft

> Sportsbooks are just inefficient financial markets — and treating them like one is a fun way to learn quant engineering.

I just shipped a project that scans live odds across DraftKings, FanDuel and BetMGM, treats Pinnacle as the "sharp" reference price, and surfaces two distinct edges:

1. **Positive expected value (+EV)** — soft-book odds priced above Pinnacle's implied fair value.
2. **Cross-book arbitrage** — situations where the inverse decimal odds across books sum to less than one.

The whole pipeline maps cleanly onto financial-markets concepts I care about:

→ Bookmaker vig is the bid-ask spread.
→ Multiplicative de-vigging is fair-mid construction.
→ +EV scanning is statistical arbitrage.
→ Kelly sizing is optimal-fraction position sizing.
→ Closing-line value is post-trade alpha attribution.
→ Steam-move detection is order-flow imbalance.

The technical highlights I am most proud of:

• **257 tests at 95% coverage**, including hypothesis property tests for the math invariants
• Vectorized numpy math layer that runs **~91× faster** than the scalar reference
• Async aiohttp ingestion with semaphore throttling, TCP connection pooling, and a 30-second TTL cache (~98% hit rate under load)
• A 5-tab Streamlit dashboard with Plotly charts, SQLite-backed line-history tracking, a Monte-Carlo bankroll simulator, and optional Telegram alerts
• GitHub Actions CI gating on black, isort, flake8, mypy, and a coverage threshold

This is not a betting service. It is a research and engineering portfolio piece — and a good excuse to practice the same patterns you would use to build a market-making or stat-arb stack.

Code, README, math, and benchmark numbers in the comments.

#Python #Quant #DataEngineering #StatisticalArbitrage #MarketMicrostructure
