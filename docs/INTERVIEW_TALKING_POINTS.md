# Five interview talking points

Concrete answers I can give when this project comes up in a quant / data-engineering interview.

---

## 1. "Walk me through the project end to end."

Live odds for five sports across four bookmakers come in via an async aiohttp client that's semaphore-throttled to ten in-flight requests, sits behind a 30-second TTL cache, and retries 429 / 503 with exponential backoff. Pinnacle is treated as the sharp reference: I take its implied probabilities, multiplicatively normalize them to remove the vig, and that gives me a fair-odds estimate. The scanner then does two things in one pass — it looks for cross-book arbitrage (any case where the inverse of the best decimal odds on each side sums to less than one) and for positive expected value (any soft-book quote whose offered odds exceed Pinnacle's fair value above a configurable threshold). Hits get sized via Kelly, written to a SQLite line-history store, and surfaced in a five-tab Streamlit dashboard with Plotly charts. Optional Telegram alerts fire on high-confidence opportunities.

## 2. "Why is this a finance project, not a gambling project?"

The vocabulary changes; the mechanics don't. Bookmaker vig is the bid-ask spread. Multiplicative de-vigging is fair-mid construction across a fragmented order book. +EV scanning is a statistical-arbitrage signal — you are betting that one venue is mispriced relative to the consensus of more-informed venues. Kelly sizing is optimal-fraction sizing under uncertainty about the edge. Steam moves are order-flow imbalance. Closing-line value is post-trade alpha attribution. Every component has a direct counterpart in equity / FX / crypto market making, and the portfolio piece is really about demonstrating that I can build that stack — async ingestion, vectorized math, persistence, risk sizing, monitoring — at production quality.

## 3. "Why Pinnacle as the reference?"

Three reasons. First, Pinnacle accepts sharp action, where most US books restrict or ban winning customers — that means its prices reflect informed flow. Second, Pinnacle's hold sits at roughly 2-3%, versus 5-8% for soft books, so its implied probabilities are closer to true after de-vigging. Third, this is the consensus choice in the public literature on betting-market efficiency (Levitt 2004, Vaughan Williams 1999, Thaler & Ziemba 1988). You could improve on it by building a weighted consensus across multiple sharps (Circa, Bookmaker.eu) — but for a single-reference baseline Pinnacle is the textbook pick.

## 4. "What are the limitations and how would you address them?"

Three honest limitations. First, my "true probability" is only as good as Pinnacle — if Pinnacle is wrong, my fair value is wrong, and basis risk against a single reference is real. A weighted-consensus model would help. Second, latency: I'm running 30-second cache TTL, which is fine for portfolio-piece purposes but too slow for line-shopping in fast-moving markets like in-game; production would need a websocket feed and sub-second invalidation. Third, execution: I detect opportunities but I don't execute, and real soft-book operators slow-walk and limit accounts that show edge. Modeling fill probability and book-specific limits would be the next layer.

## 5. "How would this scale to an institutional setup?"

Replace the per-call aiohttp ingestion with a streaming pipeline (websockets in, Kafka in the middle, a tick-store like ClickHouse or a TAQ-style Parquet lake on the side). Lift the math out of LRU-cached Python into a Numba or Rust hot path so we can scan tens of thousands of markets continuously. Add a portfolio layer on top of the per-bet Kelly so positions across correlated outcomes (same game, same sport, same evening) are sized jointly. Add a risk manager: per-book exposure caps, per-sport caps, drawdown circuit breakers. Wire CLV into a daily attribution report so we can tell skill from variance early. And finally turn the dashboard into a monitoring layer — alerts on cache miss-rate, stale data, broken bookmaker feeds, p99 latency.
