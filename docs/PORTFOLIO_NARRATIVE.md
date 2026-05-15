# Two-project portfolio narrative

I build systems that make opaque markets legible.

The first project, **Executive Evasion Index**, scrapes corporate earnings-call transcripts, scores executives on linguistic evasion (hedging, modal verbs, change-of-subject), and surfaces a leaderboard of who is dodging which questions. It's NLP applied to a domain — corporate disclosure — that mostly ships unstructured. The output is a quant-friendly signal where there used to be only narrative.

The second project, **Sports Betting Statistical Arbitrage & +EV Scanner**, treats US sportsbooks as a fragmented order book. It ingests live odds asynchronously, de-vigs Pinnacle to build a fair-mid, and scans soft books (DraftKings, FanDuel, BetMGM) for cross-venue arbitrage and positive-expected-value mispricings. Risk is sized via Kelly, persisted to SQLite line-history, and surfaced through a five-tab Streamlit dashboard. 257 tests at 95% coverage, ~2 ms scanner latency, ~91× vectorized speedup. The vocabulary is sports betting; the mechanics are market microstructure — vig is bid-ask spread, de-vigging is fair-mid construction, +EV scanning is statistical arbitrage, CLV is post-trade alpha attribution.

Together they tell one story: I take messy, asymmetric markets — corporate disclosure, sportsbooks — and turn them into structured, testable, decision-ready signals.
