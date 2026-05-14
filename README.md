# 📊 Sports Betting Statistical Arbitrage & +EV Scanner

A production-grade market microstructure system that identifies pricing inefficiencies across fragmented sportsbook order books (DraftKings, FanDuel, BetMGM).

## What It Does

| Feature | Description |
|---|---|
| **De-Vig Engine** | Strips bookmaker margin from Pinnacle (sharp book) to derive true probabilities |
| **+EV Scanner** | Flags soft-book odds that exceed fair value — the core alpha signal |
| **Arbitrage Detector** | Finds guaranteed-profit hedges across books |
| **Kelly Sizing** | Recommends optimal bet sizes with configurable fractional Kelly |
| **Live Dashboard** | Real-time Streamlit UI with KPI cards and sortable tables |

## Target Markets

- **Moneylines (h2h)** — highly efficient, thin edges
- **Player Props** — where the modern edge lives (points, rebounds, assists)

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd sports-betting-scanner
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Add your Odds API key, or leave blank for mock mode

# 3. Launch dashboard
streamlit run app.py
```

## Project Structure

```
sports-betting/
├── src/
│   ├── __init__.py
│   ├── math_engine.py      # Odds math, de-vig, EV, Kelly
│   ├── data_ingestion.py   # Async API client + mock mode
│   └── scanner.py          # Arb detection & +EV alpha gen
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── .env.example
└── README.md
```

## Data Source

[The Odds API](https://the-odds-api.com/) — real-time odds from 70+ bookmakers.

## License

MIT
