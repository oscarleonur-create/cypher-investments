# Advisor

An always-on portfolio advisor. It watches the positions you actually hold,
understands *why* you hold them, and tells you when something changes that
warrants a decision.

Not a screener and not a backtester — the unit of monitoring is the **thesis**,
not the ticker. "NVDA -3%" is noise; "NVDA -3% on a hyperscaler capex cut, which
is driver #2 of your thesis, and your stated invalidation was 'capex guidance
turns down'" is the output this system is built to produce.

## Architecture

One daemon spine feeding two pillars:

```
                    advisor daemon
              jobs · watermarks · events
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  Position mechanics                 Macro & exposure
  DTE, strike breach,                factor sensitivities,
  assignment risk, delta             book exposure, regime,
  drift, IV rank, stops              rotation, residuals
        └─────────────────┬─────────────────┘
                          ▼
      relevance gate → thesis eval → Action Card → push
                          ▼
                   outcome scoring
```

Both pillars emit into one normalized event stream and pass through the same
deterministic relevance gate, so LLM calls only ever run on events that touch
the book.

### Data sources

Free tier only: TastyTrade (positions, balances, market metrics, option
chains), SEC EDGAR via `edgartools` (8-K, 10-Q/K, Form 4, 13D/G), yfinance
(prices, headlines, earnings dates), and a hand-maintained macro calendar.

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/)

## Setup

```bash
poetry install
cp .env.example .env   # fill in API keys
```

## CLI

```bash
advisor --help
advisor -v  # verbose logging
```

### Research

Deep fundamental research, cached to SQLite (`data/research.db`).

```bash
advisor research ticker AAPL          # full 7-layer report
advisor research refresh AAPL         # force rebuild, bypass cache
advisor research view AAPL            # render the cached report
advisor research list                 # all cached reports
advisor research monitor AAPL         # re-check KPIs, show thesis health
advisor research memo AAPL            # IC-format investment memo
advisor research catalysts AAPL       # upcoming catalysts + risk register
advisor research network AAPL         # peers, holders, competitors
advisor research portfolio            # review every holding across accounts
```

Component layers can also be run individually: `statements`, `ratios`,
`valuation`, `ecosystem`, `compare`.

### Data

```bash
advisor data fetch AAPL --start 2023-01-01 --end 2024-01-01
advisor data options AAPL
advisor data inspect AAPL
advisor data cache
```

### Web

```bash
advisor web serve
```

Runs the FastAPI backend and serves the built React SPA on one port. Build the
frontend first with `cd frontend && npm install && npm run build`.

## Layout

| Path | Purpose |
|---|---|
| `src/advisor/research/` | fundamental research engine, portfolio review, theses, KPI tracking |
| `src/advisor/macro/` | regime detection, sector/factor reference data |
| `src/advisor/market/` | TastyTrade client, IV analysis, option chain search |
| `src/advisor/risk/` | position sizing and the go/no-go gate |
| `src/advisor/agent/` | LLM tool-calling loop |
| `src/advisor/api/` | FastAPI routers + WebSocket quote stream |
| `src/advisor/data/` | price/news providers and disk cache |
| `src/advisor/core/` | pricing, greeks, shared models |
| `src/research_agent/` | standalone AI research agent (separate CLI) |
| `frontend/` | React SPA |

## Tests

```bash
poetry run pytest
poetry run ruff check src tests
```
