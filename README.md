# Options Advisor

A CLI-based backtesting and strategy system for equities and options. Includes live signal scanning, multi-layer confluence analysis, market-wide screening, walk-forward testing, Monte Carlo simulation, ML signal prediction, an integrated pipeline orchestrator, and an AI-powered research agent.

## Requirements

- Python 3.12+
- [Poetry](https://python-poetry.org/)

## Setup

```bash
poetry install
```

Copy `.env.example` to `.env` and fill in API keys if using the research agent:

```bash
cp .env.example .env
```

## CLI

All commands are accessible through the `advisor` entrypoint:

```bash
advisor --help
advisor -v  # Enable verbose logging
```

### Strategy

```bash
# List all registered strategies
advisor strategy list

# Show details for a specific strategy
advisor strategy info sma_crossover
```

### Backtest

```bash
# Run a backtest
advisor backtest run sma_crossover --symbol AAPL --start 2023-01-01 --end 2024-01-01

# Override strategy parameters
advisor backtest run pead --symbol TSLA --start 2023-01-01 --end 2024-01-01 \
  --param hold_days=30 --param volume_spike_factor=2.5

# Use ATR-based position sizing and slippage
advisor backtest run mean_reversion --symbol MSFT --start 2023-01-01 --end 2024-01-01 \
  --sizer atr --slippage 0.002

# Options strategies use the Black-Scholes backtester
advisor backtest run naked_put --symbol AAPL --start 2023-01-01 --end 2024-01-01

# Monte Carlo simulation for put credit spreads
advisor backtest run put_credit_spread --symbol AAPL --monte-carlo --mc-paths 10000

# List saved results
advisor backtest results

# Show a specific run
advisor backtest show <run_id>

# Walk-forward analysis
advisor backtest walk-forward sma_crossover --symbol AAPL \
  --start 2022-01-01 --end 2024-01-01 --windows 5 --train-pct 0.7
```

### Data

```bash
# Fetch historical price data
advisor data fetch AAPL --start 2023-01-01 --end 2024-01-01

# Fetch options chain
advisor data options AAPL

# Inspect ticker metadata (sector, market cap, P/E, etc.)
advisor data inspect AAPL

# View or clear the disk cache
advisor data cache
advisor data cache --clear
```

### Signal

```bash
# Scan a symbol for live signals across all equity strategies
advisor signal scan AAPL

# Scan with a specific strategy
advisor signal scan AAPL --strategy pead

# Scan with a different data interval
advisor signal scan AAPL --interval 1h
```

### Confluence

Runs a 3-layer analysis pipeline (technical + sentiment + fundamental) and produces an ENTER / CAUTION / PASS verdict:

```bash
advisor confluence scan AAPL --strategy momentum_breakout --verbose

# Force all layers to run even without a technical breakout (for dip analysis)
advisor confluence scan AAPL --force
```

### Market

Market-wide scanning through layered filters, then confluence on qualifiers:

```bash
# Full market scan
advisor market scan --strategy pead

# Filter by sector and minimum thresholds
advisor market scan --strategy buy_the_dip \
  --include-sector Technology --min-volume 1000000 --min-cap 5.0

# Dry run (filters only, no confluence)
advisor market scan --strategy momentum_breakout --dry-run

# Scan a different universe
advisor market scan --strategy pead --universe semiconductors
```

#### Smart Money Scanner

Scans for insider buying, congressional trades, technicals, and options activity:

```bash
# Full S&P 500 scan
advisor market smart-money

# Single ticker deep-dive
advisor market smart-money --ticker AAPL
```

#### Mispricing Scanner

Finds mispriced stocks using fundamental, options market, and estimate revision signals:

```bash
# Full scan
advisor market mispricing

# Filter by sector
advisor market mispricing --sector Technology

# Single ticker
advisor market mispricing --ticker AAPL
```

#### Alpha Score

Computes a unified conviction score (0-100) across all signal layers:

```bash
# Full S&P 500 scan
advisor market alpha

# Single ticker with all layers
advisor market alpha --ticker AAPL

# Skip optional layers
advisor market alpha --skip-sentiment --skip-ml
```

### Options

Options scanning, analysis, trade tracking, and simulation:

```bash
# Scan for naked puts and credit spreads
advisor options scan --account-size 5000 --universe wheel
advisor options scan --tickers AAPL,MSFT,GOOG --live-iv

# Smart premium scanner with composite scoring
advisor options premium-scan --min-iv-pctile 30 --min-dte 25 --max-dte 45
advisor options premium-scan --strategies naked_put --live-iv --top 10

# Analyze an options chain (Greeks, yields, strikes)
advisor options analyze AAPL
advisor options analyze AAPL --expiry 2026-04-17

# Historical max-move drawdown analysis for tail risk sizing
advisor options max-move AAPL --dte 21,30,45 --lookback 252

# Show live TastyTrade account balances and positions
advisor options account
```

#### Monte Carlo Simulator

```bash
# Rank put credit spreads by risk-adjusted EV
advisor options simulate --account-size 5000 --universe wheel --paths 10000

# Deep simulation with custom tickers
advisor options simulate --tickers AAPL,MSFT --deep-paths 100000 --top 10

# Validate expired predictions against historical prices (Brier scores)
advisor options validate

# Historical replay validation
advisor options backtest-validate --symbol AAPL --start 2025-06-01 --end 2025-12-31
```

#### Trade Tracking

```bash
# Log a new trade
advisor options track open naked_put AAPL 150.0 2026-04-17 2.50
advisor options track open put_credit_spread AAPL 150.0 2026-04-17 1.20 --long-strike 145.0

# Close a trade
advisor options track close <trade_id> 0.50 --reason profit

# View open positions
advisor options track status

# View closed trades and win rate
advisor options track history
```

### Simulator

```bash
# Launch the Streamlit Monte Carlo simulator GUI
advisor simulator ui
advisor simulator ui --port 8501 --no-browser
```

### ML

ML signal tracking, model training, and prediction:

```bash
# Check ML system status
advisor ml status

# Train a model (logistic, lightgbm, or ensemble)
advisor ml train --model lightgbm --symbols AAPL,MSFT --lookback 5y
advisor ml train --compare  # Compare all model types

# Train with meta-labeling for improved precision
advisor ml train-meta --model lightgbm

# Run live prediction for a symbol
advisor ml predict AAPL
advisor ml predict AAPL --explain  # Show feature breakdown

# Inspect computed ML features
advisor ml features AAPL

# Backtest ML signals on out-of-sample data
advisor ml backtest AAPL --buy-threshold 0.65

# Precision at high probability thresholds
advisor ml precision --model lightgbm

# Walk-forward validation (IS vs OOS gap)
advisor ml walk-forward --model lightgbm --windows 5

# Detect current market regime (HMM)
advisor ml regime
advisor ml regime --fit --lookback 5y  # Force re-fit

# Compute HRP portfolio weights
advisor ml allocate AAPL,MSFT,GOOG

# Model versioning
advisor ml versions
advisor ml rollback <version_id>

# Manual signal logging and resolution
advisor ml log AAPL scanner_name 0.85 --verdict BUY --price 180.0
advisor ml resolve
advisor ml stats --scanner scanner_name
```

### Pipeline

Integrated daily trading pipeline — discovers opportunities, validates fundamentals, times IV, runs Monte Carlo simulation, and scores by conviction:

```bash
# Run the full pipeline on a universe
advisor pipeline run --universe wheel --account-size 5000

# Run on specific tickers with custom thresholds
advisor pipeline run --tickers AAPL,MSFT --min-conviction 60 --top 10

# Verbose mode shows per-layer conviction breakdown
advisor pipeline run --universe semiconductors --verbose

# Available universes: wheel, leveraged, blue_chip, sp500, semiconductors
```

## Strategies

All strategies extend `StrategyBase` and register via the `@StrategyRegistry.register` decorator.

### Equity

| Strategy | Key | Description |
|---|---|---|
| Buy & Hold | `buy_hold` | Benchmark strategy; buys on the first bar and holds |
| SMA Crossover | `sma_crossover` | Golden cross / death cross on SMA(20) vs SMA(50) |
| Momentum Breakout | `momentum_breakout` | Breakout above SMA(20) with volume confirmation |
| Buy the Dip | `buy_the_dip` | RSI oversold + price below SMA(50) but above SMA(200) |
| Mean Reversion | `mean_reversion` | Short-term bounce when RSI < 25, price > 2 ATR below EMA(20), volume spike |
| PEAD | `pead` | Post-Earnings Announcement Drift; volume spike detection, fade entry, 45-day hold |

### Options

| Strategy | Key | Description |
|---|---|---|
| Covered Call | `covered_call` | Buys shares in lots of 100 and simulates selling OTM calls |
| Naked Put | `naked_put` | Sells OTM puts; uses Black-Scholes backtester |
| Put Credit Spread | `put_credit_spread` | Sells put spreads; supports Monte Carlo simulation |
| Wheel | `wheel` | Sells puts, takes assignment, sells calls; BS backtester |

## Walk-Forward Testing

The walk-forward engine splits a date range into rolling train/test windows to measure out-of-sample performance and detect overfitting:

```bash
advisor backtest walk-forward sma_crossover --symbol AAPL \
  --start 2020-01-01 --end 2024-01-01 --windows 5
```

Output includes per-window IS/OOS metrics, average OOS return, Sharpe, max drawdown, and an IS-vs-OOS gap indicator.

## Research Agent

A standalone AI-powered research tool that produces evidence-backed "Opportunity Cards" for buy-the-dip analysis.

```bash
# Run research on a ticker
research_agent run --ticker AAPL

# Run on a sector or thesis
research_agent run --sector Technology
research_agent run --thesis "AI chip demand"

# View past results
research_agent history
research_agent show <run_id>
```

Requires `RESEARCH_AGENT_TAVILY_API_KEY` and `RESEARCH_AGENT_ANTHROPIC_API_KEY` in `.env`.

## Configuration

Default settings live in `config/default.yaml`:

| Section | Key defaults |
|---|---|
| `broker` | `initial_cash=100000`, `commission=0.001`, `slippage_perc=0.001` |
| `data` | `cache_dir=data/cache`, `cache_ttl_hours=24` |
| `sizer.atr` | `risk_pct=0.02`, `atr_multiplier=2.0` |
| `backtest` | `results_dir=data/results`, `risk_free_rate=0.04` |
| `walk_forward` | `default_windows=3`, `default_train_pct=0.7` |
| `market` | `min_avg_volume=500000`, `min_market_cap_billions=2.0`, `default_workers=4` |

All commands support `--output json` for structured output.

## Development

```bash
# Run tests
poetry run pytest

# Lint
poetry run ruff check src tests

# Format
poetry run ruff format src tests
```

## Project Structure

```
src/
  advisor/
    backtesting/   # Options backtester (Black-Scholes)
    cli/           # Typer CLI commands
    confluence/    # 3-layer analysis, alpha scorer, smart money, mispricing
    core/          # Pricing utilities
    data/          # Data providers (Yahoo Finance), cache, universe
    engine/        # Backtest runner, signal scanner, walk-forward
    market/        # Options scanner, premium screener, drawdown analysis, trade tracker
    ml/            # ML pipeline, features, regime detection, HRP, meta-labeling
    pipeline/      # Integrated orchestrator with conviction scoring
    simulator/     # Monte Carlo PCS simulator, Streamlit GUI, validation
    storage/       # Results persistence
    strategies/
      equity/      # Equity strategies
      options/     # Options strategies (naked_put, put_credit_spread, wheel)
      base.py      # Abstract strategy base class
      registry.py  # Strategy discovery and registration
    verification/  # Grounding verification
  research_agent/  # AI-powered research tool
config/            # YAML configuration
data/              # Cache and results (gitignored)
tests/             # Pytest test suite
```
