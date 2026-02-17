# Options Advisor

A CLI-based backtesting and strategy system for equities and options. Includes live signal scanning, multi-layer confluence analysis, market-wide screening, walk-forward testing, and an AI-powered research agent.

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
```

### Confluence

Runs a 3-layer analysis pipeline (technical + sentiment + fundamental) and produces an ENTER / CAUTION / PASS verdict:

```bash
advisor confluence scan AAPL --strategy momentum_breakout --verbose
```

### Market

Scans a stock universe (default: S&P 500) through layered filters, then runs the confluence pipeline on qualifiers:

```bash
# Full market scan
advisor market scan --strategy pead

# Filter by sector and minimum thresholds
advisor market scan --strategy buy_the_dip \
  --include-sector Technology --min-volume 1000000 --min-cap 5.0

# Dry run (filters only, no confluence)
advisor market scan --strategy momentum_breakout --dry-run
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
    cli/           # Typer CLI commands
    confluence/    # 3-layer analysis pipeline
    core/          # Pricing utilities
    data/          # Data providers (Yahoo Finance)
    engine/        # Backtest runner, signal scanner, walk-forward
    market/        # Market-wide scanner and filters
    strategies/
      equity/      # Equity strategies
      options/     # Options strategies
      base.py      # Abstract strategy base class
      registry.py  # Strategy discovery and registration
  research_agent/  # AI-powered research tool
config/            # YAML configuration
data/              # Cache and results (gitignored)
tests/             # Pytest test suite
```
