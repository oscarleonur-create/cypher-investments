# Monte Carlo PCS Simulator — Architecture & Critique

## Overview

The simulator replaces the synthetic Black-Scholes backtester with a **forward-looking Monte Carlo engine** that ingests live TastyTrade option chain data (bid/ask/greeks/IV), simulates thousands of correlated price+IV paths with fat tails, and ranks put credit spread (PCS) opportunities by risk-adjusted expected value. Chain snapshots are persisted in SQLite for historical accumulation. Calibration quality is tracked via Brier scores to detect model drift.

---

## Architecture

```
                         ┌──────────────┐
                         │  Streamlit   │
                         │   app.py     │
                         └──────┬───────┘
                                │
                         ┌──────┴───────┐
                         │  CLI Layer   │
                         │ options_cmds │
                         │ backtest_cmds│
                         └──────┬───────┘
                                │
                         ┌──────▼───────┐
                         │   Pipeline   │
                         │  Orchestrator│
                         └──┬───┬───┬───┘
                            │   │   │
              ┌─────────────┘   │   └─────────────┐
              │                 │                  │
       ┌──────▼──────┐  ┌──────▼──────┐   ┌──────▼──────┐
       │  Candidates  │  │   Engine    │   │ Calibration │
       │  Generator   │  │ Monte Carlo │   │  Student-t  │
       └──────┬───────┘  │  + VR Suite │   │  + AR(1)    │
              │          └─────────────┘   └─────────────┘
       ┌──────▼──────┐
       │  TastyTrade  │
       │   Client     │
       │ (Streamer)   │
       └──────┬───────┘        ┌─────────────┐
              │                │  SQLite DB   │
              └────────────────▶  simulator.db│
                               │ (4 tables)  │
                               └─────────────┘
```

### Package Layout

```
src/advisor/simulator/
├── __init__.py        # Package marker
├── models.py          # Pydantic: SimConfig, PCSCandidate, SimResult, CalibrationRecord, PipelineResult
├── db.py              # SimulatorStore — SQLite with 4 tables + Brier score computation
├── calibration.py     # Fit Student-t + vol dynamics (AR(1)) from yfinance history
├── engine.py          # MonteCarloEngine — vectorized paths + PCS sim + variance reduction
├── candidates.py      # Generate + score PCS candidates with NaN filtering
├── pipeline.py        # SimulatorPipeline — per-symbol calibration, tiered sim
├── charts.py          # Plotly chart factories for Streamlit GUI
└── app.py             # Streamlit GUI — sidebar config, ranking table, 4 charts, history + Brier

Modified:
├── market/tastytrade_client.py   # +get_option_quotes(), +get_option_greeks(), +get_enriched_chain()
├── cli/options_cmds.py           # +simulate command
├── cli/backtest_cmds.py          # +--monte-carlo flag
└── strategies/options/put_credit_spread.py  # v3.0.0

Tests:
tests/test_simulator/
├── test_engine.py       # 27 tests: paths, BSM vec, PCS sim, seed, VoV, variance reduction
├── test_calibration.py  # 10 tests: Student-t fit, vol dynamics, calibrate()
├── test_candidates.py   # 16 tests: delta, credit, scoring, NaN filtering
└── test_db.py           # 16 tests: schema, CRUD, joins, calibration tracking, Brier scores
                         # Total: 77 tests passing
```

---

## Data Flow

### 1. TastyTrade Client (`tastytrade_client.py`)

Three async functions extend the existing client:

- **`get_option_quotes()`** — Opens a `DXLinkStreamer`, subscribes to `Quote` events for all put streamer symbols, collects `{bid, ask, mid, bid_size, ask_size}` per symbol with a 5s timeout.
- **`get_option_greeks()`** — Same pattern, subscribes to `Greeks` events, collects `{delta, gamma, theta, vega, iv}`.
- **`get_enriched_chain()`** — Orchestrator: fetches the chain skeleton via `get_option_chain()`, gets the underlying price (DXLinkStreamer equity quote with yfinance fallback), fetches quotes + greeks in parallel via `asyncio.gather()`, then merges everything into enriched records.

Each enriched record:
```python
{
    "symbol": "PLTR", "expiration": "2026-04-03", "dte": 35,
    "strike": 125.0, "put_symbol": "...", "put_streamer": ".PLTR260403P125",
    "bid": 1.50, "ask": 1.65, "mid": 1.575,
    "delta": -0.28, "gamma": 0.015, "theta": -0.12, "vega": 0.14, "iv": 0.54,
    "underlying_price": 137.19
}
```

### 2. Candidate Generation (`candidates.py`)

`generate_pcs_candidates()` iterates expirations, finds short puts near the adaptive delta target (0.16/0.28/0.35 based on IV percentile), pairs each with long puts at $2-$10 width. Net credit uses the **conservative** formula: `short_bid - long_ask`. **NaN and zero bid/ask records are explicitly filtered** via `math.isnan()` checks to handle illiquid streamer data. Filters: min credit > $0.10, buying power within limit.

`compute_sell_score()` produces a 0-100 composite:
- IV Percentile: 25 pts
- POP (1 - |delta|): 20 pts
- Annualized yield: 15 pts
- Liquidity (bid-ask tightness): 15 pts
- Credit/width ratio: 15 pts
- Delta quality vs target: 10 pts

### 3. Calibration (`calibration.py`)

Downloads data once per symbol via `_get_daily_returns()` (yfinance) and passes the returns series to both fitters:

- **`fit_student_t(returns)`** — Fits `scipy.stats.t` to daily log returns. Returns `(df, loc, scale)`. df clamped to [2.5, 30].
- **`estimate_vol_dynamics(returns)`** — Computes rolling 30-day HV. `vol_mean_level` = median HV. `vol_mean_revert_speed` via **AR(1) regression**: `b = cov(x,y)/var(x)`, `kappa = -log(b) * 252`, clamped to [0.1, 5.0]. `leverage_effect` = correlation of returns vs vol changes.
- **`calibrate(symbol)`** — Downloads once, calls both fitters, returns updated `SimConfig`. Falls back to defaults on failure.

### 4. Monte Carlo Engine (`engine.py`)

**Path generation** (`_generate_paths`):
```
z_price = student_t.rvs(df)                         # fat-tailed innovation
z_vol = leverage * z_price + sqrt(1-lev^2) * N(0,1)  # correlated vol shock
S[t] = S[t-1] * exp(drift + vol*sqrt(dt)*z_price)    # GBM with fat tails
iv[t] = iv[t-1] + kappa*(theta-iv[t-1])*dt + xi*vol*sqrt(dt)*z_vol  # OU with vol-of-vol
```

**Variance reduction** (configurable via `SimConfig` flags):
- **Antithetic variates** (default ON): Pair each Z draw with -Z, halving base draws and reducing variance from symmetric payoff components.
- **Stratified sampling** (opt-in): Divide [0,1] CDF into n strata, apply inverse CDF (Student-t or normal). Shuffled per timestep to break ordering correlation.
- **Control variate** (default ON): Uses European PCS payoff (no early exit) as a correlated control. Analytical BSM spread value anchors the adjustment. Beta auto-shrinks when early exits dominate.
- **Importance sampling** (opt-in): Drift-tilted paths shift toward the short strike for better tail-risk estimation. Gaussian likelihood ratios correct for bias. Returns CVaR95_IS with standard error.

**Vectorized BSM** (`bsm_put_price_vec`): Takes numpy arrays of S and sigma, returns put prices. Handles T<=0 (intrinsic) and avoids division by zero.

**PCS simulation** (`_simulate_pcs`): Mark-to-market approach:
1. Compute initial BSM spread value at entry using per-strike IV ratios
2. Each day: reprice spread with IV skew ratios preserved, compute unrealized P&L = entry_value - current_value
3. Check exits: DTE threshold first, then profit target (unrealized >= 50% of credit), then stop loss (unrealized loss >= 2x credit), finally expiration (intrinsic)
4. Final P&L = credit - (change in spread value) - slippage
5. Cap losses at max_loss = (width - credit) * 100

**MC precision reporting**: `mc_std_err` (standard error of EV) and `variance_reduction_factor` (var_raw / var_adjusted) are computed and stored per result.

### 5. Pipeline (`pipeline.py`)

```
PER-SYMBOL CALIBRATE → SCAN → PRE-SCORE (top 200) → QUICK SIM (10K, top 20)
→ DEEP SIM (100K, top 5) → PERSIST (candidates + results + calibration records) → RETURN
```

Key improvements:
- **Per-symbol calibration**: `_calibrate_per_symbol()` creates a separate `MonteCarloEngine` with individually fitted Student-t df and vol dynamics for each ticker. A fallback engine with default params is used if calibration fails.
- **Nested asyncio.run safety**: Detects whether an event loop is already running (Jupyter, Streamlit) and uses `concurrent.futures.ThreadPoolExecutor` to avoid "cannot call asyncio.run() from a running event loop" errors.
- **Calibration tracking**: Saves `CalibrationRecord` for each final result with predicted POP, touch, stop, and EV. Actuals are filled in later via `update_calibration_outcome()`.
- **Deferred candidate saving**: Only final top results are saved to the candidates/sim_results tables (scan_and_generate saves only chain snapshots).

### 6. SQLite Store (`db.py`)

Four tables: `chain_snapshots`, `candidates`, `sim_results`, `calibration_tracking`. Follows the `research_agent/store.py` pattern with schema-as-string, `sqlite3.Row` factory, UUID-based IDs.

Key methods:
- `save_chain_snapshot()`, `get_chain_snapshots()`
- `save_candidates_batch()`, `save_sim_result()`
- `get_top_results()` (join candidates + results, rank by EV/BP)
- `get_run_history()`, `get_results_by_date_range()`
- `save_calibration_record()`, `update_calibration_outcome()`
- `compute_brier_scores(symbol?, lookback_days=90)` — Brier score = avg((predicted - actual)^2). Scores < 0.10 excellent, < 0.20 good.

### 7. Streamlit GUI (`app.py`)

Full-featured web interface with:
- **Sidebar**: Universe/ticker selection, simulation paths, exit rules, spread filters, variance reduction toggles
- **Run Sim tab**: Progress feedback via `st.status`, calibration metrics row, summary stats
- **Results tab**: Ranking table + 4 Plotly charts (P&L distribution box plot, exit breakdown stacked bar, risk comparison grouped bar, risk-return scatter with POP color scale)
- **History tab**: Date-range query of historical results, Brier score calibration quality display, EV/BP trend scatter over time

### 8. Charts (`charts.py`)

Four Plotly chart factory functions:
- `pnl_distribution_chart()` — Box plot from percentile data (p5/p25/p50/p75/p95)
- `exit_breakdown_chart()` — Horizontal stacked bar of exit fractions
- `risk_comparison_chart()` — Grouped bar of POP/touch/stop probabilities
- `risk_return_scatter()` — EV/BP vs CVaR95, bubble size = credit, color = POP

### 9. CLI (`options_cmds.py`, `backtest_cmds.py`)

- `advisor options simulate --tickers PLTR --paths 10000 --top 5`
- `advisor backtest run put_credit_spread --symbol PLTR --monte-carlo --mc-paths 10000`

---

## Critique

### Correctness Issues

**1. Redundant BSM calls in the daily loop (High severity)**

In `_simulate_pcs` (engine.py:260-272), after processing profit exits, the stop-loss check re-computes `bsm_put_price_vec` for all remaining active paths instead of slicing the already-computed `unrealized_pnl` and `current_spread` arrays. This roughly **doubles the BSM calls per timestep** — the hot path. For 100K paths x 35 days, that's ~7M extra BSM evaluations.

*Fix: After removing profit exits, use boolean indexing on the pre-computed arrays. The profit exits only change which paths are active, not the computed values for the remaining paths.*

**2. Single-IV path for both strikes (Medium severity)**

The engine generates ONE IV path per simulation and applies IV ratios (short_iv/long_iv) as fixed multipliers throughout. In reality, each strike has its own IV driven by the volatility smile. When the underlying moves, the smile shifts, and the $125 and $120 puts experience different IV dynamics. The fixed-ratio approach overstates the IV correlation between strikes, compressing the spread value distribution and potentially understating tail risk.

*Fix: Model the skew as a function of moneyness (e.g., SABR-like parameterization) so that as S changes, each strike's IV evolves differently.*

**3. Importance sampling LR approximation (Medium severity)**

The importance sampling in `_estimate_cvar_is()` (engine.py:506-510) uses a Gaussian Girsanov-like likelihood ratio as a proxy for the Student-t distribution. This is an approximation — the exact LR for Student-t innovations is different from the Gaussian case. The approximation quality degrades for very fat tails (low df). The method also uses crude drift-tilting rather than optimal exponential tilting.

*Note: The IS estimate carries a standard error via `cvar_95_se`, so users can judge reliability. But the systematic bias from the Gaussian LR proxy is not captured by the SE.*

**4. `entry_spread_value` is per-path but identical (Low severity)**

In `_simulate_pcs` (engine.py:188-195), `entry_spread_value` is computed for each of the n_paths at t=0. Since all paths start from the same S0 and iv0, this array has identical values across all paths. It could be a scalar, saving a BSM call and memory.

**5. Expiration P&L formula inconsistency (Low severity)**

At expiration (engine.py:293), P&L is `credit - (spread_intrinsic - entry_vals)`. For mid-life exits (engine.py:236), P&L is `credit - (current_spread - entry_vals) - slippage`. The expiration path omits slippage, which is correct (no exit cost at expiration), but the asymmetry in the formulas could confuse future readers.

### Performance Issues

**6. Sequential candidate simulation (Medium severity)**

The pipeline simulates candidates one-at-a-time in a Python for-loop (`_simulate_batch_per_symbol`). Since each simulation is CPU-bound numpy, this could benefit from `concurrent.futures.ProcessPoolExecutor` for the deep sim phase. With 20 candidates x 100K paths x 35 days, sequential deep sim takes ~30+ seconds.

**7. Stratified sampling is O(n_paths * dte) with Python loop (Low severity)**

`_stratified_student_t` and `_stratified_normal` (engine.py:133-157) iterate over `dte` timesteps in a Python for-loop calling `student_t.ppf()` per timestep. The `ppf` call itself is vectorized over paths, but the timestep loop adds overhead vs the pure array approach used by crude MC. For 100K paths x 35 DTE this is noticeable but not dominant.

**8. Three separate DXLinkStreamer connections (Low severity)**

`get_enriched_chain()` opens three streamer connections (equity quote, option quotes, option greeks). Each incurs websocket handshake + auth overhead (~1s each). Could be reduced to one connection subscribing to all event types.

### Design Issues

**9. No mechanism to fill in calibration actuals (Design gap)**

The pipeline saves `CalibrationRecord` with predictions, but `update_calibration_outcome()` is never called automatically. There's no scheduled job, CLI command, or workflow to fill in actual outcomes after expiration. The Brier score computation will always show 0 samples until someone manually calls `update_calibration_outcome()`.

*Fix: Add an `advisor options calibrate-update` CLI command that queries expired candidates and fills in actuals from historical price data.*

**10. Stop-loss threshold scales with credit, not width (Design note)**

With `stop_loss_multiplier=2.0`, a $0.14 credit spread ($14 per contract) has a stop at $28 unrealized loss. On a $5-wide spread, small IV moves can produce $28 MTM swings. The stop may trigger too frequently on low-credit spreads while being too lenient on high-credit ones. Consider scaling with width or max-loss instead.

**11. Streamlit app.py imports `charts.py` module — no fallback (Low severity)**

The `app.py` Streamlit GUI imports from `advisor.simulator.charts`, which requires `plotly`. If plotly is not installed, the import fails even for non-GUI usage. The import is only in `app.py` so it doesn't affect the CLI or pipeline, but it means `charts.py` should be guarded or made an optional dependency.

**12. `SimulatorStore` lacks context manager pattern (Low severity)**

The store creates a connection in `__init__` and requires manual `close()`. No `__enter__`/`__exit__` pattern means exceptions before `close()` leak the connection. The Streamlit app uses try/finally but the pipeline doesn't.

**13. UUID truncation for IDs (Low severity)**

`str(uuid.uuid4())[:8]` produces 8-char hex IDs. At ~100 candidates per run, collision probability is negligible, but this is non-standard. Full UUID4 would be safer for long-term accumulated data.

### Test Gaps

**14. No integration test for the full pipeline**

The 77 tests cover each module in isolation (engine, calibration, candidates, DB) but never test the pipeline end-to-end with mocked TastyTrade data. An integration test that mocks `get_enriched_chain()` and `get_market_metrics()`, then runs `pipeline.run()`, would catch wiring issues between the modules.

**15. No test for `get_enriched_chain()` or the streamer functions**

The TastyTrade client extensions have zero test coverage. They should be tested with mocked DXLinkStreamer responses.

**16. No test for the Streamlit app or charts**

The `app.py` and `charts.py` files are untested. Chart factory functions could be tested by verifying the returned `go.Figure` has the expected traces and layout.

**17. Importance sampling not tested**

`_estimate_cvar_is()` has no dedicated test. A test with a known tail distribution could verify that the IS CVaR estimate is closer to the true value than crude MC's estimate.

---

## Resolved from Previous Critique

The following items from the initial critique have been addressed:

| # | Issue | Resolution |
|---|-------|------------|
| 1 | NaN bid/ask propagation | Fixed: `math.isnan()` checks in `candidates.py` + test coverage |
| 2 | Calibration uses only first symbol | Fixed: `_calibrate_per_symbol()` in `pipeline.py` creates per-symbol engines |
| 3 | `asyncio.run()` nesting | Fixed: `ThreadPoolExecutor` fallback detects running event loops |
| 4 | No seed for reproducibility | Fixed: `seed` field in `SimConfig`, passed to `np.random.default_rng()` |
| 5 | No variance reduction | Fixed: Antithetic, control variate, stratified, and importance sampling implemented |
| 6 | No calibration tracking | Fixed: `calibration_tracking` table + `CalibrationRecord` model + Brier score computation |
| 7 | Stochastic test fragility | Fixed: Seed-based reproducibility + dedicated `TestSeedReproducibility` tests |
| 8 | Vol dynamics estimation | Improved: AR(1) regression replaces simple autocorrelation for mean-reversion speed |

---

## Verdict

The simulator has matured significantly since the initial implementation. The core architecture is sound and well-tested (77 tests, all passing). The main areas still needing attention, in priority order:

1. **Redundant BSM calls** in the hot loop (easy fix, ~2x speedup in `_simulate_pcs`)
2. **Calibration actuals workflow** — the Brier score infrastructure exists but has no way to fill in actual outcomes
3. **Pipeline parallelization** — `ProcessPoolExecutor` for the deep sim phase
4. **IV smile modeling** — move from fixed ratios to moneyness-based skew (longer-term)
5. **Integration and streamer tests** — pipeline end-to-end + mocked TastyTrade client tests
