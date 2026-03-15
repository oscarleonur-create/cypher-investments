"""CLI commands for backtesting."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

import typer

from advisor.cli.formatters import (
    output_error,
    output_json,
    print_result_summary,
    print_results_list,
    print_walk_forward_summary,
)

app = typer.Typer(name="backtest", help="Run and manage backtests")


def _parse_params(param_list: list[str] | None) -> dict:
    """Parse key=value parameter pairs."""
    if not param_list:
        return {}
    params = {}
    for p in param_list:
        if "=" not in p:
            output_error(f"Invalid param format: '{p}'. Expected key=value")
            continue
        key, value = p.split("=", 1)
        # Try to parse as number
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        params[key] = value
    return params


@app.command("run")
def backtest_run(
    strategy: Annotated[str, typer.Argument(help="Strategy name")],
    symbol: Annotated[str, typer.Option("--symbol", help="Ticker symbol")],
    start: Annotated[str, typer.Option("--start", help="Start date (YYYY-MM-DD)")],
    end: Annotated[str, typer.Option("--end", help="End date (YYYY-MM-DD)")],
    cash: Annotated[float, typer.Option("--cash", help="Initial cash")] = 100_000.0,
    interval: Annotated[
        str, typer.Option("--interval", "-i", help="Data interval (1m, 5m, 15m, 1h, 1d, 1wk)")
    ] = "1d",
    slippage: Annotated[float, typer.Option("--slippage", help="Slippage percentage")] = 0.001,
    sizer: Annotated[
        Optional[str], typer.Option("--sizer", help="Position sizer (atr, none)")
    ] = "atr",
    max_drawdown_pct: Annotated[
        float, typer.Option("--max-drawdown-pct", help="Circuit breaker drawdown threshold (%)")
    ] = 15.0,
    param: Annotated[
        Optional[list[str]], typer.Option("--param", help="Strategy params (k=v)")
    ] = None,
    monte_carlo: Annotated[
        bool, typer.Option("--monte-carlo", help="Use Monte Carlo simulator for PCS")
    ] = False,
    mc_paths: Annotated[int, typer.Option("--mc-paths", help="MC paths (default 10000)")] = 10_000,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Run a backtest for a strategy."""
    from advisor.storage.results_store import ResultsStore
    from advisor.strategies.options import OPTIONS_STRATEGIES
    from advisor.strategies.registry import StrategyRegistry

    if strategy in OPTIONS_STRATEGIES and monte_carlo and strategy == "put_credit_spread":
        _run_mc_backtest(symbol, cash, mc_paths, output)
        return

    if strategy in OPTIONS_STRATEGIES:
        _run_options_backtest(strategy, symbol, start, end, cash, output, _parse_params(param))
        return

    from advisor.engine.runner import BacktestRunner

    # Ensure strategies are discovered
    registry = StrategyRegistry()
    registry.discover()

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as e:
        output_error(f"Invalid date format: {e}")
        return

    params = _parse_params(param)

    # "none" disables sizer
    effective_sizer = None if sizer == "none" else sizer

    try:
        runner = BacktestRunner(
            initial_cash=cash,
            slippage_perc=slippage,
            sizer=effective_sizer,
            max_drawdown_pct=max_drawdown_pct,
        )
        result = runner.run(
            strategy_name=strategy,
            symbol=symbol,
            start=start_date,
            end=end_date,
            params=params,
            interval=interval,
        )
    except KeyError as e:
        output_error(str(e))
        return
    except Exception as e:
        output_error(f"Backtest failed: {e}")
        return

    # Save result
    store = ResultsStore()
    store.save(result)

    if output == "json":
        output_json(result)
    else:
        print_result_summary(result.model_dump())


@app.command("results")
def backtest_results(
    strategy: Annotated[
        Optional[str], typer.Option("--strategy", help="Filter by strategy")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Max results")] = 20,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """List stored backtest results."""
    from advisor.storage.results_store import ResultsStore

    store = ResultsStore()
    results = store.list_results(strategy_name=strategy, limit=limit)

    if output == "json":
        output_json(results)
    else:
        if not results:
            typer.echo("No results found.")
            return
        print_results_list([r.model_dump() for r in results])


@app.command("show")
def backtest_show(
    run_id: Annotated[str, typer.Argument(help="Run ID to show")],
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Show details of a specific backtest run."""
    from advisor.storage.results_store import ResultsStore

    store = ResultsStore()
    try:
        result = store.load(run_id)
    except FileNotFoundError:
        output_error(f"Result not found: {run_id}")
        return

    if output == "json":
        output_json(result)
    else:
        print_result_summary(result.model_dump())


@app.command("walk-forward")
def backtest_walk_forward(
    strategy: Annotated[str, typer.Argument(help="Strategy name")],
    symbol: Annotated[str, typer.Option("--symbol", help="Ticker symbol")],
    start: Annotated[str, typer.Option("--start", help="Start date (YYYY-MM-DD)")],
    end: Annotated[str, typer.Option("--end", help="End date (YYYY-MM-DD)")],
    windows: Annotated[int, typer.Option("--windows", help="Number of windows")] = 3,
    train_pct: Annotated[
        float, typer.Option("--train-pct", help="Train fraction per window")
    ] = 0.7,
    cash: Annotated[float, typer.Option("--cash", help="Initial cash")] = 100_000.0,
    interval: Annotated[str, typer.Option("--interval", "-i", help="Data interval")] = "1d",
    slippage: Annotated[float, typer.Option("--slippage", help="Slippage percentage")] = 0.001,
    sizer: Annotated[
        Optional[str], typer.Option("--sizer", help="Position sizer (atr, none)")
    ] = "atr",
    max_drawdown_pct: Annotated[
        float, typer.Option("--max-drawdown-pct", help="Circuit breaker drawdown threshold (%)")
    ] = 15.0,
    param: Annotated[
        Optional[list[str]], typer.Option("--param", help="Strategy params (k=v)")
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Run walk-forward analysis with rolling train/test windows."""
    from advisor.engine.runner import BacktestRunner
    from advisor.engine.walk_forward import WalkForwardRunner
    from advisor.strategies.registry import StrategyRegistry

    registry = StrategyRegistry()
    registry.discover()

    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as e:
        output_error(f"Invalid date format: {e}")
        return

    params = _parse_params(param)

    effective_sizer = None if sizer == "none" else sizer

    try:
        runner = BacktestRunner(
            initial_cash=cash,
            slippage_perc=slippage,
            sizer=effective_sizer,
            max_drawdown_pct=max_drawdown_pct,
        )
        wf_runner = WalkForwardRunner(runner)
        result = wf_runner.run(
            strategy_name=strategy,
            symbol=symbol,
            start=start_date,
            end=end_date,
            n_windows=windows,
            train_pct=train_pct,
            params=params,
            interval=interval,
        )
    except KeyError as e:
        output_error(str(e))
        return
    except Exception as e:
        output_error(f"Walk-forward failed: {e}")
        return

    if output == "json":
        output_json(result)
    else:
        print_walk_forward_summary(result)


# ── Options backtest helper ──────────────────────────────────────────────────


def _run_mc_backtest(symbol: str, cash: float, mc_paths: int, output: str | None) -> None:
    """Run a Monte Carlo simulation for put credit spreads."""
    from advisor.simulator.db import SimulatorStore
    from advisor.simulator.models import SimConfig
    from advisor.simulator.pipeline import SimulatorPipeline

    config = SimConfig(n_paths=mc_paths, max_buying_power=cash)
    store = SimulatorStore()

    try:
        pipeline = SimulatorPipeline(config=config, store=store)
        result = pipeline.run(symbols=[symbol.upper()], top_n=5, quick_paths=mc_paths)
    except Exception as e:
        output_error(f"MC simulation failed: {e}")
        return
    finally:
        store.close()

    if output == "json":
        output_json(result.model_dump())
    else:
        import typer

        typer.echo(
            f"MC Simulation: {result.candidates_generated} candidates, "
            f"{result.candidates_simulated} simulated"
        )
        for r in result.top_results:
            typer.echo(
                f"  {r.symbol} ${r.short_strike}/{r.long_strike} "
                f"EV=${r.ev:+.2f} POP={r.pop:.0%} EV/BP={r.ev_per_bp:.4f}"
            )


@app.command("optimize-exits")
def backtest_optimize_exits(
    strategy: Annotated[str, typer.Argument(help="Options strategy name")],
    symbol: Annotated[str, typer.Option("--symbol", help="Ticker symbol")],
    start: Annotated[str, typer.Option("--start", help="Start date (YYYY-MM-DD)")],
    end: Annotated[str, typer.Option("--end", help="End date (YYYY-MM-DD)")],
    cash: Annotated[float, typer.Option("--cash", help="Initial cash")] = 100_000.0,
    profit_targets: Annotated[
        str, typer.Option("--profit-targets", help="Comma-separated profit target pcts")
    ] = "0.25,0.50,0.75",
    stop_losses: Annotated[
        str, typer.Option("--stop-losses", help="Comma-separated stop loss multipliers")
    ] = "1.5,2.0,3.0,4.0",
    dte_exits: Annotated[
        str, typer.Option("--dte-exits", help="Comma-separated DTE exit values")
    ] = "7,14,21,28",
    metric: Annotated[str, typer.Option("--metric", help="Metric to rank by")] = "sharpe_ratio",
    top_n: Annotated[int, typer.Option("--top-n", help="Number of top results to show")] = 10,
    workers: Annotated[
        Optional[int], typer.Option("--workers", help="Max parallel workers")
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Grid-search exit parameters (profit target, stop loss, DTE) for optimal results."""
    from advisor.backtesting.exit_optimizer import (
        ExitOptimizer,
        ExitParameterGrid,
        render_ranking_table,
    )

    pt_list = [float(x.strip()) for x in profit_targets.split(",")]
    sl_list = [float(x.strip()) for x in stop_losses.split(",")]
    dte_list = [int(x.strip()) for x in dte_exits.split(",")]

    grid = ExitParameterGrid(
        profit_target_pcts=pt_list,
        stop_loss_multipliers=sl_list,
        close_at_dtes=dte_list,
    )

    typer.echo(f"Running {grid.n_combos} exit parameter combos for {strategy} on {symbol}...")

    def _progress(done: int, total: int) -> None:
        typer.echo(f"  [{done}/{total}] combos complete", err=True)

    optimizer = ExitOptimizer(
        symbol=symbol,
        start=start,
        end=end,
        strategy=strategy,
        cash=cash,
    )

    try:
        results = optimizer.optimize(
            grid=grid,
            metric=metric,
            max_workers=workers,
            progress_callback=_progress,
        )
    except Exception as e:
        output_error(f"Exit optimization failed: {e}")
        return

    if output == "json":
        output_json([r.model_dump() for r in results[:top_n]])
    else:
        render_ranking_table(results, top_n=top_n)


@app.command("pipeline")
def backtest_pipeline(
    strategy: Annotated[
        str, typer.Argument(help="Options strategy (naked_put, wheel, put_credit_spread, ...)")
    ],
    symbol: Annotated[
        Optional[str], typer.Option("--symbol", "-t", help="Single ticker (skip scanner)")
    ] = None,
    start: Annotated[
        str, typer.Option("--start", help="Backtest start date (YYYY-MM-DD)")
    ] = "2023-01-01",
    end: Annotated[
        str, typer.Option("--end", help="Backtest end date (YYYY-MM-DD)")
    ] = "2024-01-01",
    cash: Annotated[float, typer.Option("--cash", help="Initial cash")] = 100_000.0,
    scan_strategy: Annotated[
        str, typer.Option("--scan-strategy", help="Equity scan strategy for filtering")
    ] = "momentum_breakout",
    universe: Annotated[
        str, typer.Option("--universe", "-u", help="Universe for scanner (sp500, semiconductors)")
    ] = "sp500",
    min_score: Annotated[
        float, typer.Option("--min-score", help="Minimum dip/alpha score to backtest")
    ] = 45.0,
    top_n: Annotated[int, typer.Option("--top-n", help="Max symbols to backtest")] = 5,
    mode: Annotated[str, typer.Option("--mode", help="Scanner mode: dip, scan, or alpha")] = "dip",
    workers: Annotated[int, typer.Option("--workers", help="Parallel scanner workers")] = 3,
    skip_ml: Annotated[bool, typer.Option("--skip-ml", help="Skip ML signal layer")] = False,
    skip_sentiment: Annotated[
        bool, typer.Option("--skip-sentiment", help="Skip sentiment layer")
    ] = False,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Scan → score → adaptive-exit backtest pipeline.

    Runs a scanner to find candidates, scores conviction, configures adaptive
    exits based on regime + conviction, then backtests each symbol.

    Examples:
        # Single ticker — score + adaptive backtest
        advisor backtest pipeline naked_put --symbol AAPL --start 2023-01-01 --end 2024-01-01

        # Full pipeline — scan universe, pick top 5, backtest each
        advisor backtest pipeline naked_put --universe sp500 --top-n 5

        # Confluence scan mode instead of dip scoring
        advisor backtest pipeline wheel --mode scan --scan-strategy buy_the_dip
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.table import Table

    from advisor.backtesting.adaptive_exits import AdaptiveExitPolicy
    from advisor.backtesting.options_backtester import BacktestConfig, Backtester, print_summary
    from advisor.cli.formatters import console

    # ── Regime mapping: dip_analyzer uses low_vol/normal/high_vol,
    #    AdaptiveExitPolicy uses low/mid/high ──
    _REGIME_MAP = {"low_vol": "low", "normal": "mid", "high_vol": "high"}

    # ── Step 1: Gather candidates with conviction scores ──────────────

    candidates: list[dict] = []  # {symbol, score, regime, verdict, reasoning}

    if symbol:
        # Single-ticker mode: run dip analysis for conviction + regime
        typer.echo(f"Scoring {symbol.upper()}...")
        try:
            from advisor.confluence.dip_analyzer import analyze_dip

            skip_layers: set[str] = set()
            if skip_ml:
                skip_layers.add("ml_signal")
            if skip_sentiment:
                skip_layers.add("confluence")

            dip = analyze_dip(symbol, skip_layers=skip_layers)
            candidates.append(
                {
                    "symbol": dip.symbol,
                    "score": dip.dip_score,
                    "regime": dip.regime,
                    "verdict": dip.verdict.value,
                    "reasoning": dip.reasoning,
                }
            )
        except Exception as e:
            typer.echo(f"[warning] Dip analysis failed for {symbol}: {e}", err=True)
            # Fall back: backtest without conviction data
            candidates.append(
                {
                    "symbol": symbol.upper(),
                    "score": 50.0,
                    "regime": "normal",
                    "verdict": "N/A",
                    "reasoning": "Dip analysis unavailable — using default exits",
                }
            )

    elif mode == "scan":
        # Confluence scanner mode → pick ENTER verdicts
        from advisor.data.cache import DiskCache
        from advisor.market.scanner import MarketScanner

        typer.echo(f"Scanning {universe} with {scan_strategy}...")
        scanner = MarketScanner(cache=DiskCache())
        scan_result = scanner.scan(
            strategy_name=scan_strategy,
            max_workers=workers,
            universe=universe,
        )
        # Use ENTER and CAUTION results
        for r in scan_result.results:
            if r.verdict.value in ("ENTER", "CAUTION"):
                # Run quick dip analysis for conviction score
                try:
                    from advisor.confluence.dip_analyzer import analyze_dip

                    dip = analyze_dip(r.symbol, skip_layers={"ml_signal"})
                    candidates.append(
                        {
                            "symbol": r.symbol,
                            "score": dip.dip_score,
                            "regime": dip.regime,
                            "verdict": f"{r.verdict.value} → {dip.verdict.value}",
                            "reasoning": r.reasoning[:100],
                        }
                    )
                except Exception:
                    candidates.append(
                        {
                            "symbol": r.symbol,
                            "score": 70.0 if r.verdict.value == "ENTER" else 50.0,
                            "regime": "normal",
                            "verdict": r.verdict.value,
                            "reasoning": r.reasoning[:100],
                        }
                    )

    else:
        # Dip / alpha scan mode — score entire universe
        from advisor.confluence.dip_analyzer import analyze_dip
        from advisor.confluence.smart_money_screener import get_sp500_tickers

        tickers = get_sp500_tickers()
        typer.echo(f"Scoring {len(tickers)} tickers (mode={mode})...")

        skip_layers: set[str] = set()
        if skip_ml:
            skip_layers.add("ml_signal")
        if skip_sentiment:
            skip_layers.add("confluence")

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_dip, t, skip_layers): t for t in tickers}
            for future in as_completed(futures):
                try:
                    dip = future.result()
                    if dip.dip_score >= min_score:
                        candidates.append(
                            {
                                "symbol": dip.symbol,
                                "score": dip.dip_score,
                                "regime": dip.regime,
                                "verdict": dip.verdict.value,
                                "reasoning": dip.reasoning[:100],
                            }
                        )
                except Exception:
                    pass

    if not candidates:
        typer.echo("No candidates passed the minimum score threshold.")
        return

    # Sort by score descending, take top N
    candidates.sort(key=lambda c: c["score"], reverse=True)
    candidates = candidates[:top_n]

    # ── Step 2: Display candidates ────────────────────────────────────

    cand_table = Table(title="Pipeline Candidates")
    cand_table.add_column("Symbol", style="cyan")
    cand_table.add_column("Score", justify="right")
    cand_table.add_column("Verdict")
    cand_table.add_column("Regime")
    cand_table.add_column("Reasoning", max_width=60)

    verdict_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "LEAN_BUY": "cyan",
        "WATCH": "yellow",
        "PASS": "red",
        "ENTER": "green",
        "CAUTION": "yellow",
    }

    for c in candidates:
        color = "white"
        for key in verdict_colors:
            if key in c["verdict"]:
                color = verdict_colors[key]
                break
        regime_label = {"low_vol": "Calm", "normal": "Normal", "high_vol": "Stressed"}.get(
            c["regime"], c["regime"]
        )
        cand_table.add_row(
            c["symbol"],
            f"{c['score']:.1f}",
            f"[{color}]{c['verdict']}[/{color}]",
            regime_label,
            c["reasoning"],
        )

    console.print(cand_table)

    # ── Step 3: Backtest each candidate with adaptive exits ───────────

    typer.echo(f"\nBacktesting {len(candidates)} candidates with {strategy}...")

    pipeline_results: list[dict] = []

    for c in candidates:
        sym = c["symbol"]
        score = c["score"]
        regime = _REGIME_MAP.get(c["regime"], "mid")

        # Build adaptive config
        config = BacktestConfig()
        policy = AdaptiveExitPolicy(config, vol_regime=regime, conviction_score=score)
        adapted = policy.adapt()

        typer.echo(
            f"  {sym}: conviction={score:.0f}, regime={regime} → "
            f"PT={adapted.profit_target_pct:.0%}, "
            f"SL={adapted.stop_loss_multiplier:.1f}x, "
            f"DTE={adapted.close_at_dte}d",
            err=True,
        )

        try:
            bt = Backtester(sym, start, end, cash, config=adapted)
            result = bt.run(strategy)
            result["conviction_score"] = score
            result["regime"] = c["regime"]
            result["adapted_exits"] = {
                "profit_target_pct": adapted.profit_target_pct,
                "stop_loss_multiplier": adapted.stop_loss_multiplier,
                "close_at_dte": adapted.close_at_dte,
            }
            pipeline_results.append(result)
        except Exception as e:
            typer.echo(f"  [warning] {sym} backtest failed: {e}", err=True)

    if not pipeline_results:
        typer.echo("All backtests failed.")
        return

    # ── Step 4: Output ────────────────────────────────────────────────

    if output == "json":
        output_json(pipeline_results)
        return

    # Summary comparison table
    console.print()
    summary_table = Table(title=f"Pipeline Results — {strategy}")
    summary_table.add_column("Symbol", style="cyan")
    summary_table.add_column("Conviction", justify="right")
    summary_table.add_column("Regime")
    summary_table.add_column("PT / SL / DTE")
    summary_table.add_column("Trades", justify="right")
    summary_table.add_column("Win Rate", justify="right")
    summary_table.add_column("Total P&L", justify="right")
    summary_table.add_column("Return", justify="right")
    summary_table.add_column("Sharpe", justify="right")
    summary_table.add_column("Max DD", justify="right")

    for r in sorted(pipeline_results, key=lambda x: x["total_return_pct"], reverse=True):
        pnl_color = "green" if r["total_pnl"] >= 0 else "red"
        exits = r["adapted_exits"]
        regime_label = {"low_vol": "Calm", "normal": "Normal", "high_vol": "Stressed"}.get(
            r["regime"], r["regime"]
        )
        summary_table.add_row(
            r["symbol"],
            f"{r['conviction_score']:.0f}",
            regime_label,
            f"{exits['profit_target_pct']:.0%} / "
            f"{exits['stop_loss_multiplier']:.1f}x / {exits['close_at_dte']}d",
            str(r["num_trades"]),
            f"{r['win_rate_pct']:.0f}%",
            f"[{pnl_color}]${r['total_pnl']:,.0f}[/{pnl_color}]",
            f"[{pnl_color}]{r['total_return_pct']:+.1f}%[/{pnl_color}]",
            f"{r['sharpe_ratio']:.2f}",
            f"{r['max_drawdown_pct']:.1f}%",
        )

    console.print(summary_table)

    # Per-symbol detail
    if len(pipeline_results) == 1:
        console.print()
        print_summary(pipeline_results[0])


def _run_options_backtest(
    strategy: str,
    symbol: str,
    start: str,
    end: str,
    cash: float,
    output: str | None,
    params: dict | None = None,
) -> None:
    """Run an options backtest using the Black-Scholes backtester."""
    from advisor.backtesting.options_backtester import BacktestConfig, Backtester, print_summary

    config = BacktestConfig()
    if params:
        for key, value in params.items():
            if hasattr(config, key):
                # Handle booleans from string
                if isinstance(getattr(config, key), bool) and isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                setattr(config, key, value)

    try:
        bt = Backtester(symbol, start, end, cash, config=config)
        result = bt.run(strategy)
    except Exception as e:
        output_error(f"Options backtest failed: {e}")
        return

    if output == "json":
        output_json(result)
    else:
        print_summary(result)
