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
    interval: Annotated[str, typer.Option("--interval", "-i", help="Data interval (1m, 5m, 15m, 1h, 1d, 1wk)")] = "1d",
    slippage: Annotated[float, typer.Option("--slippage", help="Slippage percentage")] = 0.001,
    sizer: Annotated[Optional[str], typer.Option("--sizer", help="Position sizer (atr)")] = None,
    param: Annotated[Optional[list[str]], typer.Option("--param", help="Strategy params (k=v)")] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Run a backtest for a strategy."""
    from advisor.engine.runner import BacktestRunner
    from advisor.storage.results_store import ResultsStore
    from advisor.strategies.registry import StrategyRegistry

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

    try:
        runner = BacktestRunner(initial_cash=cash, slippage_perc=slippage, sizer=sizer)
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
    strategy: Annotated[Optional[str], typer.Option("--strategy", help="Filter by strategy")] = None,
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
    train_pct: Annotated[float, typer.Option("--train-pct", help="Train fraction per window")] = 0.7,
    cash: Annotated[float, typer.Option("--cash", help="Initial cash")] = 100_000.0,
    interval: Annotated[str, typer.Option("--interval", "-i", help="Data interval")] = "1d",
    slippage: Annotated[float, typer.Option("--slippage", help="Slippage percentage")] = 0.001,
    sizer: Annotated[Optional[str], typer.Option("--sizer", help="Position sizer (atr)")] = None,
    param: Annotated[Optional[list[str]], typer.Option("--param", help="Strategy params (k=v)")] = None,
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

    try:
        runner = BacktestRunner(initial_cash=cash, slippage_perc=slippage, sizer=sizer)
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
