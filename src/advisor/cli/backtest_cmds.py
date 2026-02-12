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
        runner = BacktestRunner(initial_cash=cash)
        result = runner.run(
            strategy_name=strategy,
            symbol=symbol,
            start=start_date,
            end=end_date,
            params=params,
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
