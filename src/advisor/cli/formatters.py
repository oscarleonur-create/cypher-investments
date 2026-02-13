"""Output formatters for the CLI - JSON and rich table output."""

from __future__ import annotations

import json
import sys
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()
err_console = Console(stderr=True)


def output_json(data: Any, file=sys.stdout) -> None:
    """Write JSON output to stdout."""
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif isinstance(data, list) and data and hasattr(data[0], "model_dump"):
        data = [item.model_dump() for item in data]
    print(json.dumps(data, indent=2, default=str), file=file)


def output_error(message: str, code: int = 1) -> None:
    """Write JSON error to stderr and exit."""
    output_json({"error": message, "code": code}, file=sys.stderr)
    raise SystemExit(code)


def print_strategies_table(strategies: list[dict[str, Any]]) -> None:
    """Print a rich table of strategies."""
    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Version")
    table.add_column("Description")

    for s in strategies:
        table.add_row(s["name"], s["type"], s["version"], s["description"])

    console.print(table)


def print_result_summary(result: dict[str, Any]) -> None:
    """Print a rich summary of a backtest result."""
    table = Table(title=f"Backtest Result: {result['run_id']}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Strategy", result["strategy_name"])
    table.add_row("Symbol", result["symbol"])
    table.add_row("Period", f"{result['start_date']} to {result['end_date']}")
    table.add_row("Initial Cash", f"${result['initial_cash']:,.2f}")
    table.add_row("Final Value", f"${result['final_value']:,.2f}")

    ret_pct = result.get("total_return_pct", 0)
    color = "green" if ret_pct >= 0 else "red"
    table.add_row("Total Return", f"[{color}]{ret_pct:+.2f}%[/{color}]")

    if result.get("sharpe_ratio") is not None:
        table.add_row("Sharpe Ratio", f"{result['sharpe_ratio']:.4f}")
    if result.get("max_drawdown_pct") is not None:
        table.add_row("Max Drawdown", f"{result['max_drawdown_pct']:.2f}%")

    table.add_row("Total Trades", str(result.get("total_trades", 0)))
    if result.get("win_rate") is not None:
        table.add_row("Win Rate", f"{result['win_rate']:.1f}%")

    console.print(table)


def print_results_list(results: list[dict[str, Any]]) -> None:
    """Print a rich table listing multiple backtest results."""
    table = Table(title="Backtest Results")
    table.add_column("Run ID", style="cyan")
    table.add_column("Strategy")
    table.add_column("Symbol")
    table.add_column("Period")
    table.add_column("Return %", justify="right")
    table.add_column("Sharpe", justify="right")

    for r in results:
        ret_pct = r.get("total_return_pct", 0)
        color = "green" if ret_pct >= 0 else "red"
        sharpe = f"{r['sharpe_ratio']:.4f}" if r.get("sharpe_ratio") is not None else "N/A"
        table.add_row(
            r["run_id"],
            r["strategy_name"],
            r["symbol"],
            f"{r['start_date']} to {r['end_date']}",
            f"[{color}]{ret_pct:+.2f}%[/{color}]",
            sharpe,
        )

    console.print(table)


_SIGNAL_COLORS = {
    "BUY": "green",
    "SELL": "red",
    "HOLD": "yellow",
    "NEUTRAL": "dim",
}


def print_signal_scan(result: Any) -> None:
    """Print a rich table for a signal scan result."""
    from advisor.engine.signals import ScanResult

    if not isinstance(result, ScanResult):
        output_json(result)
        return

    if not result.signals:
        console.print("[dim]No signals generated.[/dim]")
        return

    price = result.signals[0].price
    table = Table(title=f"Signal Scan: {result.symbol} — ${price:,.2f}")
    table.add_column("Strategy", style="cyan")
    table.add_column("Signal")
    table.add_column("Reason")

    for sig in result.signals:
        color = _SIGNAL_COLORS.get(sig.action.value, "white")
        table.add_row(
            sig.strategy_name,
            f"[{color}]{sig.action.value}[/{color}]",
            sig.reason,
        )

    console.print(table)
    console.print(f"  Summary: {result.summary}")
