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


def print_walk_forward_summary(result: Any) -> None:
    """Print a rich summary of walk-forward analysis results."""
    # Header table with aggregate metrics
    header = Table(title=f"Walk-Forward Analysis: {result.strategy} on {result.symbol}")
    header.add_column("Metric", style="cyan")
    header.add_column("Value", justify="right")

    header.add_row("Run ID", result.run_id)
    header.add_row("Period", f"{result.start_date} to {result.end_date}")
    header.add_row("Windows", str(result.n_windows))
    header.add_row("Train %", f"{result.train_pct * 100:.0f}%")
    header.add_row("", "")

    color = "green" if result.oos_avg_return_pct >= 0 else "red"
    header.add_row("OOS Avg Return", f"[{color}]{result.oos_avg_return_pct:+.2f}%[/{color}]")
    if result.oos_avg_sharpe is not None:
        header.add_row("OOS Avg Sharpe", f"{result.oos_avg_sharpe:.4f}")
    header.add_row("OOS Avg Max DD", f"{result.oos_avg_max_dd_pct:.2f}%")
    header.add_row("", "")

    gap_color = "green" if result.is_vs_oos_gap <= 5.0 else "yellow" if result.is_vs_oos_gap <= 15.0 else "red"
    header.add_row("IS Avg Return", f"{result.is_avg_return_pct:+.2f}%")
    header.add_row("IS-vs-OOS Gap", f"[{gap_color}]{result.is_vs_oos_gap:+.2f}pp[/{gap_color}]")

    console.print(header)

    # Per-window table
    windows_table = Table(title="Per-Window Results")
    windows_table.add_column("#", style="dim")
    windows_table.add_column("Train Period")
    windows_table.add_column("Test Period")
    windows_table.add_column("IS Return %", justify="right")
    windows_table.add_column("OOS Return %", justify="right")
    windows_table.add_column("OOS Sharpe", justify="right")

    for w in result.windows:
        is_ret = w.in_sample.total_return_pct
        oos_ret = w.out_of_sample.total_return_pct
        is_color = "green" if is_ret >= 0 else "red"
        oos_color = "green" if oos_ret >= 0 else "red"
        sharpe = f"{w.out_of_sample.sharpe_ratio:.4f}" if w.out_of_sample.sharpe_ratio is not None else "N/A"

        windows_table.add_row(
            str(w.window_index + 1),
            f"{w.train_start} to {w.train_end}",
            f"{w.test_start} to {w.test_end}",
            f"[{is_color}]{is_ret:+.2f}%[/{is_color}]",
            f"[{oos_color}]{oos_ret:+.2f}%[/{oos_color}]",
            sharpe,
        )

    console.print(windows_table)


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
