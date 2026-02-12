"""CLI commands for strategy management."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import (
    output_error,
    output_json,
    print_strategies_table,
)
from advisor.strategies.registry import StrategyRegistry

app = typer.Typer(name="strategy", help="Manage trading strategies")


def _ensure_discovered() -> StrategyRegistry:
    registry = StrategyRegistry()
    registry.discover()
    return registry


@app.command("list")
def strategy_list(
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """List all available strategies."""
    registry = _ensure_discovered()
    strategies = registry.list_strategies()

    if output == "json":
        output_json(strategies)
    else:
        print_strategies_table(strategies)


@app.command("info")
def strategy_info(
    name: Annotated[str, typer.Argument(help="Strategy name")],
    output: Annotated[Optional[str], typer.Option("--output", help="Output format")] = None,
) -> None:
    """Show detailed information about a strategy."""
    registry = _ensure_discovered()
    try:
        cls = registry.get_strategy(name)
    except KeyError as e:
        output_error(str(e))
        return

    info = cls.get_metadata()
    if output == "json":
        output_json(info)
    else:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()
        text = (
            f"[cyan]Name:[/cyan] {info['name']}\n"
            f"[cyan]Type:[/cyan] {info['type']}\n"
            f"[cyan]Version:[/cyan] {info['version']}\n"
            f"[cyan]Description:[/cyan] {info['description']}\n"
            f"[cyan]Parameters:[/cyan]"
        )
        for k, v in info.get("params", {}).items():
            text += f"\n  {k} = {v}"
        console.print(Panel(text, title=f"Strategy: {name}"))
