"""CLI commands for live signal scanning."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import output_error, output_json, print_signal_scan

app = typer.Typer(name="signal", help="Live signal scanning")


@app.command("scan")
def signal_scan(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol to scan")],
    strategy: Annotated[
        Optional[str],
        typer.Option("--strategy", "-s", help="Specific strategy to run"),
    ] = None,
    output: Annotated[
        Optional[str], typer.Option("--output", help="Output format (json)")
    ] = None,
) -> None:
    """Scan a symbol with registered strategies to get live signals."""
    from advisor.engine.scanner import SignalScanner

    strategy_names = [strategy] if strategy else None

    try:
        scanner = SignalScanner()
        result = scanner.scan(symbol.upper(), strategy_names=strategy_names)
    except KeyError as e:
        output_error(str(e))
        return
    except Exception as e:
        output_error(f"Scan failed: {e}")
        return

    if output == "json":
        output_json(result)
    else:
        print_signal_scan(result)
