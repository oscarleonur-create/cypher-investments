"""Root CLI application for the options advisor."""

from __future__ import annotations

import logging

import typer

from advisor.cli.backtest_cmds import app as backtest_app
from advisor.cli.confluence_cmds import app as confluence_app
from advisor.cli.data_cmds import app as data_app
from advisor.cli.signal_cmds import app as signal_app
from advisor.cli.strategy_cmds import app as strategy_app

app = typer.Typer(
    name="advisor",
    help="Options financial advisor - backtesting and strategy system",
    no_args_is_help=True,
)

app.add_typer(strategy_app, name="strategy")
app.add_typer(backtest_app, name="backtest")
app.add_typer(data_app, name="data")
app.add_typer(signal_app, name="signal")
app.add_typer(confluence_app, name="confluence")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
) -> None:
    """Options financial advisor CLI."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s | %(name)s | %(message)s",
    )


if __name__ == "__main__":
    app()
