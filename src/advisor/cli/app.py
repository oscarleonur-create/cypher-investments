"""Root CLI application for the options advisor."""

from __future__ import annotations

import logging

import typer

from advisor.cli.backtest_cmds import app as backtest_app
from advisor.cli.case_cmds import app as case_app
from advisor.cli.confluence_cmds import app as confluence_app
from advisor.cli.dashboard_cmds import app as dashboard_app
from advisor.cli.data_cmds import app as data_app
from advisor.cli.ml_cmds import app as ml_app
from advisor.cli.options_cmds import app as options_app
from advisor.cli.research_cmds import app as research_app
from advisor.cli.scenario_cmds import app as scenario_app
from advisor.cli.simulator_cmds import app as simulator_app
from advisor.cli.strategy_cmds import app as strategy_app
from advisor.cli.web_cmds import app as web_app

app = typer.Typer(
    name="advisor",
    help="Options financial advisor - backtesting and strategy system",
    no_args_is_help=True,
)

app.add_typer(strategy_app, name="strategy")
app.add_typer(case_app, name="case")
app.add_typer(backtest_app, name="backtest")
app.add_typer(data_app, name="data")
app.add_typer(confluence_app, name="confluence")
app.add_typer(options_app, name="options")
app.add_typer(simulator_app, name="simulator")
app.add_typer(ml_app, name="ml")
app.add_typer(research_app, name="research")
app.add_typer(scenario_app, name="scenario")
app.add_typer(dashboard_app, name="dashboard")
app.add_typer(web_app, name="web")


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
