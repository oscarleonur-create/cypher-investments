"""Root CLI application for the options advisor."""

from __future__ import annotations

import logging

import typer

from advisor.cli.daemon_cmds import app as daemon_app
from advisor.cli.data_cmds import app as data_app
from advisor.cli.entry_cmds import app as entry_app
from advisor.cli.research_cmds import app as research_app
from advisor.cli.scan_cmds import app as scan_app
from advisor.cli.thesis_cmds import app as thesis_app
from advisor.cli.web_cmds import app as web_app

app = typer.Typer(
    name="advisor",
    help="Always-on portfolio advisor - research, positions, macro exposure",
    no_args_is_help=True,
)

app.add_typer(daemon_app, name="daemon")
app.add_typer(data_app, name="data")
app.add_typer(entry_app, name="entry")
app.add_typer(research_app, name="research")
app.add_typer(scan_app, name="scan")
app.add_typer(thesis_app, name="thesis")
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
