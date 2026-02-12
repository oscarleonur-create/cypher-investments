"""Typer CLI for the research agent."""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from research_agent.card import render_markdown
from research_agent.config import ResearchConfig
from research_agent.models import InputMode, ResearchInput
from research_agent.store import Store

app = typer.Typer(
    name="research_agent",
    help="Market research agent for buy-the-dip opportunity analysis.",
    no_args_is_help=True,
)
console = Console()


class OutputFormat(StrEnum):
    JSON = "json"
    MARKDOWN = "markdown"
    BOTH = "both"


@app.command()
def run(
    ticker: Annotated[str, typer.Option("--ticker", "-t", help="Stock ticker to research")],
    output: Annotated[
        OutputFormat, typer.Option("--output", "-o", help="Output format")
    ] = OutputFormat.BOTH,
    offline: Annotated[bool, typer.Option("--offline", help="Use cached data only")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Verbose logging")] = False,
) -> None:
    """Run the research pipeline for a ticker."""
    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    config = ResearchConfig()
    if offline:
        config.offline_mode = True

    research_input = ResearchInput(mode=InputMode.TICKER, value=ticker.upper())

    with console.status(f"[bold green]Researching {ticker.upper()}...", spinner="dots"):
        from research_agent.pipeline import run as run_pipeline

        card = run_pipeline(research_input, config)

    from research_agent.result import write_outputs

    json_path, md_path = write_outputs(card, config)

    # Display results
    if output in (OutputFormat.MARKDOWN, OutputFormat.BOTH):
        md_text = render_markdown(card)
        console.print(Markdown(md_text))

    if output in (OutputFormat.JSON, OutputFormat.BOTH):
        console.print(f"\n[dim]JSON: {json_path}[/dim]")

    console.print(f"[dim]Markdown: {md_path}[/dim]")
    console.print(f"[bold green]Done.[/bold green] Run ID: {card.id}")


@app.command()
def show(
    run_id: Annotated[str, typer.Argument(help="Run ID to display")],
) -> None:
    """Display a past opportunity card."""
    config = ResearchConfig()
    store = Store(config.db_path)
    try:
        card = store.load_run(run_id)
    finally:
        store.close()

    if card is None:
        console.print(f"[red]Run '{run_id}' not found.[/red]")
        raise typer.Exit(1)

    md_text = render_markdown(card)
    console.print(Markdown(md_text))


@app.command()
def history(
    ticker: Annotated[str | None, typer.Option("--ticker", "-t", help="Filter by ticker")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max results")] = 20,
) -> None:
    """List past research runs."""
    config = ResearchConfig()
    store = Store(config.db_path)
    try:
        runs = store.list_runs(ticker=ticker, limit=limit)
    finally:
        store.close()

    if not runs:
        console.print("[dim]No runs found.[/dim]")
        return

    table = Table(title="Research Runs")
    table.add_column("ID", style="cyan")
    table.add_column("Mode")
    table.add_column("Input")
    table.add_column("Verdict", style="bold")
    table.add_column("Dip Type")
    table.add_column("Date")

    for r in runs:
        verdict_style = {
            "BUY_THE_DIP": "[green]BUY_THE_DIP[/green]",
            "WATCH": "[yellow]WATCH[/yellow]",
            "AVOID": "[red]AVOID[/red]",
        }.get(r["verdict"] or "", r["verdict"] or "")

        table.add_row(
            r["id"],
            r["mode"],
            r["input_value"],
            verdict_style,
            r["dip_type"] or "",
            r["created_at"] or "",
        )

    console.print(table)
