"""CLI commands for confluence scanning."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="confluence", help="Confluence scanner")


@app.command("scan")
def confluence_scan(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol to scan")],
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="Strategy to use for the technical check")
    ] = "momentum_breakout",
    output: Annotated[
        Optional[str], typer.Option("--output", help="Output format (json)")
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show detailed agent results")
    ] = False,
) -> None:
    """Run all three confluence checks (technical, sentiment, fundamental) on a symbol."""
    from rich.panel import Panel
    from rich.table import Table

    from advisor.confluence.orchestrator import run_confluence

    try:
        result = run_confluence(symbol.upper(), strategy_name=strategy)
    except Exception as e:
        output_error(f"Confluence scan failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    # Verdict panel
    verdict_colors = {"ENTER": "green", "CAUTION": "yellow", "PASS": "red"}
    color = verdict_colors.get(result.verdict.value, "white")
    console.print(
        Panel(
            f"[bold {color}]{result.verdict.value}[/bold {color}]\n\n"
            f"{result.reasoning}\n\n"
            f"Suggested hold: {result.suggested_hold_days} days",
            title=f"Confluence: {result.symbol} ({result.strategy_name})",
        )
    )

    # Agent results table
    table = Table(title="Agent Results")
    table.add_column("Agent", style="cyan")
    table.add_column("Status")
    table.add_column("Detail")

    # Technical
    tech_color = "green" if result.technical.is_bullish else "red"
    tech_status = f"[{tech_color}]{'BULLISH' if result.technical.is_bullish else 'BEARISH'}[/{tech_color}]"
    tech_detail = f"Signal: {result.technical.signal}, Price: ${result.technical.price:,.2f}"
    table.add_row("Technical", tech_status, tech_detail)

    # Sentiment
    sent_color = "green" if result.sentiment.is_bullish else "red"
    sent_status = f"[{sent_color}]{'BULLISH' if result.sentiment.is_bullish else 'BEARISH'}[/{sent_color}]"
    sent_detail = f"Score: {result.sentiment.score:.0f}/100, Positive: {result.sentiment.positive_pct:.0f}%"
    table.add_row("Sentiment", sent_status, sent_detail)

    # Fundamental
    fund_color = "green" if result.fundamental.is_clear else "red"
    fund_status = f"[{fund_color}]{'CLEAR' if result.fundamental.is_clear else 'RISK'}[/{fund_color}]"
    fund_parts = []
    if result.fundamental.earnings_within_7_days:
        fund_parts.append(f"Earnings: {result.fundamental.earnings_date}")
    else:
        fund_parts.append("No imminent earnings")
    if result.fundamental.insider_buying_detected:
        fund_parts.append("Insider buying detected")
    table.add_row("Fundamental", fund_status, ", ".join(fund_parts))

    console.print(table)

    # Verbose: show headlines and sources
    if verbose:
        if result.sentiment.key_headlines:
            console.print("\n[bold]Key Headlines:[/bold]")
            for headline in result.sentiment.key_headlines:
                console.print(f"  - {headline}")

        if result.sentiment.sources:
            source_table = Table(title="Sources")
            source_table.add_column("ID", style="dim")
            source_table.add_column("Tier", justify="center")
            source_table.add_column("Title")
            source_table.add_column("URL", style="dim")
            for src in result.sentiment.sources:
                tier_color = {1: "green", 2: "yellow"}.get(src.tier, "dim")
                source_table.add_row(
                    src.source_id,
                    f"[{tier_color}]{src.tier}[/{tier_color}]",
                    src.title,
                    src.url,
                )
            console.print(source_table)
