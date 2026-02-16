"""CLI commands for market-wide scanning."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="market", help="Market-wide scanning with layered pre-filters")


@app.command("scan")
def market_scan(
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="Strategy to route qualifiers to")
    ] = "momentum_breakout",
    min_volume: Annotated[
        int, typer.Option("--min-volume", help="Minimum average daily volume")
    ] = 500_000,
    min_cap: Annotated[
        float, typer.Option("--min-cap", help="Minimum market cap in billions")
    ] = 2.0,
    include_sector: Annotated[
        Optional[list[str]],
        typer.Option("--include-sector", help="Include only these sectors (repeatable)"),
    ] = None,
    exclude_sector: Annotated[
        Optional[list[str]],
        typer.Option("--exclude-sector", help="Exclude these sectors (repeatable)"),
    ] = None,
    workers: Annotated[int, typer.Option("--workers", help="Parallel confluence workers")] = 4,
    universe: Annotated[
        str, typer.Option("--universe", "-u", help="Universe to scan (sp500, semiconductors)")
    ] = "sp500",
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Filters only, skip confluence (no API cost)")
    ] = False,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show filter rejection details")
    ] = False,
) -> None:
    """Scan a universe through layered filters, then run confluence on qualifiers."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn

    from advisor.data.cache import DiskCache
    from advisor.market.filters import FilterConfig
    from advisor.market.scanner import MarketScanner

    config = FilterConfig(
        min_avg_volume=min_volume,
        min_market_cap=min_cap * 1e9,
        include_sectors=include_sector,
        exclude_sectors=exclude_sector,
    )

    scanner = MarketScanner(cache=DiskCache())

    # Set up progress display
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    task_ids: dict[str, int] = {}

    def on_progress(phase: str, advance: int = 1) -> None:
        if phase == "universe":
            task_ids["universe"] = progress.add_task(
                f"[cyan]Loading {universe} universe...", total=None
            )
        elif phase == "universe_done":
            if "universe" in task_ids:
                progress.update(task_ids["universe"], completed=advance, total=advance)
            task_ids["ticker_info"] = progress.add_task(
                "[cyan]Layer 1: Fetching ticker info...", total=advance
            )
        elif phase == "ticker_info":
            if "ticker_info" in task_ids:
                progress.update(task_ids["ticker_info"], advance=advance)
        elif phase == "sector_done":
            pass
        elif phase == "technical_start":
            task_ids["technical"] = progress.add_task(
                "[cyan]Layer 3: Technical pre-screen...", total=None
            )
        elif phase == "technical_download_done":
            if "technical" in task_ids:
                progress.update(
                    task_ids["technical"], description="[cyan]Layer 3: Screening...", total=1
                )
        elif phase == "technical_done":
            if "technical" in task_ids:
                progress.update(task_ids["technical"], completed=1, total=1)
        elif phase == "confluence_start":
            task_ids["confluence"] = progress.add_task(
                "[cyan]Running confluence pipeline...", total=advance
            )
        elif phase == "confluence_tick":
            if "confluence" in task_ids:
                progress.update(task_ids["confluence"], advance=advance)

    try:
        with progress:
            result = scanner.scan(
                strategy_name=strategy,
                filter_config=config,
                max_workers=workers,
                dry_run=dry_run,
                on_progress=on_progress,
                universe=universe,
            )
    except KeyError as e:
        output_error(str(e))
        return
    except Exception as e:
        output_error(f"Market scan failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    # ── Human-readable output ─────────────────────────────────────────
    print_market_scan(result, strategy, dry_run, verbose)


def print_market_scan(
    result,
    strategy: str,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Print rich-formatted market scan results."""
    from rich.panel import Panel
    from rich.table import Table

    stats = result.filter_stats

    # Funnel summary
    mode_label = " [dim](dry run)[/dim]" if dry_run else ""
    funnel_text = (
        f"[bold]Strategy:[/bold] {result.strategy_name}{mode_label}\n"
        f"[bold]Universe:[/bold] {stats.universe_total} tickers\n\n"
        f"  {stats.universe_total}"
        f" → [cyan]{stats.after_volume_cap}[/cyan] (volume + cap)"
        f" → [cyan]{stats.after_sector}[/cyan] (sector)"
        f" → [green]{stats.after_technical}[/green] (technical)\n\n"
        f"[bold]Qualifiers:[/bold] {len(result.qualifiers)}"
    )
    if result.qualifiers:
        funnel_text += f"  ({', '.join(result.qualifiers)})"

    console.print(Panel(funnel_text, title="Filter Funnel", border_style="blue"))

    if verbose:
        detail_table = Table(title="Filter Details")
        detail_table.add_column("Layer", style="cyan")
        detail_table.add_column("Passed", justify="right")
        detail_table.add_column("Rejected", justify="right")
        detail_table.add_row(
            "Volume + Cap",
            str(stats.after_volume_cap),
            str(stats.volume_cap_rejected_count),
        )
        detail_table.add_row(
            "Sector",
            str(stats.after_sector),
            str(stats.sector_rejected_count),
        )
        detail_table.add_row(
            "Technical",
            str(stats.after_technical),
            str(stats.technical_rejected_count),
        )
        if stats.fetch_error_count > 0:
            detail_table.add_row(
                "Fetch Errors",
                "-",
                str(stats.fetch_error_count),
            )
        console.print(detail_table)

    # Confluence results table
    if result.results:
        verdict_colors = {"ENTER": "green", "CAUTION": "yellow", "PASS": "red"}
        results_table = Table(title="Confluence Results")
        results_table.add_column("Symbol", style="cyan")
        results_table.add_column("Verdict")
        results_table.add_column("Tech", justify="center")
        results_table.add_column("Sent", justify="center")
        results_table.add_column("Fund", justify="center")
        results_table.add_column("Reasoning", max_width=60)

        for r in result.results:
            color = verdict_colors.get(r.verdict.value, "white")
            tech_icon = "[green]✓[/green]" if r.technical.is_bullish else "[red]✗[/red]"
            sent_icon = "[green]✓[/green]" if r.sentiment.is_bullish else "[red]✗[/red]"
            fund_icon = "[green]✓[/green]" if r.fundamental.is_clear else "[red]✗[/red]"
            results_table.add_row(
                r.symbol,
                f"[bold {color}]{r.verdict.value}[/bold {color}]",
                tech_icon,
                sent_icon,
                fund_icon,
                r.reasoning[:120],
            )

        console.print(results_table)

    if result.errors:
        console.print(f"\n[yellow]Errors ({len(result.errors)}):[/yellow]")
        for err in result.errors:
            console.print(f"  [dim]{err['symbol']}:[/dim] {err['error']}")

    # Timing
    elapsed = result.elapsed_seconds
    console.print(f"\n[dim]Completed in {elapsed:.1f}s[/dim]")
