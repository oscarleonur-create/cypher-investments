"""CLI commands for market-wide scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

if TYPE_CHECKING:
    from advisor.confluence.mispricing_screener import MispricingResult
    from advisor.confluence.smart_money_screener import SmartMoneyResult

app = typer.Typer(name="market", help="Market-wide scanning with layered pre-filters")


@app.command("smart-money")
def smart_money_scan(
    ticker: Annotated[
        Optional[str], typer.Option("--ticker", "-t", help="Check a single ticker")
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", help="Show top N results")] = 10,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Scan for smart money signals (insider buying, congressional trades, technicals)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from advisor.confluence.smart_money_screener import (
        get_sp500_tickers,
        screen_smart_money,
    )

    if ticker:
        # Single ticker mode
        result = screen_smart_money(ticker)
        if output == "json":
            output_json(result)
            return
        _print_smart_money_detail(result)
        return

    # Full scan mode
    tickers = get_sp500_tickers()
    results: list[SmartMoneyResult] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Scanning smart money signals...", total=len(tickers))

        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(screen_smart_money, t): t for t in tickers}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                except Exception:
                    pass
                progress.update(task, advance=1)

    # Sort by score, take top N
    results.sort(key=lambda r: r.total_score, reverse=True)
    results = results[:top]

    if output == "json":
        output_json([r.model_dump(mode="json") for r in results])
        return

    # Rich table
    signal_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "WATCH": "yellow",
        "HOLD": "dim",
        "SELL": "bold red",
    }

    table = Table(title=f"Smart Money — Top {top}")
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Signal")
    table.add_column("Insider", justify="right")
    table.add_column("Congress", justify="right")
    table.add_column("Technical", justify="right")
    table.add_column("Options", justify="right")
    table.add_column("Details", max_width=50)

    for r in results:
        color = signal_colors.get(r.signal.value, "white")
        details_parts = []
        if r.insider.cluster_buys:
            details_parts.append(f"{r.insider.cluster_buys} insider buys")
        if r.insider.cluster_sells:
            details_parts.append(f"{r.insider.cluster_sells} insider sells")
        if r.congress.recent_buys:
            details_parts.append(f"{r.congress.recent_buys} congress")
        if r.technical.above_sma50:
            details_parts.append(f"{r.technical.pct_from_high:.1f}% from high")
        if r.options_activity.score > 0:
            details_parts.append(f"opt vol {r.options_activity.volume_ratio:.1f}x")

        insider_label = f"{r.insider.score:+.0f}"
        table.add_row(
            r.symbol,
            f"{r.total_score:.0f}",
            f"[{color}]{r.signal.value}[/{color}]",
            f"{insider_label}/35",
            f"{r.congress.score:.0f}/20",
            f"{r.technical.score:.0f}/25",
            f"{r.options_activity.score:.0f}/20",
            ", ".join(details_parts) if details_parts else "-",
        )

    console.print(table)


def _print_smart_money_detail(r: "SmartMoneyResult") -> None:
    """Print detailed smart money result for a single ticker."""
    from rich.panel import Panel

    signal_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "WATCH": "yellow",
        "HOLD": "dim",
        "SELL": "bold red",
    }
    color = signal_colors.get(r.signal.value, "white")

    opts = r.options_activity
    notable = ", ".join(opts.notable_strikes[:3]) if opts.notable_strikes else "none"
    insider_sign = f"{r.insider.score:+.0f}"
    text = (
        f"[bold]{r.symbol}[/bold] — [{color}]{r.signal.value}[/{color}]"
        f" (Score: {r.total_score:.0f}/100)\n\n"
        f"[bold]Insider Score:[/bold] {insider_sign}/35"
        f"  (buys: {r.insider.cluster_buys}, sells: {r.insider.cluster_sells},"
        f" large buys: {r.insider.large_buys}, large sells: {r.insider.large_sells},"
        f" C-suite buys: {r.insider.csuite_buys}, C-suite sells: {r.insider.csuite_sells})\n"
        f"[bold]Congress Score:[/bold] {r.congress.score:.0f}/20"
        f"  (buys: {r.congress.recent_buys},"
        f" politicians: {', '.join(r.congress.politicians) or 'none'})\n"
        f"[bold]Technical Score:[/bold] {r.technical.score:.0f}/25"
        f"  (price: ${r.technical.price:.2f},"
        f" {r.technical.pct_from_high:.1f}% from 20d high,"
        f" {'above' if r.technical.above_sma50 else 'below'} 50 SMA,"
        f" vol {'↑' if r.technical.volume_trending_up else '↓'})\n"
        f"[bold]Options Activity:[/bold] {opts.score:.0f}/20"
        f"  (vol: {opts.total_volume:,}, ratio: {opts.volume_ratio:.1f}x,"
        f" P/C: {opts.put_call_ratio:.2f}, strikes: {notable})"
    )

    console.print(Panel(text, title="Smart Money Analysis", border_style="blue"))


@app.command("mispricing")
def mispricing_scan(
    ticker: Annotated[
        Optional[str], typer.Option("--ticker", "-t", help="Check a single ticker")
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", help="Show top N results")] = 10,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
    sector: Annotated[
        Optional[str], typer.Option("--sector", "-s", help="Filter by sector")
    ] = None,
) -> None:
    """Scan for mispriced stocks using fundamental, options, and estimate signals."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from advisor.confluence.mispricing_screener import (
        screen_mispricing,
    )
    from advisor.confluence.smart_money_screener import get_sp500_tickers

    if ticker:
        # Single ticker mode
        result = screen_mispricing(ticker)
        if output == "json":
            output_json(result)
            return
        _print_mispricing_detail(result)
        return

    # Full scan mode
    tickers = get_sp500_tickers()

    # Pre-filter by sector using Wikipedia data if --sector given
    if sector:
        from advisor.confluence.smart_money_screener import get_sp500_by_sector

        by_sector = get_sp500_by_sector()
        sector_lower = sector.lower()
        matched_tickers: set[str] = set()
        for s, s_tickers in by_sector.items():
            if sector_lower in s.lower():
                matched_tickers.update(s_tickers)
        if matched_tickers:
            tickers = [t for t in tickers if t in matched_tickers]
        else:
            console.print(f"[yellow]No tickers found for sector '{sector}'[/yellow]")
            return

    results: list[MispricingResult] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Scanning for mispricing...", total=len(tickers))

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(screen_mispricing, t): t for t in tickers}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                except Exception:
                    pass
                progress.update(task, advance=1)

    # Sort by score, take top N
    results.sort(key=lambda r: r.total_score, reverse=True)
    results = results[:top]

    if output == "json":
        output_json([r.model_dump(mode="json") for r in results])
        return

    # Rich table
    signal_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "WATCH": "yellow",
        "HOLD": "dim",
    }

    table = Table(title=f"Mispricing Scanner — Top {top}")
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Signal")
    table.add_column("Fundamental", justify="right")
    table.add_column("Options Mkt", justify="right")
    table.add_column("Estimates", justify="right")
    table.add_column("Details", max_width=60)

    for r in results:
        color = signal_colors.get(r.signal.value, "white")
        details_parts = []
        if r.fundamental.f_score:
            details_parts.append(f"F={r.fundamental.f_score}")
        if r.fundamental.discount_pct is not None:
            details_parts.append(f"P/E disc {r.fundamental.discount_pct:+.0f}%")
        if r.options_market.iv_rv_ratio is not None:
            details_parts.append(f"IV/RV {r.options_market.iv_rv_ratio:.2f}")
        if r.estimate_revisions.upside_pct is not None:
            details_parts.append(f"upside {r.estimate_revisions.upside_pct:+.0f}%")

        table.add_row(
            r.symbol,
            f"{r.total_score:.0f}",
            f"[{color}]{r.signal.value}[/{color}]",
            f"{r.fundamental.score:.0f}/40",
            f"{r.options_market.score:.0f}/30",
            f"{r.estimate_revisions.score:.0f}/30",
            ", ".join(details_parts) if details_parts else "-",
        )

    console.print(table)


def _print_mispricing_detail(r: "MispricingResult") -> None:
    """Print detailed mispricing result for a single ticker."""
    from rich.panel import Panel

    signal_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "WATCH": "yellow",
        "HOLD": "dim",
    }
    color = signal_colors.get(r.signal.value, "white")

    f = r.fundamental
    o = r.options_market
    e = r.estimate_revisions

    # F-Score details
    f_details_str = ""
    if f.f_score_details:
        checks = [f"{'✓' if v else '✗'} {k}" for k, v in f.f_score_details.items()]
        f_details_str = f"\n    {', '.join(checks)}"

    text = (
        f"[bold]{r.symbol}[/bold] — [{color}]{r.signal.value}[/{color}]"
        f" (Score: {r.total_score:.0f}/100)\n\n"
        f"[bold]Fundamental:[/bold] {f.score:.0f}/40"
        f"  (F-Score: {f.f_score}/9{f_details_str})\n"
        f"  P/E: {f.pe_ratio or 'N/A'} (sector: {f.sector_pe or 'N/A'})"
        f"  P/B: {f.pb_ratio or 'N/A'} (sector: {f.sector_pb or 'N/A'})"
        f"  EV/EBITDA: {f.ev_ebitda or 'N/A'} (sector: {f.sector_ev_ebitda or 'N/A'})\n\n"
        f"[bold]Options Market:[/bold] {o.score:.0f}/30"
        f"  (IV: {o.implied_vol or 'N/A'}, RV20d: {o.realized_vol_20d or 'N/A'},"
        f" IV/RV: {o.iv_rv_ratio or 'N/A'},"
        f" IV Rank: {o.iv_rank or 'N/A'},"
        f" P/C OI: {o.put_call_oi_ratio or 'N/A'},"
        f" Skew: {o.skew_pct or 'N/A'}%)\n"
        f"  Notable: {', '.join(o.notable_strikes[:3]) if o.notable_strikes else 'none'}\n\n"
        f"[bold]Estimate Revisions:[/bold] {e.score:.0f}/30"
        f"  (Price: ${e.current_price or 0:.2f},"
        f" Target: ${e.target_price or 0:.2f},"
        f" Upside: {e.upside_pct or 0:+.1f}%)\n"
        f"  Upgrades: {e.recent_upgrades}, Downgrades: {e.recent_downgrades},"
        f" Rec Mean: {e.recommendation_mean or 'N/A'},"
        f" EPS Growth: {e.earnings_growth_est or 'N/A'}%"
    )

    console.print(Panel(text, title="Mispricing Analysis", border_style="blue"))


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
