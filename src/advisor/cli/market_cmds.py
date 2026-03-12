"""CLI commands for market-wide scanning."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

if TYPE_CHECKING:
    from advisor.confluence.alpha_scorer import AlphaResult
    from advisor.confluence.dip_analyzer import DipAnalysisResult
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


# ── Alpha Score ──────────────────────────────────────────────────────────


@app.command("alpha")
def alpha_scan(
    ticker: Annotated[
        Optional[str], typer.Option("--ticker", "-t", help="Single ticker deep-dive")
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", help="Top N results from S&P 500")] = 10,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
    skip_sentiment: Annotated[
        bool, typer.Option("--skip-sentiment", help="Skip sentiment layer")
    ] = False,
    skip_ml: Annotated[bool, typer.Option("--skip-ml", help="Skip ML signal layer")] = False,
) -> None:
    """Compute a unified alpha conviction score (0-100) across all signal layers."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from advisor.confluence.alpha_scorer import compute_alpha
    from advisor.confluence.smart_money_screener import get_sp500_tickers

    skip_layers: set[str] = set()
    if skip_sentiment:
        skip_layers.add("sentiment")
    if skip_ml:
        skip_layers.add("ml_signal")

    if ticker:
        result = compute_alpha(ticker, skip_layers=skip_layers)
        if output == "json":
            output_json(result)
            return
        _print_alpha_detail(result)
        return

    # Full scan mode
    tickers = get_sp500_tickers()
    results: list[AlphaResult] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Computing alpha scores...", total=len(tickers))

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {pool.submit(compute_alpha, t, None, skip_layers): t for t in tickers}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                except Exception:
                    pass
                progress.update(task, advance=1)

    results.sort(key=lambda r: r.alpha_score, reverse=True)
    results = results[:top]

    if output == "json":
        output_json([r.model_dump(mode="json") for r in results])
        return

    # Rich table
    signal_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "LEAN_BUY": "cyan",
        "NEUTRAL": "yellow",
        "LEAN_SELL": "magenta",
        "AVOID": "bold red",
    }

    table = Table(title=f"Alpha Score — Top {top}")
    table.add_column("Symbol", style="cyan")
    table.add_column("Alpha", justify="right")
    table.add_column("Signal")
    table.add_column("Tech", justify="right")
    table.add_column("Sent", justify="right")
    table.add_column("Fund", justify="right")
    table.add_column("Smart$", justify="right")
    table.add_column("Misprice", justify="right")
    table.add_column("ML", justify="right")
    table.add_column("Layers", justify="right")

    for r in results:
        color = signal_colors.get(r.signal.value, "white")
        layer_map = {ls.name: ls for ls in r.layers}

        def _fmt(name: str) -> str:
            ls = layer_map.get(name)
            if ls is None or not ls.available:
                return "[dim]-[/dim]"
            return f"{ls.normalized:.0f}"

        table.add_row(
            r.symbol,
            f"{r.alpha_score:.0f}",
            f"[{color}]{r.signal.value}[/{color}]",
            _fmt("technical"),
            _fmt("sentiment"),
            _fmt("fundamental"),
            _fmt("smart_money"),
            _fmt("mispricing"),
            _fmt("ml_signal"),
            f"{r.active_layers}/{r.total_layers}",
        )

    console.print(table)


def _print_alpha_detail(r: "AlphaResult") -> None:
    """Print detailed alpha result for a single ticker."""
    from rich.panel import Panel
    from rich.table import Table

    signal_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "LEAN_BUY": "cyan",
        "NEUTRAL": "yellow",
        "LEAN_SELL": "magenta",
        "AVOID": "bold red",
    }
    color = signal_colors.get(r.signal.value, "white")

    header = (
        f"[bold]{r.symbol}[/bold] — [{color}]{r.signal.value}[/{color}]"
        f"  (Alpha: {r.alpha_score:.1f}/100)"
        f"  [{r.active_layers}/{r.total_layers} layers active]"
    )
    console.print(Panel(header, title="Alpha Score", border_style="blue"))

    # Layer breakdown table
    table = Table(title="Layer Breakdown")
    table.add_column("Layer", style="cyan")
    table.add_column("Norm", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Contrib", justify="right")
    table.add_column("Status")

    for ls in r.layers:
        if ls.available:
            status = "[green]OK[/green]"
            table.add_row(
                ls.name,
                f"{ls.normalized:.0f}",
                f"{ls.weight:.0%}",
                f"{ls.weighted_contribution:.1f}",
                status,
            )
        else:
            status = f"[dim]{ls.error or 'unavailable'}[/dim]"
            table.add_row(ls.name, "-", "-", "-", status)

    console.print(table)


# ── Dip Analysis ─────────────────────────────────────────────────────────


@app.command("dip")
def dip_scan(
    ticker: Annotated[
        Optional[str], typer.Option("--ticker", "-t", help="Single ticker deep-dive")
    ] = None,
    top: Annotated[int, typer.Option("--top", "-n", help="Top N results from S&P 500")] = 10,
    sector: Annotated[
        Optional[str], typer.Option("--sector", "-s", help="Filter by GICS sector")
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
    skip_ml: Annotated[bool, typer.Option("--skip-ml", help="Skip ML signal layer")] = False,
    skip_sentiment: Annotated[
        bool, typer.Option("--skip-sentiment", help="Skip sentiment in confluence layer")
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show dip screener detail tables")
    ] = False,
    workers: Annotated[int, typer.Option("--workers", "-w", help="Parallel workers")] = 3,
) -> None:
    """Unified dip-buying analysis combining 6 signal layers with regime adjustment."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from advisor.confluence.dip_analyzer import analyze_dip

    skip_layers: set[str] = set()
    if skip_ml:
        skip_layers.add("ml_signal")
    if skip_sentiment:
        skip_layers.add("confluence")

    if ticker:
        result = analyze_dip(ticker, skip_layers=skip_layers)
        if output == "json":
            output_json(result)
            return
        _print_dip_detail(result, verbose=verbose)
        return

    # ── Universe scan mode ───────────────────────────────────────────
    from advisor.confluence.smart_money_screener import get_sp500_by_sector, get_sp500_tickers

    tickers = get_sp500_tickers()

    if sector:
        by_sector = get_sp500_by_sector()
        sector_lower = sector.lower()
        matched: set[str] = set()
        for s, s_tickers in by_sector.items():
            if sector_lower in s.lower():
                matched.update(s_tickers)
        if matched:
            tickers = [t for t in tickers if t in matched]
        else:
            console.print(f"[yellow]No tickers found for sector '{sector}'[/yellow]")
            return

    results: list[DipAnalysisResult] = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("[cyan]Scanning dip opportunities...", total=len(tickers))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_dip, t, skip_layers): t for t in tickers}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results.append(r)
                except Exception:
                    pass
                progress.update(task, advance=1)

    results.sort(key=lambda r: r.dip_score, reverse=True)
    results = results[:top]

    if output == "json":
        output_json([r.model_dump(mode="json") for r in results])
        return

    # ── Summary table ────────────────────────────────────────────────
    verdict_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "LEAN_BUY": "cyan",
        "WATCH": "yellow",
        "PASS": "red",
    }

    title = f"Dip Analysis — Top {top}"
    if sector:
        title += f" ({sector})"

    table = Table(title=title)
    table.add_column("Symbol", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Verdict")
    table.add_column("Dip", justify="right")
    table.add_column("Smart$", justify="right")
    table.add_column("Misprice", justify="right")
    table.add_column("Conflu", justify="right")
    table.add_column("ML", justify="right")
    table.add_column("Tech", justify="right")
    table.add_column("Regime")

    for r in results:
        color = verdict_colors.get(r.verdict.value, "white")
        layer_map = {ls.name: ls for ls in r.layers}

        def _fmt(name: str) -> str:
            ls = layer_map.get(name)
            if ls is None or not ls.available:
                return "[dim]-[/dim]"
            return f"{ls.normalized:.0f}"

        regime_label = {"low_vol": "Calm", "normal": "Normal", "high_vol": "Stressed"}.get(
            r.regime, r.regime
        )

        table.add_row(
            r.symbol,
            f"{r.dip_score:.0f}",
            f"[{color}]{r.verdict.value}[/{color}]",
            _fmt("dip_screener"),
            _fmt("smart_money"),
            _fmt("mispricing"),
            _fmt("confluence"),
            _fmt("ml_signal"),
            _fmt("technical_dip"),
            regime_label,
        )

    console.print(table)


def _print_dip_detail(r: "DipAnalysisResult", *, verbose: bool = False) -> None:
    """Print detailed dip analysis for a single ticker."""
    from rich.panel import Panel
    from rich.table import Table

    verdict_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "LEAN_BUY": "cyan",
        "WATCH": "yellow",
        "PASS": "red",
    }
    color = verdict_colors.get(r.verdict.value, "white")

    regime_label = {"low_vol": "Calm", "normal": "Normal", "high_vol": "Stressed"}.get(
        r.regime, r.regime
    )
    adj_str = f"{r.regime_adjustment:+.0f}" if r.regime_adjustment else "+0"

    header = (
        f"[bold]{r.symbol}[/bold] — [{color}]{r.verdict.value}[/{color}]"
        f"  (Score: {r.dip_score:.1f}/100)"
        f"  Price: ${r.price:.2f}"
        f"  Regime: {regime_label} ({adj_str})"
        f"\n\n{r.reasoning}"
    )
    console.print(Panel(header, title="Dip Analysis", border_style="blue"))

    # Layer breakdown table
    table = Table(title="Layer Breakdown")
    table.add_column("Layer", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Weight", justify="right")
    table.add_column("Contrib", justify="right")
    table.add_column("Status")

    for ls in r.layers:
        if ls.available:
            table.add_row(
                ls.name,
                f"{ls.normalized:.0f}",
                f"{ls.weight:.0%}",
                f"{ls.weighted_contribution:.1f}",
                "[green]OK[/green]",
            )
        else:
            table.add_row(
                ls.name,
                "-",
                "-",
                "-",
                f"[dim]{ls.error or 'unavailable'}[/dim]",
            )

    console.print(table)

    if not verbose:
        return

    # ── Verbose: dip screener detail ─────────────────────────────────
    # Re-run dip screener to get raw detail (already cached in most cases)
    try:
        from advisor.confluence.dip_screener import check_dip_fundamental

        fund = check_dip_fundamental(r.symbol)
        ds = fund.dip_screener
        if ds is None:
            return

        detail = Table(title="Dip Screener Detail")
        detail.add_column("Check", style="cyan")
        detail.add_column("Value", justify="right")
        detail.add_column("Status")

        # Safety gate
        s = ds.safety
        detail.add_row(
            "Current Ratio",
            f"{s.current_ratio:.2f}" if s.current_ratio is not None else "N/A",
            "[green]OK[/green]" if s.current_ratio_ok else "[red]FAIL[/red]",
        )
        detail.add_row(
            "Debt/Equity",
            f"{s.debt_to_equity:.2f}" if s.debt_to_equity is not None else "N/A",
            "[green]OK[/green]" if s.debt_to_equity_ok else "[red]FAIL[/red]",
        )
        detail.add_row(
            "FCF (4Q > 0)",
            str(len([f for f in s.fcf_values if f > 0])) + f"/{len(s.fcf_values)}",
            "[green]OK[/green]" if s.fcf_ok else "[red]FAIL[/red]",
        )

        # Value trap
        if ds.value_trap:
            vt = ds.value_trap
            detail.add_row(
                "P/E Discount",
                f"{vt.pe_discount_pct:.0f}%" if vt.pe_discount_pct is not None else "N/A",
                "[green]ON SALE[/green]" if vt.pe_on_sale else "[dim]no[/dim]",
            )
            detail.add_row(
                "RSI Divergence",
                f"{vt.price_change_pct:.1f}% drop" if vt.price_change_pct else "N/A",
                "[green]YES[/green]" if vt.rsi_divergence else "[dim]no[/dim]",
            )

        # Fast fundamentals
        if ds.fast_fundamentals:
            ff = ds.fast_fundamentals
            detail.add_row(
                "Insider Buying",
                f"{len(ff.insider_details)} txns",
                "[green]YES[/green]" if ff.insider_buying else "[dim]no[/dim]",
            )
            detail.add_row(
                "Analyst Upside",
                f"{ff.analyst_upside_pct:+.0f}%" if ff.analyst_upside_pct else "N/A",
                "[green]BULLISH[/green]" if ff.analyst_bullish else "[dim]no[/dim]",
            )

        console.print(detail)
    except Exception as e:
        console.print(f"[dim]Could not load dip screener detail: {e}[/dim]")
