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


# ── Pipeline command ──────────────────────────────────────────────────────────


@app.command("pipeline")
def market_pipeline(
    strategy: Annotated[
        str, typer.Option("--strategy", "-s", help="Equity strategy for signals")
    ] = "momentum_breakout",
    ticker: Annotated[
        Optional[str], typer.Option("--ticker", "-t", help="Single ticker (skip scanner)")
    ] = None,
    universe: Annotated[
        str, typer.Option("--universe", "-u", help="Universe to scan (sp500, semiconductors)")
    ] = "sp500",
    min_volume: Annotated[
        int, typer.Option("--min-volume", help="Minimum avg daily volume")
    ] = 500_000,
    min_cap: Annotated[
        float, typer.Option("--min-cap", help="Minimum market cap in billions")
    ] = 2.0,
    top_n: Annotated[int, typer.Option("--top-n", help="Max stocks to show")] = 10,
    workers: Annotated[int, typer.Option("--workers", help="Parallel workers")] = 3,
    skip_ml: Annotated[bool, typer.Option("--skip-ml", help="Skip ML signal layer")] = False,
    skip_sentiment: Annotated[
        bool, typer.Option("--skip-sentiment", help="Skip sentiment layer")
    ] = False,
    cash: Annotated[float, typer.Option("--cash", help="Portfolio equity for sizing")] = 100_000.0,
    risk_pct: Annotated[
        float, typer.Option("--risk-pct", help="Risk per trade as decimal (e.g. 0.02 = 2%%)")
    ] = 0.02,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Signal → score → size pipeline for live equity trades.

    Scans for stocks with active entry signals, scores conviction via the
    dip/alpha layers, computes ATR-based position sizes, and recommends
    stop-loss / trailing-stop levels — all in one shot.

    Examples:
        # Single ticker deep-dive
        advisor market pipeline -t AAPL

        # Full universe scan → signal → size
        advisor market pipeline --strategy buy_the_dip --top-n 5

        # Custom sizing
        advisor market pipeline -t NVDA --cash 50000 --risk-pct 0.01
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from rich.panel import Panel
    from rich.table import Table

    from advisor.confluence.dip_analyzer import analyze_dip
    from advisor.engine.scanner import SignalScanner

    # ── Step 1: Gather candidate symbols ──────────────────────────────

    if ticker:
        symbols = [ticker.upper()]
    else:
        # Use the market scanner to filter the universe down to qualifiers
        from advisor.data.cache import DiskCache
        from advisor.market.filters import FilterConfig
        from advisor.market.scanner import MarketScanner

        console.print(f"Filtering {universe} universe...")

        filter_config = FilterConfig(
            min_avg_volume=min_volume,
            min_market_cap=min_cap * 1e9,
        )
        scanner = MarketScanner(cache=DiskCache())
        scan_result = scanner.scan(
            strategy_name=strategy,
            filter_config=filter_config,
            max_workers=workers,
            dry_run=True,  # filters only, no confluence API cost
            universe=universe,
        )
        symbols = scan_result.qualifiers
        console.print(
            f"  {scan_result.filter_stats.universe_total} tickers " f"-> {len(symbols)} qualifiers"
        )

    if not symbols:
        console.print("[yellow]No qualifying stocks found.[/yellow]")
        return

    # ── Step 2: Live signals + conviction scoring (parallel) ──────────

    console.print(f"Scanning signals + conviction for {len(symbols)} stocks...")

    skip_layers: set[str] = set()
    if skip_ml:
        skip_layers.add("ml_signal")
    if skip_sentiment:
        skip_layers.add("confluence")

    sig_scanner = SignalScanner()
    candidates: list[dict] = []

    def _analyze_symbol(sym: str) -> dict | None:
        """Run signal scan + dip analysis + ATR sizing for one symbol."""
        try:
            # Signal scan
            scan = sig_scanner.scan(sym, strategy_names=[strategy])
            sig = scan.signals[0] if scan.signals else None
            if sig is None:
                return None

            # Conviction score
            dip_result = None
            try:
                dip_result = analyze_dip(sym, skip_layers=skip_layers)
                conviction = dip_result.dip_score
                regime = dip_result.regime
                verdict = dip_result.verdict.value
            except Exception:
                conviction = 50.0
                regime = "normal"
                verdict = "N/A"

            # ATR-based sizing from live data
            price = sig.price
            atr = _compute_live_atr(sym)
            if atr > 0 and price > 0:
                risk_amount = cash * risk_pct
                shares = int(risk_amount / (atr * 2.0))
                shares = min(shares, int(cash * 0.25 / price))  # max 25% in one name
                position_value = shares * price
            else:
                shares = 0
                position_value = 0.0
                atr = 0.0

            # Stop levels
            stop_loss = price - (atr * 2.0) if atr > 0 else 0.0
            trailing_stop_pct = 0.08  # default 8% trail

            # Layer breakdown
            layers = {}
            regime_adj = 0.0
            reasoning = ""
            if dip_result is not None:
                for ls in dip_result.layers:
                    layers[ls.name] = {
                        "normalized": ls.normalized,
                        "weight": ls.weight,
                        "contribution": ls.weighted_contribution,
                        "available": ls.available,
                        "error": ls.error,
                    }
                regime_adj = dip_result.regime_adjustment
                reasoning = dip_result.reasoning

            return {
                "symbol": sym,
                "price": price,
                "signal": sig.action.value,
                "reason": sig.reason,
                "conviction": conviction,
                "regime": regime,
                "regime_adj": regime_adj,
                "verdict": verdict,
                "reasoning": reasoning,
                "layers": layers,
                "atr": atr,
                "shares": shares,
                "position_value": position_value,
                "pct_of_portfolio": position_value / cash * 100 if cash > 0 else 0,
                "stop_loss": stop_loss,
                "trailing_stop_pct": trailing_stop_pct,
                "risk_per_share": atr * 2.0,
                "risk_total": shares * atr * 2.0,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_symbol, sym): sym for sym in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                candidates.append(result)

    if not candidates:
        console.print("[yellow]No signals generated.[/yellow]")
        return

    # Sort: BUY/HOLD signals first, then by conviction descending
    signal_priority = {"BUY": 0, "HOLD": 1, "SELL": 2, "NEUTRAL": 3}
    candidates.sort(
        key=lambda c: (signal_priority.get(c["signal"], 9), -c["conviction"]),
    )
    candidates = candidates[:top_n]

    # ── Step 3: Output ────────────────────────────────────────────────

    if output == "json":
        output_json(candidates)
        return

    # Signal colors
    sig_colors = {"BUY": "bold green", "HOLD": "cyan", "SELL": "red", "NEUTRAL": "dim"}
    verdict_colors = {
        "STRONG_BUY": "bold green",
        "BUY": "green",
        "LEAN_BUY": "cyan",
        "WATCH": "yellow",
        "PASS": "red",
    }

    table = Table(title=f"Pipeline — {strategy} Signals + Sizing")
    table.add_column("Sym", style="cyan")
    table.add_column("Price", justify="right")
    table.add_column("Signal")
    table.add_column("Conv", justify="right")
    table.add_column("Verdict")
    table.add_column("ATR", justify="right")
    table.add_column("Shares", justify="right")
    table.add_column("Size", justify="right")
    table.add_column("Port%", justify="right")
    table.add_column("Stop", justify="right")
    table.add_column("Risk$", justify="right")

    for c in candidates:
        sc = sig_colors.get(c["signal"], "white")
        vc = verdict_colors.get(c["verdict"], "white")

        # Conviction color
        if c["conviction"] >= 60:
            conv_str = f"[green]{c['conviction']:.0f}[/green]"
        elif c["conviction"] >= 40:
            conv_str = f"[yellow]{c['conviction']:.0f}[/yellow]"
        else:
            conv_str = f"[dim]{c['conviction']:.0f}[/dim]"

        table.add_row(
            c["symbol"],
            f"${c['price']:.2f}",
            f"[{sc}]{c['signal']}[/{sc}]",
            conv_str,
            f"[{vc}]{c['verdict']}[/{vc}]",
            f"${c['atr']:.2f}",
            str(c["shares"]),
            f"${c['position_value']:,.0f}",
            f"{c['pct_of_portfolio']:.1f}%",
            f"${c['stop_loss']:.2f}",
            f"${c['risk_total']:.0f}",
        )

    console.print(table)

    # ── Per-symbol detail ─────────────────────────────────────────────

    regime_labels = {"low_vol": "Calm", "normal": "Normal", "high_vol": "Stressed"}

    for c in candidates:
        sc = sig_colors.get(c["signal"], "white")
        vc = verdict_colors.get(c["verdict"], "white")
        regime_label = regime_labels.get(c["regime"], c["regime"])

        console.print(
            f"\n[bold cyan]{'─' * 60}[/bold cyan]"
            f"\n[bold cyan]{c['symbol']}[/bold cyan]"
            f"  ${c['price']:.2f}"
        )

        # Signal
        console.print(f"\n  [bold]Signal:[/bold]  [{sc}]{c['signal']}[/{sc}]" f"  — {c['reason']}")

        # Conviction layers
        console.print(
            f"\n  [bold]Conviction:[/bold]  [{vc}]{c['verdict']}[/{vc}]"
            f"  ({c['conviction']:.1f}/100)"
            f"  |  Regime: {regime_label}"
            f" ({c['regime_adj']:+.0f} pts)"
        )

        if c["layers"]:
            layer_table = Table(show_header=True, header_style="dim", box=None, padding=(0, 2))
            layer_table.add_column("Layer")
            layer_table.add_column("Score", justify="right")
            layer_table.add_column("Weight", justify="right")
            layer_table.add_column("Contrib", justify="right")
            layer_table.add_column("Status")

            for name, info in c["layers"].items():
                if info["available"]:
                    score_val = info["normalized"]
                    if score_val >= 60:
                        score_str = f"[green]{score_val:.0f}[/green]"
                    elif score_val >= 40:
                        score_str = f"[yellow]{score_val:.0f}[/yellow]"
                    else:
                        score_str = f"[dim]{score_val:.0f}[/dim]"
                    layer_table.add_row(
                        name,
                        score_str,
                        f"{info['weight']:.0%}",
                        f"{info['contribution']:.1f}",
                        "[green]OK[/green]",
                    )
                else:
                    layer_table.add_row(
                        name,
                        "-",
                        "-",
                        "-",
                        f"[dim]{info['error'] or 'skipped'}[/dim]",
                    )
            console.print(layer_table)

        if c["reasoning"]:
            console.print(f"\n  [dim]{c['reasoning']}[/dim]")

        # Sizing
        risk_budget = cash * risk_pct
        console.print("\n  [bold]Position Sizing:[/bold]")
        console.print(
            f"    ATR(14):        ${c['atr']:.2f}/day\n"
            f"    Risk budget:    ${risk_budget:,.0f}"
            f"  ({risk_pct:.0%} of ${cash:,.0f})\n"
            f"    Risk/share:     ${c['risk_per_share']:.2f}"
            f"  (2 x ATR)\n"
            f"    Shares:         {c['shares']}"
            f"  = ${risk_budget:,.0f} / ${c['risk_per_share']:.2f}\n"
            f"    Position value: ${c['position_value']:,.0f}"
            f"  ({c['pct_of_portfolio']:.1f}% of portfolio)"
        )

        # Risk management
        console.print("\n  [bold]Risk Management:[/bold]")
        console.print(
            f"    Stop loss:      ${c['stop_loss']:.2f}"
            f"  (entry - 2×ATR)\n"
            f"    Trailing stop:  {c['trailing_stop_pct']:.0%} from peak\n"
            f"    Max loss:       ${c['risk_total']:.0f}"
            f"  ({c['risk_total'] / cash * 100:.2f}% of portfolio)"
        )

    # ── Action plan ───────────────────────────────────────────────────

    buys = [c for c in candidates if c["signal"] == "BUY"]
    if buys:
        total_capital = sum(c["position_value"] for c in buys)
        total_risk = sum(c["risk_total"] for c in buys)
        lines = [
            f"  [bold]Active BUY signals: {len(buys)}[/bold]",
            f"  Total capital needed: ${total_capital:,.0f} "
            f"({total_capital / cash * 100:.1f}% of ${cash:,.0f})",
            f"  Total risk: ${total_risk:,.0f} ({total_risk / cash * 100:.2f}% of portfolio)",
            "",
        ]
        for c in buys:
            lines.append(
                f"  [cyan]{c['symbol']}[/cyan]: "
                f"buy {c['shares']} @ ${c['price']:.2f} = ${c['position_value']:,.0f}, "
                f"stop ${c['stop_loss']:.2f} (risk ${c['risk_total']:.0f})"
            )
        console.print(Panel("\n".join(lines), title="Action Plan", border_style="green"))

    console.print()


def _compute_live_atr(symbol: str, period: int = 14) -> float:
    """Compute current ATR from live price data."""
    try:
        import yfinance as yf

        df = yf.download(symbol, period="3mo", interval="1d", progress=False)
        if df.empty or len(df) < period + 1:
            return 0.0

        # Handle multi-level columns from yfinance
        if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
            df.columns = df.columns.droplevel(1)

        highs = df["High"].values
        lows = df["Low"].values
        closes = df["Close"].values

        trs = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            return 0.0

        return float(sum(trs[-period:]) / period)
    except Exception:
        return 0.0
