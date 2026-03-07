"""CLI commands for the integrated pipeline orchestrator."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="pipeline", help="Integrated daily trading pipeline")


@app.command("run")
def pipeline_run(
    tickers: Annotated[
        Optional[str],
        typer.Option("--tickers", "-t", help="Comma-separated ticker symbols"),
    ] = None,
    universe: Annotated[
        Optional[str],
        typer.Option(
            "--universe",
            "-u",
            help="Universe name: wheel, leveraged, blue_chip, sp500, semiconductors",
        ),
    ] = None,
    account_size: Annotated[
        float, typer.Option("--account-size", help="Account size in dollars")
    ] = 5_000.0,
    max_bp_pct: Annotated[
        float, typer.Option("--max-bp-pct", help="Max buying power usage %")
    ] = 50.0,
    max_risk_pct: Annotated[
        float, typer.Option("--max-risk-pct", help="Max loss per trade as % of account")
    ] = 5.0,
    min_conviction: Annotated[
        float, typer.Option("--min-conviction", help="Minimum conviction score (0-100)")
    ] = 50.0,
    top_n: Annotated[
        int, typer.Option("--top", help="Number of top recommendations to return")
    ] = 5,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show per-layer breakdown")
    ] = False,
) -> None:
    """Run the full pipeline: discover, validate, time IV, simulate, score."""
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    from advisor.pipeline.models import PipelineConfig
    from advisor.pipeline.orchestrator import PipelineOrchestrator

    # Resolve symbols
    if tickers:
        symbols = [t.strip().upper() for t in tickers.split(",")]
    elif universe:
        symbols = _resolve_universe(universe)
    else:
        output_error("Provide --tickers or --universe")
        return

    if not symbols:
        output_error(f"No symbols found for universe '{universe}'")
        return

    config = PipelineConfig(
        account_size=account_size,
        max_bp_pct=max_bp_pct,
        max_risk_pct=max_risk_pct,
        min_conviction=min_conviction,
    )

    is_json = output == "json"

    if is_json:
        # Quiet mode for JSON output
        orchestrator = PipelineOrchestrator(config=config)
        result = orchestrator.run(symbols, top_n=top_n)
        output_json(result)
        return

    # Rich progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Pipeline", total=5)

        def _progress_cb(msg: str) -> None:
            progress.update(task, description=msg)
            # Advance on layer transitions
            if msg.startswith("Layer "):
                progress.advance(task)

        orchestrator = PipelineOrchestrator(config=config, progress_callback=_progress_cb)
        result = orchestrator.run(symbols, top_n=top_n)
        progress.update(task, completed=5, description="Done")

    # Summary
    console.print()
    console.print(
        f"[bold]Pipeline complete[/bold] in {result.elapsed_seconds:.1f}s — "
        f"scanned {result.symbols_scanned}, discovered {result.symbols_discovered}, "
        f"validated {result.symbols_validated}, simulated {result.symbols_simulated}"
    )

    if result.errors:
        console.print(f"[yellow]Warnings: {len(result.errors)}[/yellow]")
        if verbose:
            for err in result.errors:
                console.print(f"  [dim]{err}[/dim]")

    if not result.recommendations:
        console.print("[yellow]No recommendations above minimum conviction.[/yellow]")
        return

    # Main results table
    table = Table(title=f"Top {len(result.recommendations)} Recommendations")
    table.add_column("#", style="dim", width=3)
    table.add_column("Symbol", style="cyan bold")
    table.add_column("Strikes", style="white")
    table.add_column("DTE", justify="right")
    table.add_column("Credit", justify="right", style="green")
    table.add_column("BP", justify="right")
    table.add_column("Conv", justify="right", style="bold")
    table.add_column("Tier", style="bold")
    table.add_column("POP", justify="right")
    table.add_column("Edge", justify="right")
    table.add_column("EV", justify="right")
    table.add_column("Qty", justify="right")
    table.add_column("Reasoning")

    tier_colors = {"AUTO_ALERT": "green", "WATCH": "yellow", "SKIP": "red"}

    for i, rec in enumerate(result.recommendations, 1):
        tier_color = tier_colors.get(rec.conviction_tier.value, "white")
        table.add_row(
            str(i),
            rec.symbol,
            f"${rec.short_strike}/{rec.long_strike}",
            str(rec.dte),
            f"${rec.credit:.2f}",
            f"${rec.bp:.0f}",
            f"{rec.conviction_score:.0f}",
            f"[{tier_color}]{rec.conviction_tier.value}[/{tier_color}]",
            f"{rec.mc_pop:.1%}",
            f"{rec.pop_edge:+.1%}",
            f"${rec.ev:.2f}",
            str(rec.suggested_contracts),
            rec.reasoning[:60],
        )

    console.print(table)

    # Verbose: per-recommendation breakdown
    if verbose:
        for rec in result.recommendations:
            console.print()
            bd = rec.signal_breakdown
            detail = Table(title=f"{rec.symbol} — Conviction Breakdown")
            detail.add_column("Layer", style="cyan")
            detail.add_column("Score", justify="right")
            detail.add_column("Max", justify="right", style="dim")
            detail.add_row("Signal Strength", f"{bd.signal_strength:.1f}", "20")
            detail.add_row("Fundamental Safety", f"{bd.fundamental_safety:.1f}", "20")
            detail.add_row("IV Environment", f"{bd.iv_environment:.1f}", "20")
            detail.add_row("MC Edge", f"{bd.mc_edge:.1f}", "25")
            detail.add_row("Sizing Feasibility", f"{bd.sizing_feasibility:.1f}", "15")
            detail.add_row(
                "[bold]Total[/bold]",
                f"[bold]{rec.conviction_score:.1f}[/bold]",
                "[bold]100[/bold]",
            )
            console.print(detail)

            # IV + sizing details
            iv_table = Table(title=f"{rec.symbol} — IV & Sizing Details")
            iv_table.add_column("Metric", style="cyan")
            iv_table.add_column("Value", justify="right")
            iv_table.add_row("IV Rank", f"{rec.iv_rank:.0f}" if rec.iv_rank is not None else "N/A")
            iv_table.add_row("IV Percentile", f"{rec.iv_percentile:.1f}")
            iv_table.add_row("Current IV", f"{rec.current_iv:.2%}")
            iv_table.add_row("IV Timing Score", f"{rec.iv_timing_score:.1f}")
            iv_table.add_row("Contracts", str(rec.suggested_contracts))
            iv_table.add_row("Risk %", f"{rec.risk_pct:.1f}%")
            iv_table.add_row("CVaR 95", f"${rec.cvar_95:.2f}")
            iv_table.add_row("Stop Prob", f"{rec.stop_prob:.1%}")
            console.print(iv_table)


def _resolve_universe(name: str) -> list[str]:
    """Resolve universe name to list of symbols."""
    from advisor.market.options_scanner import UNIVERSES

    name = name.lower()

    # Built-in scanner universes
    if name in UNIVERSES:
        return UNIVERSES[name]

    # Special universes from data module
    if name == "sp500":
        from advisor.data.universe import fetch_universe

        stocks = fetch_universe()
        return [s.symbol for s in stocks]

    if name == "semiconductors":
        from advisor.data.universe import SEMICONDUCTORS

        return [sym for sym, _name in SEMICONDUCTORS]

    return []
