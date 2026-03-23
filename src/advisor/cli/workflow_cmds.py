"""CLI commands for morning and afternoon trading workflows."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

morning_app = typer.Typer(name="morning", help="Pre-market morning analysis workflow")
afternoon_app = typer.Typer(name="afternoon", help="End-of-day afternoon review workflow")


# ── Morning ──────────────────────────────────────────────────────────────────


@morning_app.command("run")
def morning_run(
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Analysis mode: stocks or options"),
    ] = "options",
    account_size: Annotated[
        float,
        typer.Option("--account-size", "-a", help="Account size in dollars"),
    ] = 5_000,
    universe: Annotated[
        Optional[str],
        typer.Option("--universe", "-u", help="Comma-separated universes"),
    ] = None,
    watchlist: Annotated[
        Optional[str],
        typer.Option("--watchlist", "-w", help="Extra tickers (comma-separated)"),
    ] = None,
    quick: Annotated[
        bool,
        typer.Option("--quick", "-q", help="Skip heavy phases (research, MC sim)"),
    ] = False,
    lean: Annotated[
        bool,
        typer.Option(
            "--lean", "-l", help="Lean discovery: free yfinance pre-screen instead of API scanners"
        ),
    ] = False,
    skip_account: Annotated[
        bool,
        typer.Option("--skip-account", help="Skip TastyTrade fetch (offline mode)"),
    ] = False,
    account: Annotated[
        Optional[str],
        typer.Option("--account", help="TastyTrade account number (or suffix)"),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option("--output", help="Output format (json)"),
    ] = None,
) -> None:
    """Run the morning pre-market analysis workflow."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from advisor.workflow import rendering as r

    mode = mode.lower()
    if mode not in ("stocks", "options"):
        output_error("--mode must be 'stocks' or 'options'")

    universes = _parse_list(universe) if universe else None
    extra_tickers = _parse_list(watchlist) if watchlist else []

    if lean and mode == "options":
        console.print(
            "[yellow]Warning: --lean is designed for stocks mode;"
            " options universe is already small[/yellow]"
        )

    if mode == "stocks":
        from advisor.workflow.morning_stocks import StocksMorningWorkflow

        workflow = StocksMorningWorkflow(
            account_size=account_size,
            universes=universes,
            watchlist=extra_tickers,
            quick=quick,
            skip_account=skip_account,
            account_number=account,
            lean=lean,
        )
    else:
        from advisor.workflow.morning_options import OptionsMorningWorkflow

        workflow = OptionsMorningWorkflow(
            account_size=account_size,
            universes=universes,
            watchlist=extra_tickers,
            quick=quick,
            skip_account=skip_account,
            account_number=account,
        )

    is_json = output == "json"

    if is_json:
        result = workflow.run()
        output_json(result)
        return

    # Rich progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Running {mode} morning workflow...", total=None)
        result = workflow.run()

    # Render results
    console.print()
    console.print(
        f"[bold]Morning Analysis ({mode.upper()})[/bold]  "
        f"completed in {result.elapsed_seconds:.1f}s"
    )

    r.render_regime(result.regime)
    if not skip_account:
        r.render_account(result.account)
    r.render_position_health(result.position_health)
    r.render_scanner_hits(result.scanner_hits, lean=lean)
    r.render_validated(result.validated)
    r.render_deep_dives(result.deep_dives, mode)
    r.render_morning_actions(result)

    if result.errors:
        console.print()
        console.print(f"[dim]Warnings: {len(result.errors)}[/dim]")
        for err in result.errors:
            console.print(f"  [dim]{err}[/dim]")


# ── Afternoon ────────────────────────────────────────────────────────────────


@afternoon_app.command("run")
def afternoon_run(
    mode: Annotated[
        str,
        typer.Option("--mode", "-m", help="Analysis mode: stocks or options"),
    ] = "options",
    skip_account: Annotated[
        bool,
        typer.Option("--skip-account", help="Skip TastyTrade fetch"),
    ] = False,
    skip_validation: Annotated[
        bool,
        typer.Option("--skip-validation", help="Skip Brier score check (options only)"),
    ] = False,
    account: Annotated[
        Optional[str],
        typer.Option("--account", help="TastyTrade account number (or suffix)"),
    ] = None,
    output: Annotated[
        Optional[str],
        typer.Option("--output", help="Output format (json)"),
    ] = None,
) -> None:
    """Run the afternoon end-of-day review workflow."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from advisor.workflow import rendering as r
    from advisor.workflow.models import WorkflowState

    mode = mode.lower()
    if mode not in ("stocks", "options"):
        output_error("--mode must be 'stocks' or 'options'")

    state = WorkflowState.load()

    if mode == "stocks":
        from advisor.workflow.afternoon_stocks import StocksAfternoonWorkflow

        workflow = StocksAfternoonWorkflow(skip_account=skip_account, account_number=account)
    else:
        from advisor.workflow.afternoon_options import OptionsAfternoonWorkflow

        workflow = OptionsAfternoonWorkflow(
            skip_account=skip_account,
            skip_validation=skip_validation,
            account_number=account,
        )

    is_json = output == "json"

    if is_json:
        result = workflow.run()
        output_json(result)
        return

    # Rich progress display
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Running {mode} afternoon workflow...", total=None)
        result = workflow.run()

    # Render results
    console.print()
    console.print(
        f"[bold]Afternoon Review ({mode.upper()})[/bold]  "
        f"completed in {result.elapsed_seconds:.1f}s"
    )

    r.render_account_comparison(
        result.account,
        state.morning_account,
    )

    if mode == "stocks":
        r.render_afternoon_positions(result.position_statuses)
        r.render_watch_updates(result.watch_updates)
    else:
        r.render_afternoon_positions(result.position_statuses)
        r.render_brier_summary(result.brier_summary)

    r.render_next_day_prep(
        result.earnings_alerts,
        result.tomorrow_watchlist,
    )
    r.render_afternoon_actions(result, mode)

    if result.errors:
        console.print()
        console.print(f"[dim]Warnings: {len(result.errors)}[/dim]")
        for err in result.errors:
            console.print(f"  [dim]{err}[/dim]")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]
