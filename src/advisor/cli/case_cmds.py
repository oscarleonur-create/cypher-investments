"""CLI commands for the strategy case pipeline."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="case", help="Strategy case builder — deep single-stock analysis")


@app.command("build")
def case_build(
    ticker: Annotated[
        str,
        typer.Option("--ticker", "-t", help="Ticker symbol to analyze"),
    ],
    strategy: Annotated[
        Optional[str],
        typer.Option("--strategy", "-s", help="Override auto-detected strategy"),
    ] = None,
    account_size: Annotated[
        float, typer.Option("--account-size", help="Account size in dollars")
    ] = 5_000.0,
    research: Annotated[
        bool, typer.Option("--research", help="Enable deep fundamental research (~$0.15-0.50)")
    ] = False,
    mc: Annotated[bool, typer.Option("--mc", help="Enable Monte Carlo simulation (~10s)")] = False,
    min_dte: Annotated[int, typer.Option("--min-dte", help="Minimum DTE for options scan")] = 25,
    max_dte: Annotated[int, typer.Option("--max-dte", help="Maximum DTE for options scan")] = 45,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
) -> None:
    """Build a strategy case for a single ticker."""
    from advisor.strategy_case.models import OptionsStrategyType, StrategyCaseConfig

    # Parse strategy override
    strategy_override = None
    if strategy:
        try:
            strategy_override = OptionsStrategyType(strategy.lower())
        except ValueError:
            valid = ", ".join(s.value for s in OptionsStrategyType)
            output_error(f"Invalid strategy '{strategy}'. Valid: {valid}")
            return

    config = StrategyCaseConfig(
        account_size=account_size,
        enable_research=research,
        enable_mc=mc,
        strategy_override=strategy_override,
        min_dte=min_dte,
        max_dte=max_dte,
    )

    is_json = output == "json"

    if is_json:
        from advisor.strategy_case.builder import StrategyCaseBuilder

        builder = StrategyCaseBuilder(config=config)
        case = builder.build(ticker)
        output_json(case)
        return

    # Rich TUI output
    from rich.panel import Panel
    from rich.table import Table

    from advisor.strategy_case.builder import StrategyCaseBuilder

    status_lines: list[str] = []

    def _progress(msg: str) -> None:
        status_lines.append(msg)
        console.print(f"  [dim]{msg}[/dim]")

    console.print(f"\n[bold cyan]Building strategy case for {ticker.upper()}[/bold cyan]")
    console.print()

    builder = StrategyCaseBuilder(config=config, progress_callback=_progress)
    case = builder.build(ticker)

    console.print()

    if not case.synthesis:
        console.print("[red]Case build failed — no synthesis generated.[/red]")
        if case.errors:
            for err in case.errors:
                console.print(f"  [red]{err}[/red]")
        return

    # ── Verdict Panel ────────────────────────────────────────────────────
    verdict = case.synthesis.verdict.value
    conviction = case.synthesis.conviction_score
    verdict_colors = {"STRONG": "green", "MODERATE": "yellow", "WEAK": "red", "REJECT": "red bold"}
    v_color = verdict_colors.get(verdict, "white")

    console.print(
        Panel(
            f"[{v_color}]{verdict}[/{v_color}] — Conviction: [bold]{conviction:.0f}/100[/bold]\n\n"
            f"{case.synthesis.thesis_summary}",
            title=f"[bold]{case.symbol} Strategy Case[/bold]",
            border_style=v_color.split()[0],
        )
    )

    # ── Scenario ─────────────────────────────────────────────────────────
    if case.scenario:
        console.print(
            f"\n[bold]Scenario:[/bold] {case.scenario.scenario_type.value} "
            f"({case.scenario.confidence:.0%} confidence)"
        )
        console.print(f"  {case.scenario.summary}")

    # ── Strategy Ranking ─────────────────────────────────────────────────
    if case.ranking and case.ranking.matches:
        rank_table = Table(title="Strategy Ranking")
        rank_table.add_column("#", style="dim", width=3)
        rank_table.add_column("Strategy", style="cyan")
        rank_table.add_column("Fit", justify="right")
        rank_table.add_column("Reasoning")

        for i, m in enumerate(case.ranking.matches[:5], 1):
            selected_mark = (
                " *"
                if case.ranking.selected and m.strategy == case.ranking.selected.strategy
                else ""
            )
            rank_table.add_row(
                str(i),
                f"{m.strategy.value}{selected_mark}",
                f"{m.fit_score:.0f}",
                m.reasoning[:70],
            )
        console.print(rank_table)

    # ── Strike Recommendations ───────────────────────────────────────────
    if case.options and case.options.recommendations:
        strike_table = Table(title="Strike Recommendations")
        strike_table.add_column("#", style="dim", width=3)
        strike_table.add_column("Strategy", style="cyan")
        strike_table.add_column("Strikes", style="white")
        strike_table.add_column("Exp", style="dim")
        strike_table.add_column("DTE", justify="right")
        strike_table.add_column("Credit", justify="right", style="green")
        strike_table.add_column("Delta", justify="right")
        strike_table.add_column("POP", justify="right")
        strike_table.add_column("Score", justify="right", style="bold")
        strike_table.add_column("Yield", justify="right")
        strike_table.add_column("Flags")

        for i, rec in enumerate(case.options.recommendations[:5], 1):
            strikes = f"${rec.strike:.0f}"
            if rec.long_strike:
                strikes += f"/${rec.long_strike:.0f}"
            strike_table.add_row(
                str(i),
                rec.strategy,
                strikes,
                str(rec.expiry) if rec.expiry else "",
                str(rec.dte),
                f"${rec.credit:.2f}",
                f"{rec.delta:.2f}",
                f"{rec.pop:.0%}",
                f"{rec.sell_score:.0f}",
                f"{rec.annualized_yield:.0%}",
                ", ".join(rec.flags) if rec.flags else "",
            )
        console.print(strike_table)

    # ── Risk Profile ─────────────────────────────────────────────────────
    if case.risk:
        risk_table = Table(title=f"Risk Profile ({case.risk.source.upper()})")
        risk_table.add_column("Metric", style="cyan")
        risk_table.add_column("Value", justify="right")

        risk_table.add_row("POP", f"{case.risk.pop:.1%}")
        risk_table.add_row("EV/contract", f"${case.risk.ev:.2f}")
        if case.risk.source == "mc":
            risk_table.add_row("CVaR 95", f"${case.risk.cvar_95:.2f}")
            risk_table.add_row("Stop Prob", f"{case.risk.stop_prob:.1%}")
        risk_table.add_row("Contracts", str(case.risk.suggested_contracts))
        risk_table.add_row("Max Loss", f"${case.risk.max_loss_total:.0f}")
        risk_table.add_row("Risk %", f"{case.risk.risk_pct:.1f}%")
        risk_table.add_row(
            "Feasible",
            "[green]Yes[/green]" if case.risk.sizing_feasible else "[red]No[/red]",
        )
        console.print(risk_table)

    # ── Research Summary ─────────────────────────────────────────────────
    if case.research:
        console.print(f"\n[bold]Research:[/bold] {case.research.verdict}")
        if case.research.bull_case:
            console.print(f"  Bull: {'; '.join(case.research.bull_case[:3])}")
        if case.research.bear_case:
            console.print(f"  Bear: {'; '.join(case.research.bear_case[:3])}")

    # ── Trade Plan ───────────────────────────────────────────────────────
    syn = case.synthesis
    if syn.entry_criteria:
        console.print(f"\n[bold]Entry:[/bold] {' | '.join(syn.entry_criteria)}")
    if syn.exit_plan:
        console.print(f"[bold]Exit:[/bold] {' | '.join(syn.exit_plan)}")
    if syn.invalidation:
        console.print(f"[bold]Invalidation:[/bold] {' | '.join(syn.invalidation)}")
    if syn.risks:
        console.print(f"[bold]Risks:[/bold] {' | '.join(syn.risks)}")
    if syn.management_plan:
        console.print(f"[bold]Management:[/bold] {' | '.join(syn.management_plan)}")

    # ── Footer ───────────────────────────────────────────────────────────
    if case.errors:
        console.print(f"\n[yellow]Warnings ({len(case.errors)}):[/yellow]")
        for err in case.errors:
            console.print(f"  [dim]{err}[/dim]")

    console.print(f"\n[dim]Completed in {case.elapsed_seconds:.1f}s[/dim]")
