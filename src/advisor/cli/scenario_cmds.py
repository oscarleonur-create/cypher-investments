"""CLI commands for scenario simulation."""

from __future__ import annotations

from typing import Annotated, Optional

import typer

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="scenario", help="Forward scenario simulation + strategy evaluation")


@app.command("run")
def scenario_run(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol (e.g. AAPL)")],
    strategies: Annotated[
        Optional[str],
        typer.Option(
            "--strategies", "-s", help="Comma-separated strategy names (default: all equity)"
        ),
    ] = None,
    scenarios: Annotated[
        Optional[str],
        typer.Option("--scenarios", help="Comma-separated scenarios: bull,sideways,bear,crash"),
    ] = None,
    dte: Annotated[int, typer.Option("--dte", help="Trading days to simulate")] = 30,
    paths: Annotated[int, typer.Option("--paths", help="MC paths per scenario")] = 500,
    include_signals: Annotated[
        bool,
        typer.Option("--include-signals", help="Fetch alpha/confluence to weight scenarios"),
    ] = False,
    seed: Annotated[
        Optional[int], typer.Option("--seed", help="RNG seed for reproducibility")
    ] = None,
    output: Annotated[Optional[str], typer.Option("--output", help="Output format (json)")] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Show per-scenario breakdowns")
    ] = False,
    max_workers: Annotated[
        Optional[int],
        typer.Option("--workers", help="Max parallel workers (default: 4)"),
    ] = None,
) -> None:
    """Simulate forward price scenarios and evaluate equity strategies."""
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from advisor.scenario.models import ScenarioConfig
    from advisor.scenario.pipeline import run_scenario_simulation

    symbol = symbol.upper()
    config = ScenarioConfig(
        dte=dte,
        n_paths=paths,
        seed=seed,
    )

    strat_list = [s.strip() for s in strategies.split(",")] if strategies else None
    scenario_list = [s.strip() for s in scenarios.split(",")] if scenarios else None

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Running scenario simulation for {symbol}...", total=None)
            result = run_scenario_simulation(
                symbol=symbol,
                config=config,
                strategy_names=strat_list,
                scenario_names=scenario_list,
                include_signals=include_signals,
                max_workers=max_workers,
            )
    except ValueError as e:
        output_error(str(e))
        return
    except Exception as e:
        output_error(f"Simulation failed: {e}")
        return

    if output == "json":
        output_json(result)
        return

    # Rich table output
    _render_result(result, verbose=verbose)


def _render_result(result, verbose: bool = False) -> None:
    """Render scenario simulation results as rich tables."""
    from rich.table import Table

    # Header
    console.print()
    console.print(
        f"[bold]Scenario Simulation: {result.symbol}[/bold]  "
        f"({result.config.dte}d horizon, {result.config.n_paths} paths/scenario)"
    )
    console.print()

    # Signal context if available
    if result.signal_context:
        ctx = result.signal_context
        parts = []
        if ctx.alpha_score is not None:
            parts.append(f"Alpha: {ctx.alpha_score:.0f}/100 ({ctx.alpha_signal})")
        if ctx.confluence_verdict:
            parts.append(f"Confluence: {ctx.confluence_verdict}")
        if parts:
            console.print(f"  Signals: {' | '.join(parts)}")

        if ctx.adjusted_weights:
            weights_str = "  Adjusted weights: " + " | ".join(
                f"{k}: {v:.0%}" for k, v in ctx.adjusted_weights.items()
            )
            console.print(weights_str)
        console.print()

    # Strategy ranking table
    table = Table(title="Strategy Ranking (by risk-adjusted score)")
    table.add_column("Rank", justify="right", style="dim", width=4)
    table.add_column("Strategy", style="bold")
    table.add_column("E[Return]", justify="right")
    table.add_column("E[MaxDD]", justify="right")
    table.add_column("Prob(+)", justify="right")
    table.add_column("Worst p5", justify="right")
    table.add_column("Score", justify="right", style="bold")

    for i, comp in enumerate(result.composites, 1):
        ret_color = "green" if comp.expected_return > 0 else "red"
        score_color = "green" if comp.risk_adjusted_score > 0 else "red"

        table.add_row(
            str(i),
            comp.strategy_name,
            f"[{ret_color}]{comp.expected_return:+.2f}%[/{ret_color}]",
            f"{comp.expected_max_dd:.2f}%",
            f"{comp.prob_positive:.0%}",
            f"{comp.worst_case_return_p5:+.2f}%",
            f"[{score_color}]{comp.risk_adjusted_score:.1f}[/{score_color}]",
        )

    console.print(table)

    if result.best_strategy:
        console.print(
            f"\n  [bold green]Best strategy:[/bold green] {result.best_strategy} "
            f"(score: {result.best_score:.1f})"
        )

    # Verbose: per-scenario breakdown
    if verbose:
        for comp in result.composites:
            console.print()
            console.print(f"[bold]{comp.strategy_name}[/bold] — per-scenario breakdown:")
            sc_table = Table()
            sc_table.add_column("Scenario")
            sc_table.add_column("Mean Ret", justify="right")
            sc_table.add_column("Median Ret", justify="right")
            sc_table.add_column("p5", justify="right")
            sc_table.add_column("p95", justify="right")
            sc_table.add_column("Prob(+)", justify="right")
            sc_table.add_column("Avg DD", justify="right")
            sc_table.add_column("Avg Trades", justify="right")

            for sr in comp.scenario_results:
                sc_table.add_row(
                    sr.scenario_name,
                    f"{sr.mean_return_pct:+.2f}%",
                    f"{sr.median_return_pct:+.2f}%",
                    f"{sr.p5_return_pct:+.2f}%",
                    f"{sr.p95_return_pct:+.2f}%",
                    f"{sr.prob_positive:.0%}",
                    f"{sr.mean_max_dd_pct:.2f}%",
                    f"{sr.avg_trades:.1f}",
                )
            console.print(sc_table)

    console.print()
