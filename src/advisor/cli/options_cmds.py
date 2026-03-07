"""Options CLI commands — scan, track, analyze."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from advisor.market.options_scanner import UNIVERSES, scan_options
from advisor.market.trades import TradeRecord
from advisor.market.trades import load_trades as _load_trades
from advisor.market.trades import save_trades as _save_trades

logger = logging.getLogger(__name__)
console = Console()

app = typer.Typer(
    name="options", help="Options scanning, tracking, and analysis.", no_args_is_help=True
)
track_app = typer.Typer(name="track", help="Track options trades.", no_args_is_help=True)
app.add_typer(track_app, name="track")


# ── Account command ───────────────────────────────────────────────────────────


@app.command("account")
def account(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Show live TastyTrade account balances and open positions."""
    from advisor.market.tastytrade_client import get_balances, get_positions, get_session

    async def _fetch():
        session = await get_session()
        balances = await get_balances(session)
        positions = await get_positions(session)
        return balances, positions

    try:
        balances, positions = asyncio.run(_fetch())
    except Exception as e:
        console.print(f"[red]Failed to connect to TastyTrade: {e}[/red]")
        raise typer.Exit(1)

    if output == "json":
        from advisor.cli.formatters import output_json

        output_json({"balances": balances, "positions": positions})
        return

    # Balances table
    bal_table = Table(title="Account Balances")
    bal_table.add_column("Metric", style="bold")
    bal_table.add_column("Value", justify="right", style="green")
    bal_table.add_row("Account", balances["account"])
    bal_table.add_row("Net Liquidating Value", f"${balances['net_liq']:,.2f}")
    bal_table.add_row("Cash Balance", f"${balances['cash']:,.2f}")
    bal_table.add_row("Buying Power", f"${balances['buying_power']:,.2f}")
    console.print(bal_table)
    console.print()

    # Positions table
    if not positions:
        console.print("[yellow]No open positions.[/yellow]")
        return

    pos_table = Table(title=f"Open Positions ({len(positions)})")
    pos_table.add_column("Symbol", style="cyan")
    pos_table.add_column("Type")
    pos_table.add_column("Qty", justify="right")
    pos_table.add_column("Avg Open", justify="right")
    pos_table.add_column("Current", justify="right")
    pos_table.add_column("P&L", justify="right")

    for p in positions:
        avg = p["average_open_price"]
        cur = p["close_price"]
        pnl = (avg - cur) * 100 * int(p["quantity"])
        color = "green" if pnl >= 0 else "red"
        inst = str(p["instrument_type"]).replace("InstrumentType.", "")
        pos_table.add_row(
            p["symbol"],
            inst,
            str(p["quantity"]),
            f"${avg:.2f}",
            f"${cur:.2f}",
            f"[{color}]${pnl:+,.2f}[/{color}]",
        )

    console.print(pos_table)


# ── Max-Move command ──────────────────────────────────────────────────────────


@app.command("max-move")
def max_move(
    symbol: str = typer.Argument(..., help="Symbol to analyze"),
    dtes: str = typer.Option("21,30,45,60", "--dte", "-d", help="Comma-separated DTEs"),
    lookback: int = typer.Option(252, "--lookback", "-l", help="Trading days of history"),
    regimes: bool = typer.Option(
        True, "--regimes/--no-regimes", help="Include vol regime breakdown"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Historical intra-window max drawdown analysis for tail risk sizing."""
    from advisor.market.drawdown_analysis import analyze_max_move

    dte_list = [int(d.strip()) for d in dtes.split(",")]
    is_json = output == "json"

    if not is_json:
        console.print(f"\n[bold]Max-Move Analysis: {symbol.upper()}[/bold]")
        console.print(f"[dim]Lookback: {lookback} trading days, DTEs: {dte_list}[/dim]\n")

    try:
        result = analyze_max_move(
            symbol,
            dtes=dte_list,
            lookback=lookback,
            include_regimes=regimes,
        )
    except Exception as e:
        console.print(f"[red]Analysis failed: {e}[/red]")
        raise typer.Exit(1)

    if is_json:
        from advisor.cli.formatters import output_json

        output_json(result.model_dump())
        return

    # Header
    regime_color = {"low": "green", "mid": "yellow", "high": "red"}.get(
        result.current_regime, "dim"
    )
    console.print(
        f"  Price: ${result.current_price:,.2f}  |  "
        f"HV20: {result.hv20_current:.1%}  |  "
        f"Regime: [{regime_color}]{result.current_regime.upper()}[/{regime_color}]\n"
    )

    # Drawdown Quantiles table
    if result.quantiles:
        table = Table(title="Drawdown Quantiles (intra-window worst drop)")
        table.add_column("DTE", justify="right", style="cyan")
        table.add_column("TradDays", justify="right")
        table.add_column("p95", justify="right", style="yellow")
        table.add_column("p97.5", justify="right", style="yellow")
        table.add_column("p99", justify="right", style="red")
        table.add_column("Max", justify="right", style="bold red")
        table.add_column("Windows", justify="right", style="dim")

        for q in result.quantiles:
            table.add_row(
                str(q.dte),
                str(q.trading_days),
                f"{q.dd_p95:.1%}",
                f"{q.dd_p97_5:.1%}",
                f"{q.dd_p99:.1%}",
                f"{q.dd_max:.1%}",
                str(q.n_windows),
            )
        console.print(table)
        console.print()

    # Breach Speed table
    breach_rows = []
    for q in result.quantiles:
        for b in q.breach_speed:
            breach_rows.append((q.dte, b))

    if breach_rows:
        table = Table(title="Breach Speed (how fast drawdown thresholds are hit)")
        table.add_column("DTE", justify="right", style="cyan")
        table.add_column("Threshold", justify="right")
        table.add_column("Breach%", justify="right", style="yellow")
        table.add_column("Median Days", justify="right")
        table.add_column("p25", justify="right", style="dim")
        table.add_column("p75", justify="right", style="dim")

        for dte_val, b in breach_rows:
            table.add_row(
                str(dte_val),
                f"{b.threshold_pct:.0%}",
                f"{b.breach_probability:.1%}",
                f"{b.median_days:.0f}" if b.median_days is not None else "-",
                f"{b.p25_days:.0f}" if b.p25_days is not None else "-",
                f"{b.p75_days:.0f}" if b.p75_days is not None else "-",
            )
        console.print(table)
        console.print()

    # Vol Regime table
    if regimes and result.regime_drawdowns:
        table = Table(title="Vol Regime Drawdowns")
        table.add_column("Regime", style="bold")
        table.add_column("DTE", justify="right", style="cyan")
        table.add_column("p95", justify="right", style="yellow")
        table.add_column("p99", justify="right", style="red")
        table.add_column("Max", justify="right", style="bold red")
        table.add_column("HV20 Range", justify="right", style="dim")
        table.add_column("Windows", justify="right", style="dim")

        for rd in result.regime_drawdowns:
            is_current = rd.regime == result.current_regime
            marker = " *" if is_current else ""
            label = f"{rd.regime.upper()}{marker}"
            regime_cell = f"[bold]{label}[/bold]" if is_current else label
            table.add_row(
                regime_cell,
                str(rd.dte),
                f"{rd.dd_p95:.1%}",
                f"{rd.dd_p99:.1%}",
                f"{rd.dd_max:.1%}",
                f"{rd.hv20_low:.0%}-{rd.hv20_high:.0%}",
                str(rd.n_windows),
            )
        console.print(table)
        console.print("[dim]* = current regime[/dim]")

    console.print()


# ── Scan command ───────────────────────────────────────────────────────────────


@app.command("scan")
def scan(
    account_size: float = typer.Option(5000.0, "--account-size", "-a", help="Account size in USD"),
    universe: str = typer.Option(
        "wheel", "--universe", "-u", help="Ticker universe: leveraged, wheel, blue_chip"
    ),
    tickers: Optional[str] = typer.Option(
        None, "--tickers", "-t", help="Comma-separated tickers (overrides universe)"
    ),
    live_iv: bool = typer.Option(False, "--live-iv", help="Fetch live IV data from TastyTrade"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Scan for options opportunities (naked puts + credit spreads)."""
    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
        universe_name = "custom"
    else:
        ticker_list = UNIVERSES.get(universe, UNIVERSES["wheel"])
        universe_name = universe

    is_json = output == "json"

    if not is_json:
        console.print(
            f"\n[bold]Scanning {len(ticker_list)} tickers[/bold]"
            f" (universe={universe_name}, account=${account_size:,.0f})\n"
        )

    # Optionally fetch live IV from TastyTrade
    tt_iv_data: dict = {}
    if live_iv:
        try:
            import asyncio

            from advisor.market.tastytrade_client import get_market_metrics, get_session

            async def _fetch_iv():
                session = await get_session()
                return await get_market_metrics(session, ticker_list)

            tt_iv_data = asyncio.run(_fetch_iv())
            if not is_json:
                console.print(f"[green]TastyTrade IV loaded for {len(tt_iv_data)} symbols[/green]")
        except Exception as e:
            if not is_json:
                console.print(f"[yellow]TastyTrade IV unavailable: {e}[/yellow]")

    result = scan_options(
        ticker_list,
        account_size=account_size,
        universe_name=universe_name,
        iv_overrides=tt_iv_data or None,
    )

    if is_json:
        from advisor.cli.formatters import output_json

        output_json(result)
        return

    # Check if any result has IV rank data
    has_iv_rank = any(p.iv_rank is not None for p in result.naked_puts)

    # Naked puts table
    if result.naked_puts:
        table = Table(title=f"🔻 Naked Puts ({len(result.naked_puts)} found)", show_lines=False)
        table.add_column("Symbol", style="cyan")
        table.add_column("Strike", justify="right")
        table.add_column("Expiry")
        table.add_column("DTE", justify="right")
        table.add_column("Bid", justify="right", style="green")
        table.add_column("OTM%", justify="right")
        table.add_column("IV", justify="right")
        if has_iv_rank:
            table.add_column("IVR", justify="right", style="yellow")
        table.add_column("Margin", justify="right")
        table.add_column("Ann.Yield", justify="right", style="bold green")
        table.add_column("⚠️", justify="center")

        for p in result.naked_puts[:15]:
            flag = "🔴" if p.exceeds_account_limit else ""
            row = [
                p.symbol,
                f"${p.strike:.2f}",
                str(p.expiry),
                str(p.dte),
                f"${p.bid:.2f}",
                f"{p.otm_pct:.1%}",
                f"{p.iv:.0%}",
            ]
            if has_iv_rank:
                row.append(f"{p.iv_rank:.0%}" if p.iv_rank is not None else "-")
            row.extend(
                [
                    f"${p.margin_req:,.0f}",
                    f"{p.annualized_yield:.0%}",
                    flag,
                ]
            )
            table.add_row(*row)
        console.print(table)
    else:
        console.print("[yellow]No naked put candidates found.[/yellow]")

    # Check if any spread has IV rank data
    has_spread_ivr = any(s.iv_rank is not None for s in result.spreads)

    # Credit spreads table
    if result.spreads:
        table = Table(
            title=f"\n📊 Put Credit Spreads ({len(result.spreads)} found)", show_lines=False
        )
        table.add_column("Symbol", style="cyan")
        table.add_column("Short", justify="right")
        table.add_column("Long", justify="right")
        table.add_column("Expiry")
        table.add_column("DTE", justify="right")
        table.add_column("Credit", justify="right", style="green")
        table.add_column("Max Loss", justify="right", style="red")
        if has_spread_ivr:
            table.add_column("IVR", justify="right", style="yellow")
        table.add_column("RoR", justify="right", style="bold green")
        table.add_column("Ann.Ret", justify="right", style="bold green")
        table.add_column("⚠️", justify="center")

        for s in result.spreads[:15]:
            flag = "🔴" if s.exceeds_account_limit else ""
            row = [
                s.symbol,
                f"${s.short_strike:.2f}",
                f"${s.long_strike:.2f}",
                str(s.expiry),
                str(s.dte),
                f"${s.net_credit:.2f}",
                f"${s.max_loss:,.0f}",
            ]
            if has_spread_ivr:
                row.append(f"{s.iv_rank:.0%}" if s.iv_rank is not None else "-")
            row.extend(
                [
                    f"{s.return_on_risk:.0%}",
                    f"{s.annualized_return:.0%}",
                    flag,
                ]
            )
            table.add_row(*row)
        console.print(table)
    else:
        console.print("[yellow]No credit spread candidates found.[/yellow]")

    if result.errors:
        console.print(f"\n[dim]Errors: {', '.join(result.errors[:5])}[/dim]")

    console.print(
        f"\n[dim]Scanned {result.tickers_scanned} tickers at {result.scanned_at:%H:%M:%S}[/dim]\n"
    )


# ── Simulate command ──────────────────────────────────────────────────────


@app.command("simulate")
def simulate(
    account_size: float = typer.Option(5000.0, "--account-size", "-a", help="Account size in USD"),
    universe: str = typer.Option(
        "wheel", "--universe", "-u", help="Ticker universe: leveraged, wheel, blue_chip"
    ),
    tickers: Optional[str] = typer.Option(
        None, "--tickers", "-t", help="Comma-separated tickers (overrides universe)"
    ),
    paths: int = typer.Option(10_000, "--paths", "-p", help="MC paths for quick sim"),
    deep_paths: int = typer.Option(100_000, "--deep-paths", help="MC paths for deep sim"),
    top: int = typer.Option(5, "--top", help="Number of top results to show"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Monte Carlo PCS simulator — rank put credit spreads by risk-adjusted EV."""
    from advisor.market.options_scanner import UNIVERSES
    from advisor.simulator.db import SimulatorStore
    from advisor.simulator.models import SimConfig
    from advisor.simulator.pipeline import SimulatorPipeline

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        ticker_list = UNIVERSES.get(universe, UNIVERSES["wheel"])

    is_json = output == "json"

    config = SimConfig(
        n_paths=paths,
        max_buying_power=account_size,
    )

    store = SimulatorStore()

    def _progress(msg: str):
        if not is_json:
            console.print(f"[dim]{msg}[/dim]")

    try:
        pipeline = SimulatorPipeline(
            config=config,
            store=store,
            progress_callback=_progress,
        )
        result = pipeline.run(
            symbols=ticker_list,
            top_n=top,
            quick_paths=paths,
            deep_paths=deep_paths,
        )
    except Exception as e:
        console.print(f"[red]Simulation failed: {e}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()

    if is_json:
        from advisor.cli.formatters import output_json

        output_json(result.model_dump())
        return

    # Summary
    console.print(
        f"\n[bold]MC Simulation Results[/bold]"
        f" — {result.symbols_scanned} symbols, {result.candidates_generated} candidates,"
        f" {result.candidates_simulated} simulated\n"
    )

    if not result.top_results:
        console.print("[yellow]No simulation results.[/yellow]")
        return

    table = Table(title=f"Top {len(result.top_results)} Put Credit Spreads by EV/BP")
    table.add_column("Symbol", style="cyan")
    table.add_column("Short", justify="right")
    table.add_column("Long", justify="right")
    table.add_column("DTE", justify="right")
    table.add_column("Credit", justify="right", style="green")
    table.add_column("EV", justify="right", style="bold green")
    table.add_column("POP", justify="right")
    table.add_column("Touch%", justify="right")
    table.add_column("CVaR95", justify="right", style="red")
    table.add_column("Stop%", justify="right")
    table.add_column("Hold", justify="right")
    table.add_column("EV/BP", justify="right", style="bold")

    for r in result.top_results:
        ev_color = "green" if r.ev > 0 else "red"
        table.add_row(
            r.symbol,
            f"${r.short_strike:.2f}",
            f"${r.long_strike:.2f}",
            str(r.dte),
            f"${r.net_credit:.2f}",
            f"[{ev_color}]${r.ev:+.2f}[/{ev_color}]",
            f"{r.pop:.0%}",
            f"{r.touch_prob:.0%}",
            f"${r.cvar_95:.2f}",
            f"{r.stop_prob:.0%}",
            f"{r.avg_hold_days:.0f}d",
            f"{r.ev_per_bp:.4f}",
        )

    console.print(table)

    # Exit breakdown for top result
    if result.top_results:
        best = result.top_results[0]
        console.print(
            f"\n[dim]Best exit breakdown: "
            f"Profit={best.exit_profit_target:.0%}, "
            f"Stop={best.exit_stop_loss:.0%}, "
            f"DTE={best.exit_dte:.0%}, "
            f"Expiry={best.exit_expiration:.0%}[/dim]"
        )

    # Calibration info — cal_params is {symbol: {param: value, ...}}
    cal = result.calibration_params
    if cal:
        for sym, params in cal.items():
            if isinstance(params, dict):
                vol_source = params.get("vol_source", "historical")
                vol_label = "IV" if vol_source == "live_iv" else "HV"
                console.print(
                    f"[dim]Calibration ({sym}): "
                    f"t-df={params.get('student_t_df', 0):.1f}, "
                    f"vol={params.get('vol_mean_level', 0):.0%} ({vol_label}), "
                    f"kappa={params.get('vol_mean_revert_speed', 0):.2f}, "
                    f"leverage={params.get('leverage_effect', 0):.2f}[/dim]"
                )
        console.print()


# ── Validate command ──────────────────────────────────────────────────────


@app.command("validate")
def validate(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Resolve expired MC predictions against historical prices and compute Brier scores."""
    from advisor.simulator.db import SimulatorStore
    from advisor.simulator.validation import resolve_outcomes

    store = SimulatorStore()
    try:
        outcomes = resolve_outcomes(store, data_provider=None)
        brier = store.compute_brier_scores(lookback_days=90)
    finally:
        store.close()

    if output == "json":
        from advisor.cli.formatters import output_json

        output_json(
            {
                "outcomes": [
                    {
                        "candidate_id": o.candidate_id,
                        "symbol": o.symbol,
                        "actual_profit": o.actual_profit,
                        "actual_touch": o.actual_touch,
                        "actual_stop": o.actual_stop,
                        "actual_pnl": o.actual_pnl,
                        "exit_reason": o.exit_reason,
                        "exit_day": o.exit_day,
                    }
                    for o in outcomes
                ],
                "brier_scores": brier,
            }
        )
        return

    if not outcomes:
        console.print(
            "[yellow]No pending predictions to resolve (all resolved or none expired).[/yellow]"
        )
        # Still show Brier scores if available
        if brier["n_samples"] > 0:
            _print_brier_scores(brier)
        return

    # Results table
    table = Table(title=f"Resolved Outcomes ({len(outcomes)} predictions)")
    table.add_column("Symbol", style="cyan")
    table.add_column("Candidate", style="dim")
    table.add_column("P&L", justify="right")
    table.add_column("Profitable", justify="center")
    table.add_column("Touched", justify="center")
    table.add_column("Stopped", justify="center")
    table.add_column("Exit Reason")
    table.add_column("Hold Days", justify="right")

    for o in outcomes:
        pnl_color = "green" if o.actual_pnl > 0 else "red"
        table.add_row(
            o.symbol,
            o.candidate_id[:8],
            f"[{pnl_color}]${o.actual_pnl:+.2f}[/{pnl_color}]",
            "Yes" if o.actual_profit > 0 else "No",
            "Yes" if o.actual_touch > 0 else "No",
            "Yes" if o.actual_stop > 0 else "No",
            o.exit_reason,
            str(o.exit_day),
        )

    console.print(table)
    _print_brier_scores(brier)


@app.command("backtest-validate")
def backtest_validate_cmd(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Symbol to validate"),
    start: str = typer.Option("2025-06-01", "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2025-12-31", "--end", help="End date (YYYY-MM-DD)"),
    paths: int = typer.Option(10_000, "--paths", "-p", help="MC paths per simulation"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Historical replay — re-run MC on past chain snapshots and validate against actuals."""
    from advisor.simulator.db import SimulatorStore
    from advisor.simulator.models import SimConfig
    from advisor.simulator.validation import backtest_validate

    is_json = output == "json"

    def _progress(msg: str):
        if not is_json:
            console.print(f"[dim]{msg}[/dim]")

    config = SimConfig(n_paths=paths)
    store = SimulatorStore()

    try:
        result = backtest_validate(
            store=store,
            symbol=symbol.upper(),
            start=start,
            end=end,
            config=config,
            n_paths=paths,
            progress_callback=_progress,
        )
    finally:
        store.close()

    if is_json:
        from advisor.cli.formatters import output_json

        output_json(
            {
                "symbol": symbol.upper(),
                "n_predictions": result.n_predictions,
                "n_resolved": result.n_resolved,
                "pop_brier": result.pop_brier,
                "touch_brier": result.touch_brier,
                "stop_brier": result.stop_brier,
                "ev_mae": result.ev_mae,
                "ev_correlation": result.ev_correlation,
                "calibration_buckets": result.calibration_buckets,
                "per_trade": result.per_trade,
            }
        )
        return

    console.print(
        f"\n[bold]Backtest Validation: {symbol.upper()}[/bold]"
        f" ({start} to {end}, {paths:,} paths)\n"
    )

    if result.n_resolved == 0:
        console.print(
            "[yellow]No predictions could be resolved. Ensure chain snapshots exist "
            "for the date range and expirations have passed.[/yellow]"
        )
        return

    console.print(
        f"Snapshot dates: {result.n_predictions} | Predictions resolved: {result.n_resolved}\n"
    )

    # Brier scores table
    table = Table(title="Brier Scores")
    table.add_column("Metric", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Quality")

    for metric, score in [
        ("POP", result.pop_brier),
        ("Touch", result.touch_brier),
        ("Stop", result.stop_brier),
    ]:
        if score is not None:
            quality = _brier_quality_label(score)
            table.add_row(metric, f"{score:.4f}", quality)

    console.print(table)

    # EV accuracy
    if result.ev_mae is not None:
        console.print("\n[bold]EV Accuracy[/bold]")
        console.print(f"  MAE: ${result.ev_mae:.2f}")
        if result.ev_correlation is not None:
            console.print(f"  Correlation: {result.ev_correlation:.4f}")

    console.print()


def _brier_quality_label(score: float) -> str:
    """Return a colored quality label for a Brier score."""
    if score < 0.10:
        return "[bold green]Excellent[/bold green]"
    elif score < 0.20:
        return "[green]Good[/green]"
    elif score < 0.25:
        return "[yellow]Fair[/yellow]"
    else:
        return "[red]Poor[/red]"


def _print_brier_scores(brier: dict) -> None:
    """Print Brier score summary to console."""
    if brier["n_samples"] == 0:
        return

    console.print(f"\n[bold]Calibration Quality[/bold] ({brier['n_samples']} samples)")

    table = Table()
    table.add_column("Metric", style="bold")
    table.add_column("Brier Score", justify="right")
    table.add_column("Quality")

    for metric, key in [("POP", "pop_brier"), ("Touch", "touch_brier"), ("Stop", "stop_brier")]:
        score = brier[key]
        if score is not None:
            quality = _brier_quality_label(score)
            table.add_row(metric, f"{score:.4f}", quality)

    console.print(table)
    console.print()


# ── Track commands ─────────────────────────────────────────────────────────────


@track_app.command("open")
def track_open(
    trade_type: str = typer.Argument(
        ..., help="Trade type: naked_put, put_credit_spread, covered_call, wheel"
    ),
    symbol: str = typer.Argument(..., help="Underlying symbol"),
    strike: float = typer.Argument(..., help="Short strike price"),
    expiry: str = typer.Argument(..., help="Expiration date (YYYY-MM-DD)"),
    premium: float = typer.Argument(..., help="Premium collected per contract"),
    long_strike: Optional[float] = typer.Option(
        None, "--long-strike", "-l", help="Long strike (for spreads)"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Log a new options trade."""
    from pydantic import ValidationError

    trades = _load_trades()
    try:
        record = TradeRecord(
            trade_type=trade_type,
            symbol=symbol.upper(),
            strike=strike,
            long_strike=long_strike,
            expiry=expiry,
            premium=premium,
        )
    except ValidationError:
        msg = (
            f"Invalid trade_type '{trade_type}'."
            " Must be: naked_put, put_credit_spread, covered_call, wheel"
        )
        if output == "json":
            from advisor.cli.formatters import output_json

            output_json({"error": msg})
        else:
            console.print(f"[red]{msg}[/red]")
        raise typer.Exit(1)

    trades.append(record)
    _save_trades(trades)

    if output == "json":
        from advisor.cli.formatters import output_json

        output_json(record.model_dump())
        return

    console.print(
        f"[green]Opened trade {record.id}:[/green]"
        f" {trade_type} {symbol.upper()} ${strike}"
        f" exp {expiry} @ ${premium:.2f}"
    )


@track_app.command("close")
def track_close(
    trade_id: str = typer.Argument(..., help="Trade ID to close"),
    close_price: float = typer.Argument(..., help="Closing price per contract"),
    reason: str = typer.Option(
        "profit", "--reason", "-r", help="Close reason: expired, profit, stop, assigned"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Close an existing trade."""
    trades = _load_trades()
    closed_trade = None
    for t in trades:
        if t.id == trade_id and t.status == "open":
            t.close_price = close_price
            t.close_reason = reason
            t.closed_at = datetime.now().isoformat()
            t.status = "closed"
            closed_trade = t
            break

    if not closed_trade:
        if output == "json":
            from advisor.cli.formatters import output_json

            output_json({"error": f"Trade {trade_id} not found or already closed"})
        else:
            console.print(f"[red]Trade {trade_id} not found or already closed.[/red]")
        raise typer.Exit(1)

    _save_trades(trades)

    if output == "json":
        from advisor.cli.formatters import output_json

        output_json(closed_trade.model_dump())
    else:
        pnl = closed_trade.pnl or 0
        color = "green" if pnl > 0 else "red"
        console.print(
            f"[{color}]Closed {closed_trade.id}:"
            f" {closed_trade.symbol} P&L=${pnl:+.2f}"
            f" ({reason})[/{color}]"
        )


@track_app.command("status")
def track_status(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Show open positions and P&L."""
    trades = _load_trades()
    open_trades = [t for t in trades if t.status == "open"]

    if output == "json":
        from advisor.cli.formatters import output_json

        output_json([t.model_dump() for t in open_trades])
        return

    if not open_trades:
        console.print("[yellow]No open trades.[/yellow]")
        return

    table = Table(title=f"📈 Open Positions ({len(open_trades)})")
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Symbol", style="cyan")
    table.add_column("Strike", justify="right")
    table.add_column("Expiry")
    table.add_column("Premium", justify="right", style="green")
    table.add_column("Opened")

    for t in open_trades:
        strike_str = f"${t.strike:.2f}"
        if t.long_strike:
            strike_str += f" / ${t.long_strike:.2f}"
        table.add_row(
            t.id,
            t.trade_type,
            t.symbol,
            strike_str,
            t.expiry,
            f"${t.premium:.2f}",
            t.opened_at[:10],
        )

    console.print(table)


@track_app.command("history")
def track_history(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Show closed trades and win rate."""
    trades = _load_trades()
    closed = [t for t in trades if t.status == "closed"]

    if output == "json":
        from advisor.cli.formatters import output_json

        wins = sum(1 for t in closed if (t.pnl or 0) > 0)
        total_pnl = sum(t.pnl or 0 for t in closed)
        output_json(
            {
                "trades": [t.model_dump() for t in closed],
                "win_rate": wins / len(closed) if closed else 0,
                "total_pnl": total_pnl,
            }
        )
        return

    if not closed:
        console.print("[yellow]No closed trades.[/yellow]")
        return

    wins = sum(1 for t in closed if (t.pnl or 0) > 0)
    total_pnl = sum(t.pnl or 0 for t in closed)
    win_rate = wins / len(closed) if closed else 0

    table = Table(title=f"📜 Trade History ({len(closed)} trades)")
    table.add_column("ID", style="dim")
    table.add_column("Type")
    table.add_column("Symbol", style="cyan")
    table.add_column("Strike", justify="right")
    table.add_column("Premium", justify="right")
    table.add_column("Close", justify="right")
    table.add_column("P&L", justify="right")
    table.add_column("Reason")

    for t in closed:
        pnl = t.pnl or 0
        color = "green" if pnl > 0 else "red"
        table.add_row(
            t.id,
            t.trade_type,
            t.symbol,
            f"${t.strike:.2f}",
            f"${t.premium:.2f}",
            f"${t.close_price:.2f}" if t.close_price is not None else "-",
            f"[{color}]${pnl:+.2f}[/{color}]",
            t.close_reason or "-",
        )

    console.print(table)
    console.print(f"\n[bold]Win Rate:[/bold] {win_rate:.0%} ({wins}/{len(closed)})")
    color = "green" if total_pnl > 0 else "red"
    console.print(f"[bold]Total P&L:[/bold] [{color}]${total_pnl:+,.2f}[/{color}]\n")


# ── Analyze command ────────────────────────────────────────────────────────────


@app.command("analyze")
def analyze(
    symbol: str = typer.Argument(..., help="Symbol to analyze"),
    expiry: Optional[str] = typer.Option(
        None, "--expiry", "-e", help="Specific expiry (YYYY-MM-DD)"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Analyze options chain for a symbol — Greeks, yields, best strikes."""
    from advisor.market.options_scanner import analyze_chain

    try:
        chain = analyze_chain(symbol, expiry=expiry)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if output == "json":
        from advisor.cli.formatters import output_json

        output_json(chain)
        return

    console.print(f"\n[bold]{chain.symbol}[/bold] @ ${chain.price:.2f}")
    console.print(f"Available expirations: {', '.join(chain.expirations)}\n")
    console.print(f"[bold]Expiry: {chain.expiry} ({chain.dte} DTE)[/bold]\n")

    if chain.puts:
        table = Table(title="PUT Chain (OTM)", show_lines=False)
        table.add_column("Strike", justify="right")
        table.add_column("Bid", justify="right", style="green")
        table.add_column("Ask", justify="right")
        table.add_column("IV", justify="right")
        table.add_column("OI", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("OTM%", justify="right")
        table.add_column("Yield", justify="right", style="bold")

        for p in chain.puts:
            table.add_row(
                f"${p.strike:.2f}",
                f"${p.bid:.2f}",
                f"${p.ask:.2f}",
                f"{p.iv:.0%}",
                str(p.oi),
                str(p.volume),
                f"{p.otm_pct:.1%}",
                f"{p.annualized_yield:.0%}" if p.annualized_yield else "-",
            )
        console.print(table)

    if chain.calls:
        table = Table(title="\nCALL Chain (OTM)", show_lines=False)
        table.add_column("Strike", justify="right")
        table.add_column("Bid", justify="right", style="green")
        table.add_column("Ask", justify="right")
        table.add_column("IV", justify="right")
        table.add_column("OI", justify="right")
        table.add_column("Vol", justify="right")
        table.add_column("OTM%", justify="right")

        for c in chain.calls:
            table.add_row(
                f"${c.strike:.2f}",
                f"${c.bid:.2f}",
                f"${c.ask:.2f}",
                f"{c.iv:.0%}",
                str(c.oi),
                str(c.volume),
                f"{c.otm_pct:.1%}",
            )
        console.print(table)

    console.print()


# ── Premium Scan command ──────────────────────────────────────────────────────


@app.command("premium-scan")
def premium_scan(
    account_size: float = typer.Option(5000.0, "--account-size", "-a", help="Account size in USD"),
    universe: str = typer.Option(
        "wheel", "--universe", "-u", help="Ticker universe: leveraged, wheel, blue_chip"
    ),
    tickers: Optional[str] = typer.Option(
        None, "--tickers", "-t", help="Comma-separated tickers (overrides universe)"
    ),
    min_iv_pctile: float = typer.Option(
        30.0, "--min-iv-pctile", help="Minimum IV percentile to qualify (0-100)"
    ),
    strategies: Optional[str] = typer.Option(
        None, "--strategies", "-s", help="Strategies: naked_put,put_credit_spread"
    ),
    min_dte: int = typer.Option(25, "--min-dte", help="Minimum days to expiration"),
    max_dte: int = typer.Option(45, "--max-dte", help="Maximum days to expiration"),
    top: int = typer.Option(15, "--top", help="Number of top results to show"),
    live_iv: bool = typer.Option(False, "--live-iv", help="Fetch live IV data from TastyTrade"),
    earnings_buffer: int = typer.Option(
        7, "--earnings-buffer", help="Days buffer around earnings to flag"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output format: json"),
):
    """Smart premium scanner — rank sell opportunities by composite score."""
    from advisor.market.options_scanner import UNIVERSES
    from advisor.market.premium_screener import PremiumScreener

    if tickers:
        ticker_list = [t.strip().upper() for t in tickers.split(",")]
    else:
        ticker_list = UNIVERSES.get(universe, UNIVERSES["wheel"])

    strategy_list = (
        [s.strip() for s in strategies.split(",")]
        if strategies
        else ["naked_put", "put_credit_spread"]
    )

    is_json = output == "json"

    if not is_json:
        console.print(
            f"\n[bold]Premium Scan[/bold] — scanning {len(ticker_list)} tickers"
            f" (min IV%ile={min_iv_pctile:.0f}, DTE={min_dte}-{max_dte})\n"
        )

    # Optionally fetch live IV from TastyTrade
    tt_iv_data: dict = {}
    if live_iv:
        try:
            import asyncio

            from advisor.market.tastytrade_client import get_market_metrics, get_session

            async def _fetch_iv():
                session = await get_session()
                return await get_market_metrics(session, ticker_list)

            tt_iv_data = asyncio.run(_fetch_iv())
            if not is_json:
                console.print(f"[green]TastyTrade IV loaded for {len(tt_iv_data)} symbols[/green]")
        except Exception as e:
            if not is_json:
                console.print(f"[yellow]TastyTrade IV unavailable: {e}[/yellow]")

    screener = PremiumScreener(
        account_size=account_size,
        min_iv_pctile=min_iv_pctile,
        strategies=strategy_list,
        min_dte=min_dte,
        max_dte=max_dte,
        earnings_buffer=earnings_buffer,
        top_n=top,
        tt_data=tt_iv_data or None,
    )
    scan_result = screener.scan(ticker_list)

    if is_json:
        from advisor.cli.formatters import output_json

        output_json(scan_result)
        return

    # Header
    console.print(
        f"[dim]Regime: {scan_result.regime} | "
        f"Target delta: {scan_result.target_delta:.2f} | "
        f"Scanned: {scan_result.tickers_scanned} tickers[/dim]\n"
    )

    if scan_result.opportunities:
        table = Table(
            title=f"Premium Opportunities ({len(scan_result.opportunities)} results)",
            show_lines=False,
        )
        table.add_column("Sym", style="cyan")
        table.add_column("Strategy")
        table.add_column("Strike", justify="right")
        table.add_column("Expiry")
        table.add_column("DTE", justify="right")
        table.add_column("Credit", justify="right", style="green")
        table.add_column("POP", justify="right")
        table.add_column("IV%ile", justify="right", style="yellow")
        table.add_column("Yield", justify="right", style="bold green")
        table.add_column("Liq", justify="right")
        table.add_column("Score", justify="right", style="bold")
        table.add_column("Flags", style="dim")

        for opp in scan_result.opportunities:
            strategy_label = "naked_put" if opp.strategy == "naked_put" else "spread"
            strike_str = f"${opp.strike:.2f}"
            if opp.long_strike is not None:
                strike_str += f"/${opp.long_strike:.0f}"

            # Color score
            if opp.sell_score >= 70:
                score_str = f"[bold green]{opp.sell_score:.0f}[/bold green]"
            elif opp.sell_score >= 50:
                score_str = f"[yellow]{opp.sell_score:.0f}[/yellow]"
            else:
                score_str = f"[dim]{opp.sell_score:.0f}[/dim]"

            table.add_row(
                opp.symbol,
                strategy_label,
                strike_str,
                str(opp.expiry),
                str(opp.dte),
                f"${opp.credit:.2f}",
                f"{opp.pop:.0%}",
                f"{opp.iv_percentile:.0f}",
                f"{opp.annualized_yield:.0%}",
                str(opp.liquidity.total),
                score_str,
                ", ".join(opp.flags) if opp.flags else "",
            )

        console.print(table)
    else:
        console.print("[yellow]No premium opportunities found matching criteria.[/yellow]")

    if scan_result.errors:
        console.print(f"\n[dim]Errors: {', '.join(scan_result.errors[:5])}[/dim]")

    console.print(f"\n[dim]Scanned at {scan_result.scanned_at:%H:%M:%S}[/dim]\n")
