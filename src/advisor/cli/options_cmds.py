"""Options CLI commands — scan, track, analyze."""

from __future__ import annotations

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
