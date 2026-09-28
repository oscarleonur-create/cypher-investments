"""CLI for the breadth layer: sync the market-wide stores and show the universe they cut."""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="breadth", help="The whole listed market: universe, bars and SEC facts")


def _db_path():
    from advisor.research.config import get_settings

    return get_settings().db_path


@app.command("sync")
def sync(
    bars: Annotated[bool, typer.Option("--bars/--no-bars", help="Pull daily bars")] = True,
    facts: Annotated[bool, typer.Option("--facts/--no-facts", help="Pull SEC frames")] = True,
    limit: Annotated[
        Optional[int], typer.Option("--limit", help="Only the first N common stocks (a sample)")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Directory → bars → E0 snapshot → SEC facts. The first run backfills four years."""
    from advisor.breadth.run import run_sync
    from advisor.daemon.market_calendar import now_et

    summary = run_sync(_db_path(), now_et(), bars=bars, facts=facts, limit=limit)
    if output == "json":
        output_json(summary)
        return
    if "error" in summary:
        output_error(summary["error"])
        return
    _print_summary(summary)
    if not summary["ok"]:
        console.print("[yellow]Incomplete: the bar source refused part of the pull.[/yellow]")
        raise typer.Exit(1)


def _print_summary(s: dict) -> None:
    d = s.get("directory", {})
    console.print(f"Directory: {d.get('listings')} listings, {d.get('common')} common stock")
    if "bars" in s:
        b = s["bars"]
        console.print(
            f"Bars: {b['fetched']} fetched, {b['current']} already current, {b['empty']} empty, "
            f"{b['skipped_empty']} skipped, {len(b['rebased'])} rebased, "
            f"{b['bars_written']} bars written"
            + (f" — RATE-LIMITED, {b['remaining']} left" if b["rate_limited"] else "")
        )
    _print_funnel(s.get("universe", {}))
    if "facts" in s:
        f = s["facts"]
        fr, gp = f["frames"], f["gaps"]
        console.print(
            f"SEC frames: {fr['frames_fetched']} read ({fr['frames_settled']} settled, "
            f"{fr['frames_missing']} absent), {fr['rows']} rows; "
            f"{gp['gaps_filled']} companies gap-filled"
        )
        console.print(f"Revenue coverage of eligible: {f['coverage_of_eligible']}")
        for e in fr["errors"] + gp["errors"]:
            console.print(f"[red]  {e}[/red]")
    console.print(f"[dim]{s.get('seconds')} s[/dim]")


def _print_funnel(u: dict) -> None:
    if not u:
        return
    console.print(
        f"Universe {u['day']} (rules {u['rules']}): {u['listings']} listings → "
        f"{u['common_stock_sec_filers']} common stock of SEC filers → "
        f"[bold]{u['eligible']} eligible[/bold]"
    )
    table = Table(title="Excluded, by reason")
    table.add_column("reason")
    table.add_column("names", justify="right")
    for reason, n in u["excluded"].items():
        table.add_row(reason, str(n))
    console.print(table)


@app.command("universe")
def universe(
    symbol: Annotated[Optional[str], typer.Argument(help="Show one name's standing")] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """The latest E0 snapshot: the funnel, or one name's standing and why."""
    from advisor.breadth.store import BreadthStore, breadth_path
    from advisor.breadth.universe import funnel

    with BreadthStore(breadth_path(_db_path())) as store:
        day = store.latest_universe_day()
        if day is None:
            output_error("No universe yet: run `advisor breadth sync`.")
            return
        rows = store.universe(day)
        last = store.last_run()
        counts = store.counts()
    if symbol:
        match = [r for r in rows if r["symbol"] == symbol.strip().upper().replace(".", "-")]
        if not match:
            output_error(f"{symbol.upper()} is not in the {day} symbol directory.")
            return
        r = match[0]
        if output == "json":
            output_json(r)
            return
        verdict = "eligible" if r["eligible"] else f"excluded: {r['reason']}"
        console.print(f"{r['symbol']} ({r['name']}, {r['exchange']}) on {day}: {verdict}")
        if r["price"] is not None:
            dv = r["dollar_volume"]
            console.print(
                f"  close ${r['price']:.2f}, median dollar volume "
                f"{'n/a' if dv is None else f'${dv / 1e6:,.1f}M'}, {r['sessions']} sessions"
            )
        return
    f = {"day": day.isoformat(), "rules": rows[0]["rules"] if rows else None, **funnel(rows)}
    if output == "json":
        output_json({"universe": f, "store": counts, "last_run": last})
        return
    _print_funnel(f)
    console.print(f"Store: {counts}")
    if last:
        console.print(f"Last sync {last['finished_at']} — {'ok' if last['ok'] else 'INCOMPLETE'}")
