"""CLI for entry work: the daily sheet per watched name."""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_json

app = typer.Typer(
    name="entry", help="Entries: daily sheet per watched name, zone as required growth"
)


def _pct(x: float | None, sign: bool = True) -> str:
    if x is None:
        return "—"
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def _stores():
    from advisor.daemon.store import DaemonStore
    from advisor.research.config import get_settings
    from advisor.scanner.store import ScannerStore

    path = get_settings().db_path
    return DaemonStore(path), ScannerStore(path)


def _default_symbols(store) -> list[str]:
    from advisor.daemon.universe import research_symbols

    book = store.load_latest_book()
    if book is None:
        raise typer.BadParameter("no book snapshot stored; name the symbols explicitly")
    symbols, errors = research_symbols(book)
    for e in errors:
        console.print(f"[yellow]{e}[/yellow]")
    return symbols


@app.command("sheet")
def entry_sheet(
    symbols: Annotated[
        Optional[list[str]], typer.Argument(help="Default: held + watchlists")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Today per name: events, move, holding, and the zone expressed as required growth."""
    from advisor.daemon.market_calendar import now_et
    from advisor.entry.sheet import build_sheet

    store, scanner = _stores()
    try:
        names = [s.upper() for s in symbols] if symbols else _default_symbols(store)
        sheets = [build_sheet(store, s, now_et(), scanner_store=scanner) for s in names]
    finally:
        store.close()
        scanner.close()

    if output == "json":
        output_json(
            [
                {
                    **s.model_dump(mode="json"),
                    "changed": s.changed,
                    "in_zone": s.zone.in_zone if s.zone else None,
                }
                for s in sheets
            ]
        )
        return

    table = Table(title="Entry sheet — zone: P/S at or below its own 2-year median")
    for col in (
        "symbol",
        "price",
        "day",
        "z",
        "5d",
        "20d",
        "events",
        "P/S",
        "2y median",
        "pctl",
        "zone top",
        "in zone",
        "held",
    ):
        table.add_column(col, no_wrap=True)
    for s in sheets:
        m, z = s.move, s.zone
        table.add_row(
            s.symbol + (" *" if s.changed else ""),
            "—" if m is None else f"{m.price:,.2f}",
            "—" if m is None else _pct(m.day),
            "—" if m is None or m.z is None else f"{m.z:+.1f}",
            "—" if m is None else _pct(m.d5),
            "—" if m is None else _pct(m.d20),
            f"{len(s.events_today)} ({s.events_week}/7d)",
            "—" if z is None else f"{z.ps_now:.1f}x",
            "—" if z is None else f"{z.median:.1f}x",
            "—" if z is None else f"{z.percentile:.0%}",
            "—" if z is None else f"{z.top:,.2f}",
            "—" if z is None else ("yes" if z.in_zone else f"no ({_pct(z.distance)})"),
            "—" if s.holding is None else f"{s.holding.weight * 100:.1f}%",
        )
    console.print(table)
    console.print("* = something changed today (tier A/B event, |z| ≥ 2, or a scanner candidate)")
    for s in sheets:
        lines = []
        for e in s.events_today[:5]:
            lines.append(f"  {e.ts.strftime('%m-%d %H:%M')} [{e.tier}] {e.text[:110]}")
        c = s.context
        if c:
            own = (
                f"{_pct(c.required_own, sign=False)} at its own {c.own_margin:.1%} FCF margin"
                if c.required_own is not None and c.own_margin is not None
                else "undefined at its own margin"
            )
            lines.append(
                f"  requires (10y revenue CAGR): {own}; "
                f"{_pct(c.required_generic, sign=False)} at the generic "
                f"{c.generic.fcf_margin:.0%} — delivers {_pct(c.delivered)} YoY"
                + (f", {c.consensus_label} implies {_pct(c.consensus)}/yr" if c.consensus else "")
            )
            for note in c.notes:
                lines.append(f"  note: {note}")
        for gap in s.gaps:
            lines.append(f"  gap: {gap}")
        if lines:
            console.print(f"[bold]{s.symbol}[/bold]")
            for line in lines:
                console.print(line, markup=False, emoji=False)
