"""CLI for entry work: the daily sheet per watched name."""

from __future__ import annotations

from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_json

app = typer.Typer(
    name="entry",
    help="Entries: daily sheet, sized proposals (trade + position legs), and their track record",
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
            each = "; ".join(
                f"{_pct(r, sign=False)} at {m:.1%} ({label})" for label, m, r in c.readings
            )
            span = (
                f"{_pct(c.low, sign=False)} to {_pct(c.high, sign=False)}"
                if c.readings and c.high - c.low > 0.0005
                else _pct(c.low, sign=False)
            )
            lines.append(
                f"  requires (10y revenue CAGR) {span}: {each} — delivers {_pct(c.delivered)} YoY"
                + (f", {c.consensus_label} implies {_pct(c.consensus)}/yr" if c.consensus else "")
            )
            for note in c.notes:
                lines.append(f"  note: {note}")
        if s.next_earnings is not None:
            lines.append(f"  results {s.next_earnings.isoformat()} (in {s.earnings_in} sessions)")
        for gap in s.gaps:
            lines.append(f"  gap: {gap}")
        if lines:
            console.print(f"[bold]{s.symbol}[/bold]")
            for line in lines:
                console.print(line, markup=False, emoji=False)


def _entry_store():
    from advisor.entry.store import EntryStore
    from advisor.research.config import get_settings

    return EntryStore(get_settings().db_path)


@app.command("propose")
def entry_propose(
    symbols: Annotated[
        Optional[list[str]], typer.Argument(help="Default: held + watchlists")
    ] = None,
    read: Annotated[
        bool,
        typer.Option("--read/--no-read", help="Ask the model where a decision is on the table"),
    ] = True,
    record: Annotated[
        bool, typer.Option("--record/--no-record", help="Write proposals to the ledger")
    ] = True,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """A sized proposal per name: ENTER / ADD / IN_ZONE / WAIT / NONE, with its reasons."""
    from advisor.daemon.market_calendar import now_et
    from advisor.entry.run import propose_all

    store, scanner = _stores()
    entries = _entry_store() if record else None
    try:
        proposals, errors = propose_all(
            store,
            now_et(),
            symbols=[s.upper() for s in symbols] if symbols else None,
            entry_store=entries,
            scanner_store=scanner,
            read=read,
        )
    finally:
        store.close()
        scanner.close()
        if entries is not None:
            entries.close()
    if output == "json":
        output_json({"errors": errors, "proposals": [p.model_dump(mode="json") for p in proposals]})
        return
    for e in errors:
        console.print(f"[yellow]{e}[/yellow]")
    _print_proposals(proposals)


def _print_proposals(proposals) -> None:
    order = {"ENTER": 0, "ADD": 1, "WAIT": 2, "IN_ZONE": 3, "NONE": 4, "CANNOT_SAY": 5}
    for p in sorted(proposals, key=lambda p: (order.get(p.action.value, 9), p.symbol)):
        head = f"{p.symbol} {p.action.value}"
        if p.price:
            head += f" @ {p.price:,.2f}"
        if p.stance:
            head += f" · reading {p.stance}"
        console.print(head, markup=False, emoji=False, style="bold")
        lines = [f"  trigger: {t}" for t in p.triggers]
        lines += [f"  · {r.text}  [{r.source}]" for r in p.reasons]
        lines += [f"  ! {b}" for b in p.blockers]
        for leg in p.legs:
            lines.append(
                f"  {leg.horizon}: buy {leg.shares} (${leg.notional:,.0f}), stop {leg.stop:,.2f} "
                f"({leg.stop_basis}), risk {leg.risk_pct:.0%} of net liq"
            )
            lines += [f"    exit: {x}" for x in leg.exit_rules] + [f"    {n}" for n in leg.notes]
        lines += [f"  reading: {s}" for s in p.reading]
        lines += [f"  gap: {g}" for g in p.gaps]
        for line in lines:
            console.print(line, markup=False, emoji=False)


@app.command("skip")
def entry_skip(
    symbol: Annotated[str, typer.Argument()],
    reason: Annotated[
        str,
        typer.Option(
            "--reason", "-r", help="late | news | size | exposed | missed | rules | other"
        ),
    ],
    note: Annotated[str, typer.Option("--note", "-n")] = "",
    session: Annotated[Optional[str], typer.Option("--date", help="YYYY-MM-DD")] = None,
) -> None:
    """Record why you did not act on today's proposals for a name."""
    from datetime import date

    from advisor.daemon.market_calendar import now_et
    from advisor.scanner.models import DecisionSource, SkipReason, TradeDecision

    try:
        why = SkipReason(reason.lower())
    except ValueError as exc:
        raise typer.BadParameter(
            f"reason must be one of: {', '.join(r.value for r in SkipReason)}"
        ) from exc
    day = date.fromisoformat(session) if session else now_et().date()
    entries = _entry_store()
    daemon_store, scanner = _stores()
    try:
        rows = [p for p in entries.list(session=day) if p.symbol == symbol.upper()]
        if not rows:
            raise typer.BadParameter(f"no proposal for {symbol.upper()} on {day}")
        for p in rows:
            scanner.record_decision(
                TradeDecision(
                    candidate_id=p.id,
                    taken=False,
                    source=DecisionSource.USER,
                    reason=why,
                    note=note,
                )
            )
            console.print(f"skipped {p.id}: {why.value}", markup=False, emoji=False)
    finally:
        entries.close()
        daemon_store.close()
        scanner.close()


@app.command("review")
def entry_review(
    since: Annotated[
        Optional[str], typer.Option("--since", help="First session YYYY-MM-DD")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """What proposals went on to do, by action and by what you did about them."""
    from datetime import date

    from advisor.entry.review import review

    entries = _entry_store()
    daemon_store, scanner = _stores()
    try:
        first = date.fromisoformat(since) if since else None
        rows = review(entries.list(since=first), scanner.latest_decisions())
    finally:
        entries.close()
        daemon_store.close()
        scanner.close()
    if output == "json":
        output_json(rows)
        return
    if not rows:
        console.print("no proposals recorded")
        return
    table = Table(title="Proposals: what followed (mean / % positive; stop-hit rate)")
    for col in (
        "action",
        "you",
        "n",
        "next close",
        "5d",
        "20d",
        "worst 20d",
        "trade stop",
        "pos. stop",
    ):
        table.add_column(col, no_wrap=True)

    def cell(row, key, rate=False):
        s = row[key]
        if s["n"] == 0:
            return "—"
        if rate:
            return f"{s['mean']:.0%} (n={s['n']})"
        return f"{_pct(s['mean'])} / {s['positive']:.0%} (n={s['n']})"

    for r in rows:
        table.add_row(
            r["action"],
            r["status"],
            str(r["n"]),
            cell(r, "next_close"),
            cell(r, "d5"),
            cell(r, "d20"),
            cell(r, "mae20"),
            cell(r, "trade_stop", True),
            cell(r, "pos_stop20", True),
        )
    console.print(table)
