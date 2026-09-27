"""CLI for the setup scanner (A, B, C) and its trade journal. Records, never alerts."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_json

app = typer.Typer(
    name="scan",
    help="Setup scanner: catalyst gaps (A), day-two runs (B), news dips (C), and a journal",
)


def _store():
    from advisor.research.config import get_settings
    from advisor.scanner.store import ScannerStore

    return ScannerStore(get_settings().db_path)


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f}%"


def _setup(value: Optional[str]):
    from advisor.scanner.models import Setup

    if value is None:
        return None
    try:
        return Setup(value.upper())
    except ValueError as exc:
        raise typer.BadParameter("setup must be A, B or C") from exc


@app.command("run")
def scan_run(
    no_news: Annotated[
        bool, typer.Option("--no-news", help="Skip news lookups (no credits)")
    ] = False,
    budget: Annotated[
        Optional[int], typer.Option("--news-budget", help="Cap news lookups per session")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Scan the market now and record any new A/C candidates."""
    import sqlite3

    from advisor.daemon.market_calendar import now_et
    from advisor.learning.actuator import active_session_thresholds
    from advisor.research.config import get_settings
    from advisor.scanner.scan import run_scan

    # The same thresholds the daemon runs: the code's, with approved changes.
    conn = sqlite3.connect(str(get_settings().db_path))
    try:
        thresholds = active_session_thresholds(conn)
    finally:
        conn.close()
    store = _store()
    try:
        result = run_scan(
            store,
            now_et(),
            news_budget=budget,
            check_news=not no_news,
            thresholds=thresholds,
        )
    finally:
        store.close()
    if output == "json":
        output_json(
            {
                "ran": result.ran,
                "summary": result.summary(),
                "new": [c.model_dump(mode="json") for c in result.new],
            }
        )
        return
    console.print(result.summary())
    if result.new:
        _print(result.new)


@app.command("premarket")
def scan_premarket(
    watchlist: Annotated[
        str, typer.Option("--watchlist", "-w", help="TastyTrade private watchlist name")
    ] = "Swing",
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Scan a TastyTrade watchlist before the bell for setups A, B and C."""
    from advisor.daemon.market_calendar import now_et
    from advisor.scanner.premarket import scan_watchlist

    store = _store()
    try:
        result = scan_watchlist(store, now_et(), watchlist)
    finally:
        store.close()
    if output == "json":
        output_json(
            {
                "ran": result.ran,
                "summary": result.summary(),
                "new": [c.model_dump(mode="json") for c in result.new],
            }
        )
        return
    console.print(result.summary())
    if result.new:
        _print(result.new)


def _resolve(store, symbol: str, session: Optional[str], setup: Optional[str]):
    """Candidates for a symbol on a session (default: the latest it appears in)."""
    rows = [c for c in store.list(limit=5000) if c.symbol == symbol.upper()]
    if setup:
        rows = [c for c in rows if c.setup is _setup(setup)]
    if session:
        rows = [c for c in rows if c.session == date.fromisoformat(session)]
    elif rows:
        latest = max(c.session for c in rows)
        rows = [c for c in rows if c.session == latest]
    if not rows:
        raise typer.BadParameter(f"no candidate for {symbol.upper()}")
    return rows


@app.command("skip")
def scan_skip(
    symbol: Annotated[str, typer.Argument(help="Candidate symbol")],
    reason: Annotated[
        str,
        typer.Option(
            "--reason", "-r", help="late | news | size | exposed | missed | rules | other"
        ),
    ],
    note: Annotated[str, typer.Option("--note", "-n")] = "",
    session: Annotated[Optional[str], typer.Option("--date", help="YYYY-MM-DD")] = None,
    setup: Annotated[Optional[str], typer.Option("--setup")] = None,
) -> None:
    """Record that you passed on a candidate, and why."""
    from advisor.scanner.models import DecisionSource, SkipReason, TradeDecision

    try:
        why = SkipReason(reason.lower())
    except ValueError as exc:
        raise typer.BadParameter(
            f"reason must be one of: {', '.join(r.value for r in SkipReason)}"
        ) from exc
    store = _store()
    try:
        for c in _resolve(store, symbol, session, setup):
            store.record_decision(
                TradeDecision(
                    candidate_id=c.id,
                    taken=False,
                    source=DecisionSource.USER,
                    reason=why,
                    note=note,
                )
            )
            # emoji=False: ids contain ":A:", which Rich would render as 🅰.
            console.print(
                f"skipped {c.id}: {why.value}" + (f" — {note}" if note else ""),
                emoji=False,
                markup=False,
            )
    finally:
        store.close()


@app.command("took")
def scan_took(
    symbol: Annotated[str, typer.Argument(help="Candidate symbol")],
    price: Annotated[Optional[float], typer.Option("--price", help="Your fill price")] = None,
    note: Annotated[str, typer.Option("--note", "-n")] = "",
    session: Annotated[Optional[str], typer.Option("--date", help="YYYY-MM-DD")] = None,
    setup: Annotated[Optional[str], typer.Option("--setup")] = None,
) -> None:
    """Record a candidate you took outside TastyTrade (fills there are found automatically)."""
    from advisor.scanner.models import DecisionSource, TradeDecision

    store = _store()
    try:
        for c in _resolve(store, symbol, session, setup):
            store.record_decision(
                TradeDecision(
                    candidate_id=c.id,
                    taken=True,
                    source=DecisionSource.USER,
                    fill_price=price,
                    note=note,
                )
            )
            console.print(f"taken {c.id}", emoji=False, markup=False)
    finally:
        store.close()


@app.command("journal")
def scan_journal(
    session: Annotated[Optional[str], typer.Option("--date", help="Session YYYY-MM-DD")] = None,
    sync: Annotated[bool, typer.Option("--sync/--no-sync", help="Match broker fills first")] = True,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Each candidate of a session: taken (from your fills), skipped (and why), or untagged."""
    from advisor.daemon.market_calendar import now_et
    from advisor.scanner.journal import status, sync_fills

    day = date.fromisoformat(session) if session else now_et().date()
    store = _store()
    try:
        synced = sync_fills(store, [day]) if sync else 0
        rows = store.list(session=day, limit=5000)
        decisions = store.latest_decisions()
    finally:
        store.close()
    entries = []
    for c in sorted(rows, key=lambda c: c.detected_at):
        d = decisions.get(c.id)
        entries.append(
            {
                "id": c.id,
                "symbol": c.symbol,
                "setup": c.setup.value,
                "phase": c.phase.value,
                "status": status(c, d),
                "fill_price": d.fill_price if d else None,
                "note": d.note if d else "",
                "close": c.outcomes.get("close"),
                "next_close": c.outcomes.get("next_close"),
            }
        )
    if output == "json":
        output_json({"session": day.isoformat(), "synced": synced, "candidates": entries})
        return
    if sync:
        console.print(f"{synced} new broker fill(s) matched")
    if not entries:
        console.print(f"no candidates on {day}")
        return
    table = Table(title=f"Journal {day}")
    for col in ("setup", "when", "symbol", "status", "fill", "close", "next close", "note"):
        table.add_column(col, no_wrap=True)
    for e in entries:
        table.add_row(
            e["setup"],
            e["phase"],
            e["symbol"],
            e["status"],
            "—" if e["fill_price"] is None else f"{e['fill_price']:.2f}",
            _pct(e["close"]),
            _pct(e["next_close"]),
            e["note"][:40],
        )
    console.print(table)
    untagged = sum(1 for e in entries if e["status"] == "untagged")
    if untagged:
        console.print(
            f"{untagged} untagged — `advisor scan skip SYMBOL --reason ...` says why you passed"
        )


@app.command("review")
def scan_review(
    since: Annotated[
        Optional[str], typer.Option("--since", help="First session YYYY-MM-DD")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """What the candidates you took, skipped, and never tagged went on to do."""
    from advisor.scanner.journal import REVIEW_HORIZONS, review

    store = _store()
    try:
        first = date.fromisoformat(since) if since else None
        rows = review(store.list(since=first, limit=100000), store.latest_decisions())
    finally:
        store.close()
    if output == "json":
        output_json(rows)
        return
    if not rows:
        console.print("no candidates recorded")
        return
    table = Table(title="Taken vs. not taken (mean / % positive)")
    for col in ("setup", "status", "n", "close", "next close", "5d", "10d", "20d"):
        table.add_column(col, no_wrap=True)
    for r in rows:
        cells = []
        for key in REVIEW_HORIZONS:
            s = r[key]
            cells.append(
                "—" if s["n"] == 0 else f"{_pct(s['mean'])} / {s['positive']:.0%} (n={s['n']})"
            )
        table.add_row(r["setup"], r["status"], str(r["n"]), *cells)
    console.print(table)


@app.command("list")
def scan_list(
    session: Annotated[Optional[str], typer.Option("--date", help="Session YYYY-MM-DD")] = None,
    setup: Annotated[Optional[str], typer.Option("--setup", help="A or C")] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Recorded candidates, newest first (chronological, never ranked)."""
    store = _store()
    try:
        day = date.fromisoformat(session) if session else None
        rows = store.list(session=day, setup=_setup(setup))
    finally:
        store.close()
    if output == "json":
        output_json([c.model_dump(mode="json") for c in rows])
        return
    if not rows:
        console.print("no candidates recorded")
        return
    _print(rows)


@app.command("outcomes")
def scan_outcomes(output: Annotated[str, typer.Option("--output", "-o")] = "table") -> None:
    """Fill in what the price did after each candidate. Safe to re-run."""
    from advisor.daemon.market_calendar import now_et
    from advisor.scanner.outcomes import fill_outcomes

    store = _store()
    try:
        result = fill_outcomes(store, now_et())
    finally:
        store.close()
    if output == "json":
        output_json(
            {"summary": result.summary(), "pending": result.pending, "updated": result.updated}
        )
    else:
        console.print(result.summary())


@app.command("report")
def scan_report(
    since: Annotated[
        Optional[str], typer.Option("--since", help="First session YYYY-MM-DD")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Outcome statistics per setup, split by whether a catalyst was found."""
    from advisor.scanner.report import summarize

    store = _store()
    try:
        first = date.fromisoformat(since) if since else None
        rows = summarize(store.list(since=first, limit=100000))
    finally:
        store.close()
    if output == "json":
        output_json(rows)
        return
    if not rows:
        console.print("no candidates recorded")
        return
    table = Table(title="Scanner outcomes (mean / % positive)")
    for col in (
        "setup",
        "catalyst",
        "n",
        "+30m",
        "+60m",
        "+120m",
        "close",
        "next open",
        "next close",
    ):
        table.add_column(col)
    for r in rows:
        cells = []
        for key in ("r30", "r60", "r120", "close", "next_open", "next_close"):
            s = r[key]
            cells.append(
                "—" if s["n"] == 0 else f"{_pct(s['mean'])} / {s['positive']:.0%} (n={s['n']})"
            )
        table.add_row(r["setup"], r["catalyst"], str(r["n"]), *cells)
    console.print(table)


def _print(rows) -> None:
    """One line per candidate, then the catalysts underneath for reading."""
    table = Table()
    for col in ("time", "setup", "symbol", "price", "chg", "gap", "rvol", "σ", "cap $B", "news"):
        table.add_column(col, no_wrap=True)
    for c in rows:
        if c.has_catalyst is None:
            cat = "unchecked"
        else:
            cat = str(len(c.catalysts))
        table.add_row(
            c.detected_at.strftime("%m-%d %H:%M"),
            c.setup.value + (" pre" if c.phase.value == "premarket" else ""),
            c.symbol,
            f"{c.price:.2f}",
            _pct(c.change),
            _pct(c.gap),
            "—" if c.rvol is None else f"{c.rvol:.1f}x",
            "—" if c.sigma is None else f"{c.sigma:.1f}",
            "—" if c.market_cap is None else f"{c.market_cap / 1e9:.1f}",
            cat,
        )
    console.print(table)
    for c in rows:
        peers = ""
        if c.setup.value == "C":
            peers = (
                f"  peers {', '.join(c.peers) or '—'}: {_pct(c.peer_move)}"
                if c.peer_move is not None
                else "  peers: not measured"
            )
        if not c.catalysts and not peers:
            continue
        console.print(f"[bold]{c.setup.value} {c.symbol}[/bold] {c.name or ''}{peers}")
        for item in c.catalysts[:4]:
            stamp = item.published_at.astimezone(c.detected_at.tzinfo).strftime("%m-%d %H:%M")
            console.print(f"  {stamp} [{item.kind}] {item.title[:100]}", markup=False)
