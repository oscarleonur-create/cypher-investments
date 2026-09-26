"""CLI for the learning loop: which rule versions exist and what each one produced."""

from __future__ import annotations

import sqlite3
from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_error, output_json

app = typer.Typer(name="learn", help="Rule versions, your trades, and what each rule has earned")


def _db() -> sqlite3.Connection:
    from advisor.research.config import get_settings

    return sqlite3.connect(str(get_settings().db_path))


def _current() -> dict[str, str]:
    """``{ruleset: version}`` of the rules this code runs now."""
    from advisor.entry.ruleset import entry_rules
    from advisor.scanner.ruleset import premarket_rules, session_rules

    return {s.ruleset: s.version for s in (session_rules(), premarket_rules(), entry_rules())}


def _current_params(ruleset: str) -> dict | None:
    from advisor.entry.ruleset import entry_rules
    from advisor.scanner.ruleset import premarket_rules, session_rules

    for s in (session_rules(), premarket_rules(), entry_rules()):
        if s.ruleset == ruleset:
            return s.params
    return None


@app.command("rules")
def rules(
    version: Annotated[
        Optional[str], typer.Argument(help="A version (or prefix) to show in full")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Rule versions on file, how many records each produced, and which one runs now."""
    from advisor.learning.rules import PRE_REGISTRY
    from advisor.learning.store import RuleStore

    conn = _db()
    try:
        store = RuleStore(conn)
        if version is not None:
            _show(store, version, output)
            return
        versions = store.list()
        counts = store.record_counts()
    finally:
        conn.close()
    current = _current()

    rows = []
    for v in versions:
        rows.append(
            {
                "ruleset": v.ruleset,
                "version": v.version,
                "current": current.get(v.ruleset) == v.version,
                "first_seen_at": v.first_seen_at.isoformat(),
                "first_code": v.first_code,
                "records": {label: c.get(v.version, 0) for label, c in counts.items()},
            }
        )
    unstamped = {label: c[PRE_REGISTRY] for label, c in counts.items() if PRE_REGISTRY in c}
    not_yet = sorted(rs for rs, ver in current.items() if ver not in {v.version for v in versions})
    if output == "json":
        output_json(
            {
                "versions": rows,
                "running": current,
                "not_yet_recorded": not_yet,
                "pre_registry": unstamped,
            }
        )
        return

    table = Table(title="Rule versions")
    for col in ("ruleset", "version", "now", "first seen", "first code", "records"):
        table.add_column(col)
    for r in rows:
        recs = ", ".join(f"{n} {label}" for label, n in r["records"].items() if n) or "—"
        table.add_row(
            r["ruleset"],
            r["version"],
            "●" if r["current"] else "",
            r["first_seen_at"][:16],
            r["first_code"],
            recs,
        )
    console.print(table)
    for rs in not_yet:
        console.print(f"[dim]{rs} {current[rs]} runs now but has produced no record yet[/dim]")
    if unstamped:
        text = ", ".join(f"{n} {label}" for label, n in unstamped.items())
        console.print(f"[dim]{PRE_REGISTRY}: {text} written before versions were stamped[/dim]")


def _show(store, version: str, output: str) -> None:
    v = store.get(version)
    if v is None:
        output_error(f"no single rule version matches {version!r}")
        return
    now = _current_params(v.ruleset) or {}
    changed = sorted(k for k in set(v.params) | set(now) if v.params.get(k) != now.get(k))
    if output == "json":
        output_json({**v.model_dump(mode="json"), "differs_from_running": changed})
        return
    table = Table(title=f"{v.ruleset} {v.version} (first run on {v.first_code})")
    for col in ("parameter", "value", "kind", "running now"):
        table.add_column(col)
    for name in sorted(set(v.params) | set(now)):
        here = v.params.get(name, "—")
        there = now.get(name, "—")
        table.add_row(
            name,
            str(here),
            v.kinds.get(name, "—"),
            "" if name not in changed else str(there),
        )
    console.print(table)
    if not changed:
        console.print("[dim]identical to the rules running now[/dim]")


@app.command("trades")
def trades(
    sync: Annotated[
        bool, typer.Option("--sync/--no-sync", help="Rebuild from the broker history first")
    ] = False,
    book: Annotated[
        Optional[str], typer.Option("--book", help="quick | hold | unclassified")
    ] = None,
    show: Annotated[int, typer.Option("--show", help="List the last N trades")] = 0,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Your round trips per book, what the system matched, and your exits vs the rule's."""
    from advisor.daemon.market_calendar import now_et
    from advisor.entry.store import EntryStore
    from advisor.learning.store import TradeStore
    from advisor.learning.trades import HISTORY_START, review, sync_trades, unmatched
    from advisor.research.config import get_settings
    from advisor.scanner.store import ScannerStore

    conn = _db()
    try:
        store = TradeStore(conn)
        synced = None
        if sync:
            path = get_settings().db_path
            scanner, entries = ScannerStore(path), EntryStore(path)
            try:
                synced = sync_trades(store, scanner, entries, HISTORY_START, now_et().date())
            finally:
                scanner.close()
                entries.close()
            if synced.error:
                output_error(f"broker unavailable: {synced.error}")
                return
        all_trades = store.list(book=book)
        coverage = _coverage_start(conn)
    finally:
        conn.close()

    rows = review(all_trades)
    missed, before = unmatched(all_trades, coverage)
    open_ = [t for t in all_trades if not t.closed]
    if output == "json":
        output_json(
            {
                "sync": synced.summary() if synced else None,
                "books": rows,
                "open": len(open_),
                "coverage_start": coverage.isoformat() if coverage else None,
                "unmatched_since_coverage": [t.id for t in missed],
                "before_coverage": before,
                "trades": [t.model_dump(mode="json") for t in all_trades[-show:]] if show else [],
            }
        )
        return
    if synced:
        console.print(f"[dim]{synced.summary()}[/dim]")
    table = Table(title="Your trades, by book")
    cols = ("book", "n", "hit", "mean", "median", "P&L", "sessions", "matched", "yours vs rule")
    for col in cols:
        table.add_column(col)
    for r in rows:
        if "vs_rule" not in r:
            table.add_row(r["book"], str(r["n"]), "", "", "", f"${r['pnl']:,.0f}", "", "", "")
            continue
        vs = r["vs_rule"]
        cmp_ = f"{vs['yours']:+.2%} vs {vs['rule']:+.2%} (n={vs['n']})" if vs["n"] else "—"
        table.add_row(
            r["book"],
            str(r["n"]),
            f"{r['hit_rate']:.0%}",
            f"{r['mean_ret']:+.2%}",
            f"{r['median_ret']:+.2%}",
            f"${r['pnl']:,.0f}",
            f"{r['mean_sessions']:.1f}",
            str(r["matched"]),
            cmp_,
        )
    console.print(table)
    since = f"since {coverage}" if coverage else "the system has recorded nothing yet"
    console.print(
        f"[dim]{len(open_)} open. {since}: {len(missed)} closed long trades matched no "
        f"candidate or proposal (things you saw that the system did not); {before} earlier "
        "trades predate it and are not counted as misses[/dim]"
    )
    if show:
        t2 = Table(title=f"Last {show} trades")
        for col in ("opened", "symbol", "book", "sessions", "ret", "P&L", "matched"):
            t2.add_column(col)
        for t in all_trades[-show:]:
            t2.add_row(
                t.entry_session.isoformat(),
                t.underlying if t.instrument == "Equity" else t.symbol,
                t.book.value,
                "open" if not t.closed else str(t.sessions_held),
                "—" if t.ret is None else f"{t.ret:+.2%}",
                "—" if t.pnl is None else f"${t.pnl:,.0f}",
                ", ".join(t.candidate_ids + ([t.proposal_id] if t.proposal_id else [])) or "",
            )
        console.print(t2)


def _coverage_start(conn: sqlite3.Connection):
    """The first session anything was recorded by the scanner or the entry module."""
    from datetime import date

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    firsts = [
        conn.execute(f"SELECT MIN(session) FROM {t}").fetchone()[0]
        for t in ("scan_candidates", "entry_proposals")
        if t in tables
    ]
    firsts = [f for f in firsts if f]
    return date.fromisoformat(min(firsts)) if firsts else None
