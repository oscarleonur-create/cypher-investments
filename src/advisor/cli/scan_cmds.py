"""CLI for the intraday setup scanner (setups A and C). Records, never alerts."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_json

app = typer.Typer(name="scan", help="Intraday setup scanner: catalyst gaps (A) and news dips (C)")


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
        raise typer.BadParameter("setup must be A or C") from exc


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
    from advisor.daemon.market_calendar import now_et
    from advisor.scanner.scan import run_scan

    store = _store()
    try:
        result = run_scan(store, now_et(), news_budget=budget, check_news=not no_news)
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
            c.setup.value,
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
        if not c.catalysts:
            continue
        console.print(f"[bold]{c.setup.value} {c.symbol}[/bold] {c.name or ''}")
        for item in c.catalysts[:4]:
            stamp = item.published_at.astimezone(c.detected_at.tzinfo).strftime("%m-%d %H:%M")
            console.print(f"  {stamp} [{item.kind}] {item.title[:100]}", markup=False)
