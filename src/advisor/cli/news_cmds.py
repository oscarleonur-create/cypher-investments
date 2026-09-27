"""CLI for the news agent: judge the archived items, list the judgments, score them."""

from __future__ import annotations

from datetime import timedelta
from typing import Annotated, Optional

import typer
from rich.table import Table

from advisor.cli.formatters import console, output_json

app = typer.Typer(
    name="news",
    help="The news agent: each item judged (direction, materiality, thesis), then measured",
)


def _stores():
    import sqlite3

    from advisor.daemon.store import DaemonStore
    from advisor.research.config import get_settings

    path = get_settings().db_path
    return DaemonStore(path), sqlite3.connect(str(path))


@app.command("judge")
def judge_cmd(
    symbols: Annotated[
        Optional[list[str]], typer.Argument(help="Default: held + watchlists")
    ] = None,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """Judge every item of the last week not yet judged under the current prompt."""
    from advisor.daemon.handlers import _company_name
    from advisor.daemon.market_calendar import now_et
    from advisor.news.judge import judge_all

    store, conn = _stores()
    try:
        if symbols:
            names = [s.upper() for s in symbols]
        else:
            from advisor.daemon.universe import research_symbols

            book = store.load_latest_book()
            if book is None:
                raise typer.BadParameter("no book snapshot stored; name the symbols")
            names, _ = research_symbols(book)
        judged, problems = judge_all(store, conn, names, now_et(), names=_company_name)
    finally:
        store.close()
        conn.close()
    if output == "json":
        output_json({"judged": [j.model_dump(mode="json") for j in judged], "problems": problems})
        return
    _table(judged, f"Judged {len(judged)} item(s)")
    for p in problems:
        console.print(f"[yellow]{p}[/yellow]")


def _table(judgments, title: str) -> None:
    table = Table(title=title)
    for col in ("date", "symbol", "call", "type", "new?", "basis", "thesis", "title", "why"):
        table.add_column(col, overflow="fold")
    for j in judgments:
        call = (
            "off-topic"
            if not j.about_company
            else f"{j.direction.value.lower()} {j.materiality.value.lower()}"
        )
        thesis = ", ".join(
            f"{'AGAINST' if c.against_thesis else 'for'} ({c.kind})" for c in j.claims
        )
        table.add_row(
            j.published_at.date().isoformat(),
            j.symbol,
            call,
            j.event_type.value.lower(),
            j.novelty.value.lower(),
            j.basis.value.lower(),
            thesis or "—",
            j.title[:70],
            j.why[:120],
        )
    console.print(table)


@app.command("list")
def list_cmd(
    symbol: Annotated[Optional[str], typer.Argument()] = None,
    days: Annotated[int, typer.Option("--days", "-d")] = 7,
    output: Annotated[str, typer.Option("--output", "-o")] = "table",
) -> None:
    """The judgments on file, newest first."""
    from advisor.daemon.market_calendar import now_et
    from advisor.news.judge import NewsJudgmentStore, prompt_version

    store, conn = _stores()
    try:
        rows = NewsJudgmentStore(conn).list(
            symbol=symbol, since=now_et() - timedelta(days=days), version=prompt_version()
        )
    finally:
        store.close()
        conn.close()
    if output == "json":
        output_json([j.model_dump(mode="json") for j in rows])
        return
    _table(rows, f"News judgments, last {days} days (prompt {prompt_version()})")


@app.command("report")
def report_cmd(output: Annotated[str, typer.Option("--output", "-o")] = "table") -> None:
    """Are its calls worth anything? Excess over each name's own drift, per call."""
    from advisor.learning.evaluate import Baselines, yahoo_closes
    from advisor.news.judge import NewsJudgmentStore, evaluate, prompt_version

    store, conn = _stores()
    try:
        rows = NewsJudgmentStore(conn).list(version=prompt_version())
    finally:
        store.close()
        conn.close()
    cells = evaluate(rows, Baselines(yahoo_closes()))
    if output == "json":
        output_json({"prompt": prompt_version(), "judgments": len(rows), "cells": cells})
        return
    table = Table(title=f"News calls vs what followed ({len(rows)} judgments)")
    for col in ("call", "horizon", "n", "windows", "excess", "95% CI", "verdict"):
        table.add_column(col)
    for c in cells:
        ci = c["ci"]
        table.add_row(
            c["group"],
            c["horizon"],
            str(c["n"]),
            str(c["windows"]),
            f"{c['excess'] * 100:+.2f}%",
            "—" if ci is None else f"{ci[0] * 100:+.2f}% … {ci[1] * 100:+.2f}%",
            f"{c['verdict']} — {c['reason']}",
        )
    console.print(table)
    console.print(
        "[dim]A NEGATIVE call that works has an interval below zero; nothing is called "
        "an edge without 10 independent windows.[/dim]"
    )
