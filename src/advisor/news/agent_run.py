"""Running the news agent for any name on demand, and reading what it judged.

The agent (``news.judge``) judges only news already archived for a name, and
only once its date has been checked — for held, watched and Swing names the
daemon keeps that archive current. A pick or a name just added has none, so
an on-demand run first pulls the name's recent news the way the entry
engine does when a decision has nothing to read (``news.ingest.explain_symbol``:
Tavily plus the Yahoo feed, dates checked by ``news.verify``, archived as
context), emits the context events that carry the date checks, and then
judges. Context and measurement only (user decision, 2026-09-27): a
judgment changes no action.

The pull costs one Tavily search per name. A run that finds nothing new to
judge still reports what it pulled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PULL_DAYS = 7  # the agent judges a week of news (``judge.JUDGE_WINDOW_DAYS``)
VIEW_DAYS = 14


def run_news_agent(
    db_path,
    symbol: str,
    now: datetime,
    *,
    pull: bool = True,
    puller=None,
    judge=None,
    company=None,
) -> dict:
    """Pull (optionally) and judge one name's news. Never raises; problems are reported."""
    import asyncio
    import sqlite3

    from advisor.daemon.store import DaemonStore

    symbol = symbol.strip().upper()
    store = DaemonStore(Path(db_path))
    conn = sqlite3.connect(str(db_path))
    out = {"symbol": symbol, "pulled": 0, "judged": 0, "material": 0, "problems": []}
    try:
        if company is None:
            try:
                from advisor.daemon.handlers import _company_name

                company = _company_name(symbol)
            except Exception:  # noqa: BLE001
                company = None
        if pull:
            try:
                if puller is None:
                    from advisor.news.ingest import context_events, explain_symbol

                    items = asyncio.run(
                        explain_symbol(
                            store, symbol, reason="ENTRY_REVIEW", company_name=company,
                            days=PULL_DAYS,
                        )
                    )  # fmt: skip
                    for event in context_events(items, reason="ENTRY_REVIEW"):
                        store.emit(event)
                else:
                    items = puller(store, symbol, company)
                out["pulled"] = len(items)
            except Exception as exc:  # noqa: BLE001
                out["problems"].append(f"{symbol}: news pull failed: {exc}")
        if judge is None:
            from advisor.news.judge import judge_symbol as judge
        judged, problems = judge(store, conn, symbol, now, company=company)
        out["judged"] = len(judged)
        out["material"] = sum(1 for j in judged if j.about_company and j.materiality.value != "LOW")
        out["problems"] += problems
    except Exception as exc:  # noqa: BLE001
        logger.exception("news agent failed for %s", symbol)
        out["problems"].append(f"{symbol}: {exc}")
    finally:
        conn.close()
        store.close()
    return out


def news_view(db_path, symbol: str, now: datetime) -> dict:
    """The name's judgments of the last ``VIEW_DAYS`` and its latest weekly synthesis."""
    import sqlite3

    from advisor.news.judge import NewsJudgmentStore

    conn = sqlite3.connect(str(db_path))
    try:
        js = NewsJudgmentStore(conn)
        judgments = js.list(symbol=symbol, since=now - timedelta(days=VIEW_DAYS))
        summary = js.latest_summary(symbol.upper())
    except sqlite3.OperationalError:
        return {"symbol": symbol.upper(), "judgments": [], "summary": None}
    finally:
        conn.close()
    judgments.sort(key=lambda j: j.published_at, reverse=True)
    return {
        "symbol": symbol.upper(),
        "judgments": [j.model_dump(mode="json") for j in judgments],
        "summary": summary.model_dump(mode="json") if summary else None,
    }
