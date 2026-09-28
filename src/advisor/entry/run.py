"""Sheets → proposals → (reading where it matters) → the ledger.

The model is expensive (10-20s a call) and its reading only matters where a
decision is on the table, so it is asked only for names whose sheet changed
today or whose proposal is to act (ENTER, ADD, WAIT). Quiet names get the
deterministic proposal and nothing else.

**A decision is not read blind.** A name with no filing and no archived news
in the reading's window comes back NO_FACTS — RDDT on 2026-09-25, a -27%
month with nothing in the store to say why. News is pulled only to explain
something already found (CLAUDE.md), and a proposal to act is such a thing:
for ENTER, ADD and WAIT the news is fetched (Tavily + Yahoo, keywords from
the direction of the move), archived as tier-C context, and the name read
again. Once per name per session, so an hourly job does not repeat a search
that came back empty.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

from advisor.entry.proposal import Action, Proposal, Reason, build_proposal
from advisor.entry.sheet import build_sheet

logger = logging.getLogger(__name__)

WORTH_READING = {Action.ENTER, Action.ADD, Action.WAIT, Action.EXIT, Action.TRIM, Action.REVIEW}
EXPLAIN_DAYS = 14  # the move being explained is recent; older news is not its cause

# (symbol, session) already searched this process. The daemon is long-lived,
# so this keeps the hourly job from repeating a search that found nothing.
_EXPLAINED: set[tuple[str, date]] = set()


def explain_reason(sheet) -> str:
    """The REASON_KEYWORDS key for a proposal's news pull, from its move."""
    day = sheet.move.day if sheet.move is not None else None
    if day is None:
        return "ENTRY_REVIEW"
    return "ENTRY_DROP" if day < 0 else "ENTRY_RALLY"


def _explain(daemon_store, symbol: str, reason: str) -> int:
    """Fetch, archive and emit news for one name. Returns the items found."""
    import asyncio

    from advisor.daemon.handlers import _company_name
    from advisor.news.ingest import context_events, explain_symbol

    items = asyncio.run(
        explain_symbol(
            daemon_store,
            symbol,
            reason=reason,
            company_name=_company_name(symbol),
            days=EXPLAIN_DAYS,
        )
    )
    for event in context_events(items, reason=reason):
        daemon_store.emit(event)
    return len(items)


NEWS_NAMED = 3  # judged news items spelled out in an acting proposal's reasons
NEWS_DAYS = 7  # the news agent's window (news.judge.JUDGE_WINDOW_DAYS)

# (symbol, session) whose news was pulled for a rationale this process.
_NEWS_PULLED: set[tuple[str, date]] = set()

_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def judged_news(db_path, symbol: str, now: datetime) -> list:
    """The news agent's judgments about the company itself, most material and newest first."""
    import sqlite3
    from datetime import timedelta

    from advisor.news.judge import NewsJudgmentStore

    conn = sqlite3.connect(str(db_path))
    try:
        rows = NewsJudgmentStore(conn).list(symbol=symbol, since=now - timedelta(days=NEWS_DAYS))
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    rows = [j for j in rows if j.about_company]
    rows.sort(key=lambda j: j.published_at, reverse=True)
    rows.sort(key=lambda j: _ORDER.get(j.materiality.value, 3))
    return rows


def news_rationale(daemon_store, symbol: str, now: datetime, *, puller=None) -> list[Reason]:
    """What the news says about ``symbol``, as reasons with their sources. Never raises.

    User decision (2026-09-28): an action carries its full rationale — an
    ADD on MDB the day its CEO left for Meta showed a P/S line and "1 event".
    The action is not changed here; the rationale is completed. When the
    name has no judged news this week its news is pulled and judged first
    (once per name per session), and the absence of news is said, not left
    blank.
    """
    from advisor.entry.proposal import Reason

    db_path = daemon_store.db_path
    try:
        news = judged_news(db_path, symbol, now)
        pulled = None
        key = (symbol, now.date())
        if not news and key not in _NEWS_PULLED:
            _NEWS_PULLED.add(key)
            if puller is None:
                from advisor.news.agent_run import run_news_agent as puller
            pulled = puller(db_path, symbol, now)
            news = judged_news(db_path, symbol, now)
    except Exception as exc:  # noqa: BLE001
        logger.warning("news rationale failed for %s: %s", symbol, exc)
        return [Reason(text=f"news not read: {exc}", source="news agent")]
    if not news:
        found = (pulled or {}).get("pulled")
        text = f"no news about {symbol} judged in the last {NEWS_DAYS} days" + (
            f" ({found} item(s) found, none judged about the company)" if found else ""
        )
        return [Reason(text=text, source="news agent (Tavily, Yahoo; dates checked)")]
    out = []
    for j in news[:NEWS_NAMED]:
        what = j.what or j.why or ""
        out.append(
            Reason(
                text=(
                    f"News {j.published_at.date().isoformat()}, {j.direction.value.lower()} "
                    f"{j.materiality.value.lower()}: {j.title}" + (f" — {what}" if what else "")
                )[:400],
                source=f"{j.provider}" + (f" {j.url}" if j.url else "") + " (news agent)",
            )
        )
    if len(news) > NEWS_NAMED:
        out.append(
            Reason(
                text=f"and {len(news) - NEWS_NAMED} more judged item(s) this week",
                source="news agent",
            )
        )
    return out


def default_news(daemon_store, symbol: str, now: datetime) -> list[Reason]:
    """The news reasons of a live run (replaced in tests: it searches and asks a model)."""
    return news_rationale(daemon_store, symbol, now)


def propose_all(
    daemon_store,
    now: datetime,
    *,
    symbols: list[str] | None = None,
    entry_store=None,
    scanner_store=None,
    read: bool = True,
    reader=None,
    sheet_builder=build_sheet,
    explainer=None,
    params=None,
    shadow=None,
    news=None,
) -> tuple[list[Proposal], list[str]]:
    """Proposals for ``symbols`` (default: held + watchlists). Returns (proposals, errors).

    ``params`` are the entry thresholds in force (default: the code's).
    ``shadow(sheet, net_liq, reading)`` is called with each final sheet so a
    challenger can decide on exactly what the live rules saw, reading
    included; it may not raise into the live run.
    """
    errors: list[str] = []
    book = daemon_store.load_latest_book()
    if symbols is None:
        if book is None:
            return [], ["no book snapshot stored"]
        from advisor.daemon.universe import research_symbols

        symbols, errors = research_symbols(book)
    net_liq = book.net_liq if book is not None else None

    if reader is None:
        from advisor.story.reading import read_symbol

        def reader(sym):
            return read_symbol(daemon_store, sym)

    if explainer is None:

        def explainer(sym, reason):
            return _explain(daemon_store, sym, reason)

    out = []
    for symbol in symbols:
        try:
            sheet = sheet_builder(daemon_store, symbol, now, scanner_store=scanner_store)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: sheet failed: {exc}")
            continue
        proposal = build_proposal(sheet, net_liq=net_liq, params=params)
        # An action to take carries what the news says, read before the model
        # reads the facts so the reading sees it too.
        news_reasons: list[Reason] = []
        if read and proposal.action in WORTH_READING:
            try:
                news_reasons = (news or default_news)(daemon_store, symbol, now)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: news rationale failed: {exc}")
        used_reading = None
        if read and (sheet.changed or proposal.action in WORTH_READING):
            try:
                reading = reader(symbol)
            except Exception as exc:  # noqa: BLE001
                reading = None
                errors.append(f"{symbol}: reading failed: {exc}")
            status = getattr(getattr(reading, "status", None), "value", None)
            key = (symbol, now.date())
            if status == "NO_FACTS" and proposal.action in WORTH_READING and key not in _EXPLAINED:
                _EXPLAINED.add(key)
                try:
                    found = explainer(symbol, explain_reason(sheet))
                    if found:
                        reading = reader(symbol)
                        status = getattr(getattr(reading, "status", None), "value", None)
                    else:
                        proposal.gaps.append("no news found to explain the move")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{symbol}: news pull failed: {exc}")
            if reading is not None and status == "OK":
                used_reading = reading
                proposal = build_proposal(sheet, net_liq=net_liq, reading=reading, params=params)
            elif reading is not None:
                proposal.gaps.append(f"reading {status or 'unavailable'}")
        if news_reasons:
            proposal.reasons.extend(news_reasons)
        if entry_store is not None:
            try:
                entry_store.add(proposal)
            except ValueError as exc:
                errors.append(str(exc))
        if shadow is not None:
            try:
                shadow(sheet, net_liq, used_reading)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: shadow failed: {exc}")
        out.append(proposal)
    return out, errors
