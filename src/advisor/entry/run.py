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

from advisor.entry.proposal import Action, Proposal, build_proposal
from advisor.entry.sheet import build_sheet

logger = logging.getLogger(__name__)

WORTH_READING = {Action.ENTER, Action.ADD, Action.WAIT}
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
) -> tuple[list[Proposal], list[str]]:
    """Proposals for ``symbols`` (default: held + watchlists). Returns (proposals, errors)."""
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
        proposal = build_proposal(sheet, net_liq=net_liq)
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
                proposal = build_proposal(sheet, net_liq=net_liq, reading=reading)
            elif reading is not None:
                proposal.gaps.append(f"reading {status or 'unavailable'}")
        if entry_store is not None:
            entry_store.add(proposal)
        out.append(proposal)
    return out, errors
