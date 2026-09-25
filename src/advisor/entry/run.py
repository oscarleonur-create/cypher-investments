"""Sheets → proposals → (reading where it matters) → the ledger.

The model is expensive (10-20s a call) and its reading only matters where a
decision is on the table, so it is asked only for names whose sheet changed
today or whose proposal is to act (ENTER, ADD, WAIT). Quiet names get the
deterministic proposal and nothing else.
"""

from __future__ import annotations

import logging
from datetime import datetime

from advisor.entry.proposal import Action, Proposal, build_proposal
from advisor.entry.sheet import build_sheet

logger = logging.getLogger(__name__)

WORTH_READING = {Action.ENTER, Action.ADD, Action.WAIT}


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
            if reading is not None and status == "OK":
                proposal = build_proposal(sheet, net_liq=net_liq, reading=reading)
            elif reading is not None:
                proposal.gaps.append(f"reading {status or 'unavailable'}")
        if entry_store is not None:
            entry_store.add(proposal)
        out.append(proposal)
    return out, errors
