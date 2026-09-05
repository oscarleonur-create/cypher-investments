"""Book ingest: broker state in, events out.

One entry point used by every job, differing only in whether standing
conditions are included. Intraday sweeps want crossings only (something just
happened); the brief and review also want the standing picture (what is true
today).
"""

from __future__ import annotations

import logging

from advisor.daemon.book import BookSnapshot, fetch_book
from advisor.daemon.mechanics import (
    MechanicsLimits,
    crossing_events,
    diff_events,
    state_events,
)
from advisor.daemon.models import Event, EventSource
from advisor.daemon.store import DaemonStore

logger = logging.getLogger(__name__)


class IngestResult:
    """What one ingest pass produced."""

    def __init__(self, book: BookSnapshot | None, events: list[Event], new: int) -> None:
        self.book = book
        self.events = events
        self.new = new  # events that were not already in the stream

    def summary(self) -> str:
        if self.book is None:
            return "book unavailable"
        parts = [
            f"{len(self.book.positions)} positions",
            f"net liq {self.book.net_liq:,.0f}",
            f"{self.new} new events" if self.new else "no new events",
        ]
        if self.book.partial:
            parts.append("PARTIAL — an account failed to load")
        return ", ".join(parts)


async def ingest_book(
    store: DaemonStore,
    *,
    include_standing: bool = False,
    limits: MechanicsLimits | None = None,
) -> IngestResult:
    """Snapshot the book, derive events against the previous snapshot, persist.

    A partial snapshot still produces crossing and standing events for the
    accounts that did load, but is never saved as the new baseline — otherwise
    the next diff would report the missing account's positions as closed.
    """
    try:
        book = await fetch_book()
    except Exception as exc:  # noqa: BLE001
        logger.warning("ingest: book fetch failed: %s", exc)
        return IngestResult(None, [], 0)

    previous = store.load_latest_book()
    # A partial snapshot is diffed only over the accounts that actually
    # returned, so a broker hiccup on one account cannot report every position
    # in it as closed.
    if previous is not None and book.partial:
        previous = previous.restricted_to_loaded(book.loaded_accounts)

    events: list[Event] = list(diff_events(previous, book, limits=limits))
    if previous is not None:
        events.extend(crossing_events(previous, book, limits=limits))
    if include_standing:
        events.extend(state_events(book, limits=limits))

    new = store.emit_many(events)
    store.save_book(book)
    store.set_watermark(EventSource.BROKER, last_seen_ts=book.as_of)

    return IngestResult(book, events, new)
