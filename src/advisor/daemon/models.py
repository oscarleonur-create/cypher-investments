"""Core daemon types: events, watermarks, and job outcomes.

An ``Event`` is the single normalized fact the whole system runs on. Every
ingest source — broker, EDGAR, yfinance, computed position mechanics, macro
factors — emits Events into one append-only stream, and every downstream layer
(relevance gate, thesis evaluation, Action Cards) reads from that stream rather
than from the sources directly.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et


class EventTier(StrEnum):
    """How loudly an event is allowed to speak.

    The tier is assigned at emit time from the event kind, and it decides
    delivery: A interrupts, B waits for the next digest, C is never delivered
    and exists only as context and as scoring material.
    """

    A = "A"  # actionable with a deadline — interrupt
    B = "B"  # material, no deadline — next digest
    C = "C"  # context only — logged, never delivered


class EventSource(StrEnum):
    """Where a fact came from. Watermarks are tracked per source."""

    BROKER = "broker"  # TastyTrade positions, balances, transactions
    EDGAR = "edgar"  # SEC filings
    YFINANCE = "yfinance"  # prices, headlines, earnings dates
    COMPUTED = "computed"  # derived from positions (DTE, strike breach, drift)
    MACRO = "macro"  # factor moves, regime, book exposure
    CALENDAR = "calendar"  # scheduled macro prints, earnings, ex-div
    DAEMON = "daemon"  # the daemon's own lifecycle


class Event(BaseModel):
    """One normalized fact, ready for the relevance gate."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    ts: datetime = Field(default_factory=now_et)
    source: EventSource
    kind: str  # e.g. ROLL_WINDOW_OPEN, FACTOR_SHOCK_HITTING_BOOK
    tier: EventTier
    symbol: str | None = None  # None for book-level and macro-wide events
    payload: dict = Field(default_factory=dict)
    dedup_key: str = ""  # emitter-supplied identity, hashed into dedup_hash

    def dedup_hash(self) -> str:
        """Stable identity for this fact, so re-ingesting it is a no-op.

        Deliberately excludes ``ts`` and ``id``: the same 8-K seen on two polls
        is one event, and the emitter decides via ``dedup_key`` what makes two
        occurrences distinct (a filing accession number, a DTE threshold, the
        session date).
        """
        raw = "|".join([self.source.value, self.kind, (self.symbol or "").upper(), self.dedup_key])
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


class Watermark(BaseModel):
    """How far a source has been consumed.

    The laptop sleeps, so the daemon cannot assume it saw everything as it
    happened. On start each source resumes from its watermark and backfills,
    which is what makes "here is what you missed overnight" possible.
    """

    source: EventSource
    last_seen_ts: datetime | None = None
    last_seen_cursor: str = ""  # source-specific (accession number, txn id, ...)
    updated_at: datetime = Field(default_factory=now_et)


class JobResult(BaseModel):
    """Outcome of one job execution."""

    job: str
    ok: bool
    events_emitted: int = 0
    detail: str = ""
    ran_at: datetime = Field(default_factory=now_et)
    duration_ms: int = 0


class Heartbeat(BaseModel):
    """Durable proof a job is alive, and the record of its last failure."""

    job: str
    last_run_at: datetime | None = None
    last_ok_at: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    last_error: str = ""
