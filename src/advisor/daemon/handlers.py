"""The three scheduled jobs.

Phase 1 wires the schedule, not the intelligence: each handler establishes its
context and records what it *would* ingest. Phase 2 fills the ingest calls in
behind the same signatures, so the schedule never has to change.
"""

from __future__ import annotations

import logging

from advisor.daemon import market_calendar as mc
from advisor.daemon.jobs import JobContext
from advisor.daemon.models import EventSource, JobResult

logger = logging.getLogger(__name__)


async def run_brief(ctx: JobContext) -> JobResult:
    """Pre-open: backfill everything missed since the last run, then report.

    This is the job that makes laptop hosting viable. It resumes each source
    from its watermark rather than from "now", so an 8-K filed at 16:40 while
    the machine slept still lands in the morning brief.
    """
    gap_from = min(
        (w.last_seen_ts for w in ctx.store.all_watermarks() if w.last_seen_ts),
        default=None,
    )
    detail = (
        f"backfill window from {gap_from.isoformat()}"
        if gap_from
        else "no watermarks yet — first run, nothing to backfill"
    )
    if ctx.catch_up:
        detail += " (catch-up: fired late for a missed slot)"

    # Phase 2: broker delta, EDGAR filings, overnight gaps, today's calendar,
    # positions entering a roll window, thesis invalidations tripped overnight.
    logger.info("brief: %s", detail)
    return JobResult(job="brief", ok=True, detail=detail)


async def run_watch(ctx: JobContext) -> JobResult:
    """Intraday sweep: cheap deterministic checks, silent unless actionable.

    Silence is the expected outcome. This job only speaks when an event clears
    the relevance gate at Tier A.
    """
    # Phase 2: position mechanics (DTE, strike breach, assignment risk, stops),
    # factor shocks against the book's exposure vector.
    session_close = mc.session_close(ctx.now.date())
    detail = f"session sweep, close {session_close.strftime('%H:%M')} ET"
    logger.debug("watch: %s", detail)
    return JobResult(job="watch", ok=True, detail=detail)


async def run_review(ctx: JobContext) -> JobResult:
    """Post-close: what moved, why, thesis status, tomorrow's calendar."""
    since = mc.previous_trading_day(ctx.now.date())
    # Phase 2: P&L attribution, per-position thesis status, next-day calendar.
    detail = f"post-close review, prior session {since.isoformat()}"
    logger.info("review: %s", detail)
    return JobResult(job="review", ok=True, detail=detail)


async def run_heartbeat(ctx: JobContext) -> JobResult:
    """Liveness tick — proves the supervisor loop is running between jobs.

    Advances the DAEMON watermark so `daemon status` can distinguish "the
    daemon is up and quiet" from "the daemon died three hours ago".
    """
    ctx.store.set_watermark(EventSource.DAEMON, last_seen_ts=ctx.now)
    return JobResult(job="heartbeat", ok=True, detail="alive")
