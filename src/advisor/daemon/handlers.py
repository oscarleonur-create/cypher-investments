"""The three scheduled jobs.

Phase 1 wires the schedule, not the intelligence: each handler establishes its
context and records what it *would* ingest. Phase 2 fills the ingest calls in
behind the same signatures, so the schedule never has to change.
"""

from __future__ import annotations

import logging

from advisor.daemon import market_calendar as mc
from advisor.daemon.ingest import ingest_book
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

    result = await ingest_book(ctx.store, include_standing=True)

    # Filings first: they are free, authoritative, and often explain whatever
    # the position and macro pillars are about to report.
    from advisor.news.ingest import ingest_filings

    book = ctx.store.load_latest_book()
    filings = await ingest_filings(ctx.store, book) if book else None
    if filings and filings.events:
        detail = (
            f"{result.summary()}; {len(filings.events)} filing event(s), "
            f"{filings.interrupts} interrupt(s); {detail}"
        )
    else:
        detail = f"{result.summary()}; no new filings; {detail}"
    logger.info("brief: %s", detail)
    return JobResult(job="brief", ok=True, detail=detail, events_emitted=result.new)


async def run_watch(ctx: JobContext) -> JobResult:
    """Intraday sweep: cheap deterministic checks, silent unless actionable.

    Silence is the expected outcome. This job only speaks when an event clears
    the relevance gate at Tier A.
    """
    # Crossings only: standing conditions belong in the digests, not in a
    # sweep that runs every 15 minutes.
    result = await ingest_book(ctx.store, include_standing=False)
    session_close = mc.session_close(ctx.now.date())
    detail = f"{result.summary()}; close {session_close.strftime('%H:%M')} ET"
    logger.debug("watch: %s", detail)
    return JobResult(job="watch", ok=True, detail=detail, events_emitted=result.new)


async def run_review(ctx: JobContext) -> JobResult:
    """Post-close: what moved, why, thesis status, tomorrow's calendar."""
    since = mc.previous_trading_day(ctx.now.date())
    result = await ingest_book(ctx.store, include_standing=True)

    # Pull news to explain what the free detectors already found, rather than
    # polling in the hope of finding something. The query is the question we
    # actually have, and a credit is only spent when something happened.
    explained = await _explain_todays_movers(ctx)
    angles = _scan_book_angles(ctx)
    detail = f"{result.summary()}; prior session {since.isoformat()}"
    if explained:
        detail += f"; explained {', '.join(explained)}"
    if angles:
        detail += f"; {angles}"
    logger.info("review: %s", detail)
    return JobResult(job="review", ok=True, detail=detail, events_emitted=result.new)


def _scan_book_angles(ctx: JobContext) -> str:
    """Search the holder's confirmed angles, largest position first. Never raises.

    The one scheduled news pull in the daemon; see ``news/angles.py`` for why
    it exists and the budget that bounds it.
    """
    from advisor.news.angles import scan_angles

    try:
        book = ctx.store.load_latest_book()
        if book is None or not book.net_liq:
            return ""
        weight: dict[str, float] = {}
        for position in book.positions:
            key = position.underlying.upper()
            weight[key] = weight.get(key, 0.0) + abs(position.signed_notional)
        symbols = sorted(weight, key=weight.get, reverse=True)
        scan = scan_angles(ctx.store, symbols, company_names={s: _company_name(s) for s in symbols})
        for problem in scan.errors:
            logger.warning("review: angle search failed: %s", problem)
        if not scan.queries:
            return ""
        note = f"angles: {scan.queries} queries, {len(scan.events)} new"
        if scan.skipped_for_budget:
            note += f", {len(scan.skipped_for_budget)} over budget"
        return note
    except Exception as exc:  # noqa: BLE001
        logger.warning("review: angle scan failed: %s", exc)
        return ""


# Event kinds that mean "something happened to this name that I cannot
# explain from price and factors alone" — the ones worth spending a query on.
EXPLAINABLE_KINDS = frozenset({"RESIDUAL_DIVERGENCE", "STOP_BREACHED", "PROFIT_TARGET_HIT"})


async def _explain_todays_movers(ctx: JobContext) -> list[str]:
    """Fetch and archive news for each symbol that fired today. Never raises."""
    from advisor.news.ingest import context_events, explain_symbol

    explained: list[str] = []
    try:
        today = ctx.now.date()
        seen: set[str] = set()
        for event in ctx.store.recent_events(limit=200):
            if event.kind not in EXPLAINABLE_KINDS or not event.symbol:
                continue
            if event.ts.date() != today or event.symbol in seen:
                continue
            seen.add(event.symbol)
            items = await explain_symbol(
                ctx.store, event.symbol, reason=event.kind, company_name=_company_name(event.symbol)
            )
            reason = event.kind
            for context in context_events(items, reason=reason):
                ctx.store.emit(context)
            if items:
                explained.append(f"{event.symbol}({len(items)})")
    except Exception as exc:  # noqa: BLE001
        logger.warning("review: news explanation failed: %s", exc)
    return explained


async def run_insiders(ctx: JobContext) -> JobResult:
    """Weekly: who inside the company has been buying or selling on the market.

    Weekly because each Form 4 is a fetch and an XML parse, and a book of a
    dozen names is several hundred round trips — enough to time out a job
    that has to finish before the open. The pattern it looks for develops
    over weeks anyway.
    """
    from advisor.daemon.book import fetch_book
    from advisor.news.ingest import scan_insiders

    try:
        book = await fetch_book()
    except Exception as exc:  # noqa: BLE001
        return JobResult(job="insiders", ok=False, detail=f"book unavailable: {exc}")

    events = await scan_insiders(ctx.store, book.symbols)
    detail = (
        ", ".join(
            f"{e.symbol} {e.payload['insider_count']} {e.payload['side'].lower()}" for e in events
        )
        if events
        else "no insider clusters"
    )
    logger.info("insiders: %s", detail)
    return JobResult(job="insiders", ok=True, detail=detail, events_emitted=len(events))


async def run_valuation(ctx: JobContext) -> JobResult:
    """Recompute what each holding's price requires the business to deliver.

    Weekly rather than daily: the inputs are a quarterly filing and a price,
    and the filing only moves four times a year. A daily refit would burn an
    XBRL parse per symbol to produce the same answer, and the shift detector
    is edge-triggered anyway.
    """
    from advisor.daemon.book import fetch_book
    from advisor.valuation.ingest import refresh_valuations

    try:
        book = await fetch_book()
    except Exception as exc:  # noqa: BLE001
        return JobResult(job="valuation", ok=False, detail=f"book unavailable: {exc}")

    prices = {p.underlying.upper(): p.price for p in book.positions if p.price}
    result = await refresh_valuations(ctx.store, book.symbols, prices)

    detail = result.summary()
    if result.events:
        moves = ", ".join(
            f"{e.symbol} {e.payload['previous_implied_cagr']:.1%}->{e.payload['implied_cagr']:.1%}"
            for e in result.events[:3]
        )
        detail += f" — {moves}"
    logger.info("valuation: %s", detail)
    return JobResult(job="valuation", ok=True, detail=detail, events_emitted=len(result.events))


def _company_name(symbol: str) -> str | None:
    """Registered name for a symbol, cached per process. None when unresolved."""
    if symbol not in _NAME_CACHE:
        from advisor.news.entities import company_name_for

        _NAME_CACHE[symbol] = company_name_for(symbol)
    return _NAME_CACHE[symbol]


_NAME_CACHE: dict[str, str | None] = {}


async def run_reconcile(ctx: JobContext) -> JobResult:
    """Daily: check every input against an independent source.

    A green test suite has never proved the inputs are right. This job is the
    only thing in the daemon that can say "I no longer trust my own data",
    and a failure is a Tier A event because advising on bad data is worse
    than not advising.
    """
    from advisor.daemon.book import fetch_book
    from advisor.daemon.reconcile import reconcile_events, run_reconciliation

    try:
        book = await fetch_book()
    except Exception as exc:  # noqa: BLE001
        return JobResult(job="reconcile", ok=False, detail=f"book unavailable: {exc}")

    report = await run_reconciliation(book, today=ctx.now.date())
    emitted = 0
    for event in reconcile_events(report):
        if ctx.store.emit(event):
            emitted += 1

    detail = report.summary()
    if report.failures:
        detail += " — " + "; ".join(
            f"{f.check}/{f.symbol}: {f.detail}" for f in report.failures[:3]
        )
    logger.info("reconcile: %s", detail)
    return JobResult(job="reconcile", ok=True, detail=detail, events_emitted=emitted)


async def run_macro_refresh(ctx: JobContext) -> JobResult:
    """Weekly: re-estimate factor loadings and rebuild the book exposure.

    Slow-moving by construction — a 250-day window barely shifts in five
    sessions — so this pays a batch download and a regression per symbol once a
    week rather than every morning.
    """
    from advisor.daemon.book import fetch_book
    from advisor.daemon.macro_ingest import refresh_sensitivities

    try:
        book = await fetch_book()
    except Exception as exc:  # noqa: BLE001
        return JobResult(job="macro_refresh", ok=False, detail=f"book unavailable: {exc}")

    exposure, skipped = refresh_sensitivities(ctx.store, book)
    if exposure is None:
        return JobResult(job="macro_refresh", ok=True, detail="nothing held to estimate")

    top = exposure.ranked()[:3]
    profile = ", ".join(f"{f.factor} {f.net_loading:+.2f}" for f in top)
    detail = f"{exposure.covered_weight:.0%} of notional covered; {profile}"
    if skipped:
        detail += f"; no estimate for {', '.join(skipped)}"
    logger.info("macro_refresh: %s", detail)
    return JobResult(job="macro_refresh", ok=True, detail=detail)


async def run_setup_scan(ctx: JobContext) -> JobResult:
    """Record setup A/C candidates. Measurement only: emits no events.

    Nothing here reaches the event stream, the relevance gate or a message.
    A setup earns an alert only after its recorded outcomes say it works.
    """
    import asyncio

    from advisor.scanner.scan import run_scan

    # yfinance, Tavily and EDGAR are blocking clients; run them off the loop.
    # The connection must be opened inside that worker thread — sqlite3
    # refuses a connection used from a thread other than its creator's.
    result = await asyncio.to_thread(_with_scanner_store, ctx.store.db_path, run_scan, ctx.now)
    ok = result.source_error is None or result.movers_seen > 0
    logger.info("scan: %s", result.summary())
    return JobResult(job="scan", ok=ok, detail=result.summary())


async def run_premarket_scan(ctx: JobContext) -> JobResult:
    """Before the bell: setups A, B and C on the TastyTrade "Swing" watchlist.

    Measurement only, like the session scan: nothing reaches the event stream.
    """
    import asyncio

    from advisor.scanner.premarket import scan_watchlist

    result = await asyncio.to_thread(
        _with_scanner_store, ctx.store.db_path, scan_watchlist, ctx.now
    )
    # A missing watchlist is a failure worth seeing in `daemon status`; an
    # empty morning is not.
    ok = result.error is None
    logger.info("premarket_scan: %s", result.summary())
    return JobResult(job="premarket_scan", ok=ok, detail=result.summary())


async def run_scan_outcomes(ctx: JobContext) -> JobResult:
    """After the close: fill what each candidate's price did, and which were taken.

    Fills are synced for the last three sessions, not just today's, so a
    laptop asleep through one 16:20 slot still catches up the next day.
    """
    import asyncio

    from advisor.daemon import market_calendar as mc
    from advisor.scanner.journal import sync_fills
    from advisor.scanner.outcomes import fill_outcomes

    result = await asyncio.to_thread(_with_scanner_store, ctx.store.db_path, fill_outcomes, ctx.now)
    today = ctx.now.date()
    sessions = [today] if mc.is_trading_day(today) else []
    cursor = today
    while len(sessions) < 3:
        cursor = mc.previous_trading_day(cursor)
        sessions.append(cursor)
    taken = await asyncio.to_thread(
        _with_scanner_store,
        ctx.store.db_path,
        lambda store, _now: sync_fills(store, sessions),
        ctx.now,
    )
    detail = f"{result.summary()}; {taken} newly taken from broker fills"
    logger.info("scan_outcomes: %s", detail)
    return JobResult(job="scan_outcomes", ok=True, detail=detail)


def _with_scanner_store(db_path, fn, now):
    """Open, use and close a ScannerStore entirely within the calling thread."""
    from advisor.scanner.store import ScannerStore

    store = ScannerStore(db_path)
    try:
        return fn(store, now)
    finally:
        store.close()


async def run_heartbeat(ctx: JobContext) -> JobResult:
    """Liveness tick — proves the supervisor loop is running between jobs.

    Advances the DAEMON watermark so `daemon status` can distinguish "the
    daemon is up and quiet" from "the daemon died three hours ago".
    """
    ctx.store.set_watermark(EventSource.DAEMON, last_seen_ts=ctx.now)
    return JobResult(job="heartbeat", ok=True, detail="alive")
