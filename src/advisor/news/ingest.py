"""Turning source items into events, under the tier ceiling.

The shape of this layer follows one decision: **news is pulled to explain
something, not polled hoping to find something.**

The free detectors already built — filings, stop crossings, residual
divergence — are what fire. Only then is a paid query spent, and the query it
produces ("why did AAOI fall 14% on 2026-08-24") is far better than the one
polling would have produced ("AAOI news"). That keeps Tavily around 70 credits
a month instead of 330, and it means every news item in the stream is attached
to something that actually happened.

EDGAR is the exception: it is free and authoritative, so it is polled directly
against a watermark.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import timedelta

from advisor.daemon.book import BookSnapshot
from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.news.classify import (
    Classification,
    FilingKind,
    Materiality,
    classify_delisting,
    classify_filing,
)
from advisor.news.enrich import offering_size_for
from advisor.news.models import SourceItem, SourceTier, capped_tier
from advisor.news.offering import offering_shape_for

logger = logging.getLogger(__name__)

# A filing this old is history, not news. Guards the case where a watermark is
# missing and EDGAR hands back a fortnight of filings at once: everything is
# archived, but only recent filings become events. Without the guard, a first
# run would fire a dozen interrupts about things that resolved weeks ago.
MAX_FILING_AGE_DAYS = 5

# How far back to look the first time a symbol is seen. A position opened
# today deserves the context of its recent filings — AMD had eight since July,
# none of them archived, because a single global watermark had moved past them
# on another symbol's behalf.
FIRST_LOOK_DAYS = 120

# Dilution below this share of market cap is real but not worth an interrupt.
MATERIAL_DILUTION_PCT = 0.02

_TIER_FOR_MATERIALITY = {
    Materiality.HIGH: EventTier.A,
    Materiality.MEDIUM: EventTier.B,
    Materiality.LOW: EventTier.C,
}


@dataclass
class NewsIngestResult:
    items_seen: int = 0
    items_stored: int = 0
    events: list[Event] = field(default_factory=list)
    symbols_covered: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def interrupts(self) -> int:
        return sum(1 for e in self.events if e.tier == EventTier.A)


def _event_for_filing(item: SourceItem, *, market_caps: dict[str, float]) -> Event | None:
    """One filing to at most one event, with the tier its source permits."""
    # A 25-NSE is graded by the class it removes, which the EDGAR adapter
    # captured into `summary`. Everything else is graded by form and items.
    if (item.doc_type or "").upper() == "25-NSE":
        classification = classify_delisting(item.summary)
    else:
        classification = classify_filing(item.doc_type or "", item.item_codes)
    proposed = _TIER_FOR_MATERIALITY[classification.materiality]
    payload: dict = {
        "form": item.doc_type,
        "items": item.item_codes,
        "kind": classification.kind.value,
        "label": classification.label,
        "url": item.url,
        "accession": item.accession,
        "accepted_at": item.published_at.isoformat(),
        "provider": item.provider,
        "match": item.entity.method.value,
    }

    # A 424B does not say what it is offering; the cover page does. AMD's
    # August supplement sold $4.75bn of senior notes — leverage, with not one
    # share issued — and calling that dilution reported the wrong fact.
    if classification.kind is FilingKind.DILUTION and item.accession:
        shape = offering_shape_for(item.accession)
        if shape is not None:
            payload["security_type"] = shape.security.value
            payload["preliminary"] = shape.preliminary
            if not shape.dilutive:
                classification = Classification(
                    FilingKind.DEBT_ISSUANCE,
                    Materiality.MEDIUM,
                    "debt offering — leverage, not dilution",
                )
                payload["kind"] = classification.kind.value
                payload["label"] = classification.label
                proposed = EventTier.B
            if shape.preliminary:
                # An unpriced preliminary is an intention, not a completed
                # deal; the pricing supplement that follows is the event.
                payload["label"] = f"{payload['label']} (preliminary, not yet priced)"
                proposed = EventTier.B if proposed is EventTier.A else proposed

    # Size every offering, debt included: $4.75bn of new notes is material
    # information about leverage even though it dilutes nobody. Only the
    # *dilution* percentage is reserved for offerings that issue shares.
    if classification.kind in (FilingKind.DILUTION, FilingKind.DEBT_ISSUANCE) and item.accession:
        size = offering_size_for(item.accession)
        if size is not None:
            payload["offering_usd"] = size.amount_usd
            payload["quote"] = size.quote
            cap = market_caps.get(item.entity.symbol)
            if cap:
                payload["market_cap"] = cap
                pct = size.dilution_pct(cap)
                if pct is not None:
                    key = (
                        "dilution_pct"
                        if classification.kind is FilingKind.DILUTION
                        else "offering_pct_of_cap"
                    )
                    payload[key] = round(pct, 4)
                    if classification.kind is FilingKind.DILUTION and pct < MATERIAL_DILUTION_PCT:
                        proposed = EventTier.B

    return Event(
        source=EventSource.EDGAR,
        kind=f"FILING_{classification.kind.value}",
        tier=capped_tier(item.tier, proposed),
        symbol=item.entity.symbol,
        dedup_key=item.dedup_key(),
        payload=payload,
    )


async def ingest_filings(
    store: DaemonStore,
    book: BookSnapshot,
    *,
    symbols: list[str] | None = None,
) -> NewsIngestResult:
    """Poll EDGAR for each held symbol since the watermark, and emit events."""
    from advisor.news.edgar import recent_filings

    result = NewsIngestResult()
    watched = symbols or book.symbols
    if not watched:
        return result

    cutoff = now_et() - timedelta(days=MAX_FILING_AGE_DAYS)
    first_look = now_et() - timedelta(days=FIRST_LOOK_DAYS)
    market_caps = await _market_caps(watched)

    newest = store.get_watermark(EventSource.EDGAR).last_seen_ts
    for symbol in watched:
        # Per symbol, from what that symbol's archive already holds. The
        # global watermark is kept only as a coarse liveness marker.
        since = store.latest_source_item_at(symbol) or first_look
        try:
            items = recent_filings(symbol, since=since)
        except Exception as exc:  # noqa: BLE001
            logger.warning("news: EDGAR failed for %s: %s", symbol, exc)
            result.errors.append(f"{symbol}: {exc}")
            continue

        result.symbols_covered.append(symbol)
        for item in items:
            result.items_seen += 1
            if store.save_source_item(item):
                result.items_stored += 1
            if newest is None or item.published_at > newest:
                newest = item.published_at
            # Archive everything; only recent filings become events.
            if item.published_at < cutoff:
                continue
            event = _event_for_filing(item, market_caps=market_caps)
            if event and store.emit(event):
                result.events.append(event)

    if newest is not None:
        store.set_watermark(EventSource.EDGAR, last_seen_ts=newest)
    return result


async def _market_caps(symbols: list[str]) -> dict[str, float]:
    """Market caps from the broker, for sizing dilution. Empty on failure."""
    try:
        from tastytrade.metrics import get_market_metrics

        from advisor.api import deps

        session = await deps.get_tt_session()
        return {
            m.symbol.upper(): float(m.market_cap)
            for m in await get_market_metrics(session, symbols)
            if m.market_cap
        }
    except Exception as exc:  # noqa: BLE001
        logger.info("news: market caps unavailable, dilution will be unsized: %s", exc)
        return {}


# Search engines are not asked questions, they are given terms. Measured on
# the AAOI offering, against Tavily's own relevance score:
#
#   "Applied Optoelectronics AAOI stock drop"          top 0.924, 5/5 kept
#   "Applied Optoelectronics AAOI"                     top 0.871, 5/5 kept
#   full sentence describing the move                  top 0.393, 0/5 kept
#
# So the event kind contributes a couple of keywords, never a sentence. The
# readable reason still travels with the event; it just stays out of the query.
REASON_KEYWORDS: dict[str, str] = {
    "RESIDUAL_DIVERGENCE": "stock move",
    "STOP_BREACHED": "stock drop",
    "PROFIT_TARGET_HIT": "stock rally",
    "FILING_DILUTION": "equity offering",
    "FILING_RESULTS": "earnings results",
    "FILING_MANAGEMENT_CHANGE": "executive",
    "FILING_ACTIVIST_STAKE": "activist stake",
}
DEFAULT_KEYWORDS = "stock news"


async def explain_symbol(
    store: DaemonStore,
    symbol: str,
    *,
    reason: str,
    company_name: str | None = None,
    days: int = 7,
) -> list[SourceItem]:
    """Pull news to explain something that already fired, and archive it.

    ``reason`` is the event kind that triggered the lookup. It selects the
    search keywords and is recorded on the resulting context events, so an
    item in the stream always says what it was fetched to explain.
    """
    from advisor.news.tavily import search_news
    from advisor.news.yahoo_feed import recent_context

    keywords = REASON_KEYWORDS.get(reason.upper(), DEFAULT_KEYWORDS)
    query = " ".join(part for part in (company_name, symbol, keywords) if part)
    items: list[SourceItem] = []
    try:
        items.extend(search_news(symbol, query, company_name=company_name, days=days))
    except Exception as exc:  # noqa: BLE001
        logger.warning("news: tavily lookup failed for %s: %s", symbol, exc)
    try:
        items.extend(recent_context(symbol, company_name=company_name, max_age_hours=days * 24))
    except Exception as exc:  # noqa: BLE001
        logger.warning("news: yfinance context failed for %s: %s", symbol, exc)

    for item in items:
        store.save_source_item(item)
    items.sort(key=lambda i: i.published_at, reverse=True)
    return items


def context_events(items: list[SourceItem], *, reason: str) -> list[Event]:
    """Tier C context rows so an explanation is visible beside its trigger."""
    events = []
    for item in items:
        events.append(
            Event(
                source=EventSource.YFINANCE
                if item.tier is SourceTier.UNTAGGED
                else EventSource.CALENDAR,
                kind="NEWS_CONTEXT",
                tier=capped_tier(item.tier, EventTier.C),
                symbol=item.entity.symbol,
                dedup_key=item.dedup_key(),
                payload={
                    "title": item.title,
                    "url": item.url,
                    "provider": item.provider,
                    "published_at": item.published_at.isoformat(),
                    "match": item.entity.method.value,
                    "confidence": item.entity.confidence,
                    "explains": reason,
                },
            )
        )
    return events


@dataclass
class CoverageReport:
    """How often a name's unexplained move had a filing to explain it.

    The measurement needs no hand-labelling, which is what makes it worth
    having. Residual divergence is an *independent* detector of "something
    company-specific happened", built before any of this and knowing nothing
    about news. So asking how many divergences had a primary-source item
    within a session of them scores the ingest layer objectively.

    A low number is not automatically a failure — plenty of real moves have no
    filing behind them — but a number that does not improve when a source is
    added means the source is not earning its place.
    """

    divergences: int = 0
    explained: int = 0
    by_symbol: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return self.explained / self.divergences if self.divergences else 0.0


def coverage(
    store: DaemonStore,
    *,
    window_days: int = 90,
    tolerance_days: int = 1,
    kinds: frozenset[str] = frozenset({"RESIDUAL_DIVERGENCE"}),
    tiers: frozenset[SourceTier] = frozenset({SourceTier.PRIMARY, SourceTier.AGGREGATOR}),
) -> CoverageReport:
    """Score the ingest layer against divergences already in the event stream."""
    report = CoverageReport()
    cutoff = now_et() - timedelta(days=window_days)

    for event in store.recent_events(limit=2000):
        if event.kind not in kinds or not event.symbol or event.ts < cutoff:
            continue
        report.divergences += 1
        window = timedelta(days=tolerance_days)
        items = store.source_items_between(event.symbol, event.ts - window, event.ts + window)
        hit = any(i.tier in tiers for i in items)
        report.explained += int(hit)
        seen, found = report.by_symbol.get(event.symbol, (0, 0))
        report.by_symbol[event.symbol] = (seen + 1, found + int(hit))

    return report
