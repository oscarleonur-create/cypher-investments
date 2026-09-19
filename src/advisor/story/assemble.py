"""Building a story from stored rows.

Pure assembly: reads the daemon's own tables and a price history, and returns
a filled `Story`. No LLM, no broker call. If this module is the only thing
running, you still get the whole narrative — the model in phase 6 adds prose
over these slots and may introduce no fact of its own.

The subtlety worth reading is `pricing_session`. An event carries the instant
it happened, but the market only responds when it is next open. AAOI's $600m
offering was accepted at 16:09 ET on a Friday, nine minutes after the close:
Friday's -3.32% had nothing to do with it, and Monday's -13.77% had everything
to do with it. Attributing the move to the wrong session inverts the story,
which is exactly the mistake a hand-written timeline made before this existed.
"""

from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta

from advisor.daemon import market_calendar as mc
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.news.classify import Materiality
from advisor.news.models import SourceTier
from advisor.story.models import (
    Anchor,
    Attribution,
    Confidence,
    Corroboration,
    PositionAtEvent,
    PriceReaction,
    Story,
    ThesisLink,
    Verdict,
)

logger = logging.getLogger(__name__)

CORROBORATION_WINDOW_DAYS = 3

# The ladder that keeps language inside the statistic. Ordered by descending
# strength; the first threshold the |z| clears wins.
VERDICT_LADDER: tuple[tuple[float, Verdict], ...] = (
    (3.0, Verdict.NOT_MARKET),
    (2.0, Verdict.UNEXPLAINED),
    (1.0, Verdict.PARTLY_SPECIFIC),
    (0.0, Verdict.CONSISTENT),
)


def verdict_for(residual_z: float | None) -> Verdict:
    """Map a residual to language that does not outrun it.

    A NaN falls through every threshold comparison — `abs(nan) >= 3.0` is
    False, and so is every other — and lands on CONSISTENT, which states
    confidently that macro explains a move nobody could measure. NBIS produced
    exactly that: "consistent with what macro did that day" from a residual of
    nan. Unknown is the only honest reading of an unmeasurable residual.
    """
    if residual_z is None or math.isnan(residual_z) or math.isinf(residual_z):
        return Verdict.UNKNOWN
    magnitude = abs(residual_z)
    for threshold, verdict in VERDICT_LADDER:
        if magnitude >= threshold:
            return verdict
    return Verdict.CONSISTENT


def pricing_session(moment: datetime) -> date:
    """The trading session that first gets to react to ``moment``.

    An event during a session is priced that day. One after the close — or on
    a weekend, a holiday, or after an early close — is priced at the next
    open. This is the difference between blaming Friday and blaming Monday.
    """
    et = mc.to_et(moment)
    day = et.date()
    if mc.is_trading_day(day) and et.time() < mc.session_close(day):
        return day
    # Walk forward to the next day the market actually opens.
    candidate = day + timedelta(days=1)
    for _ in range(10):
        if mc.is_trading_day(candidate):
            return candidate
        candidate += timedelta(days=1)
    return candidate


def _anchor(event: Event) -> Anchor:
    payload = event.payload or {}
    occurred_raw = payload.get("accepted_at") or payload.get("published_at")
    occurred = event.ts
    if isinstance(occurred_raw, str):
        try:
            occurred = mc.to_et(datetime.fromisoformat(occurred_raw))
        except ValueError:
            logger.debug("story: unparseable occurred_at %r", occurred_raw)

    headline = str(payload.get("label") or event.kind.replace("_", " ").lower())
    facts = {
        k: v
        for k, v in payload.items()
        if k
        in {
            "form",
            "items",
            "offering_usd",
            "dilution_pct",
            "market_cap",
            "residual_z",
            "actual_return",
            "expected_return",
            "factor",
            "z",
            "expected_book_move",
            "weight",
            "unrealized_pct",
            "entry",
            "price",
            "unrealized_usd",
            "threshold",
            "realized_pct",
        }
    }
    return Anchor(
        event_id=event.id,
        kind=event.kind,
        tier=event.tier,
        symbol=(event.symbol or "").upper(),
        occurred_at=occurred,
        ingested_at=event.ts,
        headline=headline,
        source=str(payload.get("provider") or event.source.value),
        url=payload.get("url"),
        quote=payload.get("quote"),
        facts=facts,
    )


def _position(store: DaemonStore, symbol: str, occurred_at: datetime) -> PositionAtEvent:
    """The position held at the event, or an explicit account of why not."""
    snapshot = store.load_book_at(occurred_at)
    covers = snapshot is not None
    note = ""

    if snapshot is None:
        # Never silently substitute today's book for a past one. Say which
        # snapshot was used and that it postdates the event.
        snapshot = store.load_latest_book()
        earliest = store.earliest_book_snapshot_at()
        if snapshot is None:
            return PositionAtEvent(
                confidence=Confidence.UNAVAILABLE,
                note="no book snapshot exists — the daemon was not recording",
            )
        note = (
            f"no snapshot from the event date; showing the position as of "
            f"{mc.to_et(snapshot.as_of).date()}"
        )
        if earliest is not None:
            note += f" (records begin {mc.to_et(earliest).date()})"

    held = [p for p in snapshot.positions if p.underlying.upper() == symbol]
    if not held:
        return PositionAtEvent(
            confidence=Confidence.MEASURED if covers else Confidence.ESTIMATED,
            quantity=0.0,
            snapshot_asof=snapshot.as_of,
            covers_event=covers,
            net_liq=snapshot.net_liq,
            note=note or "not held in this snapshot",
        )

    quantity = sum(p.quantity for p in held)
    signed = sum(p.signed_notional for p in held)
    basis = sum(p.cost_basis for p in held)
    return PositionAtEvent(
        confidence=Confidence.MEASURED if covers else Confidence.ESTIMATED,
        quantity=quantity,
        avg_open_price=(basis / abs(quantity)) if quantity else None,
        price_then=held[0].price,
        weight_of_net_liq=(signed / snapshot.net_liq) if snapshot.net_liq else None,
        net_liq=snapshot.net_liq,
        snapshot_asof=snapshot.as_of,
        covers_event=covers,
        note=note,
    )


def _self_priced(anchor: Anchor, position: PositionAtEvent) -> PriceReaction | None:
    """The reaction for an event that already knows the price it fired at.

    "What session priced this?" is the right question for a filing, which
    lands after the close and is priced at the next open. It is the wrong
    question for a crossing computed from a live quote: that event *is* a
    price, and it carries the one that tripped it.

    Asking the session instead produced a flat contradiction. CBRS breached
    its stop at 10:30 on 18 September at 199.66, down 8.03% from entry — and
    the session closed up 2.18% on the previous day, so the story was headed
    "stop breached (+2.18%)".
    """
    facts = anchor.facts
    entry, price = facts.get("entry"), facts.get("price")
    move = facts.get("unrealized_pct")
    if not isinstance(price, (int, float)) or not isinstance(move, (int, float)):
        return None

    dollars = facts.get("unrealized_usd")
    net_liq = position.net_liq
    return PriceReaction(
        confidence=Confidence.MEASURED,
        session=mc.to_et(anchor.occurred_at).date(),
        before=entry if isinstance(entry, (int, float)) else None,
        after=price,
        pct_move=move,
        dollars=dollars if isinstance(dollars, (int, float)) else None,
        pct_of_book=(dollars / net_liq) if isinstance(dollars, (int, float)) and net_liq else None,
        self_priced=True,
    )


def _reaction(
    symbol: str,
    occurred_at: datetime,
    position: PositionAtEvent,
    prices,
) -> PriceReaction:
    """What the session that priced the event did to the position."""
    if prices is None or prices.empty or symbol not in prices.columns:
        return PriceReaction(
            confidence=Confidence.UNAVAILABLE, note="no price history for this symbol"
        )

    series = prices[symbol].dropna()
    if series.empty:
        return PriceReaction(confidence=Confidence.UNAVAILABLE, note="price series is empty")

    session = pricing_session(occurred_at)
    dates = [d.date() for d in series.index]
    if session not in dates:
        return PriceReaction(
            confidence=Confidence.UNAVAILABLE,
            session=session,
            note=f"no bar for the pricing session {session} — it may not have happened yet",
        )

    index = dates.index(session)
    if index == 0:
        return PriceReaction(
            confidence=Confidence.UNAVAILABLE,
            session=session,
            note="no prior close to measure the reaction against",
        )

    before = float(series.iloc[index - 1])
    after = float(series.iloc[index])
    pct = (after / before - 1) if before else None
    dollars = position.quantity * (after - before) if position.quantity else None
    pct_of_book = dollars / position.net_liq if dollars is not None and position.net_liq else None

    return PriceReaction(
        confidence=Confidence.MEASURED,
        session=session,
        before=before,
        after=after,
        pct_move=pct,
        dollars=dollars,
        pct_of_book=pct_of_book,
        priced_next_session=session != mc.to_et(occurred_at).date(),
    )


def _attribution(store: DaemonStore, symbol: str, session: date | None, factors, prices):
    """How much of the pricing session the factor model accounts for."""
    from advisor.macro.sensitivity import expected_return, residual_z

    sensitivity = store.load_sensitivity(symbol)
    if sensitivity is None:
        return Attribution(
            confidence=Confidence.UNAVAILABLE,
            note=f"no factor estimate for {symbol} — too little price history",
        )
    if session is None or factors is None or factors.empty or prices is None or prices.empty:
        return Attribution(
            confidence=Confidence.UNAVAILABLE,
            r2=sensitivity.r2,
            resid_vol=sensitivity.resid_vol,
            note="factor panel unavailable for the pricing session",
        )

    from advisor.macro.factors import log_returns

    returns = log_returns(prices)
    days = [d for d in factors.index if d.date() == session and d in returns.index]
    if not days or symbol not in returns.columns:
        return Attribution(
            confidence=Confidence.UNAVAILABLE,
            r2=sensitivity.r2,
            resid_vol=sensitivity.resid_vol,
            note=f"no factor observation for {session}",
        )

    day = days[0]
    actual = returns.loc[day, symbol]
    if actual != actual:  # NaN
        return Attribution(
            confidence=Confidence.UNAVAILABLE,
            r2=sensitivity.r2,
            resid_vol=sensitivity.resid_vol,
            note=f"{symbol} has no bar for {session}",
        )

    moves = {
        c: float(factors.loc[day, c])
        for c in factors.columns
        if not math.isnan(float(factors.loc[day, c]))
    }
    if len(moves) < len(factors.columns):
        # A factor with no observation that day would poison every downstream
        # number with NaN. Dropping it narrows the model rather than breaking
        # it, and the caller is told how much was dropped.
        logger.info(
            "story: %s on %s — %d of %d factors had no observation",
            symbol,
            session,
            len(factors.columns) - len(moves),
            len(factors.columns),
        )
    if not moves:
        return Attribution(
            confidence=Confidence.UNAVAILABLE,
            r2=sensitivity.r2,
            resid_vol=sensitivity.resid_vol,
            note=f"no factor observations at all for {session}",
        )

    expected = expected_return(sensitivity, moves)
    z = residual_z(sensitivity, float(actual), moves)
    if math.isnan(expected) or math.isnan(z):
        return Attribution(
            confidence=Confidence.UNAVAILABLE,
            r2=sensitivity.r2,
            resid_vol=sensitivity.resid_vol,
            note=f"the factor model produced no usable number for {session}",
        )
    return Attribution(
        # A model estimate is never MEASURED, however well it fits.
        confidence=Confidence.ESTIMATED,
        verdict=verdict_for(z),
        actual_return=float(actual),
        expected_return=expected,
        residual_z=z,
        resid_vol=sensitivity.resid_vol,
        r2=sensitivity.r2,
    )


def _corroboration(
    store: DaemonStore, symbol: str, occurred_at: datetime, exclude_url: str | None
) -> Corroboration:
    window = timedelta(days=CORROBORATION_WINDOW_DAYS)
    items = store.source_items_between(
        symbol, occurred_at - timedelta(days=1), occurred_at + window
    )
    rows, counts = [], {"PRIMARY": 0, "BROKER": 0, "AGGREGATOR": 0, "UNTAGGED": 0}
    for item in items:
        if exclude_url and item.url == exclude_url:
            continue
        counts[item.tier.value] = counts.get(item.tier.value, 0) + 1
        rows.append(
            {
                "tier": item.tier.value,
                "title": item.title,
                "url": item.url,
                "published_at": item.published_at.isoformat(),
                "provider": item.provider,
            }
        )
    return Corroboration(
        window_days=CORROBORATION_WINDOW_DAYS + 1,
        items=rows,
        primary_count=counts["PRIMARY"] + counts["BROKER"],
        aggregator_count=counts["AGGREGATOR"],
        untagged_count=counts["UNTAGGED"],
    )


def _thesis(store: DaemonStore, symbol: str, event: Event) -> ThesisLink:
    """The stated thesis, and which of its claims this event tests.

    Three outcomes that must not be collapsed into one: no thesis at all, a
    thesis document that is still the blank template, and a thesis whose
    claims this event either trips or leaves alone.
    """
    from advisor.thesis.match import evaluate_thesis
    from advisor.thesis.repo import load_thesis

    try:
        thesis = load_thesis(store, symbol)
    except Exception as exc:  # noqa: BLE001
        logger.debug("story: thesis lookup failed for %s: %s", symbol, exc)
        return ThesisLink(confidence=Confidence.UNAVAILABLE, note="thesis store unreadable")

    if thesis is None:
        return ThesisLink(
            confidence=Confidence.MEASURED,
            exists=False,
            note=f"no stated thesis for {symbol} — nothing to test this against",
        )

    evaluations = [e.model_dump(mode="json") for e in evaluate_thesis(thesis, event)]
    note = thesis.prose_note
    if not thesis.substantive:
        note = note or "the thesis is empty"
    elif not thesis.claims:
        note = (
            "no testable claims yet — add invalidations to have events checked "
            "against this thesis automatically"
        )

    return ThesisLink(
        confidence=Confidence.MEASURED,
        exists=True,
        substantive=thesis.substantive,
        title=thesis.title,
        conviction=thesis.conviction,
        status=thesis.status,
        claims_total=len(thesis.claims),
        claims_monitored=len(thesis.monitored_claims),
        evaluations=evaluations,
        note=note,
    )


def build_story(store: DaemonStore, event: Event, *, prices=None, factors=None) -> Story:
    """Assemble one event into a story. Never raises on a missing slot."""
    anchor = _anchor(event)
    symbol = anchor.symbol

    if prices is None or factors is None:
        from advisor.macro.factors import build_factor_returns, fetch_prices

        if prices is None:
            prices = fetch_prices([symbol], period="1y")
        if factors is None:
            factors = build_factor_returns(period="1y")

    position = _position(store, symbol, anchor.occurred_at)
    reaction = _self_priced(anchor, position) or _reaction(
        symbol, anchor.occurred_at, position, prices
    )
    attribution = _attribution(store, symbol, reaction.session, factors, prices)
    corroboration = _corroboration(store, symbol, anchor.occurred_at, anchor.url)
    thesis = _thesis(store, symbol, event)

    return Story(
        symbol=symbol,
        assembled_at=mc.now_et(),
        anchor=anchor,
        position=position,
        reaction=reaction,
        attribution=attribution,
        corroboration=corroboration,
        thesis=thesis,
    )


# How an archived filing reads when a story is built from it. Used only to
# rank and describe — never to emit.
_ARCHIVE_TIER = {
    Materiality.HIGH: EventTier.A,
    Materiality.MEDIUM: EventTier.B,
    Materiality.LOW: EventTier.C,
}


def _anchor_from_archive(store: DaemonStore, symbol: str, limit: int) -> list[Event]:
    """Synthesise anchors from archived filings when no event exists.

    The event stream only carries what was new enough to be worth reporting;
    a position opened today, or one whose filings predate the daemon, has a
    history worth reading and no events at all. AMD had eight filings since
    July and not one story.

    Anchors carry the filing's own materiality, not a blanket Tier C. The two
    layers answer different questions: the event stream decides what
    interrupts *now*; the story describes what *happened*. CCXI's merger S-4
    was nine days old when those forms began being watched, so it was archived
    and never evented — and a story that called it routine would describe it
    wrongly.

    These anchors never enter the event stream, so materiality here cannot
    cause an interrupt after the fact.
    """
    from advisor.news.classify import classify_filing
    from advisor.news.foreign import classify_headline

    items = [
        i
        for i in store.recent_source_items(symbol, limit=60)
        if i.tier is SourceTier.PRIMARY and i.doc_type
    ]
    anchors: list[Event] = []
    for item in items[:limit]:
        # A 6-K is classified from its exhibit headline, exactly as the live
        # ingest does; classifying it by form alone makes every foreign
        # issuer's disclosure read as a generic "foreign issuer report".
        if item.doc_type.upper().startswith("6-K") and item.summary:
            classification = classify_headline(item.summary)
        else:
            classification = classify_filing(item.doc_type, item.item_codes)
        anchors.append(
            Event(
                source=EventSource.EDGAR,
                kind=f"FILING_{classification.kind.value}",
                tier=_ARCHIVE_TIER[classification.materiality],
                symbol=symbol,
                dedup_key=item.dedup_key(),
                payload={
                    "form": item.doc_type,
                    "items": item.item_codes,
                    "label": classification.label,
                    "url": item.url,
                    "accession": item.accession,
                    "accepted_at": item.published_at.isoformat(),
                    "provider": item.provider,
                    "match": item.entity.method.value,
                    "from_archive": True,
                },
            )
        )
    return anchors


def stories_for_symbol(store: DaemonStore, symbol: str, *, limit: int = 3) -> list[Story]:
    """Stories for a symbol's most recent events, newest first.

    Falls back to the archive when nothing has been evented, so every holding
    with a filing history has a story rather than a blank page.
    """
    symbol = symbol.upper()
    # Tier before recency. Sorting by date alone buried CCXI's merger — a
    # Tier A S-4 from nine days earlier — behind a routine deep-drawdown
    # notice from that morning, so the story led with the least important
    # thing that had happened.
    candidates = [
        e
        for e in store.recent_events(limit=400)
        if (e.symbol or "").upper() == symbol and e.tier.value in {"A", "B"}
    ]
    # Archived filings compete with events on the same footing: a merger the
    # daemon learned about too late to report is still the most important
    # thing that happened to this position. CCXI's S-4 was nine days old when
    # those forms started being watched, so it was archived and never evented,
    # and the story led with a routine deep-drawdown notice instead.
    # Dedup on the accession carried in the payload, not on `dedup_key`: the
    # events table stores only the hash, so an event read back from the
    # database has no key to compare and every filing would appear twice.
    seen = {
        (e.payload or {}).get("accession") for e in candidates if (e.payload or {}).get("accession")
    }
    candidates += [
        a
        for a in _anchor_from_archive(store, symbol, limit * 4)
        if a.payload.get("accession") not in seen
    ]

    tier_rank = {"A": 0, "B": 1, "C": 2}
    events = sorted(candidates, key=lambda e: (tier_rank[e.tier.value], -e.ts.timestamp()))[:limit]
    if not events:
        return []

    # One price and factor fetch for the whole batch, not one per story.
    from advisor.macro.factors import build_factor_returns, fetch_prices

    prices = fetch_prices([symbol], period="1y")
    factors = build_factor_returns(period="1y")
    return [build_story(store, e, prices=prices, factors=factors) for e in events]
