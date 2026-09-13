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
from datetime import date, datetime, timedelta

from advisor.daemon import market_calendar as mc
from advisor.daemon.models import Event
from advisor.daemon.store import DaemonStore
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
    """Map a residual to language that does not outrun it."""
    if residual_z is None:
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

    moves = {c: float(factors.loc[day, c]) for c in factors.columns}
    expected = expected_return(sensitivity, moves)
    z = residual_z(sensitivity, float(actual), moves)
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
    reaction = _reaction(symbol, anchor.occurred_at, position, prices)
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


def stories_for_symbol(store: DaemonStore, symbol: str, *, limit: int = 3) -> list[Story]:
    """Stories for a symbol's most recent events, newest first."""
    symbol = symbol.upper()
    events = [
        e
        for e in store.recent_events(limit=400)
        if (e.symbol or "").upper() == symbol and e.tier.value in {"A", "B"}
    ][:limit]
    if not events:
        return []

    # One price and factor fetch for the whole batch, not one per story.
    from advisor.macro.factors import build_factor_returns, fetch_prices

    prices = fetch_prices([symbol], period="1y")
    factors = build_factor_returns(period="1y")
    return [build_story(store, e, prices=prices, factors=factors) for e in events]
