"""Daemon endpoints: what the always-on advisor has seen, and what it concluded.

Read-only over the daemon's own store, plus two live checks. The split
matters for the frontend: everything backed by the store answers instantly and
can be polled, while `reconcile` and `refresh` reach the broker and the
network and must be user-triggered.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/daemon", tags=["daemon"])


def _store():
    from advisor.daemon.store import DaemonStore
    from advisor.research.config import get_settings

    return DaemonStore(get_settings().db_path)


def _event_row(event) -> dict:
    from advisor.daemon.summarize import summarize

    return {
        "id": event.id,
        "ts": event.ts.isoformat(),
        "source": event.source.value,
        "kind": event.kind,
        "tier": event.tier.value,
        "symbol": event.symbol,
        "payload": event.payload,
        # Computed server-side on purpose. The frontend had its own copy of
        # this logic and it had already drifted: it covered offerings and
        # factor shocks but not crossings, concentration, insider clusters or
        # news headlines, so a stop breach rendered without its numbers and
        # eight context items rendered as eight blank rows. One tested
        # implementation, served to both readers.
        "summary": summarize(event),
    }


@router.get("/status")
async def status() -> dict:
    """Heartbeats, watermarks, event counts and market state."""
    from advisor.daemon import market_calendar as mc
    from advisor.daemon.supervisor import build_registry

    store = _store()
    try:
        now = mc.now_et()
        jobs = []
        for job in build_registry():
            hb = store.get_heartbeat(job.name)
            jobs.append(
                {
                    "name": job.name,
                    "description": job.description,
                    "last_run_at": hb.last_run_at.isoformat() if hb.last_run_at else None,
                    "last_ok_at": hb.last_ok_at.isoformat() if hb.last_ok_at else None,
                    "run_count": hb.run_count,
                    "error_count": hb.error_count,
                    "last_error": hb.last_error,
                }
            )
        return {
            "now": now.isoformat(),
            "market_open": mc.is_market_open(now),
            "trading_day": mc.is_trading_day(now.date()),
            "jobs": jobs,
            "watermarks": [
                {
                    "source": w.source.value,
                    "last_seen_ts": w.last_seen_ts.isoformat() if w.last_seen_ts else None,
                    "updated_at": w.updated_at.isoformat() if w.updated_at else None,
                }
                for w in store.all_watermarks()
            ],
            "event_counts": store.event_counts_by_tier(),
        }
    finally:
        store.close()


@router.get("/events")
async def events(
    limit: int = Query(50, ge=1, le=500),
    tier: str | None = Query(None, pattern="^[ABC]$"),
    symbol: str | None = None,
) -> dict:
    """Recent events, newest first."""
    store = _store()
    try:
        rows = [_event_row(e) for e in store.recent_events(limit=limit * 3)]
        if tier:
            rows = [r for r in rows if r["tier"] == tier]
        if symbol:
            rows = [r for r in rows if (r["symbol"] or "").upper() == symbol.upper()]
        return {"events": rows[:limit]}
    finally:
        store.close()


@router.get("/exposure")
async def exposure() -> dict:
    """The book's factor profile, and who carries each bet."""
    store = _store()
    try:
        current = store.load_latest_exposure()
        if current is None:
            return {"exposure": None}
        return {
            "exposure": {
                "asof": current.asof.isoformat(),
                "net_liq": current.net_liq,
                "covered_weight": current.covered_weight,
                "uncovered": current.uncovered,
                "factors": [
                    {
                        "factor": f.factor,
                        "net_loading": f.net_loading,
                        "concentration": f.concentration(),
                        "contributors": [{"symbol": s, "share": v} for s, v in f.contributors[:6]],
                    }
                    for f in current.ranked()
                ],
            },
            # Read them as relative bets, not causal betas — the UI shows this.
            "caveat": (
                "Ridge estimates over deliberately correlated factors. Read as "
                "relative bets and use expected moves for prediction, not as "
                "standalone causal betas."
            ),
        }
    finally:
        store.close()


@router.get("/sources")
async def sources(
    symbol: str | None = None,
    limit: int = Query(40, ge=1, le=200),
) -> dict:
    """Archived source items — everything the advisor has read, with provenance."""
    store = _store()
    try:
        items = store.recent_source_items(symbol, limit=limit)
        return {
            "items": [
                {
                    "tier": i.tier.value,
                    "provider": i.provider,
                    "url": i.url,
                    "title": i.title,
                    "published_at": i.published_at.isoformat(),
                    "symbol": i.entity.symbol,
                    "doc_type": i.doc_type,
                    "item_codes": i.item_codes,
                    "accession": i.accession,
                    "match": i.entity.method.value,
                    "confidence": i.entity.confidence,
                }
                for i in items
            ]
        }
    finally:
        store.close()


@router.get("/symbol/{symbol}")
async def symbol_detail(symbol: str, days: int = Query(45, ge=7, le=365)) -> dict:
    """Everything the daemon holds on one name: filings, news, factors, events.

    This is the per-ticker view the CLI produces, assembled for the browser.
    """
    from advisor.daemon.market_calendar import now_et

    sym = symbol.upper()
    store = _store()
    try:
        cutoff = now_et() - timedelta(days=days)
        sensitivity = store.load_sensitivity(sym)
        book_exposure = store.load_latest_exposure()

        contribution = []
        if book_exposure and sensitivity:
            for factor in book_exposure.ranked():
                share = next((v for s, v in factor.contributors if s == sym), None)
                if share is not None:
                    contribution.append(
                        {
                            "factor": factor.factor,
                            "loading": sensitivity.loading(factor.factor),
                            "contribution": share,
                            "book_total": factor.net_loading,
                        }
                    )

        items = [i for i in store.recent_source_items(sym, limit=100) if i.published_at >= cutoff]
        return {
            "symbol": sym,
            "sensitivity": (
                {
                    "asof": sensitivity.asof.isoformat(),
                    "n_obs": sensitivity.n_obs,
                    "r2": sensitivity.r2,
                    "resid_vol": sensitivity.resid_vol,
                    "loadings": [
                        {
                            "factor": entry.factor,
                            "loading": entry.loading,
                            "tstat": entry.tstat,
                            "material": entry.is_material,
                        }
                        for entry in sorted(sensitivity.loadings, key=lambda e: -abs(e.loading))
                    ],
                }
                if sensitivity
                else None
            ),
            "contribution": contribution,
            "timeline": [
                {
                    "published_at": i.published_at.isoformat(),
                    "tier": i.tier.value,
                    "provider": i.provider,
                    "title": i.title,
                    "url": i.url,
                    "doc_type": i.doc_type,
                    "item_codes": i.item_codes,
                    "match": i.entity.method.value,
                }
                for i in items
            ],
            "events": [
                _event_row(e)
                for e in store.recent_events(limit=300)
                if (e.symbol or "").upper() == sym
            ][:40],
        }
    finally:
        store.close()


@router.get("/story/{symbol}")
async def story(symbol: str, limit: int = Query(1, ge=1, le=5)) -> dict:
    """Assembled stories for a symbol's recent events.

    Unlike the other read endpoints this one fetches a price history, so it is
    not free. The frontend requests it per ticker page rather than polling.
    """
    from advisor.story.assemble import stories_for_symbol
    from advisor.story.render import headline

    store = _store()
    try:
        stories = stories_for_symbol(store, symbol, limit=limit)
        return {
            "symbol": symbol.upper(),
            "stories": [
                {
                    **s.model_dump(mode="json"),
                    "headline": headline(s),
                    "unavailable": s.unavailable_slots,
                }
                for s in stories
            ],
        }
    finally:
        store.close()


@router.get("/thesis/{symbol}")
async def thesis_detail(symbol: str) -> dict:
    """The structured thesis for a symbol, with its claims and coverage."""
    from advisor.thesis.repo import ThesisReadError, load_thesis

    store = _store()
    try:
        try:
            thesis = load_thesis(store, symbol)
        except ThesisReadError as exc:
            raise HTTPException(status_code=503, detail=f"thesis store unreadable: {exc}") from exc
        if thesis is None:
            return {"symbol": symbol.upper(), "thesis": None}
        return {
            "symbol": thesis.symbol,
            "thesis": {
                "title": thesis.title,
                "conviction": thesis.conviction,
                "status": thesis.status,
                "substantive": thesis.substantive,
                "prose_note": thesis.prose_note,
                "coverage": thesis.coverage,
                "claims": [
                    {
                        **c.model_dump(mode="json"),
                        "monitored": c.monitored,
                        "trigger_description": c.trigger.describe(),
                    }
                    for c in thesis.claims
                ],
            },
        }
    finally:
        store.close()


@router.get("/thesis-coverage")
async def thesis_coverage() -> dict:
    """Which holdings have a written thesis, weighted by their place in the book.

    The number that matters is the share of capital with no stated view, not
    the count of symbols — a missing thesis on 21% of the book is a different
    problem from one on 3%.
    """
    from advisor.daemon.book import fetch_book
    from advisor.thesis.repo import load_thesis

    store = _store()
    try:
        book = await fetch_book()
        rows = []
        for sym in book.symbols:
            try:
                thesis = load_thesis(store, sym)
            except Exception:  # noqa: BLE001
                thesis = None
            position = next((p for p in book.positions if p.underlying.upper() == sym), None)
            weight = (position.signed_notional / book.net_liq) if position and book.net_liq else 0.0
            rows.append(
                {
                    "symbol": sym,
                    "weight": weight,
                    "written": bool(thesis and thesis.substantive),
                    "claims": len(thesis.claims) if thesis else 0,
                    "monitored": len(thesis.monitored_claims) if thesis else 0,
                }
            )
        rows.sort(key=lambda r: -abs(r["weight"]))
        uncovered = [r for r in rows if not r["written"]]
        return {
            "rows": rows,
            "uncovered_count": len(uncovered),
            "uncovered_weight": sum(abs(r["weight"]) for r in uncovered),
        }
    finally:
        store.close()


class ClaimInput(BaseModel):
    kind: str
    text: str
    event_kinds: list[str] = Field(default_factory=list)
    field: str | None = None
    comparator: str = "HAPPENS"
    threshold: float | None = None
    factor: str | None = None


@router.post("/thesis/{symbol}/claims")
async def add_claim(symbol: str, body: ClaimInput) -> dict:
    """Record one claim for a symbol."""
    from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger

    try:
        claim = Claim(
            kind=ClaimKind(body.kind.upper()),
            text=body.text,
            trigger=Trigger(
                event_kinds=[k.upper() for k in body.event_kinds],
                field=body.field,
                comparator=Comparator(body.comparator.upper()),
                threshold=body.threshold,
                factor=body.factor.upper() if body.factor else None,
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    store = _store()
    try:
        claim_id = store.save_claim(symbol, claim)
        return {
            "id": claim_id,
            "monitored": claim.monitored,
            "trigger_description": claim.trigger.describe(),
        }
    finally:
        store.close()


@router.delete("/thesis/claims/{claim_id}")
async def delete_claim(claim_id: str) -> dict:
    store = _store()
    try:
        if not store.delete_claim(claim_id):
            raise HTTPException(status_code=404, detail=f"no claim {claim_id}")
        return {"deleted": claim_id}
    finally:
        store.close()


@router.get("/actions")
async def actions(symbol: str | None = None) -> dict:
    """What follows from what is stored, per ticker.

    Reaches the broker for the current book, so this is not a poll-me
    endpoint; the frontend requests it when the page opens.
    """
    from advisor.action.assemble import build_all, build_card
    from advisor.daemon.book import fetch_book

    store = _store()
    try:
        book = await fetch_book()
        cards = [build_card(store, symbol, book)] if symbol else build_all(store, book)
        return {"cards": [c.model_dump(mode="json") for c in cards]}
    finally:
        store.close()


@router.post("/reconcile")
async def reconcile() -> dict:
    """Run every data-quality check live. Slow — reaches the broker and network."""
    from advisor.daemon.book import fetch_book
    from advisor.daemon.reconcile import run_reconciliation

    try:
        book = await fetch_book()
        report = await run_reconciliation(book)
    except Exception as exc:  # noqa: BLE001
        logger.warning("reconcile failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"reconciliation unavailable: {exc}") from exc

    return {
        "asof": report.asof.isoformat(),
        "ok": report.ok,
        "summary": report.summary(),
        "findings": [
            {
                "check": f.check,
                "severity": f.severity,
                "symbol": f.symbol,
                "detail": f.detail,
            }
            for f in report.findings
        ],
    }


@router.post("/run/{job}")
async def run_job(job: str) -> dict:
    """Trigger one daemon job by hand — the frontend's equivalent of `once --job`."""
    from advisor.daemon.supervisor import Supervisor, build_registry

    registry = build_registry()
    match = next((j for j in registry if j.name == job), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"unknown job: {job}")

    store = _store()
    try:
        result = await Supervisor(store, registry).run_job(match)
        return {
            "job": result.job,
            "ok": result.ok,
            "detail": result.detail,
            "events_emitted": result.events_emitted,
            "duration_ms": result.duration_ms,
            "ran_at": result.ran_at.isoformat(),
        }
    finally:
        store.close()


@router.get("/coverage")
async def coverage(window_days: int = Query(90, ge=7, le=365)) -> dict:
    """How often an unexplained move had a primary source to explain it."""
    from advisor.news.ingest import coverage as score

    store = _store()
    try:
        report = score(store, window_days=window_days)
        return {
            "window_days": window_days,
            "divergences": report.divergences,
            "explained": report.explained,
            "rate": report.rate,
            "by_symbol": [
                {"symbol": s, "divergences": seen, "explained": found}
                for s, (seen, found) in sorted(report.by_symbol.items())
            ],
        }
    finally:
        store.close()
