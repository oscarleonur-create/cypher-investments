"""Portfolio endpoints: live holdings snapshot, cached review, refresh jobs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from advisor.api import deps
from advisor.research.models import BayesianOverrides

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


# ── Holdings snapshot (drives the grid + 15s poll) ──────────────────────────────


@router.get("/holdings")
async def holdings() -> dict:
    """Equity holdings merged across accounts + balances + research summary.

    Live price/P&L are computed in the browser from the WebSocket quote stream;
    here we return the static cost-basis + research context per symbol.
    """
    from advisor.market.tastytrade_client import get_balances, get_positions
    from advisor.research.portfolio_review import DEFAULT_ACCOUNTS

    session = await deps.get_tt_session()

    rows: dict[str, dict] = {}
    balances: list[dict] = []
    for acct in DEFAULT_ACCOUNTS:
        try:
            positions = await get_positions(session, acct)
        except Exception as exc:  # noqa: BLE001
            logger.warning("positions fetch failed for %s: %s", acct, exc)
            positions = []
        try:
            balances.append(await get_balances(session, acct))
        except Exception as exc:  # noqa: BLE001
            logger.warning("balances fetch failed for %s: %s", acct, exc)

        for p in positions:
            if "OPTION" in str(p.get("instrument_type", "")).upper():
                continue  # equity rows only for the grid
            if "EQUITY" not in str(p.get("instrument_type", "")).upper():
                continue
            sym = (p.get("symbol") or "").upper()
            if not sym:
                continue
            direction = -1 if str(p.get("quantity_direction", "")).lower() == "short" else 1
            qty = float(p.get("quantity", 0)) * direction
            row = rows.get(sym)
            if row is None:
                rows[sym] = {
                    "symbol": sym,
                    "quantity": qty,
                    "average_open_price": float(p.get("average_open_price", 0) or 0),
                    "multiplier": int(p.get("multiplier", 1) or 1),
                    "close_price": float(p.get("close_price", 0) or 0),
                    "mark_price": float(p.get("mark_price", 0) or 0),
                    "accounts": [acct],
                }
            else:
                # Weighted-average the open price across accounts, then add qty.
                prev_qty = row["quantity"]
                prev_avg = row["average_open_price"]
                new_qty = prev_qty + qty
                if new_qty != 0:
                    row["average_open_price"] = (
                        prev_avg * prev_qty + float(p.get("average_open_price", 0) or 0) * qty
                    ) / new_qty
                row["quantity"] = new_qty
                if acct not in row["accounts"]:
                    row["accounts"].append(acct)

    _merge_research_summary(rows)

    net_liq = sum(b.get("net_liq", 0) for b in balances)
    cash = sum(b.get("cash", 0) for b in balances)
    buying_power = sum(b.get("buying_power", 0) for b in balances)

    # Snapshot today's NLV once per day (INSERT OR IGNORE deduplicates intra-day).
    _snapshot_balances(balances)

    return {
        "holdings": sorted(rows.values(), key=lambda r: r["symbol"]),
        "balances": {
            "net_liq": net_liq,
            "cash": cash,
            "buying_power": buying_power,
            "accounts": [b.get("account") for b in balances],
        },
        "symbols": sorted(rows),
    }


def _snapshot_balances(balances: list[dict]) -> None:
    from datetime import date as _date

    from advisor.research.store import ResearchStore

    today = _date.today().isoformat()
    store = ResearchStore(deps.db_path())
    try:
        for b in balances:
            if b.get("net_liq", 0) > 0:
                store.upsert_snapshot(today, b["account"], b["net_liq"], b.get("cash", 0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("NLV snapshot write failed: %s", exc)
    finally:
        store.close()


def _merge_research_summary(rows: dict[str, dict]) -> None:
    """Attach cached thesis/conviction/attention from the latest portfolio review."""
    from advisor.research.portfolio_review import load_latest_review

    loaded = load_latest_review()
    review = loaded[0] if loaded else None
    by_sym = {p.symbol.upper(): p for p in (review.positions if review else [])}
    for sym, row in rows.items():
        p = by_sym.get(sym)
        row["research"] = (
            {
                "thesis_status": p.thesis_status,
                "conviction": p.conviction,
                "attention": p.attention,
                "next_earnings_date": p.next_earnings_date,
                "base_upside": p.base_upside,
                "has_report": p.has_report,
                "kpi_alerts": p.kpi_alerts,
                "sector": p.sector,
                "bayes_upside": p.bayes_upside,
                "bayes_prob_undervalued": p.bayes_prob_undervalued,
                "analyst_target": p.analyst_target,
                "analyst_upside": p.analyst_upside,
                "analyst_n": p.analyst_n,
            }
            if p
            else None
        )


# ── Cached review ───────────────────────────────────────────────────────────────


@router.get("/review")
async def review() -> dict:
    from advisor.research.portfolio_review import load_latest_review

    loaded = load_latest_review()
    if loaded is None:
        return {"review": None, "fetched_at": None}
    rev, fetched_at = loaded
    return {"review": rev.model_dump(mode="json"), "fetched_at": fetched_at.isoformat()}


@router.post("/review/refresh")
async def refresh_review(rebuild_uncovered: bool = False, catalysts: bool = True) -> dict:
    """Kick off a background portfolio review build; returns a job id to poll."""
    import asyncio

    from advisor.research.portfolio_review import build_portfolio_review, save_review

    job_id = deps.new_job("portfolio_review")

    def _run() -> None:
        try:
            deps.update_job(job_id, message="pulling holdings…")
            rev = build_portfolio_review(
                None,
                rebuild_uncovered=rebuild_uncovered,
                with_catalysts=catalysts,
                concurrency=3,
            )
            save_review(rev)
            deps.update_job(
                job_id, status="done", message=f"reviewed {len(rev.positions)} holdings"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("portfolio refresh failed")
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id}


# ── Market context + sector rotation ──────────────────────────────────────────


@router.get("/market")
async def market() -> dict:
    """VIX snapshot + (optional) HMM volatility regime for the market panel."""
    import asyncio

    from advisor.research.market_context import get_market_context

    # yfinance + model load are blocking; keep the event loop free.
    return await asyncio.to_thread(get_market_context)


@router.get("/rotation")
async def rotation() -> dict:
    """Sector-ETF momentum (vs SPY) for the sectors currently held.

    Sectors come from the latest cached review; rebuild the review first if a
    holding has no sector yet. Returns ``{}`` when nothing is classified.
    """
    import asyncio

    from advisor.research.portfolio_analytics import sector_breakdown, sector_rotation
    from advisor.research.portfolio_review import load_latest_review

    loaded = load_latest_review()
    if loaded is None:
        return {"rotation": {}, "weights": {}}
    review = loaded[0]

    weights = {s: w for s, w in sector_breakdown(review).items() if s != "Unknown"}
    if not weights:
        return {"rotation": {}, "weights": {}}

    rot = await asyncio.to_thread(sector_rotation, list(weights.keys()))
    return {"rotation": rot, "weights": weights}


# ── Bayesian fair-value (what-if sliders) ────────────────────────────────────────


@router.get("/bayesian/{symbol}")
async def bayesian(symbol: str) -> dict:
    """Return the cached baseline Bayesian posterior for a held symbol.

    The baseline is computed during the portfolio review, so only reviewed
    holdings have one — non-held tickers get a 404 the UI renders as empty.
    """
    from advisor.research.models import BayesianPriceResult
    from advisor.research.store import ResearchStore

    sym = symbol.upper()
    store = ResearchStore(deps.db_path())
    try:
        loaded = store.load_artifact(sym, "bayesian", "latest")
    finally:
        store.close()

    if loaded is None:
        raise HTTPException(
            status_code=404,
            detail=f"No Bayesian pricing for {sym}. Run the portfolio review to compute it.",
        )
    payload_json, fetched_at = loaded
    result = BayesianPriceResult.model_validate_json(payload_json)
    return {"result": result.model_dump(mode="json"), "fetched_at": fetched_at.isoformat()}


@router.post("/bayesian/{symbol}")
async def recompute_bayesian_endpoint(symbol: str, overrides: BayesianOverrides) -> dict:
    """Recompute the posterior under user slider overrides (pure-compute, fast)."""
    from advisor.research.store import ResearchStore
    from advisor.research.valuation.bayesian import recompute_bayesian

    sym = symbol.upper()
    store = ResearchStore(deps.db_path())
    try:
        # Gate to reviewed holdings: a stored baseline means it went through the
        # portfolio review. Avoids a TastyTrade round-trip on every slider move.
        if store.load_artifact(sym, "bayesian", "latest") is None:
            raise HTTPException(
                status_code=404,
                detail=f"{sym} is not a reviewed holding. Run the portfolio review first.",
            )
        report = store.load_latest_report(sym)
    finally:
        store.close()

    if report is None:
        raise HTTPException(status_code=404, detail=f"No cached research for {sym}.")

    result = recompute_bayesian(report, overrides)
    return {"result": result.model_dump(mode="json")}


# ── Portfolio performance (deposit-adjusted returns) ──────────────────────────


@router.get("/performance")
async def performance(sync_cash_flows: bool = False) -> dict:
    """Return deposit-adjusted portfolio returns and equity curve.

    Pass ?sync_cash_flows=true to fetch new money-movement transactions from
    TastyTrade before computing (adds ~1-2s per account).
    """
    from advisor.research.performance import get_equity_curve, standard_period_returns
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        if sync_cash_flows:
            await _sync_cash_flows(store)
        snapshots = store.load_combined_snapshots()
        cash_flows = store.load_cash_flows()
    finally:
        store.close()

    periods = standard_period_returns(snapshots, cash_flows)
    curve = get_equity_curve(snapshots, cash_flows)
    return {
        "periods": periods,
        "equity_curve": curve,
        "snapshot_count": len(snapshots),
        "cash_flow_count": len(cash_flows),
    }


@router.post("/performance/backfill")
async def backfill_nlv() -> dict:
    """One-time backfill of historical NLV from TastyTrade.

    Call once after first deploy; thereafter /holdings auto-snapshots daily.
    """
    from advisor.market.tastytrade_client import get_nlv_history
    from advisor.research.portfolio_review import DEFAULT_ACCOUNTS
    from advisor.research.store import ResearchStore

    session = await deps.get_tt_session()
    store = ResearchStore(deps.db_path())
    inserted = 0
    try:
        for acct in DEFAULT_ACCOUNTS:
            try:
                items = await get_nlv_history(session, acct, time_back="all")
            except Exception as exc:  # noqa: BLE001
                logger.warning("NLV history fetch failed for %s: %s", acct, exc)
                continue
            for item in items:
                store.upsert_snapshot(item["date"], item["account"], item["net_liq"], cash=0.0)
                inserted += 1
    finally:
        store.close()
    return {"inserted": inserted}


async def _sync_cash_flows(store) -> None:
    from datetime import date

    from advisor.market.tastytrade_client import get_transactions
    from advisor.research.portfolio_review import DEFAULT_ACCOUNTS

    session = await deps.get_tt_session()
    latest = store.load_latest_cash_flow_date()
    start = date.fromisoformat(latest) if latest else None

    for acct in DEFAULT_ACCOUNTS:
        try:
            txns = await get_transactions(session, acct, start_date=start)
        except Exception as exc:  # noqa: BLE001
            logger.warning("cash flow sync failed for %s: %s", acct, exc)
            continue
        for t in txns:
            if not t.get("amount"):
                continue
            store.upsert_cash_flow(
                flow_date=t["date"],
                account=t["account"],
                amount=t["amount"],
                description=t.get("description", ""),
                tastytrade_id=t.get("id"),
            )
