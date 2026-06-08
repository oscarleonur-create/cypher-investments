"""Portfolio endpoints: live holdings snapshot, cached review, refresh jobs."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from advisor.api import deps

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
