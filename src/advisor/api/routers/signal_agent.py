"""Autonomous Signal agent endpoints.

Decides WHAT to scan and WHY (reasoning over a small toolset), not how to
size risk -- the deterministic gate still runs before results are returned.
Follows the same background-job pattern as scalping.py/swing.py.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Query
from pydantic import BaseModel

from advisor.api import deps
from advisor.signal_agent.agent import run_signal_agent
from advisor.signal_agent.models import SignalAgentResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/signal-agent", tags=["signal-agent"])

_results: dict[str, SignalAgentResult] = {}
_latest_job: str | None = None


class RunRequest(BaseModel):
    objective: str
    universe: str = "semiconductors"
    max_symbols: int = 30


@router.post("/run")
async def run(req: RunRequest) -> dict:
    session = await deps.try_get_tt_session()
    net_liq, open_notional = await deps.account_risk_state(session)
    job_id = deps.new_job("signal_agent", target=req.objective[:60])
    deps.update_job(job_id, message="reasoning…")

    def _run() -> None:
        try:
            result = run_signal_agent(
                req.objective,
                universe=req.universe,
                max_symbols=req.max_symbols,
                net_liq=net_liq,
                open_notional_by_symbol=open_notional,
            )
            global _latest_job
            _results[job_id] = result
            _latest_job = job_id
            approved = sum(1 for s in result.signals if s.risk_approved)
            deps.update_job(
                job_id,
                status="done",
                message=f"{len(result.signals)} signals, {approved} risk-approved",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("signal agent run failed")
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id}


@router.get("/results")
async def results(job_id: str | None = Query(default=None)) -> dict:
    key = job_id or _latest_job
    result = _results.get(key) if key else None
    if result is None:
        return {"result": None}
    return {"result": result.model_dump(mode="json")}
