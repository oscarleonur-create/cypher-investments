"""Risk agent: an LLM narrowing layer on top of the deterministic gate.

Advisory ceiling only, by design. gate.py's assess_signal/gate_signals
compute a hard ceiling from fixed-fractional risk, exposure, and notional
caps -- that math is untouched by anything in this module. The LLM may
only narrow further (reduce quantity, or add an extra veto) using cheap
qualitative context; it can never raise the quantity gate.py computed and
can never flip a blocked signal to approved.

RiskAgentVerdict has no field capable of expressing approval or a higher
quantity -- that's the enforcement mechanism, not just a post-hoc check.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel

from advisor.agent.llm import AgentLLM
from advisor.agent.runner import run_tool_loop
from advisor.risk.agent_tools import dispatch, tool_specs
from advisor.risk.gate import RiskLimits, assess_signal, gate_signals
from advisor.scalping.models import ScalpSignal

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 4

_SYSTEM_PROMPT = """\
You are the Risk agent's advisory layer in an options-trading system. A \
deterministic system has already sized and approved this position -- your \
job is ONLY to decide whether to narrow it further based on qualitative \
context (recent news, upcoming earnings), never to approve or size it up.

You may call get_recent_news and get_days_to_earnings for the symbol, then \
respond with a JSON object matching this schema (no other text). Leave \
narrow_quantity null if you see no reason to reduce the position further.
```json
{schema}
```
"""


class RiskAgentVerdict(BaseModel):
    """The LLM's proposal. No field here can express approval or a higher
    quantity -- that's deliberate, not an oversight."""

    narrow_quantity: int | None = None
    veto: bool = False
    reasoning: str = ""


def _parse_verdict(text: str) -> RiskAgentVerdict:
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    try:
        return RiskAgentVerdict.model_validate(json.loads(cleaned))
    except Exception:  # noqa: BLE001
        return RiskAgentVerdict(reasoning=text[:300] if text else "parse failed")


def _narrow_with_llm(signal: ScalpSignal, llm: AgentLLM) -> RiskAgentVerdict:
    schema = json.dumps(RiskAgentVerdict.model_json_schema(), indent=2)
    system_prompt = _SYSTEM_PROMPT.format(schema=schema)
    user_prompt = (
        f"Symbol: {signal.symbol}\n"
        f"Deterministic verdict: approved, quantity={signal.risk_quantity}, "
        f"entry={signal.entry}, stop={signal.stop}, target={signal.target}, "
        f"risk_reward={signal.risk_reward}, score={signal.score}\n"
        f"Strategy: {signal.strategy}"
    )
    loop_result = run_tool_loop(
        llm, system_prompt, user_prompt, tool_specs(), dispatch, max_iterations=_MAX_ITERATIONS
    )
    if loop_result.budget_exhausted:
        return RiskAgentVerdict(reasoning="tool-call budget exhausted, no narrowing applied")
    return _parse_verdict(loop_result.final_text)


def review_signal(
    signal: ScalpSignal,
    *,
    net_liq: float,
    existing_symbol_notional: float = 0.0,
    limits: RiskLimits | None = None,
    llm: AgentLLM | None = None,
) -> ScalpSignal:
    """Deterministic ceiling first, then an optional LLM narrowing pass.

    The LLM can only move the outcome toward more conservative: lower
    quantity, or an added veto. It can never undo what assess_signal denied.
    """
    ceiling = assess_signal(
        signal,
        net_liq=net_liq,
        existing_symbol_notional=existing_symbol_notional,
        limits=limits,
    )
    if not ceiling.risk_approved or not ceiling.risk_quantity:
        return ceiling  # already blocked -- nothing to narrow, skip the LLM call entirely

    llm = llm or AgentLLM()
    if not llm.configured:
        return ceiling  # LLM layer is optional; unavailable = untouched ceiling

    try:
        verdict = _narrow_with_llm(ceiling, llm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("risk agent narrowing failed for %s: %s", signal.symbol, exc)
        return ceiling  # any failure here must fall back to the ceiling, never block silently

    # ── HARD INVARIANT -- enforced in plain Python, never trusted from the LLM ──
    proposed = (
        max(0, verdict.narrow_quantity)
        if verdict.narrow_quantity is not None
        else ceiling.risk_quantity
    )
    final_quantity = min(ceiling.risk_quantity, proposed)  # can only go down
    final_approved = ceiling.risk_approved and not verdict.veto and final_quantity >= 1

    note = ceiling.risk_note
    if verdict.veto:
        note = f"{note}; LLM veto: {verdict.reasoning}".strip("; ")
    elif final_quantity < ceiling.risk_quantity:
        note = (
            f"{note}; LLM narrowed {ceiling.risk_quantity}->{final_quantity}: "
            f"{verdict.reasoning}"
        ).strip("; ")

    return ceiling.model_copy(
        update={
            "risk_approved": final_approved,
            "risk_quantity": final_quantity if final_approved else 0,
            "risk_note": note,
        }
    )


def review_signals(
    signals: list[ScalpSignal],
    *,
    net_liq: float,
    open_notional_by_symbol: dict[str, float] | None = None,
    limits: RiskLimits | None = None,
    use_llm: bool = False,
) -> list[ScalpSignal]:
    """Same deterministic gate as gate_signals(); optionally narrows each
    approved signal further with the LLM advisory layer."""
    gated = gate_signals(
        signals, net_liq=net_liq, open_notional_by_symbol=open_notional_by_symbol, limits=limits
    )
    if not use_llm:
        return gated
    exposure = open_notional_by_symbol or {}
    return [
        review_signal(
            s,
            net_liq=net_liq,
            existing_symbol_notional=exposure.get(s.symbol, 0.0),
            limits=limits,
        )
        for s in gated
    ]
