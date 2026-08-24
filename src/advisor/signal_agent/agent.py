"""Autonomous Signal agent: decides WHAT to scan and WHY, not how to size risk.

The LLM reasons over a small toolset (universe/watchlist lookup, catalysts,
scalp/swing scanners) and produces a rationale — the actual ScalpSignal
objects it triggers are collected via a side-channel on SignalToolContext,
never re-parsed out of the model's own (truncated) text. After the loop,
the deterministic risk gate runs over whatever scalp signals were collected,
exactly as the plain /api/scalping/run path does today — this agent's
autonomy is scoped to attention allocation, never to position sizing.
"""

from __future__ import annotations

import json
import logging
import re

from advisor.agent.llm import AgentLLM
from advisor.agent.runner import run_tool_loop
from advisor.risk.gate import gate_signals
from advisor.signal_agent.models import SignalAgentResult, SignalAgentVerdict
from advisor.signal_agent.tools import SignalToolContext, make_dispatch, tool_specs

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 6

_SYSTEM_PROMPT = """\
You are the Signal agent in an options-trading research workstation. Your job \
is to decide WHICH symbols are worth scanning right now and WHY, then run the \
appropriate scanner(s) on them — not to invent trade ideas yourself.

Use list_universe/get_watchlist to see what's available, get_catalysts to check \
whether a symbol has a near-term reason to move, then run_scalp_scan (intraday) \
and/or run_swing_scan (multi-day) on the symbols you've decided are worth it. \
Don't scan an entire large universe blindly — use catalysts and the stated \
objective to narrow down first.

You do NOT decide position sizing or risk approval — a separate deterministic \
system handles that after you're done. Your final answer must be a JSON object \
matching this schema (no other text):
```json
{schema}
```
"""


def _parse_verdict(text: str) -> SignalAgentVerdict:
    cleaned = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    try:
        return SignalAgentVerdict.model_validate(json.loads(cleaned))
    except Exception:  # noqa: BLE001
        return SignalAgentVerdict(rationale=text[:500] if text else "")


def run_signal_agent(
    objective: str,
    *,
    universe: str = "semiconductors",
    max_symbols: int = 30,
    net_liq: float = 0.0,
    open_notional_by_symbol: dict[str, float] | None = None,
    llm: AgentLLM | None = None,
) -> SignalAgentResult:
    llm = llm or AgentLLM()
    ctx = SignalToolContext(universe_cap=max_symbols)

    if not llm.configured:
        return SignalAgentResult(
            objective=objective,
            notes="LLM unavailable (no OpenRouter API key configured).",
        )

    schema = json.dumps(SignalAgentVerdict.model_json_schema(), indent=2)
    system_prompt = _SYSTEM_PROMPT.format(schema=schema)
    user_prompt = f"Objective: {objective}\nDefault universe if none is specified: {universe}"

    loop_result = run_tool_loop(
        llm,
        system_prompt,
        user_prompt,
        tool_specs(),
        make_dispatch(ctx),
        max_iterations=_MAX_ITERATIONS,
    )

    verdict = _parse_verdict(loop_result.final_text)
    # Same deterministic gate as /api/scalping/run, run with whatever account
    # state the caller passed in (net_liq=0.0 by default -> everything blocked
    # with "no account data", the same honest behavior gate.py already has
    # when no real account context is available).
    gated = (
        gate_signals(
            ctx.collected_scalp,
            net_liq=net_liq,
            open_notional_by_symbol=open_notional_by_symbol,
        )
        if ctx.collected_scalp
        else []
    )

    result = SignalAgentResult(
        objective=objective,
        rationale=verdict.rationale,
        focus_symbols=verdict.focus_symbols,
        notes=verdict.notes,
        signals=gated,
        swing_signals=ctx.collected_swing,
        budget_exhausted=loop_result.budget_exhausted,
    )

    try:
        from advisor.api import deps
        from advisor.research.store import ResearchStore

        store = ResearchStore(deps.db_path())
        try:
            result.run_id = store.save_agent_run(
                "signal", objective, loop_result.trace, result.model_dump(mode="json")
            )
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("signal agent run logging failed: %s", exc)

    return result
