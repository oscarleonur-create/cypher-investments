"""The agentic loop: drive a tool-calling conversation and emit stream events.

``run_agent`` is a synchronous generator (it runs in FastAPI's threadpool behind
a StreamingResponse). It yields plain dicts that the router serializes as SSE:

    {"type": "tool_call",   "name": str, "args": dict}
    {"type": "tool_result", "name": str, "ok": bool}
    {"type": "token",       "text": str}
    {"type": "done",        "text": str}          # full final answer
    {"type": "error",       "message": str}
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

from advisor.agent.llm import AgentLLM
from advisor.agent.tools import ToolContext, dispatch, tool_specs

logger = logging.getLogger(__name__)

_MAX_ITERATIONS = 8
_SUMMARY_SECTIONS = [
    "business_model",
    "thesis",
    "red_flags",
    "consensus",
    "kpi_monitor",
]
_SUMMARY_CHAR_CAP = 1800

_SYSTEM_TEMPLATE = """\
You are an equity research analyst assistant embedded in an investment dashboard. \
The user is analyzing {symbol} and already has a pre-computed research report.

You can call tools to read any section of that report in full, search the web, and \
re-run project compute (DCF, transcripts, deep research, Bayesian fair value, SEC \
filing text, options flow) and live portfolio data. Prefer reading the existing \
report before recomputing; only recompute or search the web when the user needs \
fresh, deeper, or missing information.

Be precise and evidence-based. Cite specific numbers and name your sources (URLs for \
web results, accession numbers for filings). If a tool returns an error or no data, \
say so plainly rather than guessing.

── Report summary for {symbol} ──
{summary}
"""


def build_system_prompt(report: Any, symbol: str) -> str:
    """Compact report summary + instructions. Full detail is fetched via tools."""
    summary = "(no cached report — use tools to gather data)"
    if report is not None:
        parts: list[str] = []
        for section in _SUMMARY_SECTIONS:
            value = getattr(report, section, None)
            if value is None:
                continue
            try:
                if hasattr(value, "model_dump"):
                    rendered = json.dumps(value.model_dump(mode="json"), default=str)
                else:
                    rendered = str(value)
            except Exception:  # noqa: BLE001
                continue
            parts.append(f"{section}: {rendered}")
        if parts:
            summary = "\n".join(parts)[:_SUMMARY_CHAR_CAP]
    return _SYSTEM_TEMPLATE.format(symbol=symbol, summary=summary)


def _describe_llm_error(exc: Exception) -> str:
    """Turn an opaque RetryError/APIStatusError into a readable message.

    tenacity wraps the real error in a RetryError after exhausting retries; the
    OpenRouter/Anthropic error underneath carries the status code and a body we
    actually want to show (rate limit, insufficient credits, bad request, …).
    """
    # Unwrap tenacity's RetryError to the last underlying exception.
    last = getattr(exc, "last_attempt", None)
    if last is not None:
        try:
            exc = last.exception() or exc
        except Exception:  # noqa: BLE001
            pass

    status = getattr(exc, "status_code", None)
    body = ""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            body = response.text[:300]
        except Exception:  # noqa: BLE001
            body = ""

    if status == 429:
        return "Rate limited by the model provider (429). Wait a moment and try again."
    if status == 402:
        return "Model request rejected (402): the OpenRouter account is out of credits."
    if status in (401, 403):
        return f"Model auth failed ({status}). Check RESEARCH_AGENT_OPENROUTER_API_KEY in .env."
    if status in (502, 503, 529):
        return f"Model provider temporarily overloaded ({status}). Try again shortly."
    if status is not None:
        return f"Model call failed (HTTP {status}): {body or exc}"
    return f"Model call failed: {exc}"


def _chunk(text: str, size: int = 24) -> Iterator[str]:
    """Yield small slices so the UI can render the answer progressively."""
    for i in range(0, len(text), size):
        yield text[i : i + size]


def run_agent(
    ctx: ToolContext,
    history: list[dict],
    user_message: str,
    report: Any,
    llm: AgentLLM | None = None,
) -> Iterator[dict]:
    """Run one user turn to completion, yielding stream events.

    ``history`` is the prior visible transcript (role/content dicts). Tool
    activity from earlier turns is not replayed to the model.
    """
    llm = llm or AgentLLM()
    if not llm.configured:
        yield {"type": "error", "message": "LLM unavailable (no OpenRouter API key configured)."}
        return

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(report, ctx.symbol)}
    ]
    for m in history:
        role = m.get("role")
        if role in ("user", "assistant") and m.get("content"):
            messages.append({"role": role, "content": m["content"]})
    messages.append({"role": "user", "content": user_message})

    specs = tool_specs()

    for _ in range(_MAX_ITERATIONS):
        try:
            response = llm.chat_with_tools(messages, specs)
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent LLM call failed for %s", ctx.symbol)
            yield {"type": "error", "message": _describe_llm_error(exc)}
            return

        choice = response.choices[0].message
        tool_calls = choice.tool_calls or []

        if not tool_calls:
            text = choice.content or ""
            for piece in _chunk(text):
                yield {"type": "token", "text": piece}
            yield {"type": "done", "text": text}
            return

        # Record the assistant's tool-call turn, then execute each call.
        messages.append(
            {
                "role": "assistant",
                "content": choice.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            }
        )

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"type": "tool_call", "name": name, "args": args}
            result = dispatch(name, args, ctx)
            yield {"type": "tool_result", "name": name, "ok": "error" not in result}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                }
            )

    yield {
        "type": "error",
        "message": "Reached the tool-call limit without a final answer. Try a narrower question.",
    }
