"""Non-streaming tool-calling loop, for background agents that need one final
structured result rather than a live SSE transcript.

Same core mechanics as ``loop.run_agent`` (build messages, call the LLM with
tools, execute any tool_calls and feed results back, repeat) without the
streaming/chat-history shape that feature is coupled to.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from advisor.agent.llm import AgentLLM
from advisor.agent.loop import _describe_llm_error

logger = logging.getLogger(__name__)


@dataclass
class ToolLoopResult:
    final_text: str
    trace: list[dict] = field(default_factory=list)
    iterations: int = 0
    budget_exhausted: bool = False


def run_tool_loop(
    llm: AgentLLM,
    system_prompt: str,
    user_prompt: str,
    tool_specs: list[dict],
    dispatch: Callable[[str, dict], dict],
    *,
    max_iterations: int = 6,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> ToolLoopResult:
    """Run a tool-calling conversation to completion and return the final answer.

    ``dispatch(name, args) -> dict`` executes one tool call; the caller owns
    binding whatever per-run context the tools need (a closure, functools.partial,
    etc.) — this function is context-agnostic.
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    trace: list[dict] = []

    for iteration in range(1, max_iterations + 1):
        try:
            response = llm.chat_with_tools(
                messages, tool_specs, temperature=temperature, max_tokens=max_tokens
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool loop LLM call failed")
            return ToolLoopResult(
                final_text=f"error: {_describe_llm_error(exc)}",
                trace=trace,
                iterations=iteration,
            )

        choice = response.choices[0].message
        tool_calls = choice.tool_calls or []

        if not tool_calls:
            return ToolLoopResult(
                final_text=choice.content or "", trace=trace, iterations=iteration
            )

        assistant_turn = {
            "role": "assistant",
            "content": choice.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_turn)
        trace.append(assistant_turn)

        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = dispatch(name, args)
            tool_turn = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": name,
                "content": json.dumps(result, default=str),
            }
            messages.append({k: v for k, v in tool_turn.items() if k != "name"})
            trace.append(tool_turn)

    return ToolLoopResult(
        final_text="", trace=trace, iterations=max_iterations, budget_exhausted=True
    )
