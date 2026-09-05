"""Tool registry for the Risk agent's advisory narrowing layer.

Deliberately narrow and cheap: only yfinance-backed lookups, no Tavily/LLM
fan-out (build_catalysts already runs upstream in the Signal agent — the
Risk agent doesn't need a second network+LLM round trip to narrow a
position that's already been through the deterministic ceiling).
"""

from __future__ import annotations

import logging
from typing import Callable

from advisor.agent.tools import _truncate

logger = logging.getLogger(__name__)


def _tool_get_recent_news(symbol: str, limit: int = 5) -> dict:
    from advisor.data.news import news_headlines

    headlines = news_headlines(symbol, limit=limit)
    return {"symbol": symbol, "headlines": headlines}


def _tool_get_days_to_earnings(symbol: str) -> dict:
    from advisor.data.news import earnings_context

    earnings_today, days_to = earnings_context(symbol)
    return {"symbol": symbol, "earnings_today": earnings_today, "days_to_earnings": days_to}


_Impl = Callable[..., dict]

_REGISTRY: dict[str, tuple[dict, _Impl]] = {
    "get_recent_news": (
        {
            "type": "function",
            "function": {
                "name": "get_recent_news",
                "description": "Fetch recent (last ~48h) news headlines for a symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "limit": {"type": "integer", "default": 5},
                    },
                    "required": ["symbol"],
                },
            },
        },
        _tool_get_recent_news,
    ),
    "get_days_to_earnings": (
        {
            "type": "function",
            "function": {
                "name": "get_days_to_earnings",
                "description": (
                    "Check whether a symbol has an earnings release today or how many days "
                    "until the next one -- useful for deciding whether to narrow a position "
                    "ahead of an earnings-driven volatility spike."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            },
        },
        _tool_get_days_to_earnings,
    ),
}


def tool_specs() -> list[dict]:
    return [spec for spec, _ in _REGISTRY.values()]


def dispatch(name: str, args: dict) -> dict:
    entry = _REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown tool '{name}'."}
    _, impl = entry
    try:
        result = impl(**(args or {}))
    except Exception as exc:  # noqa: BLE001
        logger.exception("risk agent tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return _truncate(result if isinstance(result, dict) else {"result": result})
