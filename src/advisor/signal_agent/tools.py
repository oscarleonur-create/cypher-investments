"""Tool registry for the autonomous Signal agent.

Same convention as ``advisor.agent.tools``: each tool is an OpenAI-style
function spec plus a Python callable wrapping an *existing* project
function — no new scanning/analysis logic lives here. Tool results handed
back to the model are size-capped via the same ``_truncate`` helper; the
actual signal objects are collected on ``SignalToolContext`` untruncated,
since a 3500-char cap would silently drop data for a real scan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from advisor.agent.tools import _json_safe, _truncate
from advisor.api import deps
from advisor.scalping.models import ScalpSignal
from advisor.scalping.strategies import SCALP_STRATEGIES

logger = logging.getLogger(__name__)

_MAX_SYMBOLS_PER_CALL = 20


@dataclass
class SignalToolContext:
    """Per-run context shared by all tool calls. ``collected_*`` are the
    untruncated source of truth for the signals a run actually returns —
    never re-derived from the (truncated) text the model sees."""

    universe_cap: int = 30
    collected_scalp: list[ScalpSignal] = field(default_factory=list)
    collected_swing: list[dict] = field(default_factory=list)


def _tool_list_universe(
    ctx: SignalToolContext, universe: str, symbols: list[str] | None = None
) -> dict:
    resolved = deps.resolve_symbols(universe, symbols or [], ctx.universe_cap)
    return {"universe": universe, "symbols": resolved}


def _tool_get_watchlist(ctx: SignalToolContext) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        rows = store.load_watchlist()
    finally:
        store.close()
    return {"watchlist": [{"symbol": r["symbol"], "note": r["note"]} for r in rows]}


def _tool_get_catalysts(ctx: SignalToolContext, symbol: str) -> dict:
    from advisor.research.catalysts import build_catalysts

    result = build_catalysts(symbol)
    return _json_safe(result)


def _tool_run_scalp_scan(
    ctx: SignalToolContext,
    symbols: list[str],
    strategy: str | None = None,
    min_rvol: float = 1.5,
) -> dict:
    from advisor.scalping.scanner import ScalpScanner

    if strategy and strategy not in SCALP_STRATEGIES:
        return {"error": f"Unknown strategy '{strategy}'. Valid: {', '.join(SCALP_STRATEGIES)}"}

    syms = [s.strip().upper() for s in symbols if s.strip()][:_MAX_SYMBOLS_PER_CALL]
    result = ScalpScanner().scan(
        syms,
        strategy_names=[strategy] if strategy else None,
        catalysts=True,
        min_rvol=min_rvol,
        use_llm=False,  # no nested LLM sentiment call from inside a tool call
    )
    ctx.collected_scalp.extend(result.signals)
    return _json_safe(result)


def _tool_run_swing_scan(
    ctx: SignalToolContext, symbols: list[str], strategy: str | None = None
) -> dict:
    from advisor.confluence.orchestrator import run_confluence

    strategy = strategy or "momentum_breakout"
    syms = [s.strip().upper() for s in symbols if s.strip()][:_MAX_SYMBOLS_PER_CALL]
    scanned: list[dict] = []
    errors: list[str] = []
    for sym in syms:
        try:
            r = run_confluence(sym, strategy_name=strategy, force_all=True, include_ml=False)
            row = {
                "symbol": sym,
                "strategy": strategy,
                "verdict": r.verdict.value,
                "reasoning": r.reasoning,
                "suggested_hold_days": r.suggested_hold_days,
            }
            scanned.append(row)
            ctx.collected_swing.append(row)
        except Exception as exc:  # noqa: BLE001
            logger.warning("signal agent swing scan failed for %s: %s", sym, exc)
            errors.append(f"{sym}: {exc}")
    return {"signals": scanned, "errors": errors}


_Impl = Callable[..., dict]

_REGISTRY: dict[str, tuple[dict, _Impl]] = {
    "list_universe": (
        {
            "type": "function",
            "function": {
                "name": "list_universe",
                "description": (
                    "Resolve a named universe (e.g. 'semiconductors', 'sp500') or a custom "
                    "symbol list into the actual ticker list you can scan. Call this before "
                    "run_scalp_scan/run_swing_scan to see what's available."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "universe": {
                            "type": "string",
                            "description": "'semiconductors' | 'sp500' | 'custom'",
                        },
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Only used when universe == 'custom'.",
                        },
                    },
                    "required": ["universe"],
                },
            },
        },
        _tool_list_universe,
    ),
    "get_watchlist": (
        {
            "type": "function",
            "function": {
                "name": "get_watchlist",
                "description": "Fetch the user's saved watchlist symbols and notes.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _tool_get_watchlist,
    ),
    "get_catalysts": (
        {
            "type": "function",
            "function": {
                "name": "get_catalysts",
                "description": (
                    "Fetch upcoming catalysts (earnings, product launches, regulatory events) "
                    "for a symbol, to decide whether it's worth scanning right now."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
            },
        },
        _tool_get_catalysts,
    ),
    "run_scalp_scan": (
        {
            "type": "function",
            "function": {
                "name": "run_scalp_scan",
                "description": (
                    "Run the intraday scalp scanner (technical + catalyst-adjusted) over a list "
                    "of symbols. Call this once you've decided WHICH symbols are worth checking "
                    "for an intraday setup right now — don't scan symbols with no reason to "
                    "expect a move."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Up to {_MAX_SYMBOLS_PER_CALL} tickers.",
                        },
                        "strategy": {"type": "string", "enum": list(SCALP_STRATEGIES.keys())},
                        "min_rvol": {"type": "number", "default": 1.5},
                    },
                    "required": ["symbols"],
                },
            },
        },
        _tool_run_scalp_scan,
    ),
    "run_swing_scan": (
        {
            "type": "function",
            "function": {
                "name": "run_swing_scan",
                "description": (
                    "Run the multi-day confluence scanner (technical + sentiment + fundamental) "
                    "over a list of symbols for swing-trade setups."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": f"Up to {_MAX_SYMBOLS_PER_CALL} tickers.",
                        },
                        "strategy": {
                            "type": "string",
                            "description": (
                                "e.g. momentum_breakout, buy_the_dip, pead, "
                                "mean_reversion, sma_crossover"
                            ),
                        },
                    },
                    "required": ["symbols"],
                },
            },
        },
        _tool_run_swing_scan,
    ),
}


def tool_specs() -> list[dict]:
    return [spec for spec, _ in _REGISTRY.values()]


def make_dispatch(ctx: SignalToolContext) -> Callable[[str, dict], dict]:
    """Bind a dispatch function to a specific run's context."""

    def dispatch(name: str, args: dict) -> dict:
        entry = _REGISTRY.get(name)
        if entry is None:
            return {"error": f"Unknown tool '{name}'."}
        _, impl = entry
        try:
            result = impl(ctx, **(args or {}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("signal agent tool %s failed", name)
            return {"error": f"{type(exc).__name__}: {exc}"}
        return _truncate(result if isinstance(result, dict) else {"result": result})

    return dispatch
