"""Tool registry for the interactive research agent.

Each tool is an OpenAI-style function spec plus a Python callable that wraps an
*existing* project function — there is no new analysis logic here. ``dispatch``
runs a tool by name and returns a JSON-safe dict; results are truncated before
being handed back to the model to keep the context bounded.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel
from research_agent.config import ResearchConfig

logger = logging.getLogger(__name__)

# Report sections the agent may pull in full via get_report_section.
_REPORT_SECTIONS = [
    "business_model",
    "ecosystem",
    "statements",
    "ratios",
    "red_flags",
    "multiples",
    "dcf",
    "catalyst_risk",
    "network",
    "industry",
    "transcripts",
    "market_data",
    "thesis",
    "consensus",
    "variant_perception",
    "management_quality",
    "investment_memo",
    "x_sentiment",
    "options_flow",
    "kpi_monitor",
    "filings",
    "deep_research",
]

_MAX_RESULT_CHARS = 3500


@dataclass
class ToolContext:
    """Per-conversation context shared by all tool calls."""

    symbol: str
    config: ResearchConfig


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    return obj


def _truncate(payload: dict) -> dict:
    """Cap a tool result's serialized size so it doesn't blow the context."""
    text = json.dumps(payload, default=str)
    if len(text) <= _MAX_RESULT_CHARS:
        return payload
    return {
        "_truncated": True,
        "note": f"Result truncated to {_MAX_RESULT_CHARS} chars.",
        "preview": text[:_MAX_RESULT_CHARS],
    }


def _load_report(ctx: ToolContext):
    from advisor.api import deps
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        return store.load_latest_report(ctx.symbol)
    finally:
        store.close()


def _searcher(ctx: ToolContext):
    from research_agent.search import TavilyClient
    from research_agent.store import Store

    return TavilyClient(ctx.config, Store(ctx.config.db_path))


# ── Tool implementations ──────────────────────────────────────────────────────


def _tool_get_report_section(ctx: ToolContext, section: str) -> dict:
    if section not in _REPORT_SECTIONS:
        return {"error": f"Unknown section '{section}'. Valid: {', '.join(_REPORT_SECTIONS)}"}
    report = _load_report(ctx)
    if report is None:
        return {"error": f"No cached research report for {ctx.symbol}."}
    return {"section": section, "data": _json_safe(getattr(report, section, None))}


def _tool_web_search(
    ctx: ToolContext, query: str, max_results: int = 5, sec_only: bool = False
) -> dict:
    if not ctx.config.tavily_api_key:
        return {"error": "Web search unavailable (no Tavily API key configured)."}
    searcher = _searcher(ctx)
    results = (
        searcher.search_sec(query, max_results=max_results)
        if sec_only
        else searcher.search(query, max_results=max_results)
    )
    return {
        "query": query,
        "results": [{"title": r.title, "url": r.url, "content": r.content[:1200]} for r in results],
    }


def _tool_compute_dcf(ctx: ToolContext, explicit_wacc: float | None = None) -> dict:
    from advisor.research.valuation.dcf import build_dcf

    report = _load_report(ctx)
    statements = report.statements if report else None
    result = build_dcf(ctx.symbol, statements=statements, explicit_wacc=explicit_wacc)
    return _json_safe(result)


def _tool_get_transcripts(ctx: ToolContext) -> dict:
    from advisor.research.transcripts import build_transcripts

    return _json_safe(build_transcripts(ctx.symbol))


def _tool_deep_research(ctx: ToolContext) -> dict:
    from advisor.research.deep_research import build_deep_research

    report = _load_report(ctx)
    filings = report.filings if report else None
    return _json_safe(build_deep_research(ctx.symbol, filings=filings))


def _tool_recompute_bayesian(ctx: ToolContext, overrides: dict | None = None) -> dict:
    from advisor.research.models import BayesianOverrides
    from advisor.research.valuation.bayesian import recompute_bayesian

    report = _load_report(ctx)
    if report is None:
        return {"error": f"No cached research report for {ctx.symbol}."}
    bo = BayesianOverrides(**(overrides or {}))
    return _json_safe(recompute_bayesian(report, bo))


def _tool_get_filing_text(ctx: ToolContext, accession_number: str) -> dict:
    from advisor.research.edgar import EdgarClient

    text = EdgarClient().get_filing_text(accession_number)
    # Filings are large; hand back a leading slice and let the model ask for more
    # specific things via web_search if needed.
    return {"accession_number": accession_number, "text": text[:8000]}


def _tool_get_options_flow(ctx: ToolContext) -> dict:
    from advisor.research.options_flow import build_options_flow

    result = build_options_flow(ctx.symbol)
    if result is None:
        return {"error": f"No options flow available for {ctx.symbol}."}
    return _json_safe(result)


def _tool_get_holdings(ctx: ToolContext) -> dict:
    import asyncio

    from advisor.market.tastytrade_client import get_balances, get_positions

    async def _gather() -> dict:
        balances = await get_balances()
        positions = await get_positions()
        return {"balances": balances, "positions": positions}

    try:
        data = asyncio.run(_gather())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"Could not load live portfolio data: {exc}"}
    sym = ctx.symbol.upper()
    data["positions"] = [
        p
        for p in data["positions"]
        if str(p.get("underlying_symbol", "")).upper() == sym
        or str(p.get("symbol", "")).upper() == sym
    ]
    return _json_safe(data)


def _tool_rebuild_report(ctx: ToolContext) -> dict:
    """Kick off a full report rebuild in the background (non-blocking)."""
    import threading

    from advisor.api import deps

    sym = ctx.symbol
    job_id = deps.new_job("research", target=sym)

    def _run() -> None:
        try:
            from advisor.research.report import build_report

            build_report(sym, force_refresh=True)
            deps.update_job(job_id, status="done", message="report ready")
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent-triggered rebuild failed for %s", sym)
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    threading.Thread(target=_run, daemon=True).start()
    return {
        "job_id": job_id,
        "status": "running",
        "note": "Full report rebuild started in the background; it takes a few minutes.",
    }


# ── Registry ──────────────────────────────────────────────────────────────────

_Impl = Callable[..., dict]

_REGISTRY: dict[str, tuple[dict, _Impl]] = {
    "get_report_section": (
        {
            "type": "function",
            "function": {
                "name": "get_report_section",
                "description": (
                    "Fetch the full detail of one section of the cached research report. "
                    "Use this to inspect data the summary only hints at."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": "string",
                            "enum": _REPORT_SECTIONS,
                            "description": "Which report section to retrieve.",
                        }
                    },
                    "required": ["section"],
                },
            },
        },
        _tool_get_report_section,
    ),
    "web_search": (
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web (Tavily) for recent news, data, or primary sources.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query."},
                        "max_results": {"type": "integer", "default": 5},
                        "sec_only": {
                            "type": "boolean",
                            "default": False,
                            "description": "Restrict the search to sec.gov filings.",
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        _tool_web_search,
    ),
    "compute_dcf": (
        {
            "type": "function",
            "function": {
                "name": "compute_dcf",
                "description": "Run a fresh base/bull/bear DCF valuation for this ticker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "explicit_wacc": {
                            "type": "number",
                            "description": "Optional WACC override (e.g. 0.10 for 10%).",
                        }
                    },
                },
            },
        },
        _tool_compute_dcf,
    ),
    "get_transcripts": (
        {
            "type": "function",
            "function": {
                "name": "get_transcripts",
                "description": "Fetch and summarize the latest earnings call transcripts.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _tool_get_transcripts,
    ),
    "deep_research": (
        {
            "type": "function",
            "function": {
                "name": "deep_research",
                "description": (
                    "Run a fresh citation-backed deep-research brief (SEC + news + company "
                    "website): customers, supply chain, recent developments, management quotes."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _tool_deep_research,
    ),
    "recompute_bayesian": (
        {
            "type": "function",
            "function": {
                "name": "recompute_bayesian",
                "description": (
                    "Recompute the Bayesian fair-value posterior, optionally with what-if "
                    "overrides on drivers, evidence weights, or ecosystem toggles."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "overrides": {
                            "type": "object",
                            "description": (
                                "Optional BayesianOverrides: driver_mean/driver_std "
                                "(maps of driver key→value), evidence_weight (key→0..1), "
                                "ecosystem_active (key→bool)."
                            ),
                        }
                    },
                },
            },
        },
        _tool_recompute_bayesian,
    ),
    "get_filing_text": (
        {
            "type": "function",
            "function": {
                "name": "get_filing_text",
                "description": (
                    "Fetch the text of a specific SEC filing by accession number. Get accession "
                    "numbers from get_report_section('filings')."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "accession_number": {"type": "string"},
                    },
                    "required": ["accession_number"],
                },
            },
        },
        _tool_get_filing_text,
    ),
    "get_options_flow": (
        {
            "type": "function",
            "function": {
                "name": "get_options_flow",
                "description": "Fetch live options flow (put/call ratio, IV, unusual activity).",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _tool_get_options_flow,
    ),
    "get_holdings": (
        {
            "type": "function",
            "function": {
                "name": "get_holdings",
                "description": (
                    "Fetch live TastyTrade account balances and the user's current position(s) "
                    "in this ticker."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _tool_get_holdings,
    ),
    "rebuild_report": (
        {
            "type": "function",
            "function": {
                "name": "rebuild_report",
                "description": (
                    "Trigger a full rebuild of the entire research report in the background. "
                    "Returns a job id immediately; does not block."
                ),
                "parameters": {"type": "object", "properties": {}},
            },
        },
        _tool_rebuild_report,
    ),
}


def tool_specs() -> list[dict]:
    """OpenAI tool specs for every registered tool."""
    return [spec for spec, _ in _REGISTRY.values()]


def dispatch(name: str, args: dict, ctx: ToolContext) -> dict:
    """Run a tool by name, returning a JSON-safe (and size-capped) result."""
    entry = _REGISTRY.get(name)
    if entry is None:
        return {"error": f"Unknown tool '{name}'."}
    _, impl = entry
    try:
        result = impl(ctx, **(args or {}))
    except Exception as exc:  # noqa: BLE001
        logger.exception("tool %s failed", name)
        return {"error": f"{type(exc).__name__}: {exc}"}
    return _truncate(result if isinstance(result, dict) else {"result": result})
