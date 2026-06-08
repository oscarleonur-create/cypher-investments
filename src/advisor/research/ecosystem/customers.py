"""Customer concentration extraction via Tavily + LLM.

Searches for the company's customer disclosure (10-K "Customers" / "Concentration"
section) and extracts structured entries via the OpenRouter LLM.
Falls back to empty list gracefully when keys are missing.
"""

from __future__ import annotations

import logging

from advisor.research.models import CustomerEntry

logger = logging.getLogger(__name__)


def get_customers(symbol: str, company_name: str = "") -> list[CustomerEntry]:
    """Return top customer entries extracted from SEC/news sources."""
    name = company_name or symbol

    try:
        from pydantic import BaseModel
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM
        from research_agent.search import TavilyClient
        from research_agent.store import Store

        config = ResearchConfig()
        if not config.openrouter_api_key or not config.tavily_api_key:
            return []

        store = Store(config.db_path)
        searcher = TavilyClient(config, store)

        query = (
            f"{name} {symbol} major customers customer concentration revenue "
            f"10-K site:sec.gov OR site:wsj.com"
        )
        results = searcher.search(query, max_results=5)
        if not results:
            return []

        context = "\n\n".join(r.content[:800] for r in results[:3])

        class CustomerList(BaseModel):
            customers: list[dict]  # [{name, concentration_pct, note}]

        llm = OpenRouterLLM(config)
        extracted = llm.complete(
            system_prompt=(
                "You are a financial analyst. Extract the top customers mentioned in the text. "
                "For each customer provide: name (string), concentration_pct (number or null), "
                "note (string). Only include explicitly named customers. Do not invent data."
            ),
            user_prompt=f"Company: {name} ({symbol})\n\nSource text:\n{context}",
            response_model=CustomerList,
        )
        entries: list[CustomerEntry] = []
        for item in extracted.customers[:10]:
            entries.append(
                CustomerEntry(
                    name=str(item.get("name", "")),
                    concentration_pct=_safe_float(item.get("concentration_pct")),
                    note=str(item.get("note", "")),
                )
            )
        return entries
    except Exception as exc:  # noqa: BLE001
        logger.warning("Customer extraction failed for %s: %s", symbol, exc)
        return []


def _safe_float(v: object) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None
