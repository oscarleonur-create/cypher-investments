"""Key supplier extraction via Tavily + LLM.

Same pattern as customers.py — search for supply-chain disclosures, extract
structured entries, return empty list on any failure.
"""

from __future__ import annotations

import logging

from advisor.research.models import SupplierEntry

logger = logging.getLogger(__name__)


def get_suppliers(symbol: str, company_name: str = "") -> list[SupplierEntry]:
    """Return key supplier entries extracted from SEC/news sources."""
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
            f"{name} {symbol} key suppliers supply chain sole source dependency "
            f"10-K risk factors site:sec.gov OR site:reuters.com"
        )
        results = searcher.search(query, max_results=5)
        if not results:
            return []

        context = "\n\n".join(r.content[:800] for r in results[:3])

        class SupplierList(BaseModel):
            suppliers: list[dict]  # [{name, category, note}]

        llm = OpenRouterLLM(config)
        extracted = llm.complete(
            system_prompt=(
                "You are a financial analyst. Extract key suppliers mentioned in the text. "
                "For each supplier provide: name (string), "
                "category (e.g. 'sole-source', 'key input', 'manufacturing'), "
                "note (string). Only include explicitly named suppliers."
            ),
            user_prompt=f"Company: {name} ({symbol})\n\nSource text:\n{context}",
            response_model=SupplierList,
        )
        return [
            SupplierEntry(
                name=str(item.get("name", "")),
                category=str(item.get("category", "")),
                note=str(item.get("note", "")),
            )
            for item in extracted.suppliers[:10]
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supplier extraction failed for %s: %s", symbol, exc)
        return []
