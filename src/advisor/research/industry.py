"""Industry & competitive position analysis — Porter's 5 Forces + moat.

Sources Tavily search results on the competitive landscape, then uses the
OpenRouter LLM to synthesise into structured Porter's 5 + moat assessment.
Falls back to a minimal stub when API keys are missing.
"""

from __future__ import annotations

import logging

from advisor.research.models import (
    IndustryAnalysis,
    MoatType,
    PortersForces,
    PortersForceStrength,
)

logger = logging.getLogger(__name__)

_FALLBACK = (None, MoatType.NONE, "", None, [], "", "", [], [], [])


def build_industry(
    symbol: str,
    company_name: str = "",
    sector: str = "",
    industry: str = "",
) -> IndustryAnalysis:
    """Return Porter's 5 + moat analysis, or a minimal stub on failure."""
    name = company_name or symbol

    (
        porters,
        moat_type,
        moat_desc,
        moat_strength,
        competitive_advantages,
        competitive_position,
        narrative,
        competitors,
        tailwinds,
        headwinds,
    ) = _llm_analysis(symbol, name, sector, industry)

    return IndustryAnalysis(
        symbol=symbol.upper(),
        sector=sector,
        industry=industry,
        porters_forces=porters,
        moat_type=moat_type,
        moat_description=moat_desc,
        moat_strength=moat_strength,
        competitive_advantages=competitive_advantages,
        competitive_position=competitive_position,
        market_share_narrative=narrative,
        key_competitors=competitors,
        sector_tailwinds=tailwinds,
        sector_headwinds=headwinds,
    )


# ── LLM analysis ─────────────────────────────────────────────────────────────


def _llm_analysis(
    symbol: str,
    name: str,
    sector: str,
    industry: str,
) -> tuple:
    try:
        from pydantic import BaseModel
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM
        from research_agent.search import TavilyClient
        from research_agent.store import Store

        config = ResearchConfig()
        if not config.openrouter_api_key or not config.tavily_api_key:
            return _FALLBACK

        store = Store(config.db_path)
        searcher = TavilyClient(config, store)

        query = (
            f"{name} {symbol} competitive landscape market share "
            f"moat advantages competitors industry {sector} {industry} 2025"
        )
        results = searcher.search(query, max_results=8)
        if not results:
            return _FALLBACK

        context = "\n\n".join(f"[{r.title}] {r.content[:600]}" for r in results[:5])

        class IndustryOut(BaseModel):
            competitive_rivalry: str  # LOW | MEDIUM | HIGH
            supplier_power: str
            buyer_power: str
            threat_of_new_entrants: str
            threat_of_substitutes: str
            porters_rationale: str
            moat_type: str  # see MoatType enum values
            moat_description: str
            moat_strength: float  # 0–10: 0=none, 5=narrow, 8+=wide durable moat
            competitive_advantages: list[str]  # 3–5 specific named advantages
            competitive_position: str  # "leader" | "challenger" | "niche" | "laggard"
            market_share_narrative: str
            key_competitors: list[str]
            sector_tailwinds: list[str]  # 2–4 secular growth drivers
            sector_headwinds: list[str]  # 2–4 structural risks

        llm = OpenRouterLLM(config)
        out = llm.complete(
            system_prompt=(
                "You are a strategy analyst. Analyse the competitive position of the company "
                "using Porter's 5 Forces and moat theory. Use only the provided text.\n"
                "Rate each Porter's force LOW | MEDIUM | HIGH (for the threat/power level).\n"
                "moat_type must be exactly one of: network_effect, cost_advantage, "
                "switching_costs, intangible_assets, efficient_scale, none.\n"
                "moat_strength: 0–10 float (0=no moat, 5=narrow, 8+=wide durable moat).\n"
                "competitive_position must be exactly one of: leader, challenger, niche, laggard.\n"
                "competitive_advantages: list of 3–5 specific named advantages "
                "(e.g. 'Apple ecosystem lock-in', not 'brand recognition').\n"
                "sector_tailwinds: 2–4 secular growth drivers for this sector.\n"
                "sector_headwinds: 2–4 structural risks or threats facing the sector."
            ),
            user_prompt=(
                f"Company: {name} ({symbol})\nSector: {sector}\n"
                f"Industry: {industry}\n\n{context}"
            ),
            response_model=IndustryOut,
        )

        def _strength(v: str) -> PortersForceStrength:
            v = v.strip().upper()
            if v == "LOW":
                return PortersForceStrength.LOW
            if v == "HIGH":
                return PortersForceStrength.HIGH
            return PortersForceStrength.MEDIUM

        porters = PortersForces(
            competitive_rivalry=_strength(out.competitive_rivalry),
            supplier_power=_strength(out.supplier_power),
            buyer_power=_strength(out.buyer_power),
            threat_of_new_entrants=_strength(out.threat_of_new_entrants),
            threat_of_substitutes=_strength(out.threat_of_substitutes),
            rationale=out.porters_rationale,
        )

        try:
            moat = MoatType(out.moat_type.lower().replace("-", "_").replace(" ", "_"))
        except ValueError:
            moat = MoatType.NONE

        position = out.competitive_position.lower().strip()
        if position not in {"leader", "challenger", "niche", "laggard"}:
            position = ""

        return (
            porters,
            moat,
            out.moat_description,
            max(0.0, min(10.0, out.moat_strength)),
            out.competitive_advantages,
            position,
            out.market_share_narrative,
            out.key_competitors,
            out.sector_tailwinds,
            out.sector_headwinds,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning("Industry analysis failed for %s: %s", symbol, exc)
        return _FALLBACK
