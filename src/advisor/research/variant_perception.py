"""Variant perception layer — our view vs. market consensus.

Answers the core hedge-fund question:
  "What does the market believe, exactly why are we different,
   and what specific edge do we have?"

Data flows in from already-built report layers (DCF, consensus, catalysts,
red flags, industry) so no additional API calls are made beyond what the LLM
needs to synthesise the narrative.
"""

from __future__ import annotations

import logging

from advisor.research.models import (
    AnalystConsensus,
    CatalystRiskResult,
    DcfResult,
    IndustryAnalysis,
    MispricingType,
    RedFlagList,
    VariantPerceptionResult,
)

logger = logging.getLogger(__name__)


def build_variant_perception(
    symbol: str,
    company_name: str = "",
    dcf: DcfResult | None = None,
    consensus: AnalystConsensus | None = None,
    catalyst_risk: CatalystRiskResult | None = None,
    red_flags: RedFlagList | None = None,
    industry: IndustryAnalysis | None = None,
) -> VariantPerceptionResult:
    """Construct the variant perception layer — our view vs. market consensus."""
    sym = symbol.upper()

    # Deterministic fields from existing data
    market_implied_growth = dcf.implied_growth_rate if dcf else None
    consensus_target = consensus.target_price_mean if consensus else None
    consensus_rec = consensus.recommendation_key if consensus else ""
    our_base_price = dcf.base.implied_price if dcf and dcf.base else None
    our_base_growth = dcf.base.assumptions.revenue_growth_yr1_3 if dcf and dcf.base else None

    price_vs_consensus: float | None = None
    if our_base_price and consensus_target and consensus_target > 0:
        price_vs_consensus = (our_base_price / consensus_target) - 1

    # LLM enrichment for the qualitative narrative
    llm = _llm_enrich(
        sym,
        company_name or sym,
        dcf=dcf,
        consensus=consensus,
        catalyst_risk=catalyst_risk,
        red_flags=red_flags,
        industry=industry,
    )

    return VariantPerceptionResult(
        symbol=sym,
        market_implied_growth=market_implied_growth,
        consensus_target_price=consensus_target,
        consensus_recommendation=consensus_rec,
        our_base_price=our_base_price,
        our_base_growth=our_base_growth,
        price_vs_consensus_pct=price_vs_consensus,
        our_edge=llm.get("our_edge", ""),
        consensus_misses=llm.get("consensus_misses", []),
        our_key_insight=llm.get("our_key_insight", ""),
        mispricing_type=MispricingType(llm.get("mispricing_type", MispricingType.NONE)),
        conviction=llm.get("conviction", "MEDIUM"),
    )


# ── LLM narrative ─────────────────────────────────────────────────────────────


def _llm_enrich(
    symbol: str,
    name: str,
    dcf: DcfResult | None,
    consensus: AnalystConsensus | None,
    catalyst_risk: CatalystRiskResult | None,
    red_flags: RedFlagList | None,
    industry: IndustryAnalysis | None,
) -> dict:
    try:
        from pydantic import BaseModel
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM

        config = ResearchConfig()
        if not config.openrouter_api_key:
            return {}

        parts: list[str] = []

        if dcf:
            parts.append(f"Current price: ${dcf.current_price:.2f}  WACC: {dcf.wacc:.1%}")
            if dcf.implied_growth_rate is not None:
                parts.append(
                    f"Market-implied growth rate (reverse-DCF): " f"{dcf.implied_growth_rate:.1%}"
                )
            if dcf.base:
                a = dcf.base.assumptions
                parts.append(
                    f"Our base case: {a.revenue_growth_yr1_3:.1%} growth, "
                    f"{a.target_fcf_margin:.1%} FCF margin → "
                    f"${dcf.base.implied_price:.2f} ({dcf.base.upside_pct:+.1%})"
                )

        if consensus:
            parts.append(
                f"Sell-side consensus ({consensus.n_analysts} analysts): "
                f"target ${consensus.target_price_mean or 0:.2f} "
                f"({consensus.consensus_upside_pct or 0:+.1%}), "
                f"recommendation: {consensus.recommendation_key}, "
                f"revision trend: {consensus.revision_trend}"
            )

        if catalyst_risk:
            cats = [c.description[:80] for c in catalyst_risk.catalysts[:3]]
            risks = [r.description[:80] for r in catalyst_risk.risks[:3]]
            if cats:
                parts.append("Near-term catalysts: " + "; ".join(cats))
            if risks:
                parts.append("Key risks: " + "; ".join(risks))

        if red_flags and red_flags.flags:
            flags = [f"{f.severity.value}: {f.title}" for f in red_flags.flags[:4]]
            parts.append("Red flags: " + "; ".join(flags))

        if industry:
            moat = industry.moat_type.value.replace("_", " ")
            parts.append(
                f"Sector: {industry.sector} | Industry: {industry.industry} | " f"Moat: {moat}"
            )
            if industry.moat_description:
                parts.append(f"Moat detail: {industry.moat_description[:120]}")

        context = "\n".join(parts) or f"Company: {name} ({symbol})"

        class VPOut(BaseModel):
            our_edge: str
            consensus_misses: list[str]
            our_key_insight: str
            mispricing_type: str
            conviction: str

        llm = OpenRouterLLM(config)
        return llm.complete(
            system_prompt=(
                "You are a hedge-fund portfolio manager articulating VARIANT PERCEPTION. "
                "Given the quantitative context, explain concisely: "
                "(1) why the market consensus may be wrong, "
                "(2) what specific information or analytical edge we have, "
                "(3) the single most important insight in one sentence. "
                "mispricing_type must be one of: sentiment_driven, estimate_revision, "
                "multiple_expansion, catalyst_blind, none. "
                "conviction: HIGH | MEDIUM | LOW. "
                "consensus_misses: 2-3 specific things the market is not pricing."
            ),
            user_prompt=f"Company: {name} ({symbol})\n\n{context}",
            response_model=VPOut,
        ).model_dump()

    except Exception as exc:  # noqa: BLE001
        logger.warning("variant_perception LLM failed for %s: %s", symbol, exc)
        return {}
