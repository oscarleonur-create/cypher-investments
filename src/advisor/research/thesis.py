"""Thesis builder — bull / base / bear scenarios with target prices.

Anchors target prices to the Phase-2 DCF scenarios, then uses the LLM to
write scenario descriptions, key assumptions, and what-proves-wrong items.
Falls back to price-only stubs when the LLM is unavailable.
"""

from __future__ import annotations

import logging

from advisor.research.models import (
    CatalystRiskResult,
    DcfResult,
    ThesisResult,
    ThesisScenario,
    VariantPerceptionResult,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROBS = {"bull": 0.25, "base": 0.50, "bear": 0.25}


def build_thesis(
    symbol: str,
    company_name: str = "",
    dcf: DcfResult | None = None,
    catalyst_risk: CatalystRiskResult | None = None,
    variant_perception: VariantPerceptionResult | None = None,
) -> ThesisResult:
    """Return a ThesisResult with bull/base/bear scenarios."""
    name = company_name or symbol
    current_price = dcf.current_price if dcf else None

    bull_price = dcf.bull.implied_price if dcf and dcf.bull else None
    base_price = dcf.base.implied_price if dcf and dcf.base else None
    bear_price = dcf.bear.implied_price if dcf and dcf.bear else None

    def _upside(target: float | None) -> float | None:
        if target is None or current_price is None or current_price == 0:
            return None
        return (target / current_price) - 1

    # Try LLM enrichment; fall back to price stubs
    llm_result = _llm_enrich(symbol, name, dcf, catalyst_risk, variant_perception)

    bull = ThesisScenario(
        scenario="bull",
        probability=llm_result.get("bull_prob", _DEFAULT_PROBS["bull"]),
        target_price=bull_price,
        upside_pct=_upside(bull_price),
        description=llm_result.get("bull_desc", ""),
        key_assumptions=llm_result.get("bull_assumptions", []),
        what_proves_wrong=llm_result.get("bull_wrong", []),
    )
    base = ThesisScenario(
        scenario="base",
        probability=llm_result.get("base_prob", _DEFAULT_PROBS["base"]),
        target_price=base_price,
        upside_pct=_upside(base_price),
        description=llm_result.get("base_desc", ""),
        key_assumptions=llm_result.get("base_assumptions", []),
        what_proves_wrong=llm_result.get("base_wrong", []),
    )
    bear = ThesisScenario(
        scenario="bear",
        probability=llm_result.get("bear_prob", _DEFAULT_PROBS["bear"]),
        target_price=bear_price,
        upside_pct=_upside(bear_price),
        description=llm_result.get("bear_desc", ""),
        key_assumptions=llm_result.get("bear_assumptions", []),
        what_proves_wrong=llm_result.get("bear_wrong", []),
    )

    summary = llm_result.get("summary", "")
    conviction = llm_result.get("conviction", "MEDIUM")

    return ThesisResult(
        symbol=symbol.upper(),
        current_price=current_price,
        bull=bull,
        base=base,
        bear=bear,
        summary=summary,
        conviction=conviction,
    )


# ── LLM enrichment ───────────────────────────────────────────────────────────


def _llm_enrich(
    symbol: str,
    name: str,
    dcf: DcfResult | None,
    catalyst_risk: CatalystRiskResult | None,
    variant_perception: VariantPerceptionResult | None = None,
) -> dict:
    try:
        from pydantic import BaseModel
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM

        config = ResearchConfig()
        if not config.openrouter_api_key:
            return {}

        # Build a compact context block from available data
        context_parts: list[str] = []
        if dcf:
            context_parts.append(
                f"DCF — current price: ${dcf.current_price:.2f}, WACC: {dcf.wacc:.1%}, "
                f"beta: {dcf.beta or 0:.2f}"
            )
            if dcf.base:
                a = dcf.base.assumptions
                context_parts.append(
                    f"Base: growth {a.revenue_growth_yr1_3:.1%}, "
                    f"FCF margin {a.target_fcf_margin:.1%}, "
                    f"implied ${dcf.base.implied_price:.2f} "
                    f"({dcf.base.upside_pct:+.1%})"
                )
            if dcf.bull:
                context_parts.append(
                    f"Bull: growth {dcf.bull.assumptions.revenue_growth_yr1_3:.1%}, "
                    f"implied ${dcf.bull.implied_price:.2f} "
                    f"({dcf.bull.upside_pct:+.1%})"
                )
            if dcf.bear:
                context_parts.append(
                    f"Bear: growth {dcf.bear.assumptions.revenue_growth_yr1_3:.1%}, "
                    f"implied ${dcf.bear.implied_price:.2f} "
                    f"({dcf.bear.upside_pct:+.1%})"
                )

        if catalyst_risk:
            top_risks = [r.description[:80] for r in catalyst_risk.risks[:3]]
            top_cats = [c.description[:80] for c in catalyst_risk.catalysts[:3]]
            if top_risks:
                context_parts.append("Key risks: " + "; ".join(top_risks))
            if top_cats:
                context_parts.append("Key catalysts: " + "; ".join(top_cats))

        if variant_perception:
            vp = variant_perception
            if vp.our_key_insight:
                context_parts.append(f"Our edge: {vp.our_key_insight}")
            if vp.consensus_misses:
                context_parts.append("Consensus blind spots: " + "; ".join(vp.consensus_misses[:3]))
            if vp.price_vs_consensus_pct is not None:
                direction = "above" if vp.price_vs_consensus_pct > 0 else "below"
                context_parts.append(
                    f"Our base price is {abs(vp.price_vs_consensus_pct):.1%} "
                    f"{direction} sell-side consensus target"
                )

        context = "\n".join(context_parts) or f"Company: {name} ({symbol})"

        class ThesisOut(BaseModel):
            bull_prob: float
            bull_desc: str
            bull_assumptions: list[str]
            bull_wrong: list[str]
            base_prob: float
            base_desc: str
            base_assumptions: list[str]
            base_wrong: list[str]
            bear_prob: float
            bear_desc: str
            bear_assumptions: list[str]
            bear_wrong: list[str]
            summary: str
            conviction: str  # HIGH | MEDIUM | LOW

        llm = OpenRouterLLM(config)
        return llm.complete(
            system_prompt=(
                "You are a portfolio manager writing an investment thesis. "
                "Given the valuation context, write concise bull/base/bear scenarios. "
                "Probabilities must sum to 1.0. "
                "key_assumptions: 2-3 strings. what_proves_wrong: 2-3 strings. "
                "conviction: HIGH | MEDIUM | LOW based on data quality and spread of scenarios."
            ),
            user_prompt=f"Company: {name} ({symbol})\n\n{context}",
            response_model=ThesisOut,
        ).model_dump()

    except Exception as exc:  # noqa: BLE001
        logger.warning("Thesis LLM enrichment failed for %s: %s", symbol, exc)
        return {}
