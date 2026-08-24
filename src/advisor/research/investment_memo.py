"""Investment Memo builder — IC-presentation format readable in under 90 seconds.

Deterministic fields are assembled directly from the already-built ResearchReport.
One LLM call synthesises the executive summary, max-loss scenario, and exit triggers.
Falls back gracefully when the LLM is unavailable.
"""

from __future__ import annotations

import logging

from advisor.research.models import InvestmentMemo, ResearchReport

logger = logging.getLogger(__name__)

# Conviction → position size tiers (% of portfolio, low–high)
_SIZING = {
    "HIGH": (5.0, 8.0),
    "MEDIUM": (2.0, 5.0),
    "LOW": (0.5, 2.0),
}


def build_investment_memo(
    report: ResearchReport,
    account_size: float | None = None,  # noqa: ARG001 — reserved for dollar sizing later
) -> InvestmentMemo:
    sym = report.symbol
    memo = InvestmentMemo(symbol=sym, company_name=report.business_model or sym)

    # ── §1 Edge ────────────────────────────────────────────────────────────────
    vp = report.variant_perception
    if vp:
        memo.key_insight = vp.our_key_insight
        memo.mispricing_type = (
            vp.mispricing_type.value
            if hasattr(vp.mispricing_type, "value")
            else str(vp.mispricing_type)
        )

    # ── §2 Business Quality ────────────────────────────────────────────────────
    if report.industry:
        memo.moat_type = (
            report.industry.moat_type.value
            if hasattr(report.industry.moat_type, "value")
            else str(report.industry.moat_type)
        )
        memo.moat_description = report.industry.moat_description

    if report.management_quality:
        memo.management_quality_score = report.management_quality.composite_score
        memo.management_quality_tier = report.management_quality.quality_tier

    # ── §3 Valuation ──────────────────────────────────────────────────────────
    dcf = report.dcf
    if dcf:
        memo.current_price = dcf.current_price
        if dcf.bear:
            memo.bear_target = dcf.bear.implied_price
            memo.bear_upside = dcf.bear.upside_pct
        if dcf.base:
            memo.base_target = dcf.base.implied_price
            memo.base_upside = dcf.base.upside_pct
        if dcf.bull:
            memo.bull_target = dcf.bull.implied_price
            memo.bull_upside = dcf.bull.upside_pct

    cn = report.consensus
    if cn:
        memo.consensus_target = cn.target_price_mean
        memo.consensus_upside = cn.consensus_upside_pct
        memo.n_analysts = cn.n_analysts

    # ── §4 Catalysts ──────────────────────────────────────────────────────────
    if report.catalyst_risk and report.catalyst_risk.catalysts:
        memo.key_catalysts = [c.description for c in report.catalyst_risk.catalysts[:3]]

    # ── §5 Position Sizing ────────────────────────────────────────────────────
    conviction = "MEDIUM"
    if report.thesis:
        conviction = report.thesis.conviction
    elif vp:
        conviction = vp.conviction
    memo.conviction = conviction
    lo, hi = _SIZING.get(conviction, _SIZING["MEDIUM"])
    memo.position_size_pct_low = lo
    memo.position_size_pct_high = hi
    memo.sizing_rationale = (
        f"{conviction} conviction → {lo:.1f}–{hi:.1f}% of portfolio. "
        "Size up to high end only if >3 independent signals align."
    )

    # ── LLM enrichment (executive_summary, max_loss, exit_triggers) ───────────
    llm_out = _llm_enrich(report)
    memo.executive_summary = llm_out.get("executive_summary", _fallback_summary(report))
    memo.max_loss_scenario = llm_out.get("max_loss_scenario", "")
    memo.bear_case_deep_dive = llm_out.get("bear_case_deep_dive", "")
    memo.exit_triggers = llm_out.get("exit_triggers", _fallback_exit_triggers(report))

    return memo


# ── LLM enrichment ────────────────────────────────────────────────────────────


def _llm_enrich(report: ResearchReport) -> dict:
    try:
        from pydantic import BaseModel as PydanticBase
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM

        config = ResearchConfig()
        if not config.openrouter_api_key:
            return {}

        context = _build_context(report)

        class MemoOut(PydanticBase):
            executive_summary: str
            max_loss_scenario: str
            bear_case_deep_dive: str
            exit_triggers: list[str]

        llm = OpenRouterLLM(config)
        result = llm.complete(
            system_prompt=(
                "You are a hedge-fund portfolio manager writing a concise "
                "Investment Committee memo. "
                "Use ONLY the facts and figures given in the context below. Do NOT "
                "invent additional numbers, percentages, price targets, multiples, "
                "or thresholds that are not explicitly stated or directly computable "
                "from it — a fabricated-but-plausible-sounding figure is worse than "
                "none, since this memo drives real position-sizing and exit decisions. "
                "executive_summary: 3 sentences max — what the company does, "
                "our edge, and why now. "
                "max_loss_scenario: 2 sentences — if we are completely wrong, what is "
                "the worst case; only state a downside %/multiple if one is derivable "
                "from the given context, otherwise describe it qualitatively. "
                "bear_case_deep_dive: 3–4 sentences expanding ONLY on the bear "
                "elements already given (what_proves_wrong, risks) — do not introduce "
                "new risk factors not present in the context. "
                "exit_triggers: 4–6 conditions that would force a position review — "
                "use a specific number only where the context supplies one to anchor "
                "to (e.g. reuse the given risk/threshold), otherwise state the "
                "condition qualitatively (e.g. 'thesis-defining customer relationship "
                "deteriorates') rather than inventing a threshold. "
                "No marketing language."
            ),
            user_prompt=context,
            response_model=MemoOut,
        )
        return result.model_dump()

    except Exception as exc:  # noqa: BLE001
        logger.warning("InvestmentMemo LLM enrichment failed for %s: %s", report.symbol, exc)
        return {}


def _build_context(report: ResearchReport) -> str:
    parts: list[str] = [f"Company: {report.business_model or report.symbol} ({report.symbol})"]

    vp = report.variant_perception
    if vp and vp.our_key_insight:
        parts.append(f"Our edge: {vp.our_key_insight}")
    if vp and vp.our_edge:
        parts.append(f"Why we differ: {vp.our_edge[:300]}")

    dcf = report.dcf
    if dcf:
        parts.append(f"DCF: current ${dcf.current_price:.2f}, WACC {dcf.wacc:.1%}")
        if dcf.base:
            parts.append(f"Base case: ${dcf.base.implied_price:.2f} ({dcf.base.upside_pct:+.1%})")
        if dcf.bear:
            parts.append(f"Bear case: ${dcf.bear.implied_price:.2f} ({dcf.bear.upside_pct:+.1%})")

    if report.thesis:
        if report.thesis.summary:
            parts.append(f"Thesis summary: {report.thesis.summary[:300]}")
        if report.thesis.bear and report.thesis.bear.what_proves_wrong:
            parts.append(
                "What proves wrong: " + "; ".join(report.thesis.bear.what_proves_wrong[:3])
            )

    if report.catalyst_risk and report.catalyst_risk.high_risks:
        parts.append(
            "Key risks: "
            + "; ".join(r.description[:80] for r in report.catalyst_risk.high_risks[:3])
        )

    if report.industry:
        parts.append(
            f"Moat: {report.industry.moat_type.value} — {report.industry.moat_description[:120]}"
        )

    return "\n".join(parts)


# ── Fallbacks ─────────────────────────────────────────────────────────────────


def _fallback_summary(report: ResearchReport) -> str:
    parts: list[str] = []
    if report.business_model:
        parts.append(report.business_model)
    vp = report.variant_perception
    if vp and vp.our_key_insight:
        parts.append(vp.our_key_insight)
    dcf = report.dcf
    if dcf and dcf.base:
        parts.append(
            f"Base case implies ${dcf.base.implied_price:.2f} ({dcf.base.upside_pct:+.1%} upside)."
        )
    return " ".join(parts) if parts else f"{report.symbol} — full research report available."


def _fallback_exit_triggers(report: ResearchReport) -> list[str]:
    triggers: list[str] = []
    dcf = report.dcf
    if dcf and dcf.bear:
        triggers.append(
            f"Stock falls to bear-case target ${dcf.bear.implied_price:.2f} without thesis change."
        )
    if report.thesis and report.thesis.bear and report.thesis.bear.what_proves_wrong:
        for item in report.thesis.bear.what_proves_wrong[:2]:
            triggers.append(item)
    if report.catalyst_risk and report.catalyst_risk.high_risks:
        triggers.append(report.catalyst_risk.high_risks[0].description[:100])
    triggers.append("Revenue growth decelerates 3 consecutive quarters below base assumptions.")
    return triggers[:5]
