"""Stage 6: Case synthesis — LLM generates final trade case."""

from __future__ import annotations

import logging

from advisor.strategy_case.models import (
    CaseSynthesis,
    CaseVerdict,
    OptionsAnalysisResult,
    ResearchSummary,
    RiskProfile,
    ScenarioResult,
    StrategyRanking,
)
from advisor.strategy_case.prompts import CASE_SYNTHESIS_SYSTEM, build_synthesis_prompt

logger = logging.getLogger(__name__)


def synthesize_case(
    symbol: str,
    scenario: ScenarioResult,
    ranking: StrategyRanking,
    options: OptionsAnalysisResult,
    risk: RiskProfile | None = None,
    research: ResearchSummary | None = None,
) -> CaseSynthesis:
    """Call Claude to synthesize all stage outputs into a trade case."""
    try:
        from research_agent.config import ResearchConfig
        from research_agent.llm import ClaudeLLM

        config = ResearchConfig()
        llm = ClaudeLLM(config)
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return _fallback_synthesis(scenario, ranking, options, risk, research)

    selected = ranking.selected
    strategy_name = selected.strategy.value if selected else "unknown"
    strategy_reasoning = selected.reasoning if selected else ""

    # Format strike details
    strike_lines = []
    for i, rec in enumerate(options.recommendations[:3], 1):
        line = f"{i}. {rec.strategy} — " f"${rec.strike}"
        if rec.long_strike:
            line += f"/${rec.long_strike}"
        line += (
            f" exp {rec.expiry}, {rec.dte}DTE | "
            f"credit ${rec.credit:.2f} | delta {rec.delta:.2f} | "
            f"POP {rec.pop:.0%} | sell score {rec.sell_score:.0f} | "
            f"yield {rec.annualized_yield:.0%}"
        )
        if rec.flags:
            line += f" | flags: {', '.join(rec.flags)}"
        strike_lines.append(line)

    strike_details = "\n".join(strike_lines) if strike_lines else "No qualifying strikes found."
    strike_details += (
        f"\nIV Percentile: {options.iv_percentile:.0f} | Term Structure: {options.term_structure}"
    )

    # Format risk details
    if risk:
        risk_details = (
            f"Source: {risk.source.upper()}\n"
            f"POP: {risk.pop:.1%} | EV: ${risk.ev:.2f} | "
            f"EV/BP: {risk.ev_per_bp:.4f}\n"
        )
        if risk.source == "mc":
            risk_details += (
                f"CVaR95: ${risk.cvar_95:.2f} | Stop Prob: {risk.stop_prob:.1%} | "
                f"Touch Prob: {risk.touch_prob:.1%}\n"
            )
        risk_details += (
            f"Sizing: {risk.suggested_contracts} contracts | "
            f"Max Loss: ${risk.max_loss_total:.0f} | "
            f"Risk: {risk.risk_pct:.1f}% of account"
        )
    else:
        risk_details = "No risk assessment available."

    # Format research summary
    research_text = None
    if research:
        research_text = (
            f"Verdict: {research.verdict}\n"
            f"Bull Case: {'; '.join(research.bull_case[:3])}\n"
            f"Bear Case: {'; '.join(research.bear_case[:3])}\n"
            f"Catalyst: {research.catalyst_summary}\n"
            f"Revenue Growth: {research.key_metrics_revenue_growth} | "
            f"Margins: {research.key_metrics_margins} | "
            f"FCF: {research.key_metrics_fcf}\n"
            f"Grounding: {research.grounding_score:.0%}"
        )

    user_prompt = build_synthesis_prompt(
        symbol=symbol,
        scenario_summary=scenario.summary,
        strategy_name=strategy_name,
        strategy_reasoning=strategy_reasoning,
        strike_details=strike_details,
        risk_details=risk_details,
        research_summary=research_text,
    )

    try:
        result = llm.complete(
            system_prompt=CASE_SYNTHESIS_SYSTEM,
            user_prompt=user_prompt,
            response_model=CaseSynthesis,
        )
        return result
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        return _fallback_synthesis(scenario, ranking, options, risk, research)


def _fallback_synthesis(
    scenario: ScenarioResult,
    ranking: StrategyRanking,
    options: OptionsAnalysisResult,
    risk: RiskProfile | None = None,
    research: ResearchSummary | None = None,
) -> CaseSynthesis:
    """Generate a basic synthesis without LLM when API is unavailable."""
    selected = ranking.selected
    strategy_name = selected.strategy.value if selected else "unknown"
    fit_score = selected.fit_score if selected else 0

    # Build thesis
    top_rec = options.recommendations[0] if options.recommendations else None
    if top_rec:
        thesis = (
            f"{scenario.summary} "
            f"Best fit: {strategy_name} (fit {fit_score:.0f}/100). "
            f"Top strike: ${top_rec.strike}"
        )
        if top_rec.long_strike:
            thesis += f"/${top_rec.long_strike}"
        thesis += f" at {top_rec.dte}DTE for ${top_rec.credit:.2f} credit."
    else:
        thesis = f"{scenario.summary} Strategy: {strategy_name}. No qualifying strikes found."

    # Compute conviction
    conviction = _compute_conviction(scenario, ranking, options, risk, research)

    # Determine verdict
    if conviction >= 75:
        verdict = CaseVerdict.STRONG
    elif conviction >= 50:
        verdict = CaseVerdict.MODERATE
    elif conviction >= 25:
        verdict = CaseVerdict.WEAK
    else:
        verdict = CaseVerdict.REJECT

    return CaseSynthesis(
        thesis_summary=thesis,
        entry_criteria=[
            f"Scenario: {scenario.scenario_type.value} with {scenario.confidence:.0%} confidence"
        ],
        exit_plan=[
            "Close at 50% profit",
            f"Close at {top_rec.dte // 2}DTE if open" if top_rec else "Time-based exit",
        ],
        invalidation=["Scenario conditions change", "Underlying breaks key support"],
        risks=[
            "IV crush risk",
            "Earnings proximity" if top_rec and "earnings" in top_rec.flags else "Gap risk",
        ],
        management_plan=["Monitor daily", "Roll if tested"],
        verdict=verdict,
        conviction_score=conviction,
    )


def _compute_conviction(
    scenario: ScenarioResult,
    ranking: StrategyRanking,
    options: OptionsAnalysisResult,
    risk: RiskProfile | None = None,
    research: ResearchSummary | None = None,
) -> float:
    """Compute conviction score from component weights.

    Weights:
        Scenario confidence: 20 pts
        Strategy fit: 15 pts
        Research verdict: 20 pts (redistributed if skipped)
        Options sell score: 20 pts
        Risk profile (POP + EV): 25 pts
    """
    score = 0.0

    # Scenario confidence: 0-1 → 0-20 pts
    scenario_pts = scenario.confidence * 20
    score += scenario_pts

    # Strategy fit: 0-100 → 0-15 pts
    fit = ranking.selected.fit_score if ranking.selected else 0
    strategy_pts = fit / 100 * 15
    score += strategy_pts

    # Research verdict: 0-20 pts
    research_pts = 0.0
    has_research = research is not None
    if has_research:
        verdict_map = {"BUY_THE_DIP": 20, "WATCH": 10, "AVOID": 0}
        research_pts = verdict_map.get(research.verdict, 10)
        # Adjust by grounding score
        research_pts *= research.grounding_score
        score += research_pts
    else:
        # Redistribute 20 pts: 10 to scenario, 10 to options
        score += scenario.confidence * 10  # extra scenario pts
        # Options gets extra below

    # Options sell score: 0-100 → 0-20 pts (or 0-30 if no research)
    top_sell_score = options.recommendations[0].sell_score if options.recommendations else 0
    options_max = 30.0 if not has_research else 20.0
    options_pts = top_sell_score / 100 * options_max
    score += options_pts

    # Risk profile: POP + EV → 0-25 pts
    risk_pts = 0.0
    if risk:
        # POP: 0.60-0.90 → 0-15 pts
        if risk.pop >= 0.90:
            pop_pts = 15.0
        elif risk.pop >= 0.60:
            pop_pts = (risk.pop - 0.60) / 0.30 * 15
        else:
            pop_pts = 0.0
        risk_pts += pop_pts

        # EV: positive = good, 0-10 pts
        if risk.ev > 0:
            risk_pts += min(10.0, risk.ev / 50 * 10)

        # Feasibility check
        if not risk.sizing_feasible:
            risk_pts *= 0.5
    else:
        # Use BSM POP from options
        if options.recommendations:
            pop = options.recommendations[0].pop
            if pop >= 0.90:
                risk_pts = 15.0
            elif pop >= 0.60:
                risk_pts = (pop - 0.60) / 0.30 * 15

    score += risk_pts

    return round(min(100, max(0, score)), 1)
