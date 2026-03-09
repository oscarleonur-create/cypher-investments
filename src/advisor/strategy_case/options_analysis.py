"""Stage 4: Options analysis — find optimal strikes via PremiumScreener."""

from __future__ import annotations

import logging

from advisor.strategy_case.models import (
    OptionsAnalysisResult,
    OptionsStrategyType,
    StrategyCaseConfig,
    StrategyMatch,
    StrikeRecommendation,
)

logger = logging.getLogger(__name__)

# Map OptionsStrategyType → PremiumScreener strategy names
_SCREENER_STRATEGY_MAP: dict[OptionsStrategyType, list[str]] = {
    OptionsStrategyType.PUT_CREDIT_SPREAD: ["put_credit_spread"],
    OptionsStrategyType.NAKED_PUT: ["naked_put"],
    OptionsStrategyType.COVERED_CALL: ["naked_put"],  # screen puts as proxy
    OptionsStrategyType.CALL_CREDIT_SPREAD: ["put_credit_spread"],  # similar mechanics
    OptionsStrategyType.IRON_CONDOR: ["put_credit_spread"],  # screen put side
    OptionsStrategyType.SHORT_STRANGLE: ["naked_put"],
    OptionsStrategyType.WHEEL: ["naked_put"],
}


def analyze_options(
    symbol: str,
    selected: StrategyMatch,
    config: StrategyCaseConfig,
) -> OptionsAnalysisResult:
    """Find optimal strikes for the selected strategy using PremiumScreener.

    Single-ticker call to the existing screener infrastructure.
    """
    from advisor.market.premium_screener import PremiumScreener

    screener_strategies = _SCREENER_STRATEGY_MAP.get(selected.strategy, ["put_credit_spread"])

    screener = PremiumScreener(
        account_size=config.account_size,
        min_iv_pctile=0.0,  # Don't filter — we want results even in low IV
        strategies=screener_strategies,
        min_dte=selected.preferred_dte_min,
        max_dte=selected.preferred_dte_max,
        top_n=5,
    )

    try:
        scan_result = screener.scan([symbol])
    except Exception as e:
        logger.error(f"Premium screener failed for {symbol}: {e}")
        return OptionsAnalysisResult(errors=[str(e)])

    recommendations: list[StrikeRecommendation] = []
    for opp in scan_result.opportunities:
        recommendations.append(
            StrikeRecommendation(
                strategy=opp.strategy,
                strike=opp.strike,
                long_strike=opp.long_strike,
                expiry=opp.expiry,
                dte=opp.dte,
                credit=opp.credit,
                delta=opp.delta,
                iv=opp.iv,
                pop=opp.pop,
                annualized_yield=opp.annualized_yield,
                sell_score=opp.sell_score,
                liquidity_score=opp.liquidity.total,
                expected_move=opp.expected_move,
                strike_vs_em=opp.strike_vs_em,
                max_loss=opp.max_loss,
                margin_req=opp.margin_req,
                flags=opp.flags,
            )
        )

    return OptionsAnalysisResult(
        recommendations=recommendations,
        iv_percentile=scan_result.opportunities[0].iv_percentile if recommendations else 0.0,
        term_structure=(scan_result.opportunities[0].term_structure if recommendations else "flat"),
        regime=scan_result.regime,
        errors=scan_result.errors,
    )
