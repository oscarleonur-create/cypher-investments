"""Stage 5: Risk assessment — MC simulation + position sizing."""

from __future__ import annotations

import logging

from advisor.strategy_case.models import (
    RiskProfile,
    StrategyCaseConfig,
    StrikeRecommendation,
)

logger = logging.getLogger(__name__)


def assess_risk_bsm(
    rec: StrikeRecommendation,
    config: StrategyCaseConfig,
) -> RiskProfile:
    """Fallback BSM-based risk profile from screener data."""
    from advisor.pipeline.models import PipelineConfig
    from advisor.pipeline.sizing import compute_position_size

    pipeline_config = PipelineConfig(
        account_size=config.account_size,
        max_bp_pct=config.max_bp_pct,
        max_risk_pct=config.max_risk_pct,
    )

    sizing = compute_position_size(
        max_loss_per_contract=rec.max_loss,
        buying_power_per_contract=rec.margin_req if rec.margin_req > 0 else rec.max_loss,
        config=pipeline_config,
    )

    return RiskProfile(
        source="bsm",
        pop=rec.pop,
        ev=_estimate_bsm_ev(rec),
        ev_per_bp=_estimate_bsm_ev(rec) / rec.margin_req if rec.margin_req > 0 else 0,
        suggested_contracts=sizing.suggested_contracts,
        position_bp=sizing.position_bp,
        max_loss_total=sizing.max_loss_total,
        risk_pct=sizing.risk_pct,
        sizing_feasible=sizing.sizing_feasible,
    )


def assess_risk_mc(
    symbol: str,
    rec: StrikeRecommendation,
    config: StrategyCaseConfig,
) -> RiskProfile:
    """Full MC simulation risk assessment.

    Only supports PCS currently. Other strategies fall back to BSM.
    """
    if rec.strategy != "put_credit_spread" or rec.long_strike is None:
        logger.info(f"MC simulation only supports PCS; falling back to BSM for {rec.strategy}")
        return assess_risk_bsm(rec, config)

    try:
        from advisor.pipeline.models import PipelineConfig
        from advisor.pipeline.sizing import compute_position_size
        from advisor.simulator.models import PCSCandidate, SimConfig
        from advisor.simulator.pipeline import SimulatorPipeline

        pipeline = SimulatorPipeline(
            config=SimConfig(),
            progress_callback=lambda msg: logger.debug(f"MC: {msg}"),
        )

        candidate = PCSCandidate(
            symbol=symbol.upper(),
            short_strike=rec.strike,
            long_strike=rec.long_strike,
            expiration=rec.expiry.isoformat() if rec.expiry else "",
            dte=rec.dte,
            credit=rec.credit,
            short_iv=rec.iv,
            pop_estimate=rec.pop,
            sell_score=rec.sell_score,
        )

        results = pipeline.run_single([candidate], n_paths=50_000)

        if not results:
            logger.warning(f"MC simulation returned no results for {symbol}")
            return assess_risk_bsm(rec, config)

        sim = results[0]

        # Position sizing
        pip_config = PipelineConfig(
            account_size=config.account_size,
            max_bp_pct=config.max_bp_pct,
            max_risk_pct=config.max_risk_pct,
        )
        sizing = compute_position_size(
            max_loss_per_contract=rec.max_loss,
            buying_power_per_contract=rec.margin_req if rec.margin_req > 0 else rec.max_loss,
            config=pip_config,
        )

        return RiskProfile(
            source="mc",
            pop=sim.pop,
            ev=sim.ev,
            ev_per_bp=sim.ev_per_bp,
            cvar_95=sim.cvar_95,
            stop_prob=sim.stop_prob,
            touch_prob=sim.touch_prob,
            suggested_contracts=sizing.suggested_contracts,
            position_bp=sizing.position_bp,
            max_loss_total=sizing.max_loss_total,
            risk_pct=sizing.risk_pct,
            sizing_feasible=sizing.sizing_feasible,
        )

    except Exception as e:
        logger.error(f"MC simulation failed for {symbol}: {e}")
        return assess_risk_bsm(rec, config)


def _estimate_bsm_ev(rec: StrikeRecommendation) -> float:
    """Estimate expected value from BSM POP and credit/max-loss."""
    credit_100 = rec.credit * 100
    max_loss_100 = rec.max_loss
    if rec.pop <= 0 or max_loss_100 <= 0:
        return 0.0
    return rec.pop * credit_100 - (1 - rec.pop) * max_loss_100
