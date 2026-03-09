"""Stage 2: Strategy mapping — map scenario to ranked options strategies."""

from __future__ import annotations

from advisor.strategy_case.models import (
    OptionsStrategyType,
    ScenarioResult,
    ScenarioType,
    StrategyCaseConfig,
    StrategyMatch,
    StrategyRanking,
)

# ── Scenario → Strategy mapping table ────────────────────────────────────────

_SCENARIO_MAP: dict[ScenarioType, list[tuple[OptionsStrategyType, float, str]]] = {
    ScenarioType.EARNINGS_DIP: [
        (
            OptionsStrategyType.PUT_CREDIT_SPREAD,
            85,
            "Defined risk on oversold dip with IV tailwind",
        ),
        (OptionsStrategyType.NAKED_PUT, 70, "Higher premium capture if fundamentals are solid"),
        (OptionsStrategyType.WHEEL, 55, "Long-term accumulation play on quality name"),
    ],
    ScenarioType.IV_SPIKE: [
        (OptionsStrategyType.PUT_CREDIT_SPREAD, 85, "Sell rich premium with defined risk"),
        (OptionsStrategyType.IRON_CONDOR, 75, "Neutral bet on IV mean-reversion"),
        (OptionsStrategyType.SHORT_STRANGLE, 60, "Maximum premium capture (undefined risk)"),
        (OptionsStrategyType.NAKED_PUT, 55, "Directional premium sale in high-IV"),
    ],
    ScenarioType.RANGE_BOUND: [
        (OptionsStrategyType.IRON_CONDOR, 85, "Neutral strategy for sideways market"),
        (OptionsStrategyType.SHORT_STRANGLE, 70, "Wider profit zone in low-movement stock"),
        (OptionsStrategyType.PUT_CREDIT_SPREAD, 55, "Slightly bullish lean in range"),
    ],
    ScenarioType.BREAKOUT_PULLBACK: [
        (OptionsStrategyType.NAKED_PUT, 80, "Sell support on pullback to accumulate"),
        (OptionsStrategyType.PUT_CREDIT_SPREAD, 75, "Defined risk on pullback bounce"),
        (OptionsStrategyType.WHEEL, 60, "Willing to own at lower price"),
    ],
    ScenarioType.MEAN_REVERSION: [
        (OptionsStrategyType.WHEEL, 80, "Accumulate shares at deep discount"),
        (OptionsStrategyType.NAKED_PUT, 75, "Sell fear for premium at extremes"),
        (OptionsStrategyType.PUT_CREDIT_SPREAD, 65, "Defined risk on bounce"),
    ],
    ScenarioType.MOMENTUM: [
        (OptionsStrategyType.COVERED_CALL, 80, "Monetize existing long position"),
        (OptionsStrategyType.CALL_CREDIT_SPREAD, 65, "Bearish hedge if overextended"),
        (OptionsStrategyType.PUT_CREDIT_SPREAD, 55, "Continue bullish lean with premium"),
    ],
}


def rank_strategies(
    scenario: ScenarioResult,
    config: StrategyCaseConfig,
) -> StrategyRanking:
    """Map a detected scenario to ranked strategy recommendations.

    Applies modifiers for account size, IV level, and earnings proximity.
    """
    base_map = _SCENARIO_MAP.get(scenario.scenario_type, [])

    matches: list[StrategyMatch] = []
    for strategy, base_score, reasoning in base_map:
        score = float(base_score)

        # ── Account size modifier ────────────────────────────────────────
        # Small accounts (<$10k) penalize undefined-risk strategies
        if config.account_size < 10_000:
            if strategy in (
                OptionsStrategyType.NAKED_PUT,
                OptionsStrategyType.SHORT_STRANGLE,
                OptionsStrategyType.COVERED_CALL,
                OptionsStrategyType.WHEEL,
            ):
                score -= 15
                reasoning += " (penalized: small account)"

        # ── Low IV modifier ──────────────────────────────────────────────
        # Below 25th percentile — premium selling is less attractive
        if scenario.iv_percentile < 25:
            score -= 10
            reasoning += f" (low IV: {scenario.iv_percentile:.0f}th pctile)"

        # ── High IV bonus for premium sellers ────────────────────────────
        if scenario.iv_percentile >= 70 and strategy not in (OptionsStrategyType.COVERED_CALL,):
            score += 5

        # ── Earnings proximity ───────────────────────────────────────────
        if scenario.days_since_earnings is not None and scenario.days_since_earnings <= 7:
            if scenario.scenario_type == ScenarioType.EARNINGS_DIP:
                score += 5  # This is the play for earnings dips
            else:
                score -= 5  # Others should be cautious near earnings

        # Clamp score
        score = max(0, min(100, score))

        # Set preferred parameters based on strategy type
        dte_min, dte_max = config.min_dte, config.max_dte
        delta_min, delta_max = 0.15, 0.35

        if strategy == OptionsStrategyType.IRON_CONDOR:
            delta_min, delta_max = 0.10, 0.20
        elif strategy == OptionsStrategyType.SHORT_STRANGLE:
            delta_min, delta_max = 0.15, 0.25
        elif strategy == OptionsStrategyType.COVERED_CALL:
            delta_min, delta_max = 0.25, 0.40

        matches.append(
            StrategyMatch(
                strategy=strategy,
                fit_score=round(score, 1),
                reasoning=reasoning,
                preferred_dte_min=dte_min,
                preferred_dte_max=dte_max,
                preferred_delta_min=delta_min,
                preferred_delta_max=delta_max,
            )
        )

    # Sort by fit score descending
    matches.sort(key=lambda m: m.fit_score, reverse=True)

    # Handle strategy override
    override_applied = False
    selected = matches[0] if matches else None

    if config.strategy_override is not None:
        override_match = next(
            (m for m in matches if m.strategy == config.strategy_override),
            None,
        )
        if override_match:
            selected = override_match
            override_applied = True
        else:
            # Strategy not in scenario map — create a basic match
            selected = StrategyMatch(
                strategy=config.strategy_override,
                fit_score=50.0,
                reasoning="User override — not a natural fit for this scenario",
                preferred_dte_min=config.min_dte,
                preferred_dte_max=config.max_dte,
            )
            matches.insert(0, selected)
            override_applied = True

    return StrategyRanking(
        matches=matches,
        selected=selected,
        override_applied=override_applied,
    )
