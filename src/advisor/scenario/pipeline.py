"""Pipeline — orchestrates scenario simulation end-to-end."""

from __future__ import annotations

import logging
from datetime import date, timedelta

from advisor.scenario.models import (
    BUILTIN_SCENARIOS,
    CompositeStrategyResult,
    ScenarioConfig,
    ScenarioDefinition,
    ScenarioSimResult,
    SignalContext,
)
from advisor.scenario.path_adapter import (
    build_full_feeds,
    fetch_warmup_data,
    generate_scenario_paths,
    synthesize_ohlcv,
)
from advisor.scenario.signal_integrator import (
    adjust_scenario_weights,
    fetch_full_signal_context,
)
from advisor.scenario.strategy_runner import (
    aggregate_path_results,
    run_strategy_on_paths,
)
from advisor.simulator.calibration import calibrate

logger = logging.getLogger(__name__)

# All registered equity strategies
DEFAULT_EQUITY_STRATEGIES = [
    "buy_hold",
    "sma_crossover",
    "momentum_breakout",
    "buy_the_dip",
    "mean_reversion",
    "pead",
]


def resolve_scenarios(
    names: list[str] | None = None,
) -> list[ScenarioDefinition]:
    """Resolve scenario names to definitions. Defaults to all four built-ins."""
    if not names:
        return list(BUILTIN_SCENARIOS.values())

    result = []
    for name in names:
        name_lower = name.lower()
        if name_lower not in BUILTIN_SCENARIOS:
            raise ValueError(
                f"Unknown scenario '{name}'. Available: {list(BUILTIN_SCENARIOS.keys())}"
            )
        result.append(BUILTIN_SCENARIOS[name_lower])
    return result


def resolve_strategies(names: list[str] | None = None) -> list[str]:
    """Resolve strategy names. Defaults to all equity strategies."""
    if not names:
        return DEFAULT_EQUITY_STRATEGIES

    from advisor.strategies.registry import StrategyRegistry

    registry = StrategyRegistry()
    registry.discover()
    available = registry.names

    for name in names:
        if name not in available:
            raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return list(names)


def compute_composite(
    scenario_results: list[dict],
    scenario_weights: dict[str, float],
    strategy_name: str,
) -> CompositeStrategyResult:
    """Compute probability-weighted composite from per-scenario results.

    scenario_results: list of dicts with 'scenario_name' and 'result' (StrategyScenarioResult)
    """
    from advisor.scenario.models import StrategyScenarioResult

    weighted_return = 0.0
    weighted_dd = 0.0
    weighted_prob_pos = 0.0
    all_p5s = []
    all_scenario_results: list[StrategyScenarioResult] = []

    for entry in scenario_results:
        name = entry["scenario_name"]
        result: StrategyScenarioResult = entry["result"]
        w = scenario_weights.get(name, 0.0)

        weighted_return += w * result.mean_return_pct
        weighted_dd += w * result.mean_max_dd_pct
        weighted_prob_pos += w * result.prob_positive
        all_p5s.append((w, result.p5_return_pct))
        all_scenario_results.append(result)

    # Worst case p5: probability-weighted 5th percentile
    worst_case_p5 = sum(w * p5 for w, p5 in all_p5s)

    # Risk-adjusted score: E[return] / max(|E[maxDD]|, 1%) * prob_positive * 100
    dd_denom = max(abs(weighted_dd), 1.0)
    risk_adjusted_score = (weighted_return / dd_denom) * weighted_prob_pos * 100

    return CompositeStrategyResult(
        strategy_name=strategy_name,
        expected_return=weighted_return,
        expected_max_dd=weighted_dd,
        worst_case_return_p5=worst_case_p5,
        prob_positive=weighted_prob_pos,
        risk_adjusted_score=risk_adjusted_score,
        scenario_results=all_scenario_results,
    )


def run_scenario_simulation(
    symbol: str,
    config: ScenarioConfig | None = None,
    strategy_names: list[str] | None = None,
    scenario_names: list[str] | None = None,
    include_signals: bool = False,
    max_workers: int | None = None,
) -> ScenarioSimResult:
    """Run the full scenario simulation pipeline.

    1. Calibrate MC parameters for the symbol
    2. Fetch warmup data
    3. For each scenario: generate paths -> synthesize OHLCV -> prepend warmup
    4. For each strategy x scenario: run on all paths -> aggregate
    5. Compute probability-weighted composites
    6. Select best strategy
    """
    config = config or ScenarioConfig()
    scenarios = resolve_scenarios(scenario_names)
    strategies = resolve_strategies(strategy_names)

    # Step 1: Calibrate
    logger.info("Calibrating MC parameters for %s", symbol)
    sim_config = calibrate(symbol)

    # Step 2: Fetch warmup history
    logger.info("Fetching warmup data (%d bars)", config.warmup_bars)
    warmup_df, avg_volume, avg_abs_return = fetch_warmup_data(symbol, config.warmup_bars)
    calibrated_iv = sim_config.vol_mean_level

    # Step 3: Signal integration (optional)
    signal_context: SignalContext | None = None
    if include_signals:
        logger.info("Fetching signal context for %s", symbol)
        signal_context = fetch_full_signal_context(symbol)

    # Determine scenario weights
    scenario_weights: dict[str, float]
    if signal_context:
        scenario_weights = adjust_scenario_weights(scenarios, signal_context)
        signal_context.adjusted_weights = scenario_weights
    else:
        total_prob = sum(s.base_probability for s in scenarios)
        scenario_weights = {s.name: s.base_probability / total_prob for s in scenarios}

    # Step 4: Per scenario — generate paths and run strategies
    sim_start = date.today() + timedelta(days=1)

    all_composites_data: dict[str, list[dict]] = {s: [] for s in strategies}

    for scenario in scenarios:
        logger.info(
            "Scenario '%s': generating %d paths (drift=%.1f%%, vol_mult=%.1fx)",
            scenario.name,
            config.n_paths,
            scenario.annual_drift * 100,
            scenario.vol_multiplier,
        )

        # Generate close-price paths
        close_paths = generate_scenario_paths(symbol, scenario, config, sim_config)

        # Synthesize OHLCV from close paths
        ohlcv_dfs = synthesize_ohlcv(
            close_paths,
            start_date=sim_start,
            historical_volume_mean=avg_volume,
            historical_avg_abs_return=avg_abs_return,
            iv=calibrated_iv * scenario.vol_multiplier,
            seed=config.seed,
        )

        # Prepend warmup data
        full_feeds = build_full_feeds(warmup_df, ohlcv_dfs)

        # Run each strategy on all paths
        for strat_name in strategies:
            logger.info("  Running %s on %d paths...", strat_name, len(full_feeds))
            path_results = run_strategy_on_paths(
                full_feeds, strat_name, config, max_workers=max_workers
            )
            agg = aggregate_path_results(path_results, strat_name, scenario.name)
            all_composites_data[strat_name].append({"scenario_name": scenario.name, "result": agg})

    # Step 5: Compute composites
    composites = []
    for strat_name in strategies:
        comp = compute_composite(all_composites_data[strat_name], scenario_weights, strat_name)
        composites.append(comp)

    # Step 6: Select best
    composites.sort(key=lambda c: c.risk_adjusted_score, reverse=True)
    best = composites[0] if composites else None

    return ScenarioSimResult(
        symbol=symbol,
        scenarios=[s.name for s in scenarios],
        strategies=strategies,
        config=config,
        composites=composites,
        best_strategy=best.strategy_name if best else None,
        best_score=best.risk_adjusted_score if best else 0.0,
        signal_context=signal_context,
    )
