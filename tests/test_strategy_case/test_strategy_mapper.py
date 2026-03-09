"""Tests for strategy mapping — Stage 2."""

from advisor.strategy_case.models import (
    OptionsStrategyType,
    ScenarioResult,
    ScenarioType,
    StrategyCaseConfig,
)
from advisor.strategy_case.strategy_mapper import rank_strategies


class TestStrategyMapper:
    def test_earnings_dip_maps_to_pcs(self):
        scenario = ScenarioResult(
            scenario_type=ScenarioType.EARNINGS_DIP,
            confidence=0.8,
            iv_percentile=65,
        )
        config = StrategyCaseConfig()
        ranking = rank_strategies(scenario, config)

        assert ranking.selected is not None
        assert ranking.selected.strategy == OptionsStrategyType.PUT_CREDIT_SPREAD
        assert len(ranking.matches) >= 2

    def test_iv_spike_maps_to_pcs(self):
        scenario = ScenarioResult(
            scenario_type=ScenarioType.IV_SPIKE,
            confidence=0.75,
            iv_percentile=80,
        )
        config = StrategyCaseConfig()
        ranking = rank_strategies(scenario, config)

        assert ranking.selected.strategy == OptionsStrategyType.PUT_CREDIT_SPREAD

    def test_range_bound_maps_to_iron_condor(self):
        scenario = ScenarioResult(
            scenario_type=ScenarioType.RANGE_BOUND,
            confidence=0.6,
            iv_percentile=50,
        )
        config = StrategyCaseConfig()
        ranking = rank_strategies(scenario, config)

        assert ranking.selected.strategy == OptionsStrategyType.IRON_CONDOR

    def test_momentum_maps_to_covered_call(self):
        scenario = ScenarioResult(
            scenario_type=ScenarioType.MOMENTUM,
            confidence=0.7,
            iv_percentile=45,
        )
        config = StrategyCaseConfig(account_size=50_000)
        ranking = rank_strategies(scenario, config)

        assert ranking.selected.strategy == OptionsStrategyType.COVERED_CALL

    def test_small_account_penalizes_naked(self):
        scenario = ScenarioResult(
            scenario_type=ScenarioType.BREAKOUT_PULLBACK,
            confidence=0.7,
            iv_percentile=50,
        )
        config_small = StrategyCaseConfig(account_size=5_000)
        config_large = StrategyCaseConfig(account_size=50_000)

        ranking_small = rank_strategies(scenario, config_small)
        ranking_large = rank_strategies(scenario, config_large)

        # Naked put should score lower in small account
        naked_small = next(
            m for m in ranking_small.matches if m.strategy == OptionsStrategyType.NAKED_PUT
        )
        naked_large = next(
            m for m in ranking_large.matches if m.strategy == OptionsStrategyType.NAKED_PUT
        )
        assert naked_small.fit_score < naked_large.fit_score

    def test_low_iv_penalty(self):
        scenario_low = ScenarioResult(
            scenario_type=ScenarioType.IV_SPIKE,
            confidence=0.6,
            iv_percentile=20,  # low IV
        )
        scenario_high = ScenarioResult(
            scenario_type=ScenarioType.IV_SPIKE,
            confidence=0.6,
            iv_percentile=80,  # high IV
        )
        config = StrategyCaseConfig(account_size=50_000)

        ranking_low = rank_strategies(scenario_low, config)
        ranking_high = rank_strategies(scenario_high, config)

        # Best score should be lower in low-IV environment
        assert ranking_low.selected.fit_score < ranking_high.selected.fit_score

    def test_strategy_override(self):
        scenario = ScenarioResult(
            scenario_type=ScenarioType.EARNINGS_DIP,
            confidence=0.8,
            iv_percentile=65,
        )
        config = StrategyCaseConfig(strategy_override=OptionsStrategyType.IRON_CONDOR)
        ranking = rank_strategies(scenario, config)

        assert ranking.override_applied is True
        assert ranking.selected.strategy == OptionsStrategyType.IRON_CONDOR

    def test_override_not_in_scenario(self):
        """Override with a strategy not in the scenario map creates a basic match."""
        scenario = ScenarioResult(
            scenario_type=ScenarioType.EARNINGS_DIP,
            confidence=0.8,
            iv_percentile=65,
        )
        config = StrategyCaseConfig(strategy_override=OptionsStrategyType.SHORT_STRANGLE)
        ranking = rank_strategies(scenario, config)

        assert ranking.override_applied is True
        assert ranking.selected.strategy == OptionsStrategyType.SHORT_STRANGLE
        assert ranking.selected.fit_score == 50.0  # default override score

    def test_all_scenarios_have_mappings(self):
        """Every scenario type should produce at least one strategy match."""
        config = StrategyCaseConfig()
        for scenario_type in ScenarioType:
            scenario = ScenarioResult(
                scenario_type=scenario_type,
                confidence=0.6,
                iv_percentile=50,
            )
            ranking = rank_strategies(scenario, config)
            assert len(ranking.matches) >= 1, f"No matches for {scenario_type}"
            assert ranking.selected is not None
