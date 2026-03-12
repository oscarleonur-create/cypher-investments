"""Tests for scenario simulation pipeline."""

from __future__ import annotations

import pytest
from advisor.scenario.models import (
    StrategyScenarioResult,
)
from advisor.scenario.pipeline import (
    compute_composite,
    resolve_scenarios,
    resolve_strategies,
)


class TestResolveScenarios:
    """Tests for scenario resolution."""

    def test_default_all_four(self):
        """No names returns all four built-in scenarios."""
        scenarios = resolve_scenarios()
        assert len(scenarios) == 4
        names = {s.name for s in scenarios}
        assert names == {"bull", "sideways", "bear", "crash"}

    def test_specific_names(self):
        """Specific names resolve correctly."""
        scenarios = resolve_scenarios(["bull", "bear"])
        assert len(scenarios) == 2
        assert scenarios[0].name == "bull"
        assert scenarios[1].name == "bear"

    def test_unknown_raises(self):
        """Unknown scenario name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scenario"):
            resolve_scenarios(["bull", "superbull"])

    def test_case_insensitive(self):
        """Scenario names are case-insensitive."""
        scenarios = resolve_scenarios(["BULL", "Bear"])
        assert len(scenarios) == 2

    def test_probabilities_sum_to_one(self):
        """Built-in scenario base probabilities sum to 1.0."""
        scenarios = resolve_scenarios()
        total = sum(s.base_probability for s in scenarios)
        assert total == pytest.approx(1.0, abs=0.001)


class TestResolveStrategies:
    """Tests for strategy resolution."""

    def test_default_all_equity(self):
        """No names returns all 6 equity strategies."""
        strategies = resolve_strategies()
        assert len(strategies) == 6
        assert "buy_hold" in strategies
        assert "sma_crossover" in strategies

    def test_unknown_raises(self):
        """Unknown strategy name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            resolve_strategies(["buy_hold", "nonexistent_strategy_xyz"])


class TestComputeComposite:
    """Tests for probability-weighted composite computation."""

    def _make_scenario_result(self, name, mean_ret, dd, prob_pos, p5):
        return {
            "scenario_name": name,
            "result": StrategyScenarioResult(
                strategy_name="test",
                scenario_name=name,
                n_paths=100,
                mean_return_pct=mean_ret,
                p5_return_pct=p5,
                prob_positive=prob_pos,
                mean_max_dd_pct=dd,
            ),
        }

    def test_equal_weights(self):
        """Equal scenario weights produce simple average."""
        results = [
            self._make_scenario_result("a", 10.0, 5.0, 0.8, -2.0),
            self._make_scenario_result("b", -10.0, 15.0, 0.3, -20.0),
        ]
        weights = {"a": 0.5, "b": 0.5}

        comp = compute_composite(results, weights, "test")

        assert comp.expected_return == pytest.approx(0.0, abs=0.01)
        assert comp.expected_max_dd == pytest.approx(10.0, abs=0.01)
        assert comp.prob_positive == pytest.approx(0.55, abs=0.01)

    def test_weighted_return(self):
        """Scenario weights correctly weight expected return."""
        results = [
            self._make_scenario_result("bull", 20.0, 3.0, 0.9, 5.0),
            self._make_scenario_result("bear", -10.0, 12.0, 0.2, -25.0),
        ]
        weights = {"bull": 0.75, "bear": 0.25}

        comp = compute_composite(results, weights, "test")

        expected_ret = 0.75 * 20.0 + 0.25 * (-10.0)  # 12.5
        assert comp.expected_return == pytest.approx(expected_ret, abs=0.01)

    def test_risk_adjusted_score(self):
        """Score formula: E[ret] / max(|E[DD]|, 1%) * P(+) * 100."""
        results = [
            self._make_scenario_result("only", 10.0, 5.0, 0.7, -3.0),
        ]
        weights = {"only": 1.0}

        comp = compute_composite(results, weights, "test")

        expected_score = (10.0 / 5.0) * 0.7 * 100  # 140.0
        assert comp.risk_adjusted_score == pytest.approx(expected_score, abs=0.1)

    def test_dd_floor(self):
        """MaxDD denominator floored at 1% to avoid division by zero."""
        results = [
            self._make_scenario_result("flat", 5.0, 0.0, 0.8, 1.0),
        ]
        weights = {"flat": 1.0}

        comp = compute_composite(results, weights, "test")

        expected_score = (5.0 / 1.0) * 0.8 * 100  # 400.0
        assert comp.risk_adjusted_score == pytest.approx(expected_score, abs=0.1)

    def test_scenario_results_attached(self):
        """Composite includes per-scenario breakdown."""
        results = [
            self._make_scenario_result("bull", 10.0, 3.0, 0.8, -1.0),
            self._make_scenario_result("bear", -5.0, 10.0, 0.3, -15.0),
        ]
        weights = {"bull": 0.6, "bear": 0.4}

        comp = compute_composite(results, weights, "test")

        assert len(comp.scenario_results) == 2
        assert comp.scenario_results[0].scenario_name == "bull"
        assert comp.scenario_results[1].scenario_name == "bear"
