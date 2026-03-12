"""Tests for scenario strategy runner."""

from __future__ import annotations

import pytest
from advisor.scenario.models import PathStrategyResult
from advisor.scenario.strategy_runner import aggregate_path_results


class TestAggregatePathResults:
    """Tests for aggregating per-path results."""

    def test_basic_aggregation(self):
        """Mean, median, and percentiles computed correctly."""
        results = [
            PathStrategyResult(
                total_return_pct=r, max_drawdown_pct=d, total_trades=t, win_rate=w, final_value=v
            )
            for r, d, t, w, v in [
                (10.0, 5.0, 3, 66.7, 110_000),
                (5.0, 3.0, 2, 50.0, 105_000),
                (-2.0, 8.0, 1, 0.0, 98_000),
                (15.0, 4.0, 4, 75.0, 115_000),
                (0.0, 2.0, 0, None, 100_000),
            ]
        ]

        agg = aggregate_path_results(results, "sma_crossover", "bull")

        assert agg.strategy_name == "sma_crossover"
        assert agg.scenario_name == "bull"
        assert agg.n_paths == 5
        assert agg.mean_return_pct == pytest.approx(5.6, abs=0.01)
        assert agg.prob_positive == pytest.approx(0.6, abs=0.01)

    def test_empty_results(self):
        """Empty input returns zero-value result."""
        agg = aggregate_path_results([], "buy_hold", "crash")
        assert agg.n_paths == 0
        assert agg.mean_return_pct == 0.0

    def test_all_positive(self):
        """prob_positive is 1.0 when all returns positive."""
        results = [
            PathStrategyResult(total_return_pct=r, final_value=100_000 * (1 + r / 100))
            for r in [5.0, 10.0, 1.0, 20.0]
        ]
        agg = aggregate_path_results(results, "buy_hold", "bull")
        assert agg.prob_positive == 1.0

    def test_percentiles(self):
        """Percentile values are ordered correctly."""
        results = [
            PathStrategyResult(total_return_pct=float(i), final_value=100_000.0)
            for i in range(-50, 51)
        ]
        agg = aggregate_path_results(results, "test", "sideways")

        assert agg.p5_return_pct < agg.p25_return_pct
        assert agg.p25_return_pct < agg.median_return_pct
        assert agg.median_return_pct < agg.p75_return_pct
        assert agg.p75_return_pct < agg.p95_return_pct

    def test_win_rate_excludes_none(self):
        """avg_win_rate ignores paths with no trades (win_rate=None)."""
        results = [
            PathStrategyResult(
                total_return_pct=5.0, total_trades=2, win_rate=50.0, final_value=105_000
            ),
            PathStrategyResult(
                total_return_pct=0.0, total_trades=0, win_rate=None, final_value=100_000
            ),
            PathStrategyResult(
                total_return_pct=10.0, total_trades=3, win_rate=66.7, final_value=110_000
            ),
        ]
        agg = aggregate_path_results(results, "test", "bull")
        # Average of 50.0 and 66.7
        assert agg.avg_win_rate == pytest.approx(58.35, abs=0.01)
