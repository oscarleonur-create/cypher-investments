"""Tests for walk-forward analysis."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pytest

from advisor.engine.results import BacktestResult
from advisor.engine.walk_forward import WalkForwardResult, WalkForwardRunner, WindowResult


def _mock_backtest_result(return_pct: float = 10.0, sharpe: float = 1.0) -> BacktestResult:
    """Create a BacktestResult with given metrics."""
    initial = 100_000.0
    final = initial * (1 + return_pct / 100)
    result = BacktestResult(
        run_id="mock",
        strategy_name="test",
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2023-06-01",
        initial_cash=initial,
        final_value=final,
        sharpe_ratio=sharpe,
        max_drawdown_pct=5.0,
        total_trades=3,
        winning_trades=2,
        losing_trades=1,
    )
    result.compute_derived()
    return result


class TestWindowDateComputation:
    """Test that walk-forward splits dates correctly."""

    def test_three_windows_over_365_days(self):
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        total = (end - start).days  # 364
        n = 3
        train_pct = 0.7

        window_days = total / n
        # Verify window math
        for i in range(n):
            w_start = start + timedelta(days=int(i * window_days))
            train_end = w_start + timedelta(days=int(window_days * train_pct))
            test_start = train_end + timedelta(days=1)
            test_end = w_start + timedelta(days=int(window_days))
            if i == n - 1:
                test_end = end

            assert train_end > w_start
            assert test_start > train_end
            assert test_end >= test_start

    def test_one_window_simple_split(self):
        start = date(2023, 1, 1)
        end = date(2023, 12, 31)
        total = (end - start).days
        train_days = int(total * 0.7)

        train_end = start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)

        assert train_end < end
        assert test_start < end


class TestWalkForwardResult:
    """Test aggregate metric computation."""

    def test_compute_aggregates(self):
        windows = []
        is_returns = [15.0, 12.0, 18.0]
        oos_returns = [8.0, 5.0, 10.0]

        for i, (is_r, oos_r) in enumerate(zip(is_returns, oos_returns)):
            windows.append(
                WindowResult(
                    window_index=i,
                    train_start="2023-01-01",
                    train_end="2023-06-01",
                    test_start="2023-06-02",
                    test_end="2023-12-31",
                    in_sample=_mock_backtest_result(return_pct=is_r, sharpe=1.5),
                    out_of_sample=_mock_backtest_result(return_pct=oos_r, sharpe=0.8),
                )
            )

        result = WalkForwardResult(
            run_id="test",
            strategy="sma_crossover",
            symbol="SPY",
            start_date="2023-01-01",
            end_date="2025-12-31",
            n_windows=3,
            train_pct=0.7,
            windows=windows,
        )
        result.compute_aggregates()

        expected_oos_avg = sum(oos_returns) / 3
        expected_is_avg = sum(is_returns) / 3

        assert abs(result.oos_avg_return_pct - expected_oos_avg) < 0.01
        assert abs(result.is_avg_return_pct - expected_is_avg) < 0.01
        assert abs(result.is_vs_oos_gap - (expected_is_avg - expected_oos_avg)) < 0.01
        assert result.oos_avg_sharpe is not None
        assert abs(result.oos_avg_sharpe - 0.8) < 0.01

    def test_compute_aggregates_empty_windows(self):
        result = WalkForwardResult(
            run_id="empty",
            strategy="test",
            symbol="SPY",
            start_date="2023-01-01",
            end_date="2023-12-31",
            n_windows=0,
            train_pct=0.7,
        )
        result.compute_aggregates()
        assert result.oos_avg_return_pct == 0.0
        assert result.is_vs_oos_gap == 0.0


class TestWalkForwardRunner:
    """Test the runner integration."""

    def test_runner_produces_valid_results(self):
        mock_runner = MagicMock()
        mock_runner.run.return_value = _mock_backtest_result()

        wf = WalkForwardRunner(mock_runner)
        result = wf.run(
            strategy_name="test",
            symbol="SPY",
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            n_windows=3,
            train_pct=0.7,
        )

        assert len(result.windows) == 3
        assert mock_runner.run.call_count == 6  # 2 per window (train + test)
        for w in result.windows:
            assert isinstance(w.in_sample, BacktestResult)
            assert isinstance(w.out_of_sample, BacktestResult)

    def test_single_window(self):
        mock_runner = MagicMock()
        mock_runner.run.return_value = _mock_backtest_result()

        wf = WalkForwardRunner(mock_runner)
        result = wf.run(
            strategy_name="test",
            symbol="SPY",
            start=date(2023, 1, 1),
            end=date(2023, 12, 31),
            n_windows=1,
            train_pct=0.7,
        )

        assert len(result.windows) == 1
        assert mock_runner.run.call_count == 2
