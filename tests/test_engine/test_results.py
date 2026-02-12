"""Tests for backtest result models."""

from advisor.engine.results import BacktestResult


def test_backtest_result_compute_derived():
    result = BacktestResult(
        run_id="test001",
        strategy_name="buy_hold",
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_cash=100_000.0,
        final_value=110_000.0,
        total_trades=5,
        winning_trades=3,
        losing_trades=2,
    )
    result.compute_derived()

    assert result.total_return == 10_000.0
    assert result.total_return_pct == 10.0
    assert result.win_rate == 60.0


def test_backtest_result_no_trades():
    result = BacktestResult(
        run_id="test002",
        strategy_name="buy_hold",
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_cash=100_000.0,
        final_value=100_000.0,
        total_trades=0,
    )
    result.compute_derived()

    assert result.total_return == 0.0
    assert result.total_return_pct == 0.0
    assert result.win_rate is None


def test_backtest_result_serialization():
    result = BacktestResult(
        run_id="test003",
        strategy_name="buy_hold",
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_cash=100_000.0,
        final_value=105_000.0,
    )
    json_str = result.model_dump_json()
    loaded = BacktestResult.model_validate_json(json_str)
    assert loaded.run_id == "test003"
    assert loaded.final_value == 105_000.0
