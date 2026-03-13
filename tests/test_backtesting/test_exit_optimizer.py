"""Tests for the exit parameter optimizer (grid search)."""

from __future__ import annotations

from advisor.backtesting.exit_optimizer import (
    ExitOptimizer,
    ExitParameterGrid,
    GridResult,
    render_ranking_table,
)


class TestExitParameterGrid:
    def test_default_grid(self):
        grid = ExitParameterGrid()
        assert grid.profit_target_pcts == [0.25, 0.50, 0.75]
        assert grid.stop_loss_multipliers == [1.5, 2.0, 3.0, 4.0]
        assert grid.close_at_dtes == [7, 14, 21, 28]

    def test_custom_grid(self):
        grid = ExitParameterGrid(
            profit_target_pcts=[0.50],
            stop_loss_multipliers=[2.0, 3.0],
            close_at_dtes=[7],
        )
        assert grid.n_combos == 2
        combos = grid.combos()
        assert len(combos) == 2
        assert (0.50, 2.0, 7) in combos
        assert (0.50, 3.0, 7) in combos

    def test_n_combos(self):
        grid = ExitParameterGrid(
            profit_target_pcts=[0.25, 0.50, 0.75],
            stop_loss_multipliers=[1.5, 2.0, 3.0, 4.0],
            close_at_dtes=[7, 14, 21, 28],
        )
        assert grid.n_combos == 48  # 3 * 4 * 4

    def test_combos_all_unique(self):
        grid = ExitParameterGrid()
        combos = grid.combos()
        assert len(combos) == len(set(combos))


class TestGridResult:
    def test_model_creation(self):
        r = GridResult(
            profit_target_pct=0.50,
            stop_loss_multiplier=3.0,
            close_at_dte=21,
            total_pnl=1500.0,
            win_rate_pct=65.0,
            sharpe_ratio=1.2,
            profit_factor=2.5,
            max_drawdown_pct=-5.0,
            avg_hold_days=18.5,
            num_trades=30,
        )
        assert r.total_pnl == 1500.0
        assert r.sharpe_ratio == 1.2

    def test_model_defaults(self):
        r = GridResult(
            profit_target_pct=0.50,
            stop_loss_multiplier=3.0,
            close_at_dte=21,
        )
        assert r.total_pnl == 0.0
        assert r.exit_reasons == {}

    def test_model_serialization(self):
        r = GridResult(
            profit_target_pct=0.50,
            stop_loss_multiplier=3.0,
            close_at_dte=21,
            total_pnl=500.0,
        )
        d = r.model_dump()
        assert d["profit_target_pct"] == 0.50
        assert d["total_pnl"] == 500.0


class TestRenderRankingTable:
    def test_render_empty(self):
        """Should not raise with empty results."""
        render_ranking_table([], top_n=5)

    def test_render_with_data(self):
        """Should not raise with valid results."""
        results = [
            GridResult(
                profit_target_pct=0.50,
                stop_loss_multiplier=3.0,
                close_at_dte=21,
                total_pnl=1000.0 * i,
                win_rate_pct=50.0 + i,
                sharpe_ratio=0.5 + i * 0.2,
            )
            for i in range(5)
        ]
        render_ranking_table(results, top_n=3)

    def test_render_with_negative_pnl(self):
        """Should handle negative P&L results."""
        results = [
            GridResult(
                profit_target_pct=0.25,
                stop_loss_multiplier=1.5,
                close_at_dte=7,
                total_pnl=-500.0,
            ),
        ]
        render_ranking_table(results, top_n=1)


class TestExitOptimizerInit:
    def test_default_config(self):
        optimizer = ExitOptimizer(
            symbol="SPY",
            start="2024-01-01",
            end="2024-12-31",
            strategy="put_credit_spread",
        )
        assert optimizer.symbol == "SPY"
        assert optimizer.strategy == "put_credit_spread"
        assert optimizer.cash == 100_000.0

    def test_custom_config(self):
        from advisor.backtesting.options_backtester import BacktestConfig

        config = BacktestConfig(use_adaptive_delta=False)
        optimizer = ExitOptimizer(
            symbol="AAPL",
            start="2023-01-01",
            end="2024-01-01",
            strategy="naked_put",
            cash=50_000.0,
            base_config=config,
        )
        assert optimizer.cash == 50_000.0
        assert optimizer.base_config.use_adaptive_delta is False
