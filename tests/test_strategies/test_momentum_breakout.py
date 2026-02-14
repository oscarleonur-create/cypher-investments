"""Tests for Momentum Breakout strategy."""

from __future__ import annotations

import sys
from datetime import date
from unittest.mock import MagicMock

import backtrader as bt
import numpy as np
import pandas as pd

from advisor.engine.results import BacktestResult
from advisor.engine.runner import BacktestRunner
from advisor.strategies.equity.momentum_breakout import MomentumBreakout
from advisor.strategies.registry import StrategyRegistry

_MOD = "advisor.strategies.equity.momentum_breakout"


def _fresh_discover() -> StrategyRegistry:
    """discover() that works even when module was already imported."""
    sys.modules.pop(_MOD, None)
    registry = StrategyRegistry()
    registry.discover()
    return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(
    prices: list[float],
    volumes: list[int] | None = None,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    """Build an OHLCV DataFrame from close prices and optional volumes."""
    n = len(prices)
    dates = pd.bdate_range(start=start, periods=n)
    if volumes is None:
        volumes = [1_000_000] * n
    return pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": volumes,
        },
        index=dates,
    )


def _run_cerebro(
    prices: list[float],
    volumes: list[int] | None = None,
    **strategy_kwargs,
) -> bt.Strategy:
    """Run a minimal Cerebro with MomentumBreakout on synthetic data."""
    df = _make_price_df(prices, volumes)
    cerebro = bt.Cerebro()
    feed = bt.feeds.PandasData(
        dataname=df,
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="Volume",
        openinterest=-1,
    )
    cerebro.adddata(feed)
    cerebro.addstrategy(MomentumBreakout, **strategy_kwargs)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    return results[0]


# ===========================================================================
# Registration tests
# ===========================================================================


class TestMomentumBreakoutRegistration:
    def test_discovered_by_registry(self):
        registry = _fresh_discover()
        assert "momentum_breakout" in registry.names

    def test_correct_class_from_registry(self):
        registry = _fresh_discover()
        cls = registry.get("momentum_breakout")
        assert cls is not None
        assert cls.strategy_name == "momentum_breakout"

    def test_metadata(self):
        meta = MomentumBreakout.get_metadata()
        assert meta["name"] == "momentum_breakout"
        assert meta["type"] == "equity"
        assert meta["version"] == "1.0.0"
        assert "sma_period" in meta["params"]
        assert "volume_factor" in meta["params"]
        assert "pct_invest" in meta["params"]

    def test_default_params(self):
        meta = MomentumBreakout.get_metadata()
        assert meta["params"]["sma_period"] == 20
        assert meta["params"]["volume_factor"] == 1.5
        assert meta["params"]["pct_invest"] == 0.95

    def test_description_is_nonempty(self):
        assert len(MomentumBreakout.description) > 0


# ===========================================================================
# Logic tests (synthetic data, no network)
# ===========================================================================


class TestMomentumBreakoutLogic:
    def test_flat_prices_no_trades(self):
        """Constant prices with constant volume — no breakout, no trades."""
        prices = [100.0] * 40
        volumes = [1_000_000] * 40
        strat = _run_cerebro(prices, volumes, sma_period=5)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_breakout_with_high_volume_triggers_buy(self):
        """Price rising above SMA with above-average volume should trigger buy."""
        # 25 bars of flat prices at 100, then breakout to 110 with high volume
        flat = [100.0] * 25
        breakout = [110.0] * 10
        prices = flat + breakout

        normal_vol = [1_000_000] * 25
        high_vol = [3_000_000] * 10  # 3x average — well above 1.5x threshold
        volumes = normal_vol + high_vol

        strat = _run_cerebro(prices, volumes, sma_period=5, volume_factor=1.5)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total >= 1

    def test_breakout_without_volume_no_trade(self):
        """Price above SMA but low volume should NOT trigger buy."""
        flat = [100.0] * 25
        breakout = [110.0] * 10
        prices = flat + breakout

        # Keep volume constant (not above 1.5x average)
        volumes = [1_000_000] * 35

        strat = _run_cerebro(prices, volumes, sma_period=5, volume_factor=1.5)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_volume_only_no_breakout_no_trade(self):
        """High volume but price below SMA should NOT trigger buy."""
        # Prices stay flat — no breakout above SMA
        prices = [100.0] * 35

        normal_vol = [1_000_000] * 25
        high_vol = [3_000_000] * 10  # High volume but no price breakout
        volumes = normal_vol + high_vol

        strat = _run_cerebro(prices, volumes, sma_period=5, volume_factor=1.5)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_drop_below_sma_triggers_sell(self):
        """After buying, price dropping below SMA should close position."""
        # Phase 1: flat, Phase 2: breakout with volume, Phase 3: drop
        flat = [100.0] * 25
        breakout = [120.0] * 10
        drop = [90.0] * 10
        prices = flat + breakout + drop

        normal_vol = [1_000_000] * 25
        high_vol = [3_000_000] * 10
        after_vol = [1_000_000] * 10
        volumes = normal_vol + high_vol + after_vol

        strat = _run_cerebro(prices, volumes, sma_period=5, volume_factor=1.5)
        trade_analysis = strat.analyzers.trades.get_analysis()
        closed = trade_analysis.get("total", {}).get("closed", 0)
        assert closed >= 1


# ===========================================================================
# Integration tests (mocked data provider)
# ===========================================================================


class TestMomentumBreakoutIntegration:
    @staticmethod
    def _mock_provider(n_days: int = 200) -> MagicMock:
        """Create a mock YahooDataProvider returning reproducible random data."""
        rng = np.random.default_rng(42)
        dates = pd.bdate_range(start="2025-01-02", periods=n_days)
        close = 150.0 + rng.standard_normal(n_days).cumsum()
        df = pd.DataFrame(
            {
                "Open": close * (1 + rng.uniform(-0.005, 0.005, n_days)),
                "High": close * (1 + rng.uniform(0.0, 0.02, n_days)),
                "Low": close * (1 - rng.uniform(0.0, 0.02, n_days)),
                "Close": close,
                "Volume": rng.integers(500_000, 5_000_000, n_days),
            },
            index=dates,
        )
        provider = MagicMock()
        provider.get_stock_history.return_value = df
        return provider

    def test_backtest_runner_returns_valid_result(self):
        _fresh_discover()
        provider = self._mock_provider()
        runner = BacktestRunner(provider=provider)
        result = runner.run(
            strategy_name="momentum_breakout",
            symbol="AAPL",
            start=date(2025, 1, 2),
            end=date(2025, 12, 31),
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "momentum_breakout"
        assert result.symbol == "AAPL"
        assert result.initial_cash == 100_000
        assert result.final_value > 0

    def test_custom_params_respected(self):
        _fresh_discover()
        provider = self._mock_provider()
        runner = BacktestRunner(provider=provider)
        result = runner.run(
            strategy_name="momentum_breakout",
            symbol="AAPL",
            start=date(2025, 1, 2),
            end=date(2025, 12, 31),
            params={"sma_period": 10, "volume_factor": 2.0},
        )
        assert isinstance(result, BacktestResult)
        assert result.params == {"sma_period": 10, "volume_factor": 2.0}
