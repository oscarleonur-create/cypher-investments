"""Tests for SMA Crossover strategy."""

from __future__ import annotations

import sys
from datetime import date
from unittest.mock import MagicMock

import backtrader as bt
import numpy as np
import pandas as pd
from advisor.engine.results import BacktestResult
from advisor.engine.runner import BacktestRunner
from advisor.strategies.equity.sma_crossover import SMACrossover
from advisor.strategies.registry import StrategyRegistry

# Module path used by discover(). The top-level import caches it in
# sys.modules, so discover() won't re-execute the decorator after the
# autouse reset_registry fixture clears registrations.  Popping the key
# lets discover() reimport (and re-register) the module cleanly.
_SMA_MOD = "advisor.strategies.equity.sma_crossover"


def _fresh_discover() -> StrategyRegistry:
    """discover() that works even when sma_crossover was already imported."""
    sys.modules.pop(_SMA_MOD, None)
    registry = StrategyRegistry()
    registry.discover()
    return registry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_price_df(prices: list[float], start: str = "2024-01-02") -> pd.DataFrame:
    """Build an OHLCV DataFrame from a list of close prices."""
    dates = pd.bdate_range(start=start, periods=len(prices))
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": [p * 1.01 for p in prices],
            "Low": [p * 0.99 for p in prices],
            "Close": prices,
            "Volume": [1_000_000] * len(prices),
        },
        index=dates,
    )
    return df


def _run_cerebro(prices: list[float], **strategy_kwargs) -> bt.Strategy:
    """Run a minimal Cerebro with SMACrossover on synthetic prices."""
    df = _make_price_df(prices)
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
    cerebro.addstrategy(SMACrossover, **strategy_kwargs)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    results = cerebro.run()
    return results[0]


# ===========================================================================
# Registration tests
# ===========================================================================


class TestSMACrossoverRegistration:
    def test_discovered_by_registry(self):
        registry = _fresh_discover()
        assert "sma_crossover" in registry.names

    def test_correct_class_from_registry(self):
        registry = _fresh_discover()
        cls = registry.get("sma_crossover")
        assert cls is not None
        assert cls.strategy_name == "sma_crossover"

    def test_metadata(self):
        meta = SMACrossover.get_metadata()
        assert meta["name"] == "sma_crossover"
        assert meta["type"] == "equity"
        assert meta["version"] == "1.0.0"
        assert "short_period" in meta["params"]
        assert "long_period" in meta["params"]
        assert "pct_invest" in meta["params"]

    def test_default_params(self):
        meta = SMACrossover.get_metadata()
        assert meta["params"]["short_period"] == 20
        assert meta["params"]["long_period"] == 50
        assert meta["params"]["pct_invest"] == 0.95

    def test_description_is_nonempty(self):
        assert len(SMACrossover.description) > 0


# ===========================================================================
# Logic tests (synthetic data, no network)
# ===========================================================================


class TestSMACrossoverLogic:
    def test_flat_prices_no_trades(self):
        """Constant prices should never trigger a crossover signal."""
        prices = [100.0] * 80
        strat = _run_cerebro(prices, short_period=5, long_period=10)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total = trade_analysis.get("total", {}).get("total", 0)
        assert total == 0

    def test_golden_cross_triggers_buy(self):
        """Declining then rising prices should produce a golden cross buy."""
        # Start high so short SMA < long SMA during decline,
        # then rise to trigger short SMA crossing above long SMA.
        prices = (
            [120.0] * 15  # initial plateau
            + [120 - i * 2 for i in range(1, 16)]  # decline to 90
            + [90 + i * 2 for i in range(1, 51)]  # rise to 190
        )
        strat = _run_cerebro(prices, short_period=5, long_period=15)
        # Should have entered a position
        assert strat.broker.getvalue() != 100_000  # portfolio changed

    def test_death_cross_closes_position(self):
        """Decline then rise then decline should buy on golden cross and
        close the trade on the subsequent death cross."""
        # The short SMA must be strictly below the long SMA before the rise
        # so that the CrossOver indicator detects an actual crossing event.
        prices = (
            [150.0] * 20  # flat start - SMAs converge
            + [150 - i * 2 for i in range(1, 16)]  # decline (short < long)
            + [120 + i * 3 for i in range(1, 21)]  # rise (golden cross → buy)
            + [180 - i * 3 for i in range(1, 31)]  # decline (death cross → sell)
        )
        strat = _run_cerebro(prices, short_period=5, long_period=15)
        trade_analysis = strat.analyzers.trades.get_analysis()
        closed = trade_analysis.get("total", {}).get("closed", 0)
        assert closed >= 1

    def test_sustained_uptrend_no_duplicate_orders(self):
        """Once in position during an uptrend, no additional buys should fire."""
        # Initial dip then sustained rise to trigger one golden cross only.
        prices = (
            [100] * 20
            + [100 - i for i in range(1, 11)]  # dip to 90
            + [90 + i for i in range(1, 61)]  # sustained rise to 150
        )
        strat = _run_cerebro(prices, short_period=5, long_period=15)
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_open = trade_analysis.get("total", {}).get("open", 0)
        total_closed = trade_analysis.get("total", {}).get("closed", 0)
        # At most one trade should have been opened
        assert total_open + total_closed <= 1


# ===========================================================================
# Integration tests (mocked data provider)
# ===========================================================================


class TestSMACrossoverIntegration:
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
            strategy_name="sma_crossover",
            symbol="AAPL",
            start=date(2025, 1, 2),
            end=date(2025, 12, 31),
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "sma_crossover"
        assert result.symbol == "AAPL"
        assert result.initial_cash == 100_000
        assert result.final_value > 0

    def test_custom_params_respected(self):
        _fresh_discover()
        provider = self._mock_provider()
        runner = BacktestRunner(provider=provider)
        result = runner.run(
            strategy_name="sma_crossover",
            symbol="AAPL",
            start=date(2025, 1, 2),
            end=date(2025, 12, 31),
            params={"short_period": 10, "long_period": 30},
        )
        assert isinstance(result, BacktestResult)
        assert result.params == {"short_period": 10, "long_period": 30}
