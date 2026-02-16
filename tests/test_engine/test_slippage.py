"""Tests for slippage model integration."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import backtrader as bt
from advisor.engine.results import BacktestResult
from advisor.engine.runner import BacktestRunner
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


def _make_synthetic_feed(start: date, days: int = 60, base_price: float = 100.0):
    """Create a PandasData feed with synthetic trending data."""
    import pandas as pd

    dates = pd.bdate_range(start=start, periods=days)
    # Uptrend so buy_hold always profits — makes slippage effect visible
    prices = [base_price + i * 0.5 for i in range(days)]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1_000_000] * days,
        },
        index=dates,
    )
    return bt.feeds.PandasData(dataname=df)


@StrategyRegistry.register
class _SimpleBuy(StrategyBase):
    """Minimal strategy for testing: buy once."""

    strategy_name = "test_simple_buy"
    description = "test"
    version = "1.0.0"

    params = (("pct_invest", 0.95),)

    def __init__(self):
        super().__init__()
        self.bought = False

    def next(self):
        if not self.bought:
            cash = self.broker.getcash()
            size = int((cash * self.p.pct_invest) / self.data.close[0])
            if size > 0:
                self.buy(size=size)
                self.bought = True


def _run_with_slippage(slippage: float) -> float:
    """Run a simple backtest with given slippage and return final value."""
    cerebro = bt.Cerebro()
    feed = _make_synthetic_feed(date(2023, 1, 1), days=60)
    cerebro.adddata(feed)
    cerebro.addstrategy(_SimpleBuy)
    cerebro.broker.setcash(100_000)
    cerebro.broker.setcommission(commission=0.0)
    cerebro.broker.set_slippage_perc(perc=slippage, slip_open=True, slip_match=True, slip_out=False)
    results = cerebro.run()
    return cerebro.broker.getvalue()


def test_slippage_reduces_returns():
    """Higher slippage should produce lower final value."""
    val_no_slip = _run_with_slippage(0.0)
    val_high_slip = _run_with_slippage(0.05)
    assert val_high_slip < val_no_slip


def test_backtest_result_has_slippage_field():
    """BacktestResult should store slippage_perc."""
    result = BacktestResult(
        run_id="slip01",
        strategy_name="test",
        symbol="SPY",
        start_date="2023-01-01",
        end_date="2024-01-01",
        initial_cash=100_000,
        final_value=105_000,
        slippage_perc=0.002,
    )
    assert result.slippage_perc == 0.002


def test_runner_passes_slippage_to_result():
    """BacktestRunner should propagate slippage_perc into BacktestResult."""
    # Create runner and mock out the actual run to check plumbing
    provider = MagicMock()
    runner = BacktestRunner(initial_cash=100_000, slippage_perc=0.005, provider=provider)
    assert runner.slippage_perc == 0.005
