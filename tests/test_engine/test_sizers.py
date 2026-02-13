"""Tests for ATR-based position sizer."""

from __future__ import annotations

from datetime import date

import backtrader as bt
import pytest

from advisor.engine.sizers import ATRSizer
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


def _make_feed(prices: list[float], start: date = date(2023, 1, 1)):
    """Create a PandasData feed from a list of close prices."""
    import pandas as pd

    days = len(prices)
    dates = pd.bdate_range(start=start, periods=days)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p * 1.02 for p in prices],
            "low": [p * 0.98 for p in prices],
            "close": prices,
            "volume": [1_000_000] * days,
        },
        index=dates,
    )
    return bt.feeds.PandasData(dataname=df)


@StrategyRegistry.register
class _SizerTestBuy(StrategyBase):
    """Buys once — lets sizer determine size."""

    strategy_name = "test_sizer_buy"
    description = "test"
    version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.bought = False
        self.executed_size = 0

    def next(self):
        if not self.bought and len(self.data) > 15:
            self.buy()
            self.bought = True

    def notify_order(self, order):
        if order.status == order.Completed and order.isbuy():
            self.executed_size = order.executed.size


def _run_sizer_test(prices: list[float], sizer_params: dict | None = None) -> int:
    """Run with ATRSizer and return the executed position size."""
    cerebro = bt.Cerebro()
    feed = _make_feed(prices)
    cerebro.adddata(feed)
    cerebro.addstrategy(_SizerTestBuy)
    cerebro.broker.setcash(100_000)
    params = sizer_params or {}
    cerebro.addsizer(ATRSizer, **params)
    results = cerebro.run()
    return int(results[0].executed_size)


def test_atr_sizer_produces_valid_size():
    """ATR sizer should produce a positive position size on trending data."""
    # Steady uptrend — ATR will be non-zero
    prices = [100 + i * 0.5 for i in range(40)]
    size = _run_sizer_test(prices)
    assert size > 0


def test_high_vol_smaller_positions():
    """Higher volatility data should produce smaller positions."""
    # Low volatility
    low_vol = [100 + i * 0.1 for i in range(40)]
    # High volatility — big daily swings
    high_vol = [100 + (i * 0.1) + (5 * ((-1) ** i)) for i in range(40)]

    size_low = _run_sizer_test(low_vol)
    size_high = _run_sizer_test(high_vol)

    assert size_low > size_high, (
        f"Low-vol size ({size_low}) should exceed high-vol size ({size_high})"
    )


def test_flat_prices_zero_size():
    """Truly flat OHLC (no range) should yield ATR=0 and size=0."""
    import pandas as pd

    days = 40
    dates = pd.bdate_range(start=date(2023, 1, 1), periods=days)
    p = 100.0
    df = pd.DataFrame(
        {"open": [p] * days, "high": [p] * days, "low": [p] * days,
         "close": [p] * days, "volume": [1_000_000] * days},
        index=dates,
    )
    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=df))
    cerebro.addstrategy(_SizerTestBuy)
    cerebro.broker.setcash(100_000)
    cerebro.addsizer(ATRSizer)
    results = cerebro.run()
    assert int(results[0].executed_size) == 0


def test_use_sizer_false_preserves_manual():
    """When use_sizer=False, strategies should use their manual sizing."""
    # This just confirms the flag default
    from advisor.strategies.equity.buy_hold import BuyAndHold

    defaults = BuyAndHold._get_param_defaults()
    assert defaults.get("use_sizer") is False
