"""SMA Crossover momentum strategy."""

from __future__ import annotations

from typing import ClassVar

import backtrader as bt

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class SMACrossover(StrategyBase):
    strategy_name: ClassVar[str] = "sma_crossover"
    strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
    description: ClassVar[str] = (
        "Momentum strategy using Simple Moving Average crossovers. "
        "Buys on golden cross (short SMA crosses above long SMA) "
        "and sells on death cross (short SMA crosses below long SMA)."
    )
    version: ClassVar[str] = "1.0.0"

    params: ClassVar[tuple] = (
        ("short_period", 20),
        ("long_period", 50),
        ("pct_invest", 0.95),
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.short_period
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.long_period
        )
        self.crossover = bt.indicators.CrossOver(self.sma_short, self.sma_long)

    def next(self):
        if self.order:
            return

        if self.crossover > 0 and not self.position:
            cash = self.broker.getcash()
            price = self.data.close[0]
            size = int((cash * self.p.pct_invest) / price)
            if size > 0:
                self.order = self.buy(size=size)

        elif self.crossover < 0 and self.position:
            self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
