"""Buy-and-hold reference strategy."""

from __future__ import annotations

from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class BuyAndHold(StrategyBase):
    strategy_name: ClassVar[str] = "buy_hold"
    strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
    description: ClassVar[str] = "Buy and hold the underlying asset for the entire period"
    version: ClassVar[str] = "1.0.0"

    params: ClassVar[tuple] = (
        ("pct_invest", 0.95),  # Percentage of portfolio to invest
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.bought = False

    def next(self):
        if self.order:
            return

        if not self.bought:
            cash = self.broker.getcash()
            price = self.data.close[0]
            size = int((cash * self.p.pct_invest) / price)
            if size > 0:
                self.order = self.buy(size=size)
                self.bought = True

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
