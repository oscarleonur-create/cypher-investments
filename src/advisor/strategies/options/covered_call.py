"""Covered call options strategy."""

from __future__ import annotations

import logging
from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@StrategyRegistry.register
class CoveredCall(StrategyBase):
    strategy_name: ClassVar[str] = "covered_call"
    strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
    description: ClassVar[str] = (
        "Buy underlying shares and sell OTM call options against them. "
        "Collects premium income while holding the stock."
    )
    version: ClassVar[str] = "1.0.0"

    params: ClassVar[tuple] = (
        ("pct_invest", 0.95),
        ("otm_pct", 0.05),  # How far OTM to sell calls (5%)
        ("days_to_expiry", 30),  # Target DTE for sold calls
        ("lots", 1),  # Number of 100-share lots
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.shares_held = 0
        self.call_sold = False

    def next(self):
        if self.order:
            return

        price = self.data.close[0]

        # Step 1: Buy underlying shares if we don't have them
        if self.shares_held == 0:
            cash = self.broker.getcash()
            target_shares = self.p.lots * 100
            cost = target_shares * price
            if cash >= cost:
                self.order = self.buy(size=target_shares)
                return

        # Step 2: Sell calls against held shares (simulated via log)
        # Full options implementation requires synthetic pricing from engine
        if self.shares_held > 0 and not self.call_sold:
            strike = price * (1 + self.p.otm_pct)
            logger.info(
                f"[CoveredCall] Would sell {self.p.lots} call(s) at "
                f"strike={strike:.2f}, DTE={self.p.days_to_expiry}"
            )
            self.call_sold = True

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.shares_held += order.executed.size
                self.call_sold = False
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None
