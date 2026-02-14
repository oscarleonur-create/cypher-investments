"""Momentum breakout strategy with volume confirmation.

The Backtrader strategy (MomentumBreakout) handles backtesting.
The scan() function is a thin backward-compat wrapper that delegates
to the unified confluence pipeline via StrategyBase.scan().
"""

from __future__ import annotations

from typing import ClassVar

import backtrader as bt

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


# ═══════════════════════════════════════════════════════════════════════════════
# Backtrader strategy (for backtesting)
# ═══════════════════════════════════════════════════════════════════════════════


@StrategyRegistry.register
class MomentumBreakout(StrategyBase):
    strategy_name: ClassVar[str] = "momentum_breakout"
    strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
    description: ClassVar[str] = (
        "Momentum breakout strategy that buys when price breaks above "
        "the 20-day SMA with above-average volume confirmation, validated "
        "by news sentiment (>70% positive) and no imminent earnings risk. "
        "Exits when price falls back below the 20-day SMA."
    )
    version: ClassVar[str] = "1.0.0"

    params: ClassVar[tuple] = (
        ("sma_period", 20),
        ("volume_factor", 1.5),
        ("pct_invest", 0.95),
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.sma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.sma_period
        )
        self.avg_volume = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.sma_period
        )

    def next(self):
        if self.order:
            return

        if not self.position:
            # BUY: price above SMA AND volume above average * factor
            if (
                self.data.close[0] > self.sma[0]
                and self.data.volume[0] > self.avg_volume[0] * self.p.volume_factor
            ):
                if self.p.use_sizer:
                    self.order = self.buy()
                else:
                    cash = self.broker.getcash()
                    price = self.data.close[0]
                    size = int((cash * self.p.pct_invest) / price)
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            # SELL: price drops below SMA
            if self.data.close[0] < self.sma[0]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


# ═══════════════════════════════════════════════════════════════════════════════
# Backward-compat module-level scan()
# ═══════════════════════════════════════════════════════════════════════════════


def scan(symbol: str):
    """Run the full confluence scan via the unified pipeline."""
    return MomentumBreakout.scan(symbol)
