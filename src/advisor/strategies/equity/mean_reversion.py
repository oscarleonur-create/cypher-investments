"""Mean Reversion (Short-Term Bounce) strategy.

Entry: RSI(14) < 25 AND price > 2*ATR(14) below 20 EMA AND volume > 1.5x 20-day avg
Exit:  Price returns to 20 EMA
Hold:  3-7 days typical

Catches quick 5-10% bounces in beaten-down names without requiring an
uptrend — just an oversold snap-back to the 20 EMA.

Sets force_all_confluence=True because stocks are below EMAs by design.
"""

from __future__ import annotations

from typing import ClassVar

import backtrader as bt

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class MeanReversion(StrategyBase):
    strategy_name: ClassVar[str] = "mean_reversion"
    strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
    description: ClassVar[str] = (
        "Mean reversion strategy that enters when RSI(14) is deeply oversold "
        "(<25), price is more than 2*ATR(14) below the 20 EMA, and volume "
        "spikes above 1.5x the 20-day average. Exits when price reverts back "
        "to the 20 EMA. Designed for short-term 3-7 day bounce trades in "
        "beaten-down names without requiring an intact uptrend."
    )
    version: ClassVar[str] = "1.0.0"
    force_all_confluence: ClassVar[bool] = True

    params: ClassVar[tuple] = (
        ("rsi_period", 14),
        ("rsi_threshold", 25),
        ("ema_period", 20),
        ("atr_period", 14),
        ("atr_multiplier", 2.0),
        ("volume_avg_period", 20),
        ("volume_spike_factor", 1.5),
        ("pct_invest", 0.95),
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period, safediv=True)
        self.ema = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.ema_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        self.vol_avg = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.volume_avg_period
        )

    def next(self):
        if self.order:
            return

        if not self.position:
            # Entry: deeply oversold + far below EMA + volume spike
            price_below_ema = self.ema[0] - self.data.close[0]
            if (
                self.rsi[0] < self.p.rsi_threshold
                and self.atr[0] > 0
                and price_below_ema > self.p.atr_multiplier * self.atr[0]
                and self.vol_avg[0] > 0
                and self.data.volume[0] > self.p.volume_spike_factor * self.vol_avg[0]
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
            # Exit: price reverts back to EMA
            if self.data.close[0] >= self.ema[0]:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


def scan(symbol: str):
    """Run the full confluence scan via the unified pipeline."""
    return MeanReversion.scan(symbol)
