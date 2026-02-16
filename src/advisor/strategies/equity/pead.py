"""Post-Earnings Announcement Drift (PEAD) strategy.

Entry: Volume spike >2x (proxy for earnings event) → wait 2-3 bars →
       buy if price faded below spike-day high and above SMA(200)
Exit:  Sell after hold_days bars OR if stop-loss hit

Uses volume-spike proxy since Backtrader has no earnings data concept.
Real alpha is in the live scanner (pead_screener.py), not backtesting.

Sets force_all_confluence=True because PEAD stocks have faded below
their highs and likely won't show a technical breakout.
"""

from __future__ import annotations

from typing import ClassVar

import backtrader as bt

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class PeadDrift(StrategyBase):
    strategy_name: ClassVar[str] = "pead"
    strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
    description: ClassVar[str] = (
        "Post-Earnings Announcement Drift strategy that detects volume spikes "
        "(proxy for earnings events), waits 2-3 bars for a price fade below "
        "the spike-day high while remaining above SMA(200), then enters a "
        "position held for 45 days to capture the post-earnings drift. "
        "Includes a -8% stop-loss for downside protection."
    )
    version: ClassVar[str] = "1.1.0"
    force_all_confluence: ClassVar[bool] = True

    params: ClassVar[tuple] = (
        ("hold_days", 45),
        ("volume_spike_factor", 2.0),
        ("fade_pct", -0.02),
        ("sma_long", 200),
        ("volume_avg_period", 20),
        ("wait_bars_min", 2),
        ("wait_bars_max", 3),
        ("stop_loss_pct", -0.08),
        ("pct_invest", 0.95),
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_long)
        self.vol_avg = bt.indicators.SimpleMovingAverage(
            self.data.volume, period=self.p.volume_avg_period
        )
        # Track spike events: (bar_index, spike_day_high)
        self.spike_event = None
        self.entry_bar = None
        self.entry_price = None

    def next(self):
        if self.order:
            return

        bar_idx = len(self.data)

        if not self.position:
            # Detect volume spike (proxy for earnings event)
            if (
                self.spike_event is None
                and self.vol_avg[0] > 0
                and self.data.volume[0] > self.p.volume_spike_factor * self.vol_avg[0]
            ):
                self.spike_event = (bar_idx, self.data.high[0])

            # Check for fade entry after waiting period
            if self.spike_event is not None:
                bars_since_spike = bar_idx - self.spike_event[0]
                spike_high = self.spike_event[1]

                if bars_since_spike > self.p.wait_bars_max:
                    # Window expired, reset
                    self.spike_event = None
                elif self.p.wait_bars_min <= bars_since_spike <= self.p.wait_bars_max:
                    # Check fade + above SMA(200)
                    fade = (self.data.close[0] - spike_high) / spike_high
                    if fade <= self.p.fade_pct and self.data.close[0] > self.sma_long[0]:
                        if self.p.use_sizer:
                            self.order = self.buy()
                        else:
                            cash = self.broker.getcash()
                            price = self.data.close[0]
                            size = int((cash * self.p.pct_invest) / price)
                            if size > 0:
                                self.order = self.buy(size=size)
                        self.entry_bar = bar_idx
                        self.entry_price = self.data.close[0]
                        self.spike_event = None
        else:
            # Exit: hold_days reached OR stop-loss hit
            price = self.data.close[0]
            hold_exit = self.entry_bar is not None and bar_idx - self.entry_bar >= self.p.hold_days
            stop_exit = (
                self.entry_price is not None
                and self.entry_price > 0
                and (price - self.entry_price) / self.entry_price <= self.p.stop_loss_pct
            )
            if hold_exit or stop_exit:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


def scan(symbol: str):
    """Run the full confluence scan via the unified pipeline."""
    return PeadDrift.scan(symbol)
