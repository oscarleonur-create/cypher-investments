"""Buy-the-dip strategy with RSI oversold + Bollinger Band dip detection.

Entry: RSI(14) < 30, price within 1% of lower Bollinger Band(20,2)
Exit:  RSI(14) > 70  OR  price within 1% of upper Bollinger Band

Sets force_all_confluence=True so the confluence pipeline runs sentiment
and fundamental checks even without a technical breakout — dip stocks
are below bands by design.
"""

from __future__ import annotations

from typing import ClassVar

import backtrader as bt

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class BuyTheDip(StrategyBase):
    strategy_name: ClassVar[str] = "buy_the_dip"
    strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
    description: ClassVar[str] = (
        "Buy-the-dip strategy that enters when RSI(14) is oversold (<30) "
        "and price is near the lower Bollinger Band (20-period, 2σ). "
        "Validated by a 3-layer fundamental screener (safety gate, value "
        "trap detector, timing confirmation). Exits on RSI overbought "
        "(>70) or price near the upper Bollinger Band."
    )
    version: ClassVar[str] = "2.0.0"
    force_all_confluence: ClassVar[bool] = True

    params: ClassVar[tuple] = (
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("bb_period", 20),
        ("bb_devfactor", 2.0),
        ("bb_proximity_pct", 0.01),
        ("vol_avg_period", 20),
        ("vol_min_ratio", 1.5),
        ("pct_invest", 0.95),
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.bband = bt.indicators.BollingerBands(
            self.data.close,
            period=self.p.bb_period,
            devfactor=self.p.bb_devfactor,
        )
        self.vol_sma = bt.indicators.SMA(self.data.volume, period=self.p.vol_avg_period)

    def next(self):
        if self.order:
            return

        if not self.position:
            # BUY: RSI oversold + price near lower Bollinger Band + volume confirmation
            lower = self.bband.lines.bot[0]
            near_lower = self.data.close[0] <= lower * (1 + self.p.bb_proximity_pct)
            vol_avg = self.vol_sma[0]
            vol_confirmed = vol_avg <= 0 or self.data.volume[0] >= vol_avg * self.p.vol_min_ratio
            if self.rsi[0] < self.p.rsi_oversold and near_lower and vol_confirmed:
                if self.p.use_sizer:
                    self.order = self.buy()
                else:
                    cash = self.broker.getcash()
                    price = self.data.close[0]
                    size = int((cash * self.p.pct_invest) / price)
                    if size > 0:
                        self.order = self.buy(size=size)
        else:
            # SELL: RSI overbought OR price near upper band (take profit)
            upper = self.bband.lines.top[0]
            near_upper = self.data.close[0] >= upper * (1 - self.p.bb_proximity_pct)
            if self.rsi[0] > self.p.rsi_overbought or near_upper:
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None


def scan(symbol: str):
    """Run the full confluence scan via the unified pipeline."""
    return BuyTheDip.scan(symbol)
