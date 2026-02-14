"""Buy-the-dip strategy with RSI oversold + SMA dip detection.

Entry: RSI(14) < 30, price < SMA(50), price > SMA(200)
Exit:  RSI(14) > 70  OR  price < SMA(200)

The scan() override passes force_all=True to the confluence pipeline
because dip stocks are below SMAs by design — the normal breakout
check would skip them.
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
        "and price is below the 50-day SMA but above the 200-day SMA "
        "(long-term uptrend intact). Validated by a 3-layer fundamental "
        "screener (safety gate, value trap detector, timing confirmation). "
        "Exits on RSI overbought (>70) or uptrend break (price < SMA 200)."
    )
    version: ClassVar[str] = "1.0.0"

    params: ClassVar[tuple] = (
        ("rsi_period", 14),
        ("rsi_oversold", 30),
        ("rsi_overbought", 70),
        ("sma_short", 50),
        ("sma_long", 200),
        ("pct_invest", 0.95),
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        self.sma_short = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.sma_short
        )
        self.sma_long = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.p.sma_long
        )

    def next(self):
        if self.order:
            return

        if not self.position:
            # BUY: RSI oversold + price in dip + long-term uptrend intact
            if (
                self.rsi[0] < self.p.rsi_oversold
                and self.data.close[0] < self.sma_short[0]
                and self.data.close[0] > self.sma_long[0]
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
            # SELL: RSI overbought (take profit) OR uptrend broken (cut loss)
            if (
                self.rsi[0] > self.p.rsi_overbought
                or self.data.close[0] < self.sma_long[0]
            ):
                self.order = self.close()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    @classmethod
    def scan(cls, symbol: str) -> "ConfluenceResult":
        """Run confluence with force_all=True (dip stocks are below SMAs by design)."""
        from advisor.confluence.orchestrator import run_confluence

        return run_confluence(symbol, strategy_name=cls.strategy_name, force_all=True)


def scan(symbol: str):
    """Run the full confluence scan via the unified pipeline."""
    return BuyTheDip.scan(symbol)
