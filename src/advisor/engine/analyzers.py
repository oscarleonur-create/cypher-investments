"""Custom Backtrader analyzers."""

from __future__ import annotations

import logging

import backtrader as bt

logger = logging.getLogger(__name__)


class DrawdownCircuitBreaker(bt.Analyzer):
    """Monitors portfolio drawdown and trips strategies when threshold exceeded.

    When the portfolio drops more than max_drawdown_pct from its peak,
    sets _circuit_breaker_tripped on all strategies, causing them to close
    positions and skip new entries.
    """

    params = (("max_drawdown_pct", 15.0),)

    def __init__(self):
        super().__init__()
        self._peak_value: float = 0.0
        self._tripped = False

    def next(self):
        current_value = self.strategy.broker.getvalue()
        if current_value > self._peak_value:
            self._peak_value = current_value

        if self._peak_value > 0 and not self._tripped:
            drawdown_pct = ((self._peak_value - current_value) / self._peak_value) * 100
            if drawdown_pct >= self.p.max_drawdown_pct:
                self._tripped = True
                logger.warning(
                    "Circuit breaker tripped: portfolio drawdown %.1f%% exceeds %.1f%% threshold",
                    drawdown_pct,
                    self.p.max_drawdown_pct,
                )
                # Signal all strategies to stop trading
                self.strategy._circuit_breaker_tripped = True
                # Close any open positions
                if self.strategy.position:
                    self.strategy.close()

    def get_analysis(self):
        return {"tripped": self._tripped, "peak_value": self._peak_value}


class TradeRecorder(bt.Analyzer):
    """Records detailed information about each trade."""

    def __init__(self):
        super().__init__()
        self.trades: list[dict] = []
        self._open_trades: dict[int, dict] = {}

    def notify_trade(self, trade):
        if trade.justopened:
            self._open_trades[trade.ref] = {
                "ref": trade.ref,
                "direction": "long" if trade.size > 0 else "short",
                "symbol": trade.data._name or "",
                "open_date": str(self.data.datetime.date(0)),
                "open_price": trade.price,
                "size": abs(trade.size),
                "commission": trade.commission,
            }

        if trade.isclosed:
            record = self._open_trades.pop(trade.ref, {})
            record.update(
                {
                    "close_date": str(self.data.datetime.date(0)),
                    "close_price": trade.price,
                    "pnl": trade.pnl,
                    "pnl_pct": (trade.pnl / (abs(trade.size) * trade.price)) * 100
                    if trade.price > 0 and trade.size != 0
                    else 0.0,
                    "commission": trade.commission,
                }
            )
            self.trades.append(record)

    def get_analysis(self):
        return {"trades": self.trades}
