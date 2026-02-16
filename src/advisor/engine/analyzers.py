"""Custom Backtrader analyzers."""

from __future__ import annotations

import backtrader as bt


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
