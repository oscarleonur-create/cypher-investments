"""Custom Backtrader sizers for position sizing."""

from __future__ import annotations

import backtrader as bt


class ATRSizer(bt.Sizer):
    """Size positions based on ATR so a 1-ATR move costs risk_pct of equity.

    Formula: size = (equity * risk_pct) / (ATR * atr_multiplier)

    ATR is computed manually from the data feed's high/low/close since
    Backtrader sizers don't support attaching indicators in __init__.
    """

    params = (
        ("atr_period", 14),
        ("risk_pct", 0.02),
        ("atr_multiplier", 2.0),
    )

    def _compute_atr(self, data) -> float:
        """Compute ATR from the last atr_period bars of the data feed."""
        period = self.p.atr_period
        if len(data) < period + 1:
            return 0.0

        tr_values = []
        for i in range(period):
            idx = -(i + 1)
            high = data.high[idx]
            low = data.low[idx]
            prev_close = data.close[idx - 1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_values.append(tr)

        if not tr_values:
            return 0.0
        return sum(tr_values) / len(tr_values)

    def _getsizing(self, comminfo, cash, data, isbuy):
        atr_val = self._compute_atr(data)
        if atr_val <= 0:
            return 0

        equity = self.broker.getvalue()
        risk_amount = equity * self.p.risk_pct
        size = int(risk_amount / (atr_val * self.p.atr_multiplier))

        # Cap at what we can afford
        price = data.close[0]
        if price > 0:
            affordable = int(cash / price)
            size = min(size, affordable)

        return max(size, 0)
