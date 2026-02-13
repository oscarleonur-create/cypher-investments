"""Signal data models for the live scanner."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class SignalAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"


class StrategySignal(BaseModel):
    """Signal produced by a single strategy for a symbol."""

    strategy_name: str
    symbol: str
    action: SignalAction
    reason: str
    timestamp: datetime
    price: float


class ScanResult(BaseModel):
    """Aggregated scan result across multiple strategies for a symbol."""

    symbol: str
    scanned_at: datetime
    signals: list[StrategySignal]

    @property
    def summary(self) -> str:
        counts = {a: 0 for a in SignalAction}
        for sig in self.signals:
            counts[sig.action] += 1
        parts = [f"{counts[a]} {a.value}" for a in SignalAction]
        return ", ".join(parts)
