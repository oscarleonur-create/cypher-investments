"""Neutral candidate model for the risk gate.

Replaces the scalp-specific signal the gate used to be typed on. A
``TradeCandidate`` is any proposed action with an entry, a stop and a target —
whether it came from a position-mechanics event, a macro exposure alert, or a
watchlist trigger. The gate sizes and vetoes; it never executes.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TradeCandidate(BaseModel):
    """A proposed action awaiting a sizing/approval verdict."""

    symbol: str
    source: str  # event kind or module that proposed it
    action: str  # BUY / SELL / ROLL / HEDGE / TRIM / CLOSE
    reason: str = ""
    entry: float
    stop: float
    target: float
    score: float = 0.0  # 0-100 rank, used for ordering gated output
    created_at: datetime = Field(default_factory=datetime.now)

    # ── Risk gate verdict (populated by advisor.risk.gate) ──────────────────
    risk_approved: bool | None = None  # None = not yet gated
    risk_quantity: int | None = None  # sized quantity if approved
    risk_note: str = ""  # sizing caps applied / veto reason

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        return round(reward / risk, 2) if risk > 0 else 0.0
