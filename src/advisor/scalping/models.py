"""Result models for the intraday scalping scanner."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class ScalpAction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class ScalpSignal(BaseModel):
    """A single actionable scalp setup for one symbol/strategy."""

    symbol: str
    strategy: str
    action: ScalpAction
    reason: str
    price: float  # last close at scan time
    entry: float
    stop: float
    target: float
    score: float = 0.0  # 0–100 final (catalyst-adjusted) rank
    technical_score: float | None = None  # the raw setup score before catalysts
    bar_time: datetime  # timestamp of the bar that triggered

    # ── Catalyst context (populated by the catalyst layer) ──────────────────
    rvol: float | None = None  # relative volume vs intraday baseline
    gap_pct: float | None = None  # today's open vs prior close, as a fraction
    earnings_today: bool = False
    days_to_earnings: int | None = None
    headlines: list[str] = Field(default_factory=list)
    sentiment_score: float | None = None  # 0–100 LLM news sentiment (50 = neutral)
    catalyst_note: str = ""  # one-line "why it's moving" summary

    @property
    def risk_reward(self) -> float:
        risk = abs(self.entry - self.stop)
        reward = abs(self.target - self.entry)
        return round(reward / risk, 2) if risk > 0 else 0.0


class ScalpScanResult(BaseModel):
    """Output of one scan across a universe."""

    scanned_at: datetime = Field(default_factory=datetime.now)
    interval: str = "5m"
    symbols_scanned: int = 0
    source: str = "tastytrade"  # or "yfinance" when the live feed is unavailable
    min_rvol: float = 0.0  # the RVOL gate applied (0 = no gate)
    gated_out: int = 0  # technical signals dropped for failing the RVOL gate
    signals: list[ScalpSignal] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
