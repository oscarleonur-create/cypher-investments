"""Data models for the momentum confluence system."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class ConfluenceVerdict(StrEnum):
    ENTER = "ENTER"
    CAUTION = "CAUTION"
    PASS = "PASS"


class TechnicalResult(BaseModel):
    """Result from the technical breakout check."""

    signal: str
    price: float
    sma_20: float
    volume_ratio: float
    is_bullish: bool


class SourceInfo(BaseModel):
    """A cited source from sentiment analysis."""

    source_id: str
    url: str
    title: str = ""
    tier: int = 3


class SentimentResult(BaseModel):
    """Result from the news sentiment check."""

    score: float = Field(ge=0, le=100)
    positive_pct: float = Field(ge=0, le=100)
    key_headlines: list[str] = Field(default_factory=list)
    sources: list[SourceInfo] = Field(default_factory=list)
    is_bullish: bool


class FundamentalResult(BaseModel):
    """Result from the fundamental risk check."""

    earnings_within_7_days: bool
    earnings_date: date | None = None
    insider_buying_detected: bool
    is_clear: bool


class ConfluenceResult(BaseModel):
    """Aggregated result from all three confluence agents."""

    symbol: str
    strategy_name: str = "momentum_breakout"
    verdict: ConfluenceVerdict
    technical: TechnicalResult
    sentiment: SentimentResult
    fundamental: FundamentalResult
    reasoning: str
    suggested_hold_days: int = 4
    scanned_at: datetime = Field(default_factory=datetime.now)
