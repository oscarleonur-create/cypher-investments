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


class SafetyCheckResult(BaseModel):
    """Layer 1: Safety gate — reject bankrupt or distressed companies."""

    current_ratio: float | None = None
    current_ratio_ok: bool = False
    debt_to_equity: float | None = None
    debt_to_equity_ok: bool = False
    fcf_values: list[float] = Field(default_factory=list)
    fcf_ok: bool = False
    passes: bool = False


class ValueTrapResult(BaseModel):
    """Layer 2: Value trap detector — confirm the dip is a real deal."""

    current_pe: float | None = None
    five_year_avg_pe: float | None = None
    pe_discount_pct: float | None = None
    pe_on_sale: bool = False
    price_change_pct: float | None = None
    forward_eps: float | None = None
    trailing_eps: float | None = None
    rsi_divergence: bool = False
    is_value: bool = False


class FastFundamentalsResult(BaseModel):
    """Layer 3: Timing confirmation — insider and analyst signals."""

    insider_buying: bool = False
    c_suite_buying: bool = False
    insider_details: list[dict] = Field(default_factory=list)
    analyst_target_price: float | None = None
    analyst_upside_pct: float | None = None
    n_analysts: int = 0
    analyst_bullish: bool = False
    has_confirmation: bool = False


class DipScreenerResult(BaseModel):
    """Combined 3-layer dip screener result."""

    safety: SafetyCheckResult
    value_trap: ValueTrapResult | None = None
    fast_fundamentals: FastFundamentalsResult | None = None
    overall_score: str = "FAIL"
    rejection_reason: str | None = None


class FundamentalResult(BaseModel):
    """Result from the fundamental risk check."""

    earnings_within_7_days: bool
    earnings_date: date | None = None
    insider_buying_detected: bool
    is_clear: bool
    dip_screener: DipScreenerResult | None = None


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
