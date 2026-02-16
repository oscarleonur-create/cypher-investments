"""Data models for market-wide scanning results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from advisor.confluence.models import ConfluenceResult


class FilterStatsModel(BaseModel):
    """Statistics from the filter funnel."""

    universe_total: int
    after_volume_cap: int
    after_sector: int
    after_technical: int
    volume_cap_rejected_count: int
    sector_rejected_count: int
    technical_rejected_count: int
    fetch_error_count: int = 0


class MarketScanResult(BaseModel):
    """Result of a full market-wide scan."""

    strategy_name: str
    scanned_at: datetime = Field(default_factory=datetime.now)
    universe_size: int
    filter_stats: FilterStatsModel
    qualifiers: list[str]
    results: list[ConfluenceResult] = Field(default_factory=list)
    errors: list[dict] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
