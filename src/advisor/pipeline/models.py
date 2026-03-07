"""Pydantic models for the integrated pipeline orchestrator."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ── Configuration ────────────────────────────────────────────────────────────


class PipelineConfig(BaseModel):
    """Pipeline orchestrator configuration."""

    account_size: float = Field(default=5_000.0, description="Total account value in dollars")
    max_bp_pct: float = Field(default=50.0, description="Max buying power usage as % of account")
    max_risk_pct: float = Field(default=5.0, description="Max loss per trade as % of account")
    min_conviction: float = Field(
        default=50.0, description="Minimum conviction score to include in results"
    )
    min_iv_rank: float = Field(
        default=25.0, description="Minimum IV rank (0-100) for IV timing gate"
    )

    # MC simulation paths
    quick_paths: int = Field(default=10_000, description="Quick sim path count")
    deep_paths: int = Field(default=100_000, description="Deep sim path count")

    # Conviction layer weights (must sum to 100)
    w_signal: float = Field(default=20.0, description="Signal discovery weight (0-20 pts)")
    w_fundamental: float = Field(default=20.0, description="Fundamental safety weight (0-20 pts)")
    w_iv: float = Field(default=20.0, description="IV environment weight (0-20 pts)")
    w_mc_edge: float = Field(default=25.0, description="MC edge weight (0-25 pts)")
    w_sizing: float = Field(default=15.0, description="Sizing feasibility weight (0-15 pts)")


# ── Enums ────────────────────────────────────────────────────────────────────


class ConvictionTier(StrEnum):
    AUTO_ALERT = "AUTO_ALERT"  # >= 75
    WATCH = "WATCH"  # 50-74
    SKIP = "SKIP"  # < 50


# ── Layer results ────────────────────────────────────────────────────────────


class SignalDiscoveryResult(BaseModel):
    """Per-symbol raw scanner outputs from Layer 1."""

    symbol: str
    dip_score: str | None = None  # FAIL/WEAK/WATCH/LEAN_BUY/BUY/STRONG_BUY
    pead_score: str | None = None
    mispricing_score: float | None = None  # 0-100
    smart_money_score: float | None = None  # normalized 0-100
    best_signal_score: float = 0.0  # max of normalized scores, 0-100


class ValidationResult(BaseModel):
    """Confluence + alpha validation from Layer 2."""

    symbol: str
    confluence_verdict: str = "PASS"  # ENTER / CAUTION / PASS
    alpha_score: float = 0.0  # 0-100
    fundamental_safe: bool = False


class IVTimingResult(BaseModel):
    """IV environment analysis from Layer 3."""

    symbol: str
    iv_rank: float | None = None
    iv_percentile: float = 0.0
    current_iv: float = 0.0
    term_structure: str = "flat"  # contango / backwardation / flat
    vol_mean_level: float = 0.0
    vol_mean_revert_speed: float = 0.0
    vol_direction: str = "neutral"  # mean_reverting_down / elevated / neutral
    iv_timing_score: float = 0.0  # composite 0-100


class MCEdgeResult(BaseModel):
    """Monte Carlo simulation edge from Layer 4."""

    symbol: str
    short_strike: float = 0.0
    long_strike: float = 0.0
    expiration: str = ""
    dte: int = 0
    credit: float = 0.0
    max_loss: float = 0.0
    bp: float = 0.0
    mc_pop: float = 0.0
    market_pop: float = 0.0  # from PCSCandidate.pop_estimate
    pop_edge: float = 0.0  # mc_pop - market_pop
    ev: float = 0.0
    ev_per_bp: float = 0.0
    cvar_95: float = 0.0
    stop_prob: float = 0.0
    mc_edge_score: float = 0.0  # composite 0-100


class SizingResult(BaseModel):
    """Position sizing from Layer 5."""

    suggested_contracts: int = 0
    position_bp: float = 0.0  # total BP used
    bp_utilization_pct: float = 0.0  # position_bp / available_bp * 100
    max_loss_total: float = 0.0  # max_loss_per_contract * contracts
    risk_pct: float = 0.0  # actual % of account at risk
    sizing_feasible: bool = False
    sizing_score: float = 0.0  # 0-100


class SignalBreakdown(BaseModel):
    """Per-layer weighted contributions to conviction score."""

    signal_strength: float = 0.0  # 0-20
    fundamental_safety: float = 0.0  # 0-20
    iv_environment: float = 0.0  # 0-20
    mc_edge: float = 0.0  # 0-25
    sizing_feasibility: float = 0.0  # 0-15


# ── Final output ─────────────────────────────────────────────────────────────


class TradeRecommendation(BaseModel):
    """Final ranked trade recommendation."""

    symbol: str
    short_strike: float
    long_strike: float
    expiration: str
    dte: int
    credit: float
    max_loss: float
    bp: float

    # Conviction
    conviction_score: float  # 0-100
    conviction_tier: ConvictionTier
    signal_breakdown: SignalBreakdown

    # MC stats
    mc_pop: float = 0.0
    pop_edge: float = 0.0
    ev: float = 0.0
    ev_per_bp: float = 0.0
    cvar_95: float = 0.0
    stop_prob: float = 0.0

    # Sizing
    suggested_contracts: int = 0
    risk_pct: float = 0.0

    # IV
    iv_rank: float | None = None
    iv_percentile: float = 0.0
    current_iv: float = 0.0
    iv_timing_score: float = 0.0

    # Human-readable summary
    reasoning: str = ""


class PipelineRunResult(BaseModel):
    """Full pipeline run output."""

    config: PipelineConfig = Field(default_factory=PipelineConfig)
    symbols_scanned: int = 0
    symbols_discovered: int = 0
    symbols_validated: int = 0
    symbols_simulated: int = 0
    recommendations: list[TradeRecommendation] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
    run_at: datetime = Field(default_factory=datetime.now)
