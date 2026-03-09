"""Data models for the strategy case pipeline."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ── Enums ────────────────────────────────────────────────────────────────────


class ScenarioType(StrEnum):
    EARNINGS_DIP = "EARNINGS_DIP"
    IV_SPIKE = "IV_SPIKE"
    BREAKOUT_PULLBACK = "BREAKOUT_PULLBACK"
    RANGE_BOUND = "RANGE_BOUND"
    MEAN_REVERSION = "MEAN_REVERSION"
    MOMENTUM = "MOMENTUM"


class OptionsStrategyType(StrEnum):
    PUT_CREDIT_SPREAD = "put_credit_spread"
    NAKED_PUT = "naked_put"
    COVERED_CALL = "covered_call"
    CALL_CREDIT_SPREAD = "call_credit_spread"
    IRON_CONDOR = "iron_condor"
    SHORT_STRANGLE = "short_strangle"
    WHEEL = "wheel"


class CaseVerdict(StrEnum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    REJECT = "REJECT"


# ── Stage 1: Scenario Detection ─────────────────────────────────────────────


class ScenarioResult(BaseModel):
    """Output of scenario detection."""

    scenario_type: ScenarioType
    confidence: float = Field(ge=0, le=1.0, default=0.5)
    summary: str = ""
    price: float = 0.0
    price_change_pct: float = 0.0
    iv_percentile: float = 0.0
    dip_score: str | None = None  # from dip screener
    pead_score: str | None = None  # from pead screener
    rsi: float | None = None
    sma_20_distance_pct: float | None = None
    days_since_earnings: int | None = None
    earnings_date: date | None = None


# ── Stage 2: Strategy Mapping ────────────────────────────────────────────────


class StrategyMatch(BaseModel):
    """A strategy candidate with fit score and metadata."""

    strategy: OptionsStrategyType
    fit_score: float = Field(ge=0, le=100, default=0.0)
    reasoning: str = ""
    preferred_dte_min: int = 25
    preferred_dte_max: int = 45
    preferred_delta_min: float = 0.15
    preferred_delta_max: float = 0.35


class StrategyRanking(BaseModel):
    """Ranked list of strategy candidates."""

    matches: list[StrategyMatch] = Field(default_factory=list)
    selected: StrategyMatch | None = None
    override_applied: bool = False


# ── Stage 3: Research Bridge ─────────────────────────────────────────────────


class ResearchSummary(BaseModel):
    """Condensed version of OpportunityCard for the case pipeline."""

    verdict: str = ""  # BUY_THE_DIP / WATCH / AVOID
    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)
    catalyst_summary: str = ""
    grounding_score: float = 1.0
    key_metrics_revenue_growth: str = "Unknown"
    key_metrics_margins: str = "Unknown"
    key_metrics_fcf: str = "Unknown"
    key_metrics_guidance: str = "Unknown"


# ── Stage 4: Options Analysis ────────────────────────────────────────────────


class StrikeRecommendation(BaseModel):
    """A specific strike/spread recommendation from premium screening."""

    strategy: str = ""
    strike: float = 0.0
    long_strike: float | None = None
    expiry: date | None = None
    dte: int = 0
    credit: float = 0.0
    delta: float = 0.0
    iv: float = 0.0
    pop: float = 0.0
    annualized_yield: float = 0.0
    sell_score: float = 0.0
    liquidity_score: int = 0
    expected_move: float = 0.0
    strike_vs_em: float = 0.0
    max_loss: float = 0.0
    margin_req: float = 0.0
    flags: list[str] = Field(default_factory=list)


class OptionsAnalysisResult(BaseModel):
    """Output of options analysis stage."""

    recommendations: list[StrikeRecommendation] = Field(default_factory=list)
    iv_percentile: float = 0.0
    term_structure: str = "flat"
    regime: str = "normal"
    errors: list[str] = Field(default_factory=list)


# ── Stage 5: Risk Assessment ────────────────────────────────────────────────


class RiskProfile(BaseModel):
    """Risk assessment from MC simulation or BSM estimates."""

    source: str = "bsm"  # "mc" or "bsm"
    pop: float = 0.0  # probability of profit
    ev: float = 0.0  # expected value per contract
    ev_per_bp: float = 0.0
    cvar_95: float = 0.0
    stop_prob: float = 0.0
    touch_prob: float = 0.0
    suggested_contracts: int = 0
    position_bp: float = 0.0
    max_loss_total: float = 0.0
    risk_pct: float = 0.0
    sizing_feasible: bool = False


# ── Stage 6: Synthesis ───────────────────────────────────────────────────────


class CaseSynthesis(BaseModel):
    """LLM-generated trade case synthesis."""

    thesis_summary: str = ""
    entry_criteria: list[str] = Field(default_factory=list)
    exit_plan: list[str] = Field(default_factory=list)
    invalidation: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    management_plan: list[str] = Field(default_factory=list)
    verdict: CaseVerdict = CaseVerdict.WEAK
    conviction_score: float = Field(ge=0, le=100, default=0.0)


# ── Final Output ─────────────────────────────────────────────────────────────


class StrategyCaseConfig(BaseModel):
    """Configuration for the strategy case builder."""

    account_size: float = 5_000.0
    max_risk_pct: float = 5.0
    max_bp_pct: float = 50.0
    enable_research: bool = False
    enable_mc: bool = False
    strategy_override: OptionsStrategyType | None = None
    min_dte: int = 25
    max_dte: int = 45


class StrategyCase(BaseModel):
    """Complete strategy case — final pipeline output."""

    symbol: str
    built_at: datetime = Field(default_factory=datetime.now)
    config: StrategyCaseConfig = Field(default_factory=StrategyCaseConfig)

    # Stage outputs
    scenario: ScenarioResult | None = None
    ranking: StrategyRanking | None = None
    research: ResearchSummary | None = None
    options: OptionsAnalysisResult | None = None
    risk: RiskProfile | None = None
    synthesis: CaseSynthesis | None = None

    # Errors from any stage
    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0
