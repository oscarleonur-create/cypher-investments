"""Pydantic models for the daily workflow pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_STATE_PATH = _PROJECT_ROOT / "data" / "workflow_state.json"


# ── Enums ────────────────────────────────────────────────────────────────────


class WorkflowMode(StrEnum):
    STOCKS = "stocks"
    OPTIONS = "options"


class PositionAction(StrEnum):
    HOLD = "HOLD"
    CONSIDER_PROFIT = "CONSIDER_PROFIT"
    REVIEW = "REVIEW"
    TREND_WARNING = "TREND_WARNING"
    TAKE_PROFIT = "TAKE_PROFIT"
    CONSIDER_CLOSING = "CONSIDER_CLOSING"
    APPROACHING_EXPIRY = "APPROACHING_EXPIRY"
    DELTA_WARNING = "DELTA_WARNING"
    TRAIL_STOP = "TRAIL_STOP"
    TREND_BREAK = "TREND_BREAK"
    EARNINGS_WARNING = "EARNINGS_WARNING"
    CLOSE = "CLOSE"
    ROLL = "ROLL"


class CandidateVerdict(StrEnum):
    DEEP_DIVE = "DEEP_DIVE"
    WATCH = "WATCH"
    SKIP = "SKIP"


# ── Regime ───────────────────────────────────────────────────────────────────


class RegimeContext(BaseModel):
    regime_name: str = "normal"
    vix: float = 0.0
    spy_vol: float = 0.0
    regime_prob: list[float] = Field(default_factory=list)
    warning: str | None = None


# ── Account ──────────────────────────────────────────────────────────────────


class AccountSnapshot(BaseModel):
    net_liq: float = 0.0
    cash: float = 0.0
    buying_power: float = 0.0
    positions: list[dict[str, Any]] = Field(default_factory=list)
    warning: str | None = None


# ── Position Health ──────────────────────────────────────────────────────────


class PositionHealth(BaseModel):
    symbol: str
    underlying: str = ""
    quantity: int = 0
    avg_open_price: float = 0.0
    current_price: float = 0.0
    pnl_pct: float = 0.0
    action: PositionAction = PositionAction.HOLD
    reason: str = ""
    # Options-specific
    dte: int | None = None
    delta: float | None = None
    credit_received: float | None = None


# ── Discovery ────────────────────────────────────────────────────────────────


class ScannerHit(BaseModel):
    symbol: str
    sources: list[str] = Field(default_factory=list)
    best_score: float = 0.0
    smart_money_score: float | None = None
    mispricing_score: float | None = None
    alpha_score: float | None = None
    dip_score: float | None = None


# ── Candidate Validation ─────────────────────────────────────────────────────


class ValidatedCandidate(BaseModel):
    symbol: str
    verdict: CandidateVerdict
    confluence_verdict: str = ""
    ml_signal: str | None = None
    ml_win_prob: float | None = None
    strategy_signals: list[str] = Field(default_factory=list)
    alpha_score: float | None = None
    reason: str = ""


# ── Deep Dive ────────────────────────────────────────────────────────────────


class DeepDiveResult(BaseModel):
    symbol: str
    # Scenario simulation
    scenario_expected_return: float | None = None
    scenario_max_dd: float | None = None
    scenario_prob_profit: float | None = None
    best_strategy: str | None = None
    # Research agent
    research_verdict: str | None = None
    research_conviction: float | None = None
    research_dip_type: str | None = None
    bull_case: list[str] = Field(default_factory=list)
    bear_case: list[str] = Field(default_factory=list)
    # Options pipeline
    strikes: str | None = None
    dte: int | None = None
    credit: float | None = None
    conviction: float | None = None
    pop: float | None = None
    ev: float | None = None
    suggested_contracts: int | None = None


# ── Afternoon Models ─────────────────────────────────────────────────────────


class AfternoonPositionStatus(BaseModel):
    symbol: str
    action: PositionAction = PositionAction.HOLD
    reason: str = ""
    pnl_pct: float = 0.0
    # Options-specific
    dte: int | None = None
    limit_price: float | None = None


class BrierScoreSummary(BaseModel):
    pop_brier: float | None = None
    touch_brier: float | None = None
    stop_brier: float | None = None
    n_samples: int = 0
    n_resolved: int = 0
    warning: str | None = None


class WatchCandidate(BaseModel):
    symbol: str
    new_buy_signals: list[str] = Field(default_factory=list)
    alpha_score: float | None = None
    score_change: float | None = None
    status: str = ""


class EarningsAlert(BaseModel):
    symbol: str
    earnings_date: str
    days_until: int
    has_position: bool = False


# ── Result Containers ────────────────────────────────────────────────────────


class MorningResult(BaseModel):
    mode: str
    run_at: datetime = Field(default_factory=datetime.now)
    regime: RegimeContext = Field(default_factory=RegimeContext)
    account: AccountSnapshot = Field(default_factory=AccountSnapshot)
    position_health: list[PositionHealth] = Field(default_factory=list)
    scanner_hits: list[ScannerHit] = Field(default_factory=list)
    validated: list[ValidatedCandidate] = Field(default_factory=list)
    deep_dives: list[DeepDiveResult] = Field(default_factory=list)
    watch_candidates: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


class AfternoonResult(BaseModel):
    mode: str
    run_at: datetime = Field(default_factory=datetime.now)
    account: AccountSnapshot = Field(default_factory=AccountSnapshot)
    position_statuses: list[AfternoonPositionStatus] = Field(default_factory=list)
    brier_summary: BrierScoreSummary | None = None
    watch_updates: list[WatchCandidate] = Field(default_factory=list)
    earnings_alerts: list[EarningsAlert] = Field(default_factory=list)
    tomorrow_watchlist: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    elapsed_seconds: float = 0.0


# ── Workflow State (morning→afternoon handoff) ───────────────────────────────


class WorkflowState(BaseModel):
    mode: str = ""
    morning_run_at: datetime | None = None
    morning_account: dict[str, Any] | None = None
    watchlist: list[str] = Field(default_factory=list)
    watch_candidates: list[str] = Field(default_factory=list)
    morning_alpha_scores: dict[str, float] = Field(default_factory=dict)
    lean_mode: bool = False

    @classmethod
    def load(cls) -> WorkflowState:
        if _STATE_PATH.exists():
            try:
                return cls.model_validate_json(_STATE_PATH.read_text())
            except Exception:
                return cls()
        return cls()

    def save(self) -> None:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(self.model_dump_json(indent=2))
