"""Pydantic models for scenario simulation."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class ScenarioDefinition(BaseModel):
    """Defines a market scenario with drift/vol overrides and prior probability."""

    name: str
    annual_drift: float = Field(description="Annualized GBM drift override")
    vol_multiplier: float = Field(default=1.0, description="Scale factor for calibrated volatility")
    base_probability: float = Field(
        ge=0.0, le=1.0, description="Prior probability of this scenario"
    )


# Built-in scenarios
BUILTIN_SCENARIOS: dict[str, ScenarioDefinition] = {
    "bull": ScenarioDefinition(
        name="bull", annual_drift=0.15, vol_multiplier=0.9, base_probability=0.25
    ),
    "sideways": ScenarioDefinition(
        name="sideways", annual_drift=0.03, vol_multiplier=1.0, base_probability=0.45
    ),
    "bear": ScenarioDefinition(
        name="bear", annual_drift=-0.15, vol_multiplier=1.3, base_probability=0.20
    ),
    "crash": ScenarioDefinition(
        name="crash", annual_drift=-0.40, vol_multiplier=2.0, base_probability=0.10
    ),
}


class ScenarioConfig(BaseModel):
    """Simulation parameters for scenario analysis."""

    dte: int = Field(default=30, description="Trading days to simulate")
    n_paths: int = Field(default=500, description="Paths per scenario")
    initial_cash: float = Field(default=100_000.0, description="Starting portfolio value")
    warmup_bars: int = Field(
        default=200, description="Historical bars prepended for indicator warmup"
    )
    seed: int | None = Field(default=None, description="RNG seed for reproducibility")


class PathStrategyResult(BaseModel):
    """Per-path strategy execution result."""

    total_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    win_rate: float | None = None
    final_value: float = 0.0


class StrategyScenarioResult(BaseModel):
    """Aggregated strategy results across all paths within one scenario."""

    strategy_name: str
    scenario_name: str
    n_paths: int = 0

    # Return distribution
    mean_return_pct: float = 0.0
    median_return_pct: float = 0.0
    p5_return_pct: float = 0.0
    p25_return_pct: float = 0.0
    p75_return_pct: float = 0.0
    p95_return_pct: float = 0.0

    # Risk
    prob_positive: float = 0.0
    mean_max_dd_pct: float = 0.0
    median_max_dd_pct: float = 0.0

    # Activity
    avg_trades: float = 0.0
    avg_win_rate: float | None = None


class CompositeStrategyResult(BaseModel):
    """Probability-weighted strategy results across all scenarios."""

    strategy_name: str
    expected_return: float = 0.0
    expected_max_dd: float = 0.0
    worst_case_return_p5: float = 0.0
    prob_positive: float = 0.0
    risk_adjusted_score: float = 0.0

    # Per-scenario breakdown
    scenario_results: list[StrategyScenarioResult] = Field(default_factory=list)


class SignalContext(BaseModel):
    """Captured signal state used for scenario weighting."""

    alpha_score: float | None = None
    alpha_signal: str | None = None
    confluence_verdict: str | None = None
    adjusted_weights: dict[str, float] = Field(default_factory=dict)


class ScenarioSimResult(BaseModel):
    """Top-level result of a full scenario simulation run."""

    symbol: str
    scenarios: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    config: ScenarioConfig = Field(default_factory=ScenarioConfig)

    composites: list[CompositeStrategyResult] = Field(default_factory=list)
    best_strategy: str | None = None
    best_score: float = 0.0

    signal_context: SignalContext | None = None
    run_at: datetime = Field(default_factory=datetime.now)
