"""Pydantic models for Monte Carlo PCS simulator."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SimConfig(BaseModel):
    """Monte Carlo simulation configuration."""

    # Path generation
    n_paths: int = Field(default=10_000, description="Number of MC paths (10K quick, 100K deep)")
    student_t_df: float = Field(
        default=5.0, description="Student-t degrees of freedom for fat tails"
    )
    vol_mean_level: float = Field(default=0.30, description="Long-run IV mean level")
    vol_mean_revert_speed: float = Field(default=0.5, description="IV mean-reversion speed (kappa)")
    leverage_effect: float = Field(
        default=-0.5, description="Return-vol correlation (negative = leverage)"
    )
    vol_of_vol: float = Field(
        default=0.1, description="IV diffusion coefficient (volatility of volatility)"
    )
    risk_free_rate: float = Field(default=0.05, description="Risk-free rate")
    seed: int | None = Field(
        default=None, description="RNG seed for reproducibility (None = random)"
    )

    # Exit rules
    profit_target_pct: float = Field(default=0.50, description="Close at 50% of credit collected")
    stop_loss_multiplier: float = Field(default=2.0, description="Close at 2x credit loss")
    close_at_dte: int = Field(default=7, description="Close when DTE <= this value")

    # Slippage
    slippage_pct: float = Field(
        default=0.02, description="Slippage on exit as fraction of spread width"
    )

    # Variance reduction
    use_antithetic: bool = Field(default=True, description="Antithetic variates: pair Z with -Z")
    use_control_variate: bool = Field(default=True, description="BSM analytical control variate")
    use_stratified: bool = Field(default=False, description="Stratified sampling (opt-in)")
    use_importance_sampling: bool = Field(
        default=False, description="Importance sampling for tail risk (opt-in)"
    )
    is_tail_quantile: float = Field(default=0.05, description="Target tail quantile for IS")

    # Candidate generation
    min_credit: float = Field(default=0.10, description="Minimum net credit per spread")
    min_width: float = Field(default=2.0, description="Minimum spread width in dollars")
    max_width: float = Field(default=10.0, description="Maximum spread width in dollars")
    max_buying_power: float = Field(default=5000.0, description="Maximum buying power per spread")
    delta_target: float | None = Field(default=None, description="Override adaptive delta target")


class PCSCandidate(BaseModel):
    """Put credit spread candidate with real market data."""

    symbol: str
    expiration: str
    dte: int
    short_strike: float
    long_strike: float
    width: float
    short_bid: float
    short_ask: float
    long_bid: float
    long_ask: float
    net_credit: float = Field(description="short_bid - long_ask (conservative)")
    mid_credit: float = Field(description="(short_mid - long_mid)")
    short_delta: float
    short_gamma: float
    short_theta: float
    short_vega: float
    short_iv: float
    long_delta: float
    long_iv: float
    underlying_price: float
    iv_percentile: float = 0.0
    iv_rank: float = 0.0
    pop_estimate: float = Field(default=0.0, description="1 - abs(short_delta)")
    sell_score: float = 0.0
    buying_power: float = Field(description="(width - net_credit) * 100")
    dd_cushion_ratio: float | None = Field(
        default=None,
        description="Strike OTM% / |DD_H^(95)|: >1 means strike beyond 95th pctile drawdown",
    )


class SimResult(BaseModel):
    """Per-candidate Monte Carlo simulation results."""

    candidate_id: str = ""
    symbol: str = ""
    short_strike: float = 0.0
    long_strike: float = 0.0
    dte: int = 0
    net_credit: float = 0.0

    # MC outputs
    ev: float = Field(description="Expected value per spread (dollars)")
    pop: float = Field(description="Probability of profit")
    touch_prob: float = Field(description="Probability price touches short strike")
    cvar_95: float = Field(description="Conditional VaR at 95% (worst 5% avg loss)")
    stop_prob: float = Field(description="Probability of hitting stop loss")
    avg_hold_days: float = Field(description="Average holding period in days")
    ev_per_bp: float = Field(description="EV per dollar of buying power")

    # MC precision
    mc_std_err: float = Field(default=0.0, description="MC standard error of EV estimate")
    variance_reduction_factor: float = Field(
        default=1.0, description="Variance improvement vs crude MC"
    )
    cvar_95_is: float = Field(default=0.0, description="CVaR95 from importance sampling")
    cvar_95_se: float = Field(default=0.0, description="Standard error of CVaR95 estimate")

    # P&L distribution
    pnl_p5: float = Field(description="5th percentile P&L")
    pnl_p25: float = Field(description="25th percentile P&L")
    pnl_p50: float = Field(description="Median P&L")
    pnl_p75: float = Field(description="75th percentile P&L")
    pnl_p95: float = Field(description="95th percentile P&L")

    # Exit breakdown
    exit_profit_target: float = Field(description="Fraction exiting at profit target")
    exit_stop_loss: float = Field(description="Fraction exiting at stop loss")
    exit_dte: float = Field(description="Fraction exiting at DTE threshold")
    exit_expiration: float = Field(description="Fraction held to expiration")


class CalibrationRecord(BaseModel):
    """Tracks predicted vs actual outcomes for Brier score calibration."""

    candidate_id: str
    symbol: str
    predicted_pop: float
    predicted_touch: float
    predicted_stop: float
    predicted_ev: float = 0.0
    actual_profit: float | None = None  # 1.0 if profitable, 0.0 if not
    actual_touch: float | None = None  # 1.0 if touched, 0.0 if not
    actual_stop: float | None = None  # 1.0 if stopped, 0.0 if not
    actual_pnl: float | None = None
    created_at: datetime = Field(default_factory=datetime.now)


class PipelineResult(BaseModel):
    """Full simulation pipeline output."""

    symbols_scanned: int
    candidates_generated: int
    candidates_simulated: int
    top_results: list[SimResult]
    calibration_params: dict = Field(default_factory=dict)
    run_at: datetime = Field(default_factory=datetime.now)
    config: SimConfig = Field(default_factory=SimConfig)
