"""Calibration module — fit fat-tail and vol dynamics parameters from historical data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from advisor.simulator.models import SimConfig

logger = logging.getLogger(__name__)


def _get_daily_returns(symbol: str, lookback: int = 252) -> pd.Series:
    """Fetch daily returns for a symbol using yfinance."""
    import yfinance as yf

    period = f"{max(lookback // 252 + 1, 2)}y"
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data for {symbol}")
    close = df["Close"].squeeze()
    returns = np.log(close / close.shift(1)).dropna()
    return returns.iloc[-lookback:] if len(returns) > lookback else returns


def fit_student_t(returns: pd.Series) -> tuple[float, float, float]:
    """Fit Student-t distribution to daily log returns.

    Returns (df, loc, scale). Typical df: 3-8 for equities.
    """
    df, loc, scale = student_t.fit(returns.values)
    # Clamp df to reasonable range
    df = float(np.clip(df, 2.5, 30.0))
    return df, float(loc), float(scale)


def estimate_vol_dynamics(returns: pd.Series) -> dict[str, float]:
    """Estimate volatility dynamics from historical returns.

    Returns dict with vol_mean_level, vol_mean_revert_speed, leverage_effect.
    Uses AR(1) regression on rolling HV for mean-reversion speed estimation.
    """
    # Rolling 30-day HV (annualized)
    rolling_hv = returns.rolling(30).std() * np.sqrt(252)
    rolling_hv = rolling_hv.dropna()

    if len(rolling_hv) < 30:
        return {
            "vol_mean_level": float(returns.std() * np.sqrt(252)),
            "vol_mean_revert_speed": 0.5,
            "leverage_effect": -0.5,
        }

    # Vol mean level: median HV
    vol_mean_level = float(rolling_hv.median())

    # Mean-reversion speed via AR(1) fit on HV levels: hv[t] = a + b*hv[t-1]
    # kappa = -log(b) * 252 (annualized), clamped to [0.1, 5.0]
    hv_arr = rolling_hv.values
    y = hv_arr[1:]
    x = hv_arr[:-1]
    if len(x) > 10:
        # OLS: b = cov(x,y) / var(x)
        b = np.cov(x, y)[0, 1] / max(np.var(x), 1e-10)
        b = float(np.clip(b, 0.01, 0.999))  # Must be < 1 for mean reversion
        # Convert daily AR(1) coefficient to continuous kappa
        vol_mean_revert_speed = float(np.clip(-np.log(b) * 252, 0.1, 5.0))
    else:
        vol_mean_revert_speed = 0.5

    # Leverage effect: correlation of returns vs vol changes
    aligned = pd.DataFrame({"ret": returns, "dvol": rolling_hv.diff()}).dropna()
    if len(aligned) > 10:
        leverage_effect = float(np.clip(aligned["ret"].corr(aligned["dvol"]), -0.9, 0.0))
    else:
        leverage_effect = -0.5

    return {
        "vol_mean_level": vol_mean_level,
        "vol_mean_revert_speed": vol_mean_revert_speed,
        "leverage_effect": leverage_effect,
    }


def calibrate(symbol: str, config: SimConfig | None = None) -> SimConfig:
    """Calibrate SimConfig for a symbol using historical data.

    Downloads price data once and passes to both fitters.
    """
    base = config or SimConfig()

    try:
        returns = _get_daily_returns(symbol)
        df, loc, scale = fit_student_t(returns)
        vol_params = estimate_vol_dynamics(returns)

        return base.model_copy(
            update={
                "student_t_df": df,
                "vol_mean_level": vol_params["vol_mean_level"],
                "vol_mean_revert_speed": vol_params["vol_mean_revert_speed"],
                "leverage_effect": vol_params["leverage_effect"],
            }
        )
    except Exception as e:
        logger.warning("Calibration failed for %s, using defaults: %s", symbol, e)
        return base
