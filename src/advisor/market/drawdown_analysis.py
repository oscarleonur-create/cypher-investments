"""Historical drawdown analysis — intra-window max drawdown, breach speed, vol regimes."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────


class BreachSpeedStats(BaseModel):
    """How quickly price breaches a given drawdown threshold within the window."""

    threshold_pct: float = Field(description="Drawdown threshold (e.g. -0.05 = -5%)")
    breach_probability: float = Field(description="Fraction of windows that breached this level")
    median_days: float | None = Field(
        default=None, description="Median trading days to first breach"
    )
    p25_days: float | None = Field(default=None, description="25th percentile days to breach")
    p75_days: float | None = Field(default=None, description="75th percentile days to breach")


class DrawdownQuantiles(BaseModel):
    """Per-DTE drawdown quantile summary."""

    dte: int
    trading_days: int
    dd_p95: float = Field(description="95th percentile intra-window drawdown")
    dd_p97_5: float = Field(description="97.5th percentile drawdown")
    dd_p99: float = Field(description="99th percentile drawdown")
    dd_max: float = Field(description="Maximum observed drawdown")
    n_windows: int = Field(description="Number of rolling windows used")
    breach_speed: list[BreachSpeedStats] = Field(default_factory=list)


class VolRegimeDrawdown(BaseModel):
    """Drawdown quantiles conditioned on volatility regime at window start."""

    regime: str = Field(description="low, mid, or high")
    dte: int
    dd_p95: float
    dd_p99: float
    dd_max: float
    hv20_low: float = Field(description="Lower bound of HV20 for this regime")
    hv20_high: float = Field(description="Upper bound of HV20 for this regime")
    n_windows: int


class MaxMoveResult(BaseModel):
    """Full drawdown analysis result for a symbol."""

    symbol: str
    current_price: float
    hv20_current: float
    current_regime: str
    quantiles: list[DrawdownQuantiles] = Field(default_factory=list)
    regime_drawdowns: list[VolRegimeDrawdown] = Field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_price_history(symbol: str, lookback: int = 504) -> pd.Series:
    """Fetch ~2yr daily close prices via yfinance."""
    period = f"{max(lookback // 252 + 1, 2)}y"
    df = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No price data for {symbol}")
    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.dropna()


def _dte_to_trading_days(dte: int) -> int:
    """Convert calendar DTE to approximate trading days."""
    return max(1, round(dte * 5 / 7))


# ── Core computation ─────────────────────────────────────────────────────────


def compute_intra_window_drawdowns(prices: np.ndarray, H: int) -> np.ndarray:
    """Compute intra-window max drawdown for each rolling start day.

    For each start t, drawdown = min_{k=1..H}((S_{t+k} - S_t) / S_t).
    Returns array of drawdown values (negative = loss).
    """
    n = len(prices)
    if n <= H:
        return np.array([])

    drawdowns = np.empty(n - H)
    for t in range(n - H):
        start_price = prices[t]
        window = prices[t + 1 : t + H + 1]
        returns = (window - start_price) / start_price
        drawdowns[t] = returns.min()

    return drawdowns


def compute_breach_speed(
    prices: np.ndarray,
    H: int,
    thresholds: list[float] | None = None,
) -> list[BreachSpeedStats]:
    """Compute breach speed statistics for given drawdown thresholds.

    For each threshold, finds fraction of windows that breached it and
    distribution of first-passage time in trading days.
    """
    if thresholds is None:
        thresholds = [-0.05, -0.10, -0.15]

    n = len(prices)
    if n <= H:
        return []

    results = []
    for threshold in thresholds:
        breach_days = []
        n_windows = n - H

        for t in range(n_windows):
            start_price = prices[t]
            for k in range(1, H + 1):
                ret = (prices[t + k] - start_price) / start_price
                if ret <= threshold:
                    breach_days.append(k)
                    break

        breach_prob = len(breach_days) / n_windows if n_windows > 0 else 0.0

        stats = BreachSpeedStats(
            threshold_pct=threshold,
            breach_probability=round(breach_prob, 4),
        )
        if breach_days:
            arr = np.array(breach_days)
            stats.median_days = float(np.median(arr))
            stats.p25_days = float(np.percentile(arr, 25))
            stats.p75_days = float(np.percentile(arr, 75))

        results.append(stats)

    return results


def compute_vol_regime_labels(
    prices: np.ndarray, window: int = 20
) -> tuple[np.ndarray, dict[str, tuple[float, float]], str]:
    """Compute rolling 20d HV, split into tercile regimes.

    Returns:
        labels: array of regime labels ("low", "mid", "high") per day
        boundaries: dict mapping regime name to (low_bound, high_bound) of HV20
        current_regime: regime label for the most recent observation
    """
    returns = np.diff(np.log(prices))
    if len(returns) < window:
        labels = np.array(["mid"] * len(prices), dtype="U4")
        return labels, {"low": (0.0, 0.0), "mid": (0.0, 1.0), "high": (1.0, 1.0)}, "mid"

    # Rolling HV (annualized)
    hv = pd.Series(returns).rolling(window).std().values * np.sqrt(252)

    # Tercile boundaries (ignoring NaNs from rolling warmup)
    valid_hv = hv[~np.isnan(hv)]
    if len(valid_hv) == 0:
        labels = np.array(["mid"] * len(prices), dtype="U4")
        return labels, {"low": (0.0, 0.0), "mid": (0.0, 1.0), "high": (1.0, 1.0)}, "mid"

    t33 = float(np.percentile(valid_hv, 33.3))
    t67 = float(np.percentile(valid_hv, 66.7))

    label_list = ["mid"] * len(hv)
    for i in range(len(hv)):
        if np.isnan(hv[i]):
            label_list[i] = "mid"
        elif hv[i] <= t33:
            label_list[i] = "low"
        elif hv[i] >= t67:
            label_list[i] = "high"
    labels = np.array(label_list, dtype="U4")

    # Boundaries
    boundaries = {
        "low": (float(valid_hv.min()), t33),
        "mid": (t33, t67),
        "high": (t67, float(valid_hv.max())),
    }

    # Current regime: use last valid HV value
    current_regime = labels[-1] if len(labels) > 0 else "mid"

    return labels, boundaries, str(current_regime)


# ── Main entry point ─────────────────────────────────────────────────────────


def analyze_max_move(
    symbol: str,
    dtes: list[int] | None = None,
    lookback: int = 252,
    thresholds: list[float] | None = None,
    include_regimes: bool = True,
) -> MaxMoveResult:
    """Analyze historical intra-window max drawdown for a symbol.

    Args:
        symbol: Ticker symbol
        dtes: Calendar DTEs to analyze (default: [21, 30, 45, 60])
        lookback: Trading days of history for rolling windows
        thresholds: Drawdown thresholds for breach speed (default: [-5%, -10%, -15%])
        include_regimes: Whether to compute vol-regime conditioned drawdowns

    Returns:
        MaxMoveResult with quantiles and optional regime breakdowns
    """
    symbol = symbol.upper()
    if dtes is None:
        dtes = [21, 30, 45, 60]
    if thresholds is None:
        thresholds = [-0.05, -0.10, -0.15]

    # Fetch price history (need extra for regime labels warmup)
    close = _get_price_history(symbol, lookback=lookback + 100)
    prices = close.values.astype(float)

    # Current price and HV20
    current_price = float(prices[-1])
    if len(prices) >= 21:
        recent_returns = np.diff(np.log(prices[-21:]))
        hv20_current = float(np.std(recent_returns) * np.sqrt(252))
    else:
        hv20_current = 0.0

    # Vol regime labels
    if include_regimes:
        regime_labels, boundaries, current_regime = compute_vol_regime_labels(prices)
    else:
        regime_labels = None
        boundaries = {}
        current_regime = "n/a"

    # Use only the lookback window for drawdown computation
    analysis_prices = prices[-lookback:] if len(prices) > lookback else prices
    if regime_labels is not None:
        analysis_labels = regime_labels[-len(analysis_prices) :]
    else:
        analysis_labels = None

    quantiles_list = []
    regime_drawdowns_list = []

    for dte in sorted(dtes):
        H = _dte_to_trading_days(dte)

        # Unconditional drawdowns
        dd = compute_intra_window_drawdowns(analysis_prices, H)
        if len(dd) == 0:
            continue

        breach = compute_breach_speed(analysis_prices, H, thresholds)

        quantiles_list.append(
            DrawdownQuantiles(
                dte=dte,
                trading_days=H,
                dd_p95=round(float(np.percentile(dd, 5)), 4),  # 5th pctile of dd = 95th worst
                dd_p97_5=round(float(np.percentile(dd, 2.5)), 4),
                dd_p99=round(float(np.percentile(dd, 1)), 4),
                dd_max=round(float(dd.min()), 4),
                n_windows=len(dd),
                breach_speed=breach,
            )
        )

        # Regime-conditioned drawdowns
        if include_regimes and analysis_labels is not None:
            for regime_name in ("low", "mid", "high"):
                # Mask: windows starting in this regime
                regime_mask = np.array([analysis_labels[t] == regime_name for t in range(len(dd))])
                regime_dd = dd[regime_mask]

                if len(regime_dd) < 5:
                    continue

                hv_low, hv_high = boundaries.get(regime_name, (0.0, 0.0))
                regime_drawdowns_list.append(
                    VolRegimeDrawdown(
                        regime=regime_name,
                        dte=dte,
                        dd_p95=round(float(np.percentile(regime_dd, 5)), 4),
                        dd_p99=round(float(np.percentile(regime_dd, 1)), 4),
                        dd_max=round(float(regime_dd.min()), 4),
                        hv20_low=round(hv_low, 4),
                        hv20_high=round(hv_high, 4),
                        n_windows=len(regime_dd),
                    )
                )

    return MaxMoveResult(
        symbol=symbol,
        current_price=round(current_price, 2),
        hv20_current=round(hv20_current, 4),
        current_regime=current_regime,
        quantiles=quantiles_list,
        regime_drawdowns=regime_drawdowns_list,
    )


def get_regime_matched_quantile(
    result: MaxMoveResult, dte: int, percentile: float = 95.0
) -> float | None:
    """Lookup helper: get regime-matched drawdown quantile for a DTE.

    Falls back to unconditional quantile if regime data unavailable.
    """
    # Try regime-matched first
    for rd in result.regime_drawdowns:
        if rd.dte == dte and rd.regime == result.current_regime:
            if percentile == 95.0:
                return rd.dd_p95
            elif percentile == 99.0:
                return rd.dd_p99
            break

    # Fallback to unconditional
    for q in result.quantiles:
        if q.dte == dte:
            if percentile == 95.0:
                return q.dd_p95
            elif percentile == 97.5:
                return q.dd_p97_5
            elif percentile == 99.0:
                return q.dd_p99
            break

    return None
