"""Fractional differentiation — preserve long-range memory while achieving stationarity.

Implements the Hosking/Jensen fixed-window fractional differentiation
weights from Lopez de Prado (2018).  Fractionally differenced series
retain more memory than integer-differenced (d=1) series while still
being stationary (testable via ADF), which helps ML models capture
long-term trend without losing predictive signal.

Usage::

    from advisor.ml.fracdiff import fracdiff_series, optimal_d

    # Fractionally difference a close-price series
    fd = fracdiff_series(close, d=0.4, window=100)

    # Find the minimum d that makes the series stationary
    best_d = optimal_d(close)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _fracdiff_weights(d: float, window: int, threshold: float = 1e-5) -> np.ndarray:
    """Compute fractional differentiation weights using the Hosking recursion.

    w_k = -w_{k-1} * (d - k + 1) / k

    Weights are truncated at ``window`` terms or when |w_k| < ``threshold``.

    Args:
        d: Differentiation order (0 < d < 1 typically).
        window: Maximum number of weight terms.
        threshold: Minimum absolute weight before truncation.

    Returns:
        1-D array of weights, length <= window.
    """
    weights = [1.0]
    for k in range(1, window):
        w_k = -weights[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        weights.append(w_k)
    return np.array(weights)


def fracdiff_series(
    series: pd.Series,
    d: float = 0.4,
    window: int = 100,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fixed-window fractional differentiation to a price series.

    Args:
        series: Input series (typically log-prices or close prices).
        d: Differentiation order.  d=0 is the original series, d=1 is
           standard first-difference.  Typical values: 0.3–0.5.
        window: Maximum lookback for weights.
        threshold: Minimum absolute weight to include.

    Returns:
        Fractionally differenced Series, same index as input but with
        NaN for the warm-up period.
    """
    weights = _fracdiff_weights(d, window, threshold)
    width = len(weights)

    result = pd.Series(np.nan, index=series.index)
    values = series.values

    for i in range(width - 1, len(values)):
        result.iloc[i] = np.dot(weights, values[i - width + 1 : i + 1][::-1])

    return result


def optimal_d(
    series: pd.Series,
    d_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.05,
    significance: float = 0.05,
    window: int = 100,
) -> float:
    """Find the minimum d that makes the series stationary (ADF test).

    Searches from d=0 upward in increments of ``step`` until the
    Augmented Dickey-Fuller test p-value falls below ``significance``.

    Args:
        series: Input price series.
        d_range: (min_d, max_d) search range.
        step: Increment for d search.
        significance: ADF p-value threshold for stationarity.
        window: Fracdiff weight window.

    Returns:
        Minimum d achieving stationarity, or d_range[1] if none found.
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels not installed — returning default d=0.4")
        return 0.4

    best_d = d_range[1]
    d = d_range[0]

    while d <= d_range[1]:
        fd = fracdiff_series(series, d=d, window=window).dropna()
        if len(fd) < 30:
            d += step
            continue

        try:
            adf_stat, pvalue, *_ = adfuller(fd, maxlag=1)
            if pvalue < significance:
                best_d = d
                logger.info("Optimal d=%.2f (ADF stat=%.3f, p=%.4f)", d, adf_stat, pvalue)
                break
        except Exception:
            pass
        d += step

    return round(best_d, 2)


def compute_fracdiff_features(
    close: pd.Series,
    d_values: list[float] | None = None,
    window: int = 100,
) -> pd.DataFrame:
    """Compute fractional differentiation features for multiple d values.

    Args:
        close: Close price series.
        d_values: List of d values to compute.  Defaults to [0.3, 0.4, 0.5].
        window: Weight lookback window.

    Returns:
        DataFrame with columns ``fracdiff_d{d}`` for each d, plus
        ``fracdiff_auto`` using the optimal d found via ADF test.
    """
    if d_values is None:
        d_values = [0.3, 0.4, 0.5]

    log_close = np.log(close.replace(0, np.nan)).dropna()
    features = pd.DataFrame(index=close.index)

    for d in d_values:
        col = f"fracdiff_d{d:.1f}".replace(".", "")
        fd = fracdiff_series(log_close, d=d, window=window)
        # Normalize by rolling std for stationarity
        fd_std = fd.rolling(20).std().replace(0, np.nan)
        features[col] = fd / fd_std

    # Auto-optimal d
    try:
        auto_d = optimal_d(log_close, window=window)
        fd_auto = fracdiff_series(log_close, d=auto_d, window=window)
        fd_auto_std = fd_auto.rolling(20).std().replace(0, np.nan)
        features["fracdiff_auto"] = fd_auto / fd_auto_std
        features["fracdiff_optimal_d"] = auto_d  # Scalar feature
    except Exception as e:
        logger.warning("Could not compute optimal fracdiff: %s", e)
        features["fracdiff_auto"] = 0.0
        features["fracdiff_optimal_d"] = 0.5

    return features
