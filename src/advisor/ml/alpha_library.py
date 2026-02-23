"""Alpha library — formulaic alpha factors inspired by WorldQuant 101.

Each alpha is a pure function: OHLCV DataFrame -> Series (one value per bar).
Alphas are grouped into orthogonal categories (momentum, mean-reversion,
volume, volatility, price-pattern) to maximize breadth.

Usage::

    from advisor.ml.alpha_library import compute_all_alphas

    alphas_df = compute_all_alphas(df)  # df has OHLCV columns
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helper functions ──────────────────────────────────────────────────


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percentile rank (0-1) of current value in window."""
    return series.rolling(window).rank(pct=True)


def _ts_delta(series: pd.Series, period: int) -> pd.Series:
    """Change from ``period`` bars ago."""
    return series - series.shift(period)


def _ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Pearson correlation."""
    return x.rolling(window).corr(y)


def _ts_stddev(series: pd.Series, window: int) -> pd.Series:
    """Rolling standard deviation."""
    return series.rolling(window).std()


def _ts_argmax(series: pd.Series, window: int) -> pd.Series:
    """Position of max value in window (0 = oldest, window-1 = newest), normalized."""
    return series.rolling(window).apply(
        lambda x: np.argmax(x) / (len(x) - 1) if len(x) > 1 else 0.5, raw=True
    )


def _ts_argmin(series: pd.Series, window: int) -> pd.Series:
    """Position of min value in window, normalized 0-1."""
    return series.rolling(window).apply(
        lambda x: np.argmin(x) / (len(x) - 1) if len(x) > 1 else 0.5, raw=True
    )


def _decay_linear(series: pd.Series, window: int) -> pd.Series:
    """Linearly-weighted moving average (recent bars weighted more)."""
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)


# ── Alpha factors ─────────────────────────────────────────────────────
# Each returns a pd.Series with the same index as the input.


def alpha_momentum_12_1(close: pd.Series) -> pd.Series:
    """12-month momentum skipping last month (classic cross-sectional momentum)."""
    return close.pct_change(252) - close.pct_change(21)


def alpha_short_reversal(close: pd.Series) -> pd.Series:
    """5-day mean reversion: negative of 5-day return."""
    return -close.pct_change(5)


def alpha_volume_surprise(volume: pd.Series) -> pd.Series:
    """Volume ratio: today vs 20-day average."""
    avg = volume.rolling(20).mean()
    return (volume / avg.replace(0, np.nan)) - 1.0


def alpha_price_volume_corr(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Rolling correlation between returns and volume (10d).
    Negative = divergence (smart money)."""
    ret = close.pct_change()
    return -_ts_corr(ret, volume.astype(float), 10)


def alpha_intraday_intensity(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Intraday intensity: (2*close - high - low) / (high - low) * volume.
    Measures where close falls within the high-low range, weighted by volume."""
    hl_range = (high - low).replace(0, np.nan)
    ii = (2 * close - high - low) / hl_range * volume
    return ii.rolling(10).sum() / volume.rolling(10).sum().replace(0, np.nan)


def alpha_overnight_sentiment(open_: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight return momentum (close-to-open gap, smoothed)."""
    gap = (open_ - close.shift(1)) / close.shift(1)
    return gap.rolling(5).mean()


def alpha_vol_of_vol(close: pd.Series) -> pd.Series:
    """Volatility of volatility — 2nd moment instability."""
    ret = close.pct_change()
    rolling_vol = ret.rolling(10).std()
    return rolling_vol.rolling(20).std() / rolling_vol.rolling(20).mean().replace(0, np.nan)


def alpha_skewness(close: pd.Series) -> pd.Series:
    """Rolling 20-day return skewness."""
    ret = close.pct_change()
    return ret.rolling(20).skew()


def alpha_kurtosis(close: pd.Series) -> pd.Series:
    """Rolling 20-day return kurtosis (excess)."""
    ret = close.pct_change()
    return ret.rolling(20).kurt()


def alpha_amihud_illiquidity(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Amihud illiquidity: |return| / dollar volume (20d rolling avg).
    Higher = less liquid = larger price impact."""
    abs_ret = close.pct_change().abs()
    dollar_vol = (close * volume).replace(0, np.nan)
    ratio = abs_ret / dollar_vol
    return ratio.rolling(20).mean() * 1e9  # Scale up for readability


def alpha_high_low_momentum(high: pd.Series, low: pd.Series) -> pd.Series:
    """Momentum of the high-low range: expanding range = trending, contracting = mean-reverting."""
    hl = (high - low) / low
    return _ts_delta(hl.rolling(5).mean(), 10)


def alpha_close_location(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
) -> pd.Series:
    """Close location value: where close sits in the day's range, smoothed.
    1.0 = closed at high, 0.0 = closed at low."""
    hl_range = (high - low).replace(0, np.nan)
    clv = (close - low) / hl_range
    return clv.rolling(10).mean()


def alpha_acceleration(close: pd.Series) -> pd.Series:
    """Price acceleration: 2nd derivative of price (change in momentum)."""
    ret = close.pct_change()
    mom = ret.rolling(10).mean()
    return _ts_delta(mom, 5)


def alpha_down_vol_ratio(close: pd.Series) -> pd.Series:
    """Downside volatility ratio: vol of negative returns / total vol.
    Higher = more downside risk."""
    ret = close.pct_change()
    down_ret = ret.clip(upper=0)
    total_vol = ret.rolling(20).std().replace(0, np.nan)
    down_vol = down_ret.rolling(20).std()
    return down_vol / total_vol


def alpha_ret_consistency(close: pd.Series) -> pd.Series:
    """Return consistency: fraction of positive days in 20d window.
    High = steady uptrend, low = choppy."""
    ret = close.pct_change()
    return (ret > 0).astype(float).rolling(20).mean()


def alpha_max_drawdown_20d(close: pd.Series) -> pd.Series:
    """Rolling 20-day max drawdown."""
    roll_max = close.rolling(20).max()
    drawdown = close / roll_max.replace(0, np.nan) - 1.0
    return drawdown


def alpha_vwap_deviation(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Deviation from approximate VWAP (typical price * volume weighted)."""
    typical = (high + low + close) / 3
    vwap_num = (typical * volume).rolling(20).sum()
    vwap_den = volume.rolling(20).sum().replace(0, np.nan)
    vwap = vwap_num / vwap_den
    return (close - vwap) / vwap.replace(0, np.nan)


def alpha_trend_strength(close: pd.Series) -> pd.Series:
    """ADX-like trend strength: absolute return / range over 20 bars.
    Close to 1.0 = strong trend, close to 0 = range-bound."""
    net_move = abs(close - close.shift(20))
    cum_range = close.pct_change().abs().rolling(20).sum() * close
    return net_move / cum_range.replace(0, np.nan)


def alpha_volume_price_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume-Price Trend (VPT): cumulative volume * return, then slope."""
    ret = close.pct_change()
    vpt = (ret * volume).cumsum()
    # Slope over last 10 bars, normalized by rolling vol
    slope = vpt.rolling(10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True
    )
    vol_mean = volume.rolling(20).mean().replace(0, np.nan)
    return slope / vol_mean


# ── Main API ──────────────────────────────────────────────────────────

_ALPHA_REGISTRY: list[tuple[str, callable]] = []


def _register(name: str):
    """Decorator to register an alpha computation function."""

    def wrapper(fn):
        _ALPHA_REGISTRY.append((name, fn))
        return fn

    return wrapper


def compute_all_alphas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all alpha factors from an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: Close, High, Low, Open, Volume.

    Returns:
        DataFrame with one column per alpha factor, same index as input.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    open_ = df["Open"]
    volume = df["Volume"].astype(float)

    alphas = pd.DataFrame(index=df.index)

    # Momentum category
    alphas["alpha_mom_12_1"] = alpha_momentum_12_1(close)
    alphas["alpha_short_reversal"] = alpha_short_reversal(close)
    alphas["alpha_acceleration"] = alpha_acceleration(close)
    alphas["alpha_ret_consistency"] = alpha_ret_consistency(close)

    # Mean-reversion category
    alphas["alpha_overnight_sentiment"] = alpha_overnight_sentiment(open_, close)
    alphas["alpha_close_location"] = alpha_close_location(close, high, low)

    # Volume category
    alphas["alpha_volume_surprise"] = alpha_volume_surprise(volume)
    alphas["alpha_price_volume_corr"] = alpha_price_volume_corr(close, volume)
    alphas["alpha_intraday_intensity"] = alpha_intraday_intensity(close, high, low, volume)
    alphas["alpha_volume_price_trend"] = alpha_volume_price_trend(close, volume)

    # Volatility category
    alphas["alpha_vol_of_vol"] = alpha_vol_of_vol(close)
    alphas["alpha_skewness"] = alpha_skewness(close)
    alphas["alpha_kurtosis"] = alpha_kurtosis(close)
    alphas["alpha_down_vol_ratio"] = alpha_down_vol_ratio(close)

    # Liquidity / microstructure
    alphas["alpha_amihud_illiquidity"] = alpha_amihud_illiquidity(close, volume)

    # Price pattern
    alphas["alpha_high_low_momentum"] = alpha_high_low_momentum(high, low)
    alphas["alpha_trend_strength"] = alpha_trend_strength(close)
    alphas["alpha_max_drawdown_20d"] = alpha_max_drawdown_20d(close)
    alphas["alpha_vwap_deviation"] = alpha_vwap_deviation(close, high, low, volume)

    # Clip extreme values to avoid outlier contamination
    for col in alphas.columns:
        q01 = alphas[col].quantile(0.01)
        q99 = alphas[col].quantile(0.99)
        if pd.notna(q01) and pd.notna(q99) and q01 < q99:
            alphas[col] = alphas[col].clip(q01, q99)

    return alphas


def alpha_feature_names() -> list[str]:
    """Return the list of alpha feature column names."""
    return [
        "alpha_mom_12_1",
        "alpha_short_reversal",
        "alpha_acceleration",
        "alpha_ret_consistency",
        "alpha_overnight_sentiment",
        "alpha_close_location",
        "alpha_volume_surprise",
        "alpha_price_volume_corr",
        "alpha_intraday_intensity",
        "alpha_volume_price_trend",
        "alpha_vol_of_vol",
        "alpha_skewness",
        "alpha_kurtosis",
        "alpha_down_vol_ratio",
        "alpha_amihud_illiquidity",
        "alpha_high_low_momentum",
        "alpha_trend_strength",
        "alpha_max_drawdown_20d",
        "alpha_vwap_deviation",
    ]
