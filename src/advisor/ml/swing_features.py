"""Swing trade pattern features for the ML pipeline.

15 features in 5 groups that encode the setups swing traders look for:
pullbacks to support in uptrends, consolidation/squeeze before breakouts,
multi-timeframe trend alignment, and volume behavior at key price events.

Each feature is a pure function: OHLCV DataFrame -> Series (one value per bar).

Usage::

    from advisor.ml.swing_features import compute_all_swing_features

    swing_df = compute_all_swing_features(df)  # df has OHLCV columns
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────


def _wilder_smooth(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (equivalent to EMA with alpha=1/period)."""
    return series.ewm(alpha=1.0 / period, min_periods=period).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span).mean()


# ── Group A: Trend Structure (3) ────────────────────────────────────


def swing_adx_14(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Proper Wilder-smoothed ADX (0-1 scaled).

    Uses +DM/-DM decomposition. >0.25 = trending, <0.20 = range-bound.
    """
    period = 14
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    # Wilder smoothing
    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, period) / atr.replace(0, np.nan)

    # DX and ADX
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = _wilder_smooth(dx, period)

    return adx / 100.0  # Scale 0-1


def swing_di_spread(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """(+DI - -DI) / 100. Positive = bullish trend, negative = bearish."""
    period = 14
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index
    )

    atr = _wilder_smooth(tr, period)
    plus_di = 100 * _wilder_smooth(plus_dm, period) / atr.replace(0, np.nan)
    minus_di = 100 * _wilder_smooth(minus_dm, period) / atr.replace(0, np.nan)

    return (plus_di - minus_di) / 100.0


def swing_higher_highs_ratio(high: pd.Series) -> pd.Series:
    """Fraction of 5-bar rolling highs that exceed the previous 5-bar high (last 20 bars).

    Near 1.0 = orderly uptrend structure, near 0.0 = deteriorating.
    """
    rolling_high = high.rolling(5).max()
    exceeds = (rolling_high > rolling_high.shift(5)).astype(float)
    return exceeds.rolling(20).mean()


# ── Group B: Consolidation & Squeeze (3) ────────────────────────────


def swing_range_contraction(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """ATR(5) / ATR(20). Low values = consolidation / base forming.

    Differs from existing vol_ratio which uses return vol, not range.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_5 = tr.rolling(5).mean()
    atr_20 = tr.rolling(20).mean()
    return atr_5 / atr_20.replace(0, np.nan)


def swing_bb_squeeze(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.Series:
    """0/1: Bollinger Bands contained within Keltner Channels (TTM Squeeze).

    BB: 20-SMA +/- 2*std. KC: 20-EMA +/- 2*ATR.
    """
    period = 20

    # Bollinger Bands
    bb_mid = close.rolling(period).mean()
    bb_std = close.rolling(period).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std

    # Keltner Channels (using ATR)
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_20 = tr.rolling(period).mean()
    kc_mid = _ema(close, period)
    kc_upper = kc_mid + 2 * atr_20
    kc_lower = kc_mid - 2 * atr_20

    # Squeeze: BB inside KC
    squeeze = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)
    return squeeze


def swing_consolidation_days(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Consecutive bars where H-L range stays within 2*ATR envelope, divided by 20.

    Capped at 1.0. Longer consolidation = higher breakout probability.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr_20 = tr.rolling(20).mean()

    hl_range = high - low
    within_envelope = (hl_range < 2 * atr_20).astype(float)

    # Count consecutive bars within envelope
    consec = within_envelope.copy()
    for i in range(1, len(consec)):
        if within_envelope.iloc[i] == 1.0:
            consec.iloc[i] = consec.iloc[i - 1] + 1.0
        else:
            consec.iloc[i] = 0.0

    return (consec / 20.0).clip(upper=1.0)


# ── Group C: Pullback & Support (3) ────────────────────────────────


def swing_pullback_to_sma20(close: pd.Series) -> pd.Series:
    """Proximity to SMA(20) when in uptrend (price > SMA-50, SMA-20 rising).

    Returns 0-1 measure: 1 = touching SMA-20 from above in uptrend, 0 = no setup.
    """
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    # SMA-20 slope (positive = rising)
    sma_20_slope = sma_20 - sma_20.shift(5)

    # Conditions: price > SMA-50, SMA-20 rising
    uptrend = (close > sma_50) & (sma_20_slope > 0)

    # Distance from SMA-20 (as fraction of price)
    dist = (close - sma_20) / close.replace(0, np.nan)

    # Score: closer to SMA-20 from above = higher score
    # 0 distance = score 1.0, >5% above = score 0
    score = (1.0 - dist.clip(lower=0) / 0.05).clip(lower=0)

    # Only score when in uptrend
    return (score * uptrend.astype(float)).fillna(0.0)


def swing_pullback_to_sma50(close: pd.Series) -> pd.Series:
    """Proximity to SMA(50) for institutional pullback level.

    Same concept as SMA-20 pullback but for deeper institutional pullbacks.
    """
    sma_50 = close.rolling(50).mean()
    sma_100 = close.rolling(100).mean()

    # SMA-50 slope
    sma_50_slope = sma_50 - sma_50.shift(5)

    # Conditions: price > SMA-100, SMA-50 rising
    uptrend = (close > sma_100) & (sma_50_slope > 0)

    # Distance from SMA-50
    dist = (close - sma_50) / close.replace(0, np.nan)

    score = (1.0 - dist.clip(lower=0) / 0.05).clip(lower=0)

    return (score * uptrend.astype(float)).fillna(0.0)


def swing_support_proximity(close: pd.Series) -> pd.Series:
    """Distance from 20-bar rolling low (support), as percentage, clipped [-5%, 5%]."""
    rolling_low = close.rolling(20).min()
    dist_pct = (close - rolling_low) / rolling_low.replace(0, np.nan)
    return dist_pct.clip(-0.05, 0.05)


# ── Group D: Multi-Timeframe (3) ───────────────────────────────────


def swing_weekly_trend(close: pd.Series) -> pd.Series:
    """Weekly close resampled, 5-week EMA slope sign (+1/-1/0), forward-filled to daily."""
    # Resample to weekly
    weekly = close.resample("W").last().dropna()
    weekly_ema = _ema(weekly, 5)
    weekly_slope = weekly_ema - weekly_ema.shift(1)

    # Sign: +1 (up), -1 (down), 0 (flat)
    trend = np.sign(weekly_slope)

    # Forward-fill back to daily
    return trend.reindex(close.index, method="ffill").fillna(0.0)


def swing_daily_weekly_align(close: pd.Series) -> pd.Series:
    """1.0 when daily and weekly trends agree, -1.0 when opposed, 0.0 when flat."""
    # Daily trend: SMA-20 slope sign
    sma_20 = close.rolling(20).mean()
    daily_trend = np.sign(sma_20 - sma_20.shift(5))

    # Weekly trend
    weekly = swing_weekly_trend(close)

    # Alignment
    alignment = daily_trend * weekly
    return alignment.fillna(0.0)


def swing_trend_maturity(close: pd.Series) -> pd.Series:
    """Bars since last SMA(20)/SMA(50) crossover, divided by 60 (capped at 1.0).

    Early-stage trends have more follow-through.
    """
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    # Cross signal: SMA-20 above SMA-50
    above = (sma_20 > sma_50).astype(float)
    cross = above.diff().abs()  # 1 on crossover bars

    # Count bars since last crossover
    bars_since = pd.Series(0.0, index=close.index)
    count = 0.0
    for i in range(len(bars_since)):
        if pd.notna(cross.iloc[i]) and cross.iloc[i] > 0:
            count = 0.0
        else:
            count += 1.0
        bars_since.iloc[i] = count

    return (bars_since / 60.0).clip(upper=1.0)


# ── Group E: Volume at Key Events (3) ──────────────────────────────


def swing_volume_at_breakout(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume z-score on days when price exceeds 20-bar high; 0 otherwise.

    Breakouts on high volume are more reliable.
    """
    rolling_high = close.rolling(20).max()
    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std()
    vol_zscore = (volume - vol_mean) / vol_std.replace(0, np.nan)

    # Breakout: price exceeds previous 20-bar high
    # Use shift(1) so we compare to the high before today
    breakout = close > rolling_high.shift(1)

    return (vol_zscore * breakout.astype(float)).fillna(0.0)


def swing_volume_dryup(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume 5-bar avg / 20-bar avg during consolidation (range_contraction < 0.8).

    Low = supply exhaustion before breakout.
    """
    rc = swing_range_contraction(high, low, close)
    in_consolidation = (rc < 0.8).astype(float)

    vol_5 = volume.rolling(5).mean()
    vol_20 = volume.rolling(20).mean()
    vol_ratio = vol_5 / vol_20.replace(0, np.nan)

    return (vol_ratio * in_consolidation).fillna(0.0)


def swing_accumulation_score(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """10-day fraction of bars where close is in upper half of H-L range AND volume > 20d avg.

    Detects institutional accumulation.
    """
    hl_range = (high - low).replace(0, np.nan)
    close_in_upper = ((close - low) / hl_range > 0.5).astype(float)
    vol_above_avg = (volume > volume.rolling(20).mean()).astype(float)

    accumulation_bar = close_in_upper * vol_above_avg
    return accumulation_bar.rolling(10).mean()


# ── Main API ────────────────────────────────────────────────────────


def compute_all_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 15 swing pattern features from an OHLCV DataFrame.

    Args:
        df: DataFrame with columns: Close, High, Low, Open, Volume.

    Returns:
        DataFrame with one column per swing feature, same index as input.
    """
    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"].astype(float)

    features = pd.DataFrame(index=df.index)

    # Group A: Trend Structure
    features["swing_adx_14"] = swing_adx_14(high, low, close)
    features["swing_di_spread"] = swing_di_spread(high, low, close)
    features["swing_higher_highs_ratio"] = swing_higher_highs_ratio(high)

    # Group B: Consolidation & Squeeze
    features["swing_range_contraction"] = swing_range_contraction(high, low, close)
    features["swing_bb_squeeze"] = swing_bb_squeeze(close, high, low)
    features["swing_consolidation_days"] = swing_consolidation_days(high, low, close)

    # Group C: Pullback & Support
    features["swing_pullback_to_sma20"] = swing_pullback_to_sma20(close)
    features["swing_pullback_to_sma50"] = swing_pullback_to_sma50(close)
    features["swing_support_proximity"] = swing_support_proximity(close)

    # Group D: Multi-Timeframe
    features["swing_weekly_trend"] = swing_weekly_trend(close)
    features["swing_daily_weekly_align"] = swing_daily_weekly_align(close)
    features["swing_trend_maturity"] = swing_trend_maturity(close)

    # Group E: Volume at Key Events
    features["swing_volume_at_breakout"] = swing_volume_at_breakout(close, volume)
    features["swing_volume_dryup"] = swing_volume_dryup(high, low, close, volume)
    features["swing_accumulation_score"] = swing_accumulation_score(close, high, low, volume)

    # Clip extreme values (1st/99th percentile)
    for col in features.columns:
        q01 = features[col].quantile(0.01)
        q99 = features[col].quantile(0.99)
        if pd.notna(q01) and pd.notna(q99) and q01 < q99:
            features[col] = features[col].clip(q01, q99)

    return features


def swing_feature_names() -> list[str]:
    """Return the list of swing feature column names."""
    return [
        # Group A: Trend Structure
        "swing_adx_14",
        "swing_di_spread",
        "swing_higher_highs_ratio",
        # Group B: Consolidation & Squeeze
        "swing_range_contraction",
        "swing_bb_squeeze",
        "swing_consolidation_days",
        # Group C: Pullback & Support
        "swing_pullback_to_sma20",
        "swing_pullback_to_sma50",
        "swing_support_proximity",
        # Group D: Multi-Timeframe
        "swing_weekly_trend",
        "swing_daily_weekly_align",
        "swing_trend_maturity",
        # Group E: Volume at Key Events
        "swing_volume_at_breakout",
        "swing_volume_dryup",
        "swing_accumulation_score",
    ]
