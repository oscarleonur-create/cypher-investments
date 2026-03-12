"""Volume confirmation layer for dip-buying signals.

Checks three independent volume signals that validate a dip is worth buying:

1. Capitulation spike — high volume on a recent down day (panic exhaustion)
2. Volume dry-up — declining volume after the selloff (selling pressure fading)
3. OBV divergence — On-Balance Volume rising while price falls (accumulation)

Each signal contributes to a 0-100 composite score.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np

from advisor.confluence.models import VolumeConfirmationResult
from advisor.data.yahoo import YahooDataProvider

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
CAPITULATION_VOL_MULT = 2.0  # volume must be 2x avg on a down day
LOOKBACK_DAYS = 5  # look for capitulation in last N trading days
DRYUP_WINDOW = 3  # last N days must show declining volume
OBV_LOOKBACK = 10  # OBV divergence window (trading days)
HISTORY_BARS = 60  # fetch enough bars for 20d avg + OBV lookback


def check_volume_confirmation(symbol: str) -> VolumeConfirmationResult:
    """Run volume confirmation analysis for a dip candidate.

    Fetches ~60 days of daily OHLCV and computes capitulation, dry-up,
    and OBV divergence signals.
    """
    provider = YahooDataProvider(cache=None)
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(HISTORY_BARS * 1.6))  # calendar padding

    try:
        df = provider.get_stock_history(symbol, start, end, interval="1d")
    except Exception as e:
        logger.warning(f"Volume confirmation: no data for {symbol}: {e}")
        return VolumeConfirmationResult()

    if df is None or len(df) < 25:
        return VolumeConfirmationResult()

    close = df["Close"].values.astype(float)
    volume = df["Volume"].values.astype(float)

    avg_vol_20 = float(np.mean(volume[-20:]))
    current_vol = float(volume[-1])
    vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0

    # ── 1. Capitulation spike ─────────────────────────────────────────
    capitulation = False
    cap_ratio = 0.0
    if len(close) > LOOKBACK_DAYS and avg_vol_20 > 0:
        for i in range(-LOOKBACK_DAYS, 0):
            # Down day: close < previous close
            if close[i] < close[i - 1]:
                day_ratio = volume[i] / avg_vol_20
                if day_ratio > cap_ratio:
                    cap_ratio = day_ratio
                if day_ratio >= CAPITULATION_VOL_MULT:
                    capitulation = True

    # ── 2. Volume dry-up ──────────────────────────────────────────────
    dryup = False
    if len(volume) >= DRYUP_WINDOW + 1:
        recent_vols = volume[-DRYUP_WINDOW:]
        # Each day's volume is less than the prior day
        dryup = all(recent_vols[i] < recent_vols[i - 1] for i in range(1, DRYUP_WINDOW))

    # ── 3. OBV divergence ─────────────────────────────────────────────
    obv_div = False
    if len(close) > OBV_LOOKBACK:
        obv = _compute_obv(close, volume)
        # Price trend: compare last bar to bar N days ago
        price_down = close[-1] < close[-OBV_LOOKBACK]
        # OBV trend: simple linear slope over window
        obv_window = obv[-OBV_LOOKBACK:]
        obv_slope = np.polyfit(range(len(obv_window)), obv_window, 1)[0]
        obv_up = obv_slope > 0
        obv_div = price_down and obv_up

    # ── Composite score ───────────────────────────────────────────────
    score = 0.0
    if capitulation:
        # Scale by how extreme: 2x → 30, 3x+ → 40
        score += min(40.0, 20.0 + (cap_ratio - CAPITULATION_VOL_MULT) * 10)
    if dryup:
        score += 25.0
    if obv_div:
        score += 35.0
    score = min(100.0, score)

    return VolumeConfirmationResult(
        volume_ratio=round(vol_ratio, 2),
        capitulation_detected=capitulation,
        capitulation_ratio=round(cap_ratio, 2),
        volume_dryup=dryup,
        obv_divergence=obv_div,
        score=round(score, 1),
    )


def _compute_obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Compute On-Balance Volume from close and volume arrays."""
    obv = np.zeros_like(volume)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv
