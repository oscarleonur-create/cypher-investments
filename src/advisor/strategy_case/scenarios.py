"""Stage 1: Scenario detection — classify what setup is in play."""

from __future__ import annotations

import logging

import numpy as np
import yfinance as yf

from advisor.strategy_case.models import ScenarioResult, ScenarioType

logger = logging.getLogger(__name__)


def detect_scenario(symbol: str) -> ScenarioResult:
    """Detect the current market scenario for a symbol.

    Uses existing cheap screeners (yfinance only) to classify
    the setup: earnings dip, IV spike, pullback, range-bound, etc.
    """
    symbol = symbol.upper()
    ticker = yf.Ticker(symbol)

    # ── Gather price data ────────────────────────────────────────────────
    hist = ticker.history(period="3mo")
    if hist.empty or len(hist) < 20:
        return ScenarioResult(
            scenario_type=ScenarioType.RANGE_BOUND,
            confidence=0.3,
            summary=f"Insufficient price data for {symbol}",
        )

    close = hist["Close"]
    price = float(close.iloc[-1])
    sma_20 = float(close.rolling(20).mean().iloc[-1])
    sma_20_dist = (price - sma_20) / sma_20 * 100

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
    rsi = 100 - (100 / (1 + rs))

    # Recent price change (5-day)
    if len(close) >= 6:
        price_change_5d = (price - float(close.iloc[-6])) / float(close.iloc[-6]) * 100
    else:
        price_change_5d = 0.0

    # 20-day realized volatility for range detection
    daily_returns = close.pct_change().dropna()
    vol_20 = float(daily_returns.tail(20).std() * np.sqrt(252)) if len(daily_returns) >= 20 else 0.3

    # ── IV percentile (cheap estimate) ───────────────────────────────────
    iv_percentile = 50.0
    try:
        from advisor.market.iv_analysis import compute_iv_percentile

        iv_result = compute_iv_percentile(symbol, ticker=ticker)
        iv_percentile = iv_result.iv_percentile
    except Exception as e:
        logger.debug(f"IV percentile failed for {symbol}: {e}")

    # ── PEAD screener ────────────────────────────────────────────────────
    pead_score: str | None = None
    days_since_earnings: int | None = None
    earnings_date = None
    try:
        from advisor.confluence.pead_screener import check_pead_fundamental

        pead_result = check_pead_fundamental(symbol)
        if pead_result.pead_screener:
            pead_score = pead_result.pead_screener.overall_score
            earnings_date = pead_result.earnings_date
            if pead_result.pead_screener.earnings_surprise.days_since_report is not None:
                days_since_earnings = pead_result.pead_screener.earnings_surprise.days_since_report
    except Exception as e:
        logger.debug(f"PEAD check failed for {symbol}: {e}")

    # ── Dip screener ─────────────────────────────────────────────────────
    dip_score: str | None = None
    try:
        from advisor.confluence.dip_screener import check_dip_fundamental

        dip_result = check_dip_fundamental(symbol)
        if dip_result.dip_screener:
            dip_score = dip_result.dip_screener.overall_score
    except Exception as e:
        logger.debug(f"Dip check failed for {symbol}: {e}")

    # ── Classification logic ─────────────────────────────────────────────
    base = dict(
        price=round(price, 2),
        price_change_pct=round(price_change_5d, 2),
        iv_percentile=round(iv_percentile, 1),
        dip_score=dip_score,
        pead_score=pead_score,
        rsi=round(rsi, 1),
        sma_20_distance_pct=round(sma_20_dist, 2),
        days_since_earnings=days_since_earnings,
        earnings_date=earnings_date,
    )

    # Priority 1: Earnings dip — recent earnings + price drop + PEAD signal
    if (
        pead_score is not None
        and pead_score not in ("FAIL",)
        and days_since_earnings is not None
        and days_since_earnings <= 14
        and price_change_5d < -3
    ):
        confidence = 0.8
        if pead_score in ("BUY", "STRONG_BUY"):
            confidence = 0.9
        return ScenarioResult(
            scenario_type=ScenarioType.EARNINGS_DIP,
            confidence=confidence,
            summary=(
                f"{symbol} dropped {price_change_5d:.1f}% post-earnings "
                f"({days_since_earnings}d ago). PEAD: {pead_score}."
            ),
            **base,
        )

    # Priority 2: IV spike — high IV percentile + elevated vol
    if iv_percentile >= 70:
        confidence = min(0.9, 0.5 + (iv_percentile - 70) / 60)
        return ScenarioResult(
            scenario_type=ScenarioType.IV_SPIKE,
            confidence=round(confidence, 2),
            summary=(
                f"{symbol} IV at {iv_percentile:.0f}th percentile. " f"Premium selling environment."
            ),
            **base,
        )

    # Priority 3: Breakout pullback — was above SMA, now pulling back near it
    if -5 < sma_20_dist < 2 and price_change_5d < -2 and rsi < 45:
        confidence = 0.6
        if dip_score in ("LEAN_BUY", "BUY", "STRONG_BUY"):
            confidence = 0.75
        return ScenarioResult(
            scenario_type=ScenarioType.BREAKOUT_PULLBACK,
            confidence=confidence,
            summary=(
                f"{symbol} pulling back to SMA-20 ({sma_20_dist:+.1f}% away). "
                f"RSI: {rsi:.0f}. Dip: {dip_score}."
            ),
            **base,
        )

    # Priority 4: Mean reversion — oversold + dip looks buyable
    if rsi < 35 and price_change_5d < -5:
        confidence = 0.6
        if dip_score in ("LEAN_BUY", "BUY", "STRONG_BUY"):
            confidence = 0.8
        return ScenarioResult(
            scenario_type=ScenarioType.MEAN_REVERSION,
            confidence=confidence,
            summary=(
                f"{symbol} oversold: RSI {rsi:.0f}, down {price_change_5d:.1f}% in 5d. "
                f"Dip: {dip_score}."
            ),
            **base,
        )

    # Priority 5: Momentum — above SMA, strong RSI, trending up
    if sma_20_dist > 3 and rsi > 55 and price_change_5d > 0:
        confidence = 0.6
        if rsi > 65:
            confidence = 0.7
        return ScenarioResult(
            scenario_type=ScenarioType.MOMENTUM,
            confidence=confidence,
            summary=(f"{symbol} in uptrend: {sma_20_dist:+.1f}% above SMA-20, " f"RSI {rsi:.0f}."),
            **base,
        )

    # Default: Range-bound
    confidence = 0.5
    if vol_20 < 0.20:
        confidence = 0.65
    return ScenarioResult(
        scenario_type=ScenarioType.RANGE_BOUND,
        confidence=confidence,
        summary=(
            f"{symbol} trading in range near SMA-20 ({sma_20_dist:+.1f}%). "
            f"RSI: {rsi:.0f}, 20d vol: {vol_20:.0%}."
        ),
        **base,
    )
