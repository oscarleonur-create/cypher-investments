"""Technical agent — checks momentum breakout signal via SignalScanner.

Runs the momentum_breakout strategy through the scanner for the buy/sell signal,
then fetches raw price data to compute the actual SMA-20 and volume ratio.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from advisor.confluence.models import TechnicalResult
from advisor.data.yahoo import YahooDataProvider
from advisor.engine.scanner import SignalScanner
from advisor.engine.signals import SignalAction

logger = logging.getLogger(__name__)


def _compute_indicators(symbol: str, sma_period: int = 20) -> dict:
    """Fetch recent price data and compute SMA + volume ratio.

    Returns dict with keys: price, sma_20, volume_ratio.
    """
    provider = YahooDataProvider(cache=None)
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=sma_period * 3)  # enough bars for SMA warmup

    try:
        df = provider.get_stock_history(symbol, start, end, interval="1d")
        if df is None or len(df) < sma_period:
            return {"price": 0.0, "sma_20": 0.0, "volume_ratio": 0.0}

        close = df["Close"]
        volume = df["Volume"]

        current_price = float(close.iloc[-1])
        sma_value = float(close.iloc[-sma_period:].mean())
        avg_volume = float(volume.iloc[-sma_period:].mean())
        current_volume = float(volume.iloc[-1])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0.0

        return {
            "price": current_price,
            "sma_20": round(sma_value, 2),
            "volume_ratio": round(volume_ratio, 2),
        }
    except Exception as e:
        logger.warning(f"Could not compute indicators for {symbol}: {e}")
        return {"price": 0.0, "sma_20": 0.0, "volume_ratio": 0.0}


def check_technical(symbol: str, strategy_name: str = "momentum_breakout") -> TechnicalResult:
    """Run a strategy through the signal scanner.

    Also computes the actual SMA-20 and volume ratio from raw price data
    so the result has real indicator values, not just the buy/sell signal.
    """
    scanner = SignalScanner()
    result = scanner.scan(symbol, strategy_names=[strategy_name])

    if not result.signals:
        return TechnicalResult(
            signal="NEUTRAL",
            price=0.0,
            sma_20=0.0,
            volume_ratio=0.0,
            is_bullish=False,
        )

    sig = result.signals[0]
    is_bullish = sig.action == SignalAction.BUY

    # Compute actual indicator values from raw data
    indicators = _compute_indicators(symbol)

    return TechnicalResult(
        signal=sig.action.value,
        price=sig.price,
        sma_20=indicators["sma_20"],
        volume_ratio=indicators["volume_ratio"],
        is_bullish=is_bullish,
    )
