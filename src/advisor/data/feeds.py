"""Backtrader data feed wrappers."""

from __future__ import annotations

from datetime import date

import backtrader as bt
import pandas as pd

from advisor.data.yahoo import YahooDataProvider


class PandasFeed(bt.feeds.PandasData):
    """Standard OHLCV data feed from a pandas DataFrame.

    Expects DataFrame with columns: Open, High, Low, Close, Volume
    and a DatetimeIndex.
    """

    params = (
        ("open", "Open"),
        ("high", "High"),
        ("low", "Low"),
        ("close", "Close"),
        ("volume", "Volume"),
        ("openinterest", None),
    )


def create_feed(
    symbol: str,
    start: date,
    end: date,
    provider: YahooDataProvider | None = None,
) -> PandasFeed:
    """Create a Backtrader data feed for a symbol.

    Fetches data via the provider and wraps it in a PandasData feed.
    """
    if provider is None:
        provider = YahooDataProvider()

    df = provider.get_stock_history(symbol, start, end)
    # Ensure the index is tz-naive for Backtrader compatibility
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return PandasFeed(dataname=df)
