"""Data provider protocol for market data sources."""

from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd


class DataProvider(Protocol):
    """Protocol defining the interface for market data providers."""

    def get_stock_history(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data for a stock.

        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index should be DatetimeIndex.
        """
        ...

    def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch current options chain.

        Returns dict with keys 'calls' and 'puts', each a DataFrame.
        """
        ...

    def get_ticker_info(self, symbol: str) -> dict:
        """Fetch ticker metadata."""
        ...
