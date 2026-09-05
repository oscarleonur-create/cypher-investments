"""Yahoo Finance data provider via yfinance."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from advisor.data.cache import DiskCache

logger = logging.getLogger(__name__)


class YahooDataProvider:
    """Market data provider using Yahoo Finance (yfinance)."""

    def __init__(self, cache: DiskCache | None = None):
        self.cache = cache or DiskCache()

    def get_stock_history(
        self,
        symbol: str,
        start: date,
        end: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data."""
        cache_key = ("stock", symbol, str(start), str(end), interval)

        if self.cache:
            cached = self.cache.get_dataframe(*cache_key)
            if cached is not None:
                return cached

        logger.info(f"Fetching {symbol} history: {start} to {end}")
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=str(start), end=str(end), interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} from {start} to {end}")

        # Standardize column names
        df.columns = [c.title() for c in df.columns]
        # Keep only OHLCV columns
        keep_cols = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in keep_cols if c in df.columns]
        df = df[available]

        if self.cache:
            self.cache.set_dataframe(df, *cache_key)

        return df

    def get_options_chain(
        self,
        symbol: str,
        expiration: date | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch current options chain from Yahoo Finance."""
        ticker = yf.Ticker(symbol)

        if expiration:
            chain = ticker.option_chain(str(expiration))
        else:
            expirations = ticker.options
            if not expirations:
                raise ValueError(f"No options available for {symbol}")
            chain = ticker.option_chain(expirations[0])

        return {
            "calls": chain.calls,
            "puts": chain.puts,
        }

    def get_ticker_info(self, symbol: str) -> dict:
        """Fetch ticker metadata."""
        ticker = yf.Ticker(symbol)
        return dict(ticker.info)

    def get_available_expirations(self, symbol: str) -> list[str]:
        """Get available option expiration dates."""
        ticker = yf.Ticker(symbol)
        return list(ticker.options)


def fetch_earnings_dates(symbol: str, ticker: yf.Ticker | None = None) -> list[date]:
    """Upcoming/recent earnings dates from yfinance's calendar, best-effort.

    Pass an already-constructed `ticker` to reuse it (avoids a second network
    round-trip when the caller already has one, e.g. alongside `.info`).
    Returns an empty list on any lookup failure — callers treat "no earnings
    data" as a normal, silent case rather than an error.
    """
    try:
        cal = (ticker or yf.Ticker(symbol.upper())).calendar
        raw = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
        if not isinstance(raw, list):
            raw = [raw]
        dates: list[date] = []
        for r in raw:
            try:
                dates.append(pd.Timestamp(r).date())
            except Exception:  # noqa: BLE001
                continue
        return dates
    except Exception as exc:  # noqa: BLE001
        logger.debug("fetch_earnings_dates failed for %s: %s", symbol, exc)
        return []
