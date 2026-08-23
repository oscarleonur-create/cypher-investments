"""Intraday candle fetching for the scalping scanner.

Primary feed is TastyTrade's DXLink candle stream (`subscribe_candle`), which
returns OHLCV **and** a per-candle VWAP. When no TastyTrade session is available
(or it yields nothing) we fall back to yfinance via `YahooDataProvider`.

The DataFrames returned here are normalized to lowercase columns
``open, high, low, close, volume, vwap`` indexed by a tz-naive ``DatetimeIndex``
sorted ascending.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from datetime import date, datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Calendar days of history to request from yfinance per interval (its 1m feed is
# limited to ~7 days; intraday intervals only go back ~60 days).
_YF_DAYS_BY_INTERVAL = {"1m": 5, "2m": 5, "5m": 30, "15m": 55, "30m": 55, "60m": 55, "1h": 55}

_COLS = ["open", "high", "low", "close", "volume", "vwap"]


# ── Public API ──────────────────────────────────────────────────────────────


def fetch_intraday_candles(
    symbols: list[str],
    interval: str = "5m",
    lookback_minutes: int = 1440,
    session=None,
    timeout: float = 8.0,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Fetch intraday candles for ``symbols``.

    Returns ``(candles_by_symbol, source)`` where ``source`` is ``"tastytrade"``
    or ``"yfinance"``. Symbols with no data are simply absent from the dict.
    """
    if session is not None:
        try:
            start = datetime.now() - timedelta(minutes=lookback_minutes)
            tt = _run_async(_fetch_tt_candles(session, symbols, interval, start, timeout))
            tt = {s: df for s, df in tt.items() if df is not None and not df.empty}
            if tt:
                # Backfill any symbols TastyTrade missed from yfinance.
                missing = [s for s in symbols if s not in tt]
                if missing:
                    tt.update(_fetch_yf_candles(missing, interval, lookback_minutes))
                return tt, "tastytrade"
        except Exception as exc:  # noqa: BLE001 — degrade gracefully to yfinance
            logger.warning("TastyTrade candle fetch failed, falling back to yfinance: %s", exc)

    return _fetch_yf_candles(symbols, interval, lookback_minutes), "yfinance"


def candles_to_records(df: pd.DataFrame) -> list[dict]:
    """Serialize a candle DataFrame to JSON-friendly records for the frontend."""
    out: list[dict] = []
    vwap_series = df["vwap"] if "vwap" in df.columns else None
    for ts, row in df.iterrows():
        out.append(
            {
                "t": int(pd.Timestamp(ts).timestamp() * 1000),
                "open": _f(row.get("open")),
                "high": _f(row.get("high")),
                "low": _f(row.get("low")),
                "close": _f(row.get("close")),
                "volume": _f(row.get("volume")),
                "vwap": _f(vwap_series.loc[ts]) if vwap_series is not None else None,
            }
        )
    return out


# ── TastyTrade candles ──────────────────────────────────────────────────────


async def _fetch_tt_candles(
    session, symbols: list[str], interval: str, start_time: datetime, timeout: float
) -> dict[str, pd.DataFrame]:
    """Subscribe to Candle events and assemble per-symbol OHLCV+VWAP frames."""
    from tastytrade.dxfeed import Candle
    from tastytrade.streamer import DXLinkStreamer

    rows: dict[str, dict[int, dict]] = {s: {} for s in symbols}

    async with DXLinkStreamer(session) as streamer:
        await streamer.subscribe_candle(symbols, interval, start_time)

        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                candle: Candle = await asyncio.wait_for(
                    streamer.get_event(Candle),
                    timeout=max(0.1, deadline - asyncio.get_event_loop().time()),
                )
            except asyncio.TimeoutError:
                break

            base = (candle.event_symbol or "").split("{")[0]
            if base not in rows or candle.time is None or candle.close is None:
                continue
            rows[base][int(candle.time)] = {
                "open": _f(candle.open),
                "high": _f(candle.high),
                "low": _f(candle.low),
                "close": _f(candle.close),
                "volume": _f(candle.volume),
                "vwap": _f(candle.vwap),
            }

    return {s: _frame_from_rows(r) for s, r in rows.items() if r}


def _frame_from_rows(rows: dict[int, dict]) -> pd.DataFrame:
    """Build a sorted, time-indexed frame from {epoch_ms: ohlcv} records."""
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index, unit="ms")
    df = df.sort_index()
    return df.reindex(columns=_COLS)


# ── yfinance fallback ───────────────────────────────────────────────────────


def _fetch_yf_candles(
    symbols: list[str], interval: str, lookback_minutes: int
) -> dict[str, pd.DataFrame]:
    from advisor.data.yahoo import YahooDataProvider

    provider = YahooDataProvider()
    days = max(1, min(_YF_DAYS_BY_INTERVAL.get(interval, 30), (lookback_minutes // 1440) + 1 or 1))
    start = date.today() - timedelta(days=_YF_DAYS_BY_INTERVAL.get(interval, days))
    end = date.today() + timedelta(days=1)

    out: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            raw = provider.get_stock_history(sym, start, end, interval=interval)
        except Exception as exc:  # noqa: BLE001 — skip illiquid/bad symbols
            logger.debug("yfinance candle fetch failed for %s: %s", sym, exc)
            continue
        if raw.empty:
            continue
        df = raw.rename(columns={c: c.lower() for c in raw.columns})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        from advisor.scalping.indicators import vwap as _vwap

        df["vwap"] = _vwap(df)
        out[sym] = df.reindex(columns=_COLS)
    return out


# ── Helpers ─────────────────────────────────────────────────────────────────


def _run_async(coro):
    """Run a coroutine whether or not an event loop is already running."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _f(v) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if pd.notna(f) else None
