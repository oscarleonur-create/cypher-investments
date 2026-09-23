"""Where movers, volatility and catalysts come from. All free, all fallible.

Discovery is market-wide on purpose. The user's best setup-A trades were in
names no list held beforehand (AMPG, CCXI, WOLF), so a fixed universe would
have missed exactly the trades that paid. Yahoo's screener is unofficial and
can change without notice; every function here returns empty rather than
raising, and the scan reports the failure instead of pretending the market
was quiet.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime

from advisor.scanner.models import CatalystItem, Mover

logger = logging.getLogger(__name__)

# NASDAQ (NMS/NGM/NCM), NYSE (NYQ) and NYSE American (ASE). OTC names are
# excluded: their quotes are thin and their filings often absent.
_EXCHANGES = ["NMS", "NGM", "NCM", "NYQ", "ASE"]
_PAGE = 250


def _num(raw) -> float | None:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _to_mover(q: dict) -> Mover | None:
    symbol = str(q.get("symbol") or "").strip().upper()
    if not symbol:
        return None
    return Mover(
        symbol=symbol,
        name=q.get("shortName") or q.get("longName"),
        price=_num(q.get("regularMarketPrice")),
        prev_close=_num(q.get("regularMarketPreviousClose")),
        open=_num(q.get("regularMarketOpen")),
        volume=_num(q.get("regularMarketVolume")),
        avg_volume=_num(q.get("averageDailyVolume3Month")),
        market_cap=_num(q.get("marketCap")),
        source="yahoo-screener",
    )


def screen_movers(min_move: float = 0.04, c_min_cap: float = 10e9, a_min_cap: float = 300e6):
    """Today's big movers, both directions. Returns (movers, error or None).

    The screener's filters are a coarse pre-cut to keep the page small; the
    detector re-checks everything, so a loose screen here is harmless.
    """
    try:
        import yfinance as yf
        from yfinance import EquityQuery as Q
    except Exception as exc:  # noqa: BLE001
        return [], f"yfinance unavailable: {exc}"

    pct = min_move * 100
    queries = {
        "gainers": Q(
            "and",
            [
                Q("gte", ["percentchange", pct]),
                Q("gte", ["intradaymarketcap", a_min_cap]),
                Q("is-in", ["exchange", *_EXCHANGES]),
            ],
        ),
        "losers": Q(
            "and",
            [
                Q("lte", ["percentchange", -pct]),
                Q("gte", ["intradaymarketcap", c_min_cap]),
                Q("is-in", ["exchange", *_EXCHANGES]),
            ],
        ),
    }
    movers: dict[str, Mover] = {}
    errors = []
    for label, query in queries.items():
        try:
            result = yf.screen(
                query, sortField="percentchange", sortAsc=label == "losers", size=_PAGE
            )
            quotes = result.get("quotes", []) if isinstance(result, dict) else []
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{label}: {exc}")
            continue
        for q in quotes:
            m = _to_mover(q)
            if m is not None:
                movers[m.symbol] = m
    return list(movers.values()), ("; ".join(errors) or None)


def daily_sigma(symbols: list[str], sessions: int = 60) -> dict[str, float]:
    """Std-dev of daily returns over the last ``sessions`` completed sessions.

    Today's partial bar is dropped, so the move being judged is never part of
    the volatility it is judged against. A symbol with fewer than 20 returns
    gets no estimate rather than a noisy one.
    """
    if not symbols:
        return {}
    try:
        import pandas as pd
        import yfinance as yf

        from advisor.daemon.market_calendar import now_et

        data = yf.download(symbols, period="6mo", interval="1d", progress=False, auto_adjust=False)[
            "Close"
        ]
        if isinstance(data, pd.Series):
            data = data.to_frame(symbols[0])
        today = pd.Timestamp(now_et().date())
        data = data[data.index < today]
    except Exception as exc:  # noqa: BLE001
        logger.warning("scanner: volatility download failed: %s", exc)
        return {}

    out = {}
    for sym in symbols:
        if sym not in data:
            continue
        returns = data[sym].dropna().pct_change().dropna().tail(sessions)
        if len(returns) >= 20:
            sd = float(returns.std())
            if math.isfinite(sd) and sd > 0:
                out[sym] = sd
    return out


def find_catalysts(
    symbol: str, name: str | None, since: datetime
) -> tuple[list[CatalystItem], bool]:
    """News and filings about ``symbol`` published at or after ``since``.

    Returns (items, checked). ``checked`` is False only when neither source
    could be asked — then an empty list means "unknown", not "no news".
    Keywords, never sentences: the repo measured 0.924 against 0.393.

    Known limit: both underlying clients swallow network errors and return
    empty, so an outage of both reads as "checked, nothing found". The scan
    result reports how many lookups came back empty so a run of zeros shows.
    """
    items: list[CatalystItem] = []
    asked = False

    try:
        from advisor.news.tavily import search_news

        query = " ".join(p for p in (name, symbol, "stock") if p)
        for it in search_news(symbol, query, company_name=name, days=3, max_results=8):
            if it.published_at >= since:
                items.append(
                    CatalystItem(
                        kind="news",
                        title=it.title,
                        published_at=it.published_at,
                        provider=it.provider,
                        url=it.url,
                    )
                )
        asked = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("scanner: news lookup failed for %s: %s", symbol, exc)

    try:
        from advisor.news.edgar import recent_filings

        for it in recent_filings(symbol, lookback_days=4):
            if it.published_at >= since:
                items.append(
                    CatalystItem(
                        kind=it.doc_type or "filing",
                        title=it.title,
                        published_at=it.published_at,
                        provider=it.provider,
                        url=it.url,
                    )
                )
        asked = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("scanner: filing lookup failed for %s: %s", symbol, exc)

    items.sort(key=lambda i: i.published_at, reverse=True)
    return items, asked


def intraday_bars(symbol: str, start, end):
    """5-minute regular-session bars in ET between two dates, or None.

    yfinance keeps 60 days of 5-minute history, which bounds how late an
    outcome can still be filled.
    """
    try:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="5m",
            prepost=False,
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("scanner: bars unavailable for %s: %s", symbol, exc)
        return None
    if df is None or df.empty:
        return None
    df.index = df.index.tz_convert("America/New_York")
    return df
