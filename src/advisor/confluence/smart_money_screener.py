"""Smart Money screener — detects institutional/insider/political buying signals.

Four signal sources scored -100 to +100:
  1. Insider activity (SEC Form 4 via OpenInsider) — -35 to +35 pts
  2. Congressional trading (Finnhub) — 0 to +20 pts
  3. Technical breakout proximity (yfinance) — 0 to +25 pts
  4. Unusual options activity (yfinance) — 0 to +20 pts
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from datetime import datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import yfinance as yf
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("/tmp/advisor_smart_money_cache")
CACHE_TTL = 4 * 3600  # 4 hours
INSIDER_LOOKBACK_DAYS = 180


# ── Models ───────────────────────────────────────────────────────────────────


class SmartMoneySignal(StrEnum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WATCH = "WATCH"
    HOLD = "HOLD"
    SELL = "SELL"


class InsiderTrade(BaseModel):
    filing_date: str = ""
    ticker: str = ""
    insider_name: str = ""
    title: str = ""
    trade_type: str = ""  # "Purchase" or "Sale"
    price: float = 0.0
    qty: int = 0
    value: float = 0.0


class InsiderScore(BaseModel):
    score: float = 0.0
    buy_trades: list[InsiderTrade] = Field(default_factory=list)
    sell_trades: list[InsiderTrade] = Field(default_factory=list)
    cluster_buys: int = 0
    large_buys: int = 0
    csuite_buys: int = 0
    cluster_sells: int = 0
    large_sells: int = 0
    csuite_sells: int = 0


class CongressScore(BaseModel):
    score: float = 0.0
    trades_found: int = 0
    recent_buys: int = 0
    politicians: list[str] = Field(default_factory=list)


class TechnicalBreakoutScore(BaseModel):
    score: float = 0.0
    price: float = 0.0
    high_20d: float = 0.0
    sma_50: float = 0.0
    pct_from_high: float = 0.0
    above_sma50: bool = False
    volume_trending_up: bool = False


class OptionsActivityScore(BaseModel):
    score: float = 0.0
    total_volume: int = 0
    avg_volume: float = 0.0
    volume_ratio: float = 0.0
    put_call_ratio: float = 0.0
    notable_strikes: list[str] = Field(default_factory=list)


class SmartMoneyResult(BaseModel):
    symbol: str
    total_score: float = 0.0
    signal: SmartMoneySignal = SmartMoneySignal.HOLD
    insider: InsiderScore = Field(default_factory=InsiderScore)
    congress: CongressScore = Field(default_factory=CongressScore)
    technical: TechnicalBreakoutScore = Field(default_factory=TechnicalBreakoutScore)
    options_activity: OptionsActivityScore = Field(default_factory=OptionsActivityScore)
    scanned_at: datetime = Field(default_factory=datetime.now)


# ── Cache helpers ────────────────────────────────────────────────────────────


def _cache_key(prefix: str, key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:24]
    return CACHE_DIR / f"{prefix}_{h}.json"


def _cache_get(prefix: str, key: str) -> dict | None:
    p = _cache_key(prefix, key)
    if p.exists() and (time.time() - p.stat().st_mtime) < CACHE_TTL:
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _cache_set(prefix: str, key: str, data: dict) -> None:
    """Atomic cache write — write to temp file then rename."""
    try:
        p = _cache_key(prefix, key)
        fd, tmp = tempfile.mkstemp(dir=CACHE_DIR, suffix=".tmp")
        closed = False
        try:
            os.write(fd, json.dumps(data).encode())
            os.close(fd)
            closed = True
            os.replace(tmp, p)
        except Exception:
            if not closed:
                os.close(fd)
            Path(tmp).unlink(missing_ok=True)
            raise
    except Exception as e:
        logger.debug(f"Cache write failed: {e}")


# ── Shared yfinance helper ──────────────────────────────────────────────────

_yf_lock = threading.Lock()
_yf_last_call = 0.0
_YF_MIN_INTERVAL = 0.35  # seconds between yfinance calls across all threads


def _yf_throttle() -> None:
    """Rate-limit yfinance calls to avoid 429s."""
    global _yf_last_call
    with _yf_lock:
        now = time.time()
        wait = _YF_MIN_INTERVAL - (now - _yf_last_call)
        if wait > 0:
            time.sleep(wait)
        _yf_last_call = time.time()


def _fetch_ticker_data(symbol: str) -> tuple[yf.Ticker, "pd.DataFrame"]:
    """Return a shared Ticker instance and 3-month history."""

    _yf_throttle()
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="3mo")
    return ticker, hist


# ── Column header mapping for OpenInsider ────────────────────────────────────

# Expected headers (lowercased) → our field names
_OI_HEADER_MAP = {
    "filing date": "filing_date",
    "trade date": "trade_date",
    "ticker": "ticker",
    "insider name": "insider_name",
    "title": "title",
    "trade type": "trade_type",
    "price": "price",
    "qty": "qty",
    "value": "value",
}


# ── 1. Insider activity (OpenInsider) ────────────────────────────────────────


def _parse_insider_table(html: str, symbol: str) -> list[dict]:
    """Parse OpenInsider HTML into trade dicts using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="tinytable")
    if not tables:
        return []

    table = tables[-1]  # data table is typically the last tinytable
    headers_row = table.find("thead")
    if not headers_row:
        return []

    # Build column index from actual headers (normalize \xa0 → regular space)
    ths = headers_row.find_all("th")
    col_map: dict[str, int] = {}
    for i, th in enumerate(ths):
        text = th.get_text(strip=True).replace("\xa0", " ").lower()
        for expected, field in _OI_HEADER_MAP.items():
            if expected in text:
                col_map[field] = i
                break

    tbody = table.find("tbody")
    if not tbody:
        return []

    trades = []
    for row in tbody.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) < max(col_map.values(), default=0) + 1:
            continue

        def cell_text(field: str) -> str:
            idx = col_map.get(field)
            if idx is None or idx >= len(cells):
                return ""
            return cells[idx].get_text(strip=True)

        trades.append(
            {
                "filing_date": cell_text("filing_date"),
                "ticker": symbol.upper(),
                "insider_name": cell_text("insider_name"),
                "title": cell_text("title"),
                "trade_type": cell_text("trade_type"),
                "price": cell_text("price"),
                "qty": cell_text("qty"),
                "value": cell_text("value"),
            }
        )

    return trades


def _fetch_insider_activity(symbol: str) -> InsiderScore:
    cached = _cache_get("insider", symbol)
    if cached:
        return InsiderScore(**cached)

    url = f"http://openinsider.com/search?q={symbol.lower()}"
    buy_trades: list[InsiderTrade] = []
    sell_trades: list[InsiderTrade] = []

    try:
        resp = httpx.get(
            url,
            timeout=15.0,
            headers={"User-Agent": "Mozilla/5.0"},
            follow_redirects=True,
        )
        resp.raise_for_status()
        raw_trades = _parse_insider_table(resp.text, symbol)

        cutoff = datetime.now() - timedelta(days=INSIDER_LOOKBACK_DAYS)

        for raw in raw_trades:
            # Parse and filter by date
            filing_str = raw["filing_date"]
            try:
                filing_date = datetime.strptime(filing_str[:10], "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            if filing_date < cutoff:
                continue

            # Parse numeric fields
            def clean_num(s: str) -> str:
                return s.replace(",", "").replace("$", "").replace("+", "").strip()

            try:
                price_str = clean_num(raw["price"])
                qty_str = clean_num(raw["qty"])
                value_str = clean_num(raw["value"])

                trade = InsiderTrade(
                    filing_date=filing_str,
                    ticker=symbol.upper(),
                    insider_name=raw["insider_name"],
                    title=raw["title"],
                    trade_type="Purchase" if "P - Purchase" in raw["trade_type"] else "Sale",
                    price=float(price_str) if price_str else 0.0,
                    qty=abs(int(qty_str)) if qty_str else 0,
                    value=abs(float(value_str)) if value_str else 0.0,
                )
            except (ValueError, IndexError):
                continue

            if "P - Purchase" in raw["trade_type"]:
                buy_trades.append(trade)
            elif "S - Sale" in raw["trade_type"]:
                sell_trades.append(trade)

    except Exception as e:
        logger.warning(f"OpenInsider fetch failed for {symbol}: {e}")
        result = InsiderScore()
        _cache_set("insider", symbol, result.model_dump())
        return result

    # Score buys (max +35)
    csuite_buys = csuite_sells = large_buys = large_sells = 0
    csuite_titles = ("CEO", "CFO", "COO", "PRES", "DIR")

    for t in buy_trades:
        title_upper = t.title.upper()
        if any(r in title_upper for r in csuite_titles):
            csuite_buys += 1
        if t.value >= 100_000:
            large_buys += 1

    buy_score = 0.0
    cluster_buys = len(buy_trades)
    if cluster_buys >= 3:
        buy_score += 13
    elif cluster_buys >= 2:
        buy_score += 7
    buy_score += min(large_buys * 4, 9)
    buy_score += min(csuite_buys * 4, 9)
    if buy_trades:
        buy_score += 4
    buy_score = min(buy_score, 35.0)

    # Score sells (max -35)
    for t in sell_trades:
        title_upper = t.title.upper()
        if any(r in title_upper for r in csuite_titles):
            csuite_sells += 1
        if t.value >= 100_000:
            large_sells += 1

    sell_score = 0.0
    cluster_sells = len(sell_trades)
    if cluster_sells >= 5:
        sell_score -= 15
    elif cluster_sells >= 3:
        sell_score -= 10
    sell_score -= min(large_sells * 3, 9)
    sell_score -= min(csuite_sells * 4, 9)
    sell_score = max(sell_score, -35.0)

    score = buy_score + sell_score

    result = InsiderScore(
        score=score,
        buy_trades=buy_trades[:10],
        sell_trades=sell_trades[:10],
        cluster_buys=cluster_buys,
        large_buys=large_buys,
        csuite_buys=csuite_buys,
        cluster_sells=cluster_sells,
        large_sells=large_sells,
        csuite_sells=csuite_sells,
    )
    _cache_set("insider", symbol, result.model_dump())
    return result


# ── 2. Congressional trading (Finnhub) ──────────────────────────────────────


def _fetch_congress_trades(symbol: str) -> CongressScore:
    cached = _cache_get("congress", symbol)
    if cached:
        return CongressScore(**cached)

    api_key = os.environ.get("FINNHUB_API_KEY", "")
    if not api_key:
        logger.debug("No FINNHUB_API_KEY set, skipping congress data")
        result = CongressScore()
        _cache_set("congress", symbol, result.model_dump())
        return result

    recent_buys = 0
    politicians: list[str] = []

    try:
        url = f"https://finnhub.io/api/v1/stock/congressional-trading?symbol={symbol.upper()}&token={api_key}"
        resp = httpx.get(url, timeout=15.0, follow_redirects=True)
        resp.raise_for_status()
        data = resp.json()

        cutoff = datetime.now() - timedelta(days=90)
        trades = data if isinstance(data, list) else data.get("data", [])

        for trade in trades:
            tx_date_str = trade.get("transactionDate", "")
            tx_type = trade.get("transactionType", "")
            try:
                tx_date = datetime.strptime(tx_date_str, "%Y-%m-%d")
            except (ValueError, TypeError):
                continue

            if tx_date >= cutoff and "purchase" in str(tx_type).lower():
                pol = trade.get("representative", "Unknown")
                if pol not in politicians:
                    politicians.append(pol)
                recent_buys += 1

    except Exception as e:
        logger.debug(f"Finnhub congress fetch failed for {symbol}: {e}")

    # Score (max 20)
    score = 0.0
    if recent_buys >= 3:
        score = 16.0
    elif recent_buys >= 2:
        score = 12.0
    elif recent_buys >= 1:
        score = 7.0

    if len(politicians) >= 2:
        score = min(score + 4, 20.0)

    result = CongressScore(
        score=score,
        trades_found=recent_buys,
        recent_buys=recent_buys,
        politicians=politicians[:5],
    )
    _cache_set("congress", symbol, result.model_dump())
    return result


# ── 3. Technical breakout proximity ─────────────────────────────────────────


def _check_technical_breakout(
    symbol: str,
    ticker: yf.Ticker | None = None,
    hist: "pd.DataFrame | None" = None,
) -> TechnicalBreakoutScore:
    cached = _cache_get("technical", symbol)
    if cached:
        return TechnicalBreakoutScore(**cached)

    try:
        if hist is None or hist.empty:
            _yf_throttle()
            if ticker is None:
                ticker = yf.Ticker(symbol)
            hist = ticker.history(period="3mo")

        if hist.empty or len(hist) < 50:
            return TechnicalBreakoutScore()

        close = hist["Close"]
        volume = hist["Volume"]
        current_price = float(close.iloc[-1])
        high_20d = float(close.tail(20).max())
        sma_50 = float(close.tail(50).mean())
        pct_from_high = ((high_20d - current_price) / high_20d) * 100

        vol_5 = float(volume.tail(5).mean())
        vol_20 = float(volume.tail(20).mean())
        volume_trending_up = vol_5 > vol_20

        above_sma50 = current_price > sma_50

        # Score (max 25)
        score = 0.0
        if pct_from_high <= 3.0:
            score += 10
        elif pct_from_high <= 5.0:
            score += 5

        if above_sma50:
            score += 9

        if volume_trending_up:
            score += 6

        result = TechnicalBreakoutScore(
            score=min(score, 25.0),
            price=current_price,
            high_20d=high_20d,
            sma_50=round(sma_50, 2),
            pct_from_high=round(pct_from_high, 2),
            above_sma50=above_sma50,
            volume_trending_up=volume_trending_up,
        )
        _cache_set("technical", symbol, result.model_dump())
        return result

    except Exception as e:
        logger.warning(f"Technical check failed for {symbol}: {e}")
        return TechnicalBreakoutScore()


# ── 4. Unusual options activity ──────────────────────────────────────────────

_MAX_EXPIRIES = 4  # check up to 4 nearest expirations


def _check_options_activity(
    symbol: str,
    ticker: yf.Ticker | None = None,
    hist: "pd.DataFrame | None" = None,
) -> OptionsActivityScore:
    cached = _cache_get("options", symbol)
    if cached:
        return OptionsActivityScore(**cached)

    try:
        if ticker is None:
            _yf_throttle()
            ticker = yf.Ticker(symbol)

        expirations = ticker.options
        if not expirations:
            result = OptionsActivityScore()
            _cache_set("options", symbol, result.model_dump())
            return result

        # Aggregate volume and OI across multiple expirations in a single pass
        total_call_vol = 0
        total_put_vol = 0
        total_oi = 0
        all_calls_frames = []

        for exp in expirations[:_MAX_EXPIRIES]:
            try:
                _yf_throttle()
                chain = ticker.option_chain(exp)
                total_call_vol += int(chain.calls["volume"].fillna(0).sum())
                total_put_vol += int(chain.puts["volume"].fillna(0).sum())
                total_oi += int(chain.calls["openInterest"].fillna(0).sum())
                total_oi += int(chain.puts["openInterest"].fillna(0).sum())
                all_calls_frames.append(chain.calls)
            except Exception:
                continue

        total_volume = total_call_vol + total_put_vol

        # Compare options volume against its own open interest as a baseline.
        # High volume relative to OI indicates fresh positioning (unusual activity).
        avg_daily_oi = total_oi if total_oi > 0 else 1
        volume_ratio = total_volume / avg_daily_oi

        put_call_ratio = (total_put_vol / total_call_vol) if total_call_vol > 0 else 999.0

        # Notable strikes: high-OI near-ATM calls across expirations
        import pandas as pd

        notable_strikes: list[str] = []
        if hist is None or hist.empty:
            hist = ticker.history(period="1mo")
        current_price = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0

        if current_price > 0 and all_calls_frames:
            combined_calls = pd.concat(all_calls_frames, ignore_index=True)
            atm_range = current_price * 0.05
            near_atm = combined_calls[
                (combined_calls["strike"] >= current_price - atm_range)
                & (combined_calls["strike"] <= current_price + atm_range)
            ]
            if not near_atm.empty:
                top_oi = near_atm.nlargest(3, "openInterest")
                for _, row in top_oi.iterrows():
                    notable_strikes.append(f"${row['strike']:.0f}C OI:{int(row['openInterest'])}")

        has_oi_concentration = len(notable_strikes) > 0

        # Score (max 20)
        score = 0.0
        if volume_ratio >= 0.5:
            score += 10
        elif volume_ratio >= 0.3:
            score += 7
        elif volume_ratio >= 0.15:
            score += 3

        if total_call_vol > 1.5 * total_put_vol:
            score += 7  # call-heavy skew

        if has_oi_concentration:
            score += 3

        score = min(score, 20.0)

        result = OptionsActivityScore(
            score=score,
            total_volume=total_volume,
            avg_volume=round(float(avg_daily_oi), 0),
            volume_ratio=round(volume_ratio, 2),
            put_call_ratio=round(put_call_ratio, 2),
            notable_strikes=notable_strikes,
        )
        _cache_set("options", symbol, result.model_dump())
        return result

    except Exception as e:
        logger.debug(f"Options activity check failed for {symbol}: {e}")
        result = OptionsActivityScore()
        _cache_set("options", symbol, result.model_dump())
        return result


# ── Main screener ────────────────────────────────────────────────────────────


def screen_smart_money(symbol: str) -> SmartMoneyResult:
    """Run all four smart money signal checks for a symbol."""
    # Share a single yf.Ticker + history across technical & options checks
    try:
        ticker, hist = _fetch_ticker_data(symbol)
    except Exception:
        ticker, hist = None, None  # type: ignore[assignment]

    insider = _fetch_insider_activity(symbol)
    congress = _fetch_congress_trades(symbol)
    technical = _check_technical_breakout(symbol, ticker=ticker, hist=hist)
    options_activity = _check_options_activity(symbol, ticker=ticker, hist=hist)

    total = insider.score + congress.score + technical.score + options_activity.score

    if total <= -20:
        signal = SmartMoneySignal.SELL
    elif total >= 75:
        signal = SmartMoneySignal.STRONG_BUY
    elif total >= 60:
        signal = SmartMoneySignal.BUY
    elif total >= 40:
        signal = SmartMoneySignal.WATCH
    else:
        signal = SmartMoneySignal.HOLD

    return SmartMoneyResult(
        symbol=symbol.upper(),
        total_score=total,
        signal=signal,
        insider=insider,
        congress=congress,
        technical=technical,
        options_activity=options_activity,
    )


def _fetch_sp500_table() -> dict:
    """Fetch S&P 500 table from Wikipedia, returning tickers and sector map.

    Returns dict with keys: "tickers" (sorted list) and "sector_map" (ticker -> sector).
    """
    cached = _cache_get("universe", "sp500_v2")
    if cached:
        return cached

    try:
        import io

        import pandas as pd

        resp = httpx.get(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            timeout=15.0,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        resp.raise_for_status()
        table = pd.read_html(io.StringIO(resp.text))[0]
        table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
        tickers = sorted(table["Symbol"].tolist())
        sector_map = dict(zip(table["Symbol"], table["GICS Sector"]))
        result = {"tickers": tickers, "sector_map": sector_map}
        _cache_set("universe", "sp500_v2", result)
        return result
    except Exception as e:
        logger.warning(f"Failed to fetch S&P 500 list: {e}")
        return {
            "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "UNH"],
            "sector_map": {},
        }


def get_sp500_tickers() -> list[str]:
    """Fetch S&P 500 ticker list from Wikipedia."""
    return _fetch_sp500_table()["tickers"]


def get_sp500_by_sector() -> dict[str, list[str]]:
    """Return S&P 500 tickers grouped by GICS sector.

    Returns dict mapping sector name -> list of ticker symbols.
    """
    sector_map = _fetch_sp500_table()["sector_map"]
    by_sector: dict[str, list[str]] = {}
    for sym, sector in sector_map.items():
        by_sector.setdefault(sector, []).append(sym)
    return by_sector
