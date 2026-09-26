"""Premarket scan of one TastyTrade watchlist for setups A, B and C.

Why a separate path from the session scanner:

- **Different universe.** The session scanner discovers movers market-wide.
  Before the bell there is no free market-wide screen worth trusting, and the
  user asked for their own list first: the TastyTrade watchlist named
  "Swing", read on every scan so edits in the app take effect immediately.
- **Different data.** Yahoo's quote is stale before the open. DXLink's
  extended-hours candles carry real premarket prints and volume.
- **Different entry.** Nobody fills the user's size at an 08:40 premarket
  print. A premarket candidate is entered at the official open; outcomes are
  measured from there, and the premarket price is kept to show how much of
  the move was gone before the bell.

Three facts measured against the live feed on 2026-09-24 shape the parsing:

1. Daily candles are stamped at 00:00 **UTC** of their session. Read in ET,
   Thursday's bar looks like Wednesday's.
2. Extended candles include the overnight session from 00:00 ET. Premarket
   is 04:00-09:30; earlier prints are excluded.
3. Some names do not trade premarket at all (AOSL had none that morning).
   No print is "no premarket quote", never a 0% move.
"""

from __future__ import annotations

import asyncio
import logging
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

from advisor.daemon import market_calendar as mc
from advisor.scanner import detect as rules
from advisor.scanner.models import Candidate, CatalystItem, Phase, Setup

logger = logging.getLogger(__name__)

DEFAULT_WATCHLIST = "Swing"
PREMARKET_OPEN = time(4, 0)
# A scan runs only inside this window; after 09:30 the session scanner owns it.
SCAN_FROM = time(7, 0)
SCAN_UNTIL = mc.REGULAR_OPEN


@dataclass(frozen=True)
class Bar:
    ts: datetime  # tz-aware; daily bars carry the session date at 00:00 UTC
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class PremarketQuote:
    symbol: str
    prev_close: float | None = None
    last: float | None = None  # latest premarket print, None if none traded
    premarket_volume: float = 0.0
    prev_day_change: float | None = None
    prev_day_volume: float | None = None
    avg_volume: float | None = None  # 20 sessions before the previous day
    sigma: float | None = None  # daily-return std-dev, 60 sessions to prev day
    market_cap: float | None = None
    name: str | None = None

    @property
    def change(self) -> float | None:
        if not (_ok(self.last) and _ok(self.prev_close)):
            return None
        return self.last / self.prev_close - 1

    @property
    def prev_day_rvol(self) -> float | None:
        if not (_ok(self.prev_day_volume) and _ok(self.avg_volume)):
            return None
        return self.prev_day_volume / self.avg_volume

    @property
    def volume_share(self) -> float | None:
        """Premarket volume as a share of an average full day's volume."""
        if not _ok(self.avg_volume):
            return None
        return self.premarket_volume / self.avg_volume


def _ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x) and x > 0


def daily_session(bar: Bar) -> date:
    """The session a daily bar belongs to: its UTC date, never its ET date."""
    return bar.ts.astimezone(timezone.utc).date()


def build_quote(
    symbol: str, intraday: list[Bar], daily: list[Bar], now: datetime
) -> PremarketQuote:
    """Everything the premarket rules need, from raw candles. Pure."""
    now = mc.to_et(now)
    session = now.date()
    prev_day = mc.previous_trading_day(session)
    day_before = mc.previous_trading_day(prev_day)

    by_session = {daily_session(b): b for b in daily if _ok(b.close)}
    q = PremarketQuote(symbol=symbol)

    prev_bar = by_session.get(prev_day)
    if prev_bar is not None:
        q.prev_close = prev_bar.close
        q.prev_day_volume = prev_bar.volume
    else:
        # Fall back to the last regular-session 5m bar of the previous day.
        close_t = mc.session_close(prev_day)
        rth = [
            b
            for b in intraday
            if mc.to_et(b.ts).date() == prev_day
            and mc.REGULAR_OPEN <= mc.to_et(b.ts).time() < close_t
        ]
        if rth:
            q.prev_close = max(rth, key=lambda b: b.ts).close

    before = by_session.get(day_before)
    if prev_bar is not None and before is not None:
        q.prev_day_change = prev_bar.close / before.close - 1

    history = sorted((d, b) for d, b in by_session.items() if d < prev_day)
    vols = [b.volume for _, b in history[-20:] if b.volume and b.volume > 0]
    if len(vols) >= 10:
        q.avg_volume = statistics.fmean(vols)

    closes = [b.close for d, b in sorted(by_session.items()) if d <= prev_day]
    returns = [b / a - 1 for a, b in zip(closes, closes[1:])][-60:]
    if len(returns) >= 20:
        sd = statistics.pstdev(returns)
        q.sigma = sd if sd > 0 else None

    pre = [
        b
        for b in intraday
        if mc.to_et(b.ts).date() == session
        and PREMARKET_OPEN <= mc.to_et(b.ts).time() < mc.REGULAR_OPEN
        and b.ts <= now
        and _ok(b.close)
    ]
    if pre:
        q.last = max(pre, key=lambda b: b.ts).close
        q.premarket_volume = sum(b.volume for b in pre if b.volume and b.volume > 0)
    return q


# ── Rules ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PremarketThresholds:
    a_gap_min: float = 0.04  # A: premarket at least +4% vs yesterday's close
    # A: premarket volume vs an average day. On an ordinary morning MRVL and
    # RKLB traded 3-5% of a day before the bell (2026-09-24); a catalyst
    # morning should be well above that. A first guess, to be revisited once
    # outcomes exist.
    a_volume_share_min: float = 0.05
    b_prev_day_min: float = 0.09  # B: yesterday closed at least +9%
    b_prev_rvol_min: float = 1.5  # B: on 1.5x its average volume
    b_hold_min: float = -0.03  # B: and premarket has not given back more than 3%
    c_drop_min: float = 0.04  # C: premarket at least -4%
    sigma_min: float = 2.0  # C: and at least 2 sigma
    min_premarket_dollars: float = 500e3  # A and C: thin prints are not a price
    small_min_cap: float = 300e6  # A and B
    small_min_price: float = 2.0
    c_min_cap: float = 10e9
    c_min_price: float = 5.0
    own_share_min: float = 0.5  # C: at least half the drop is its own


PM_DEFAULT = PremarketThresholds()


# Thresholds are inclusive, and 97/100 - 1 is -0.030000000000000027 in floating
# point: without a tolerance "exactly 3% given back" would fail a 3% rule.
_EPS = 1e-9


def _gte(x: float, floor: float) -> bool:
    return x >= floor - _EPS


def _liquid(q: PremarketQuote, t: PremarketThresholds) -> bool:
    return _ok(q.last) and _gte(q.last * q.premarket_volume, t.min_premarket_dollars)


def is_premarket_gap(q: PremarketQuote, t: PremarketThresholds = PM_DEFAULT) -> bool:
    c, share = q.change, q.volume_share
    if c is None or share is None or not _ok(q.market_cap):
        return False
    return (
        q.market_cap >= t.small_min_cap
        and q.last >= t.small_min_price
        and _gte(c, t.a_gap_min)
        and _gte(share, t.a_volume_share_min)
        and _liquid(q, t)
    )


def is_day_two(q: PremarketQuote, t: PremarketThresholds = PM_DEFAULT) -> bool:
    """B is decided on yesterday. Premarket only has to not be collapsing.

    A name with no premarket prints still qualifies: silence before the bell
    is not a failed hold. The record keeps ``change`` as None to say so.
    """
    rv = q.prev_day_rvol
    if q.prev_day_change is None or rv is None or not _ok(q.market_cap):
        return False
    if not _ok(q.prev_close) or q.prev_close < t.small_min_price:
        return False
    if q.market_cap < t.small_min_cap:
        return False
    held = q.change is None or _gte(q.change, t.b_hold_min)
    return _gte(q.prev_day_change, t.b_prev_day_min) and _gte(rv, t.b_prev_rvol_min) and held


def is_premarket_dip(
    q: PremarketQuote, t: PremarketThresholds = PM_DEFAULT
) -> tuple[bool, float | None]:
    """C before the peer test. Returns (qualifies, move in sigmas)."""
    c = q.change
    if c is None or not _ok(q.market_cap) or q.market_cap < t.c_min_cap:
        return False, None
    if q.last < t.c_min_price or not _liquid(q, t) or not _gte(-c, t.c_drop_min):
        return False, None
    if q.sigma is None:
        return True, None
    z = abs(c) / q.sigma
    return _gte(z, t.sigma_min), z


def detect_premarket(
    q: PremarketQuote, t: PremarketThresholds = PM_DEFAULT
) -> list[tuple[Setup, float | None]]:
    out: list[tuple[Setup, float | None]] = []
    z_any = abs(q.change) / q.sigma if q.change is not None and q.sigma else None
    if is_premarket_gap(q, t):
        out.append((Setup.CATALYST_GAP, z_any))
    if is_day_two(q, t):
        out.append((Setup.DAY_TWO, z_any))
    ok, z = is_premarket_dip(q, t)
    if ok:
        out.append((Setup.NEWS_DIP, z))
    return out


# ── The scan ──────────────────────────────────────────────────────────────


@dataclass
class PremarketResult:
    ran: bool
    reason: str = ""
    watchlist: list[str] = field(default_factory=list)
    no_data: list[str] = field(default_factory=list)
    new: list[Candidate] = field(default_factory=list)
    already_recorded: int = 0
    sector_moves: list[str] = field(default_factory=list)
    news_lookups: int = 0
    error: str | None = None

    def summary(self) -> str:
        if not self.ran:
            return f"skipped: {self.reason}"
        by = {s: sum(1 for c in self.new if c.setup is s) for s in Setup}
        text = (
            f"premarket {len(self.watchlist)} watched; new A={by[Setup.CATALYST_GAP]} "
            f"B={by[Setup.DAY_TWO]} C={by[Setup.NEWS_DIP]}; "
            f"{self.already_recorded} already recorded; {self.news_lookups} news lookups"
        )
        if self.no_data:
            text += f"; no data: {', '.join(self.no_data)}"
        if self.sector_moves:
            text += f"; sector move, not C: {', '.join(self.sector_moves)}"
        if self.error:
            text += f"; error: {self.error}"
        return text


def run_premarket_scan(
    store,
    now: datetime,
    *,
    fetch_watchlist: Callable[[], tuple[list[str], str | None]],
    fetch_quotes: Callable[[list[str], datetime], dict[str, PremarketQuote]],
    fetch_peer_move: Callable[[str, datetime], tuple[list[str], float | None]],
    fetch_catalysts: Callable[[str, str | None, datetime], tuple[list[CatalystItem], bool]],
    source: str = f"tastytrade:{DEFAULT_WATCHLIST}",
    t: PremarketThresholds = PM_DEFAULT,
) -> PremarketResult:
    from advisor.scanner.ruleset import premarket_rules
    from advisor.scanner.scan import catalyst_window_start

    now = mc.to_et(now)
    if not mc.is_trading_day(now.date()):
        return PremarketResult(ran=False, reason="not a trading day")
    if not (SCAN_FROM <= now.time() < SCAN_UNTIL):
        return PremarketResult(
            ran=False,
            reason=f"outside {SCAN_FROM.strftime('%H:%M')}-{SCAN_UNTIL.strftime('%H:%M')} ET",
        )

    symbols, error = fetch_watchlist()
    result = PremarketResult(ran=True, watchlist=symbols, error=error)
    if not symbols:
        return result

    quotes = fetch_quotes(symbols, now)
    session = now.date()
    since = catalyst_window_start(now)
    stamp = premarket_rules(t)
    for sym in symbols:
        q = quotes.get(sym)
        if q is None or q.prev_close is None:
            result.no_data.append(sym)
            continue
        for setup, z in detect_premarket(q, t):
            if store.exists(session, setup, sym, Phase.PREMARKET):
                result.already_recorded += 1
                continue
            peers, moved = [], None
            if setup is Setup.NEWS_DIP:
                peers, moved = fetch_peer_move(sym, now)
                if not rules.passes_peer_test(
                    q.change, moved, rules.Thresholds(own_share_min=t.own_share_min)
                ):
                    result.sector_moves.append(sym)
                    continue
            cand = Candidate(
                session=session,
                setup=setup,
                symbol=sym,
                detected_at=now,
                phase=Phase.PREMARKET,
                source=source,
                name=q.name,
                # Premarket print if there is one, else yesterday's close: the
                # reference the open will be compared against.
                price=q.last if _ok(q.last) else q.prev_close,
                prev_close=q.prev_close,
                change=q.change,
                sigma=z,
                market_cap=q.market_cap,
                peers=peers,
                peer_move=moved,
                prev_day_change=q.prev_day_change,
                prev_day_rvol=q.prev_day_rvol,
                premarket_volume=q.premarket_volume,
                rules=stamp,
            )
            items, checked = fetch_catalysts(sym, q.name, since)
            cand.catalysts, cand.news_checked = items, checked
            result.news_lookups += 1
            if store.add(cand):
                result.new.append(cand)
            else:
                result.already_recorded += 1
    return result


def scan_watchlist(store, now: datetime, name: str = DEFAULT_WATCHLIST) -> PremarketResult:
    """The premarket scan wired to live sources. What the daemon and CLI call."""
    from advisor.scanner.sources import find_catalysts

    return run_premarket_scan(
        store,
        now,
        fetch_watchlist=lambda: fetch_watchlist(name),
        fetch_quotes=fetch_quotes,
        fetch_peer_move=premarket_peer_move,
        fetch_catalysts=find_catalysts,
        source=f"tastytrade:{name}",
    )


# ── Live sources ──────────────────────────────────────────────────────────


def fetch_watchlist(name: str = DEFAULT_WATCHLIST) -> tuple[list[str], str | None]:
    """Symbols of the user's TastyTrade private watchlist, or ([], error)."""

    async def _get():
        from tastytrade.watchlists import PrivateWatchlist

        from advisor.market.tastytrade_client import get_session

        wl = await PrivateWatchlist.get(await get_session(), name)
        out = []
        for e in wl.watchlist_entries or []:
            sym = e.get("symbol") if isinstance(e, dict) else getattr(e, "symbol", None)
            if sym and str(sym).strip():
                out.append(str(sym).strip().upper())
        return list(dict.fromkeys(out))

    try:
        return asyncio.run(_get()), None
    except Exception as exc:  # noqa: BLE001
        return [], f"watchlist {name!r} unavailable: {exc}"


async def _collect_candles(symbols: list[str], interval: str, start: datetime, ext: bool):
    """Candles per symbol. Bounded in time: live updates never go quiet."""
    from tastytrade import DXLinkStreamer
    from tastytrade.dxfeed import Candle

    from advisor.market.tastytrade_client import get_session

    out: dict[str, dict[int, Bar]] = {s: {} for s in symbols}
    async with DXLinkStreamer(await get_session()) as st:
        await st.subscribe_candle(symbols, interval, start, extended_trading_hours=ext)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 20
        while loop.time() < deadline:
            try:
                c = await asyncio.wait_for(st.get_event(Candle), 4)
            except asyncio.TimeoutError:
                break
            vals = [c.open, c.high, c.low, c.close]
            if any(v is None for v in vals):
                continue
            o, h, lo, cl = (float(v) for v in vals)
            if not all(math.isfinite(v) for v in (o, h, lo, cl)):
                continue
            sym = c.event_symbol.split("{")[0]
            if sym in out:
                ts = datetime.fromtimestamp(c.time / 1000, tz=timezone.utc)
                out[sym][c.time] = Bar(ts, o, h, lo, cl, float(c.volume or 0))
    return {s: sorted(v.values(), key=lambda b: b.ts) for s, v in out.items()}


def fetch_quotes(
    symbols: list[str], now: datetime, *, with_profile: bool = True
) -> dict[str, PremarketQuote]:
    """Premarket quotes from DXLink candles, plus name and cap from Yahoo."""
    if not symbols:
        return {}
    now = mc.to_et(now)
    prev_day = mc.previous_trading_day(now.date())
    intraday_from = datetime.combine(prev_day, mc.REGULAR_OPEN, tzinfo=mc.MARKET_TZ)
    daily_from = now - timedelta(days=130)

    async def _both():
        intraday = await _collect_candles(symbols, "5m", intraday_from, True)
        daily = await _collect_candles(symbols, "1d", daily_from, False)
        return intraday, daily

    try:
        intraday, daily = asyncio.run(_both())
    except Exception as exc:  # noqa: BLE001
        logger.warning("premarket: candle fetch failed: %s", exc)
        return {}

    out = {}
    for sym in symbols:
        q = build_quote(sym, intraday.get(sym, []), daily.get(sym, []), now)
        if with_profile:
            q.name, q.market_cap = _profile(sym)
        out[sym] = q
    return out


def _profile(symbol: str) -> tuple[str | None, float | None]:
    try:
        import yfinance as yf

        info = yf.Ticker(symbol).info
        cap = info.get("marketCap")
        return info.get("shortName"), float(cap) if cap else None
    except Exception as exc:  # noqa: BLE001
        logger.info("premarket: no profile for %s: %s", symbol, exc)
        return None, None


def premarket_peer_move(symbol: str, now: datetime) -> tuple[list[str], float | None]:
    """Peers chosen as in the session scan; their move measured premarket."""
    from advisor.scanner.sources import select_peers

    peers = select_peers(symbol)
    if not peers:
        return [], None
    quotes = fetch_quotes(peers, now, with_profile=False)
    moves = [q.change for q in quotes.values() if q.change is not None]
    if len(moves) < 2:
        return peers, None
    return peers, statistics.median(moves)
