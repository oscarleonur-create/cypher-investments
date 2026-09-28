"""Daily bars for every listed common stock, kept current with one pull a day.

Three facts measured on 2026-09-27 shape this module:

1. **Yahoo serves the breadth, once.** A bulk download of 1,500 names and a
   year of bars took 103 s; the same request minutes later returned nothing
   for all 1,500. So the store is incremental — after the first backfill a
   day's sync asks for about two weeks per name — chunks are paced, and a
   chunk that comes back empty for every name stops the sync instead of
   recording thousands of "no data" rows. What was not fetched is fetched on
   the next run: each symbol's progress is kept on its own.
2. **Yahoo's close is split-adjusted, not dividend-adjusted.** A split
   rewrites the whole history, so bars stored before it sit on a different
   basis. Each incremental pull overlaps what is stored; when an overlapping
   close disagrees by more than ``REBASE_TOLERANCE`` the symbol's history is
   refetched in full. Dividends do not touch this close, so returns computed
   here leave them out — a fraction of a percent over the 20–120 session
   horizons the plan measures, and said so wherever a return is shown.
3. **A session's bar is final only after its close.** A pull at 11:00
   returns a bar for today built from half a session. Nothing after the last
   closed session is stored: the 16:00 bell, 13:00 on an early close, plus
   ``SETTLE_MINUTES`` for Yahoo to publish the final print.
"""

from __future__ import annotations

import logging
import math
import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from advisor.breadth.store import BreadthStore
from advisor.daemon import market_calendar as mc

logger = logging.getLogger(__name__)

BACKFILL_DAYS = 4 * 365  # the learning loop replays two years on four of data
OVERLAP_DAYS = 14  # calendar days re-read on every incremental pull
CHUNK = 150
PAUSE_SECONDS = 2.0
# A chunk this large coming back empty for every name is a refusal, not a
# chunk of delisted symbols.
RATE_LIMIT_MIN_CHUNK = 10
RATE_LIMIT_BACKOFF_SECONDS = 60.0
REBASE_TOLERANCE = 0.02
# A symbol Yahoo had nothing for is asked again after this long, not daily.
EMPTY_RETRY_DAYS = 7
SETTLE_MINUTES = 20


@dataclass(frozen=True)
class Bar:
    day: date
    open: float | None
    high: float | None
    low: float | None
    close: float
    volume: float | None


Fetch = Callable[[list[str], date], dict[str, list[Bar]]]


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def clean(bars: list[Bar]) -> tuple[list[Bar], int]:
    """(usable bars, how many were dropped). Pure.

    A bar needs a positive close. A negative or missing volume becomes None
    rather than dropping a bar whose price is fine; a zero-volume bar is a
    real, if dead, session and is kept.
    """
    out, dropped = [], 0
    for b in bars:
        close = _num(b.close)
        if close is None or close <= 0:
            dropped += 1
            continue
        volume = _num(b.volume)
        out.append(
            Bar(
                day=b.day,
                open=_num(b.open),
                high=_num(b.high),
                low=_num(b.low),
                close=close,
                volume=volume if volume is not None and volume >= 0 else None,
            )
        )
    return out, dropped


def last_closed_session(now: datetime) -> date:
    """The latest session whose final bar can be read at ``now`` (ET). Pure."""
    now = mc.to_et(now)
    today = now.date()
    if mc.is_trading_day(today):
        close = datetime.combine(today, mc.session_close(today), tzinfo=mc.MARKET_TZ)
        if now >= close + timedelta(minutes=SETTLE_MINUTES):
            return today
    return mc.previous_trading_day(today)


def needs_rebase(stored: dict[date, float], fresh: list[Bar]) -> bool:
    """True when a fresh close disagrees with a stored one for the same day. Pure."""
    for b in fresh:
        old = stored.get(b.day)
        if old and old > 0 and abs(b.close / old - 1) > REBASE_TOLERANCE:
            return True
    return False


def yahoo_fetch(symbols: list[str], start: date) -> dict[str, list[Bar]]:
    """Daily bars from ``start`` for each symbol; a symbol with none is left out."""
    import yfinance as yf

    df = yf.download(
        symbols,
        start=start.isoformat(),
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        actions=False,
        threads=True,
        progress=False,
    )
    out: dict[str, list[Bar]] = {}
    if df is None or df.empty:
        return out
    multi = getattr(df.columns, "nlevels", 1) > 1
    for s in symbols:
        try:
            frame = df[s] if multi else df
        except KeyError:
            continue
        frame = frame.dropna(subset=["Close"])
        bars = [
            Bar(
                day=ts.date(),
                open=_num(r["Open"]),
                high=_num(r["High"]),
                low=_num(r["Low"]),
                close=float(r["Close"]),
                volume=_num(r["Volume"]),
            )
            for ts, r in frame.iterrows()
        ]
        if bars:
            out[s] = bars
    return out


@dataclass
class SyncReport:
    requested: int = 0
    current: int = 0  # already up to date; not asked for
    skipped_empty: int = 0  # had no data recently; retried after EMPTY_RETRY_DAYS
    fetched: int = 0
    empty: list[str] = field(default_factory=list)
    rebased: list[str] = field(default_factory=list)
    bars_written: int = 0
    dropped_bad: int = 0
    rate_limited: bool = False
    remaining: int = 0  # not reached because the source refused

    def summary(self) -> str:
        s = (
            f"{self.requested} symbols: {self.current} current, {self.fetched} fetched, "
            f"{self.skipped_empty} skipped (no data lately), "
            f"{len(self.empty)} empty, {len(self.rebased)} rebased, {self.bars_written} bars"
        )
        if self.dropped_bad:
            s += f", {self.dropped_bad} bad bars dropped"
        if self.rate_limited:
            s += f"; RATE-LIMITED, {self.remaining} left for the next run"
        return s

    def as_dict(self) -> dict:
        return {
            "requested": self.requested,
            "current": self.current,
            "fetched": self.fetched,
            "empty": len(self.empty),
            "skipped_empty": self.skipped_empty,
            "rebased": self.rebased,
            "bars_written": self.bars_written,
            "dropped_bad": self.dropped_bad,
            "rate_limited": self.rate_limited,
            "remaining": self.remaining,
        }


def _state(store: BreadthStore) -> dict[str, date]:
    return {
        r["symbol"]: date.fromisoformat(r["last_day"])
        for r in store.conn.execute(
            "SELECT symbol, last_day FROM breadth_bar_state WHERE last_day IS NOT NULL"
        )
    }


def _recently_empty(store: BreadthStore, now: datetime) -> set[str]:
    cutoff = (now - timedelta(days=EMPTY_RETRY_DAYS)).isoformat()
    return {
        r[0]
        for r in store.conn.execute(
            "SELECT symbol FROM breadth_bar_state WHERE last_day IS NULL "
            "AND last_error IS NOT NULL AND synced_at >= ?",
            (cutoff,),
        )
    }


def _stored_closes(store: BreadthStore, symbol: str, since: date) -> dict[date, float]:
    return {
        date.fromisoformat(r[0]): r[1]
        for r in store.conn.execute(
            "SELECT day, close FROM breadth_bars WHERE symbol = ? AND day >= ?",
            (symbol, since.isoformat()),
        )
    }


def _write(store: BreadthStore, symbol: str, bars: list[Bar], now: datetime, *, replace: bool):
    with store.conn:
        if replace:
            store.conn.execute("DELETE FROM breadth_bars WHERE symbol = ?", (symbol,))
        store.conn.executemany(
            "INSERT OR REPLACE INTO breadth_bars (symbol, day, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(symbol, b.day.isoformat(), b.open, b.high, b.low, b.close, b.volume) for b in bars],
        )
        first, last = store.conn.execute(
            "SELECT MIN(day), MAX(day) FROM breadth_bars WHERE symbol = ?", (symbol,)
        ).fetchone()
        store.conn.execute(
            "INSERT INTO breadth_bar_state (symbol, first_day, last_day, synced_at, rebased_at, "
            "last_error) VALUES (?, ?, ?, ?, ?, NULL) ON CONFLICT(symbol) DO UPDATE SET "
            "first_day = excluded.first_day, last_day = excluded.last_day, "
            "synced_at = excluded.synced_at, last_error = NULL, "
            "rebased_at = COALESCE(excluded.rebased_at, breadth_bar_state.rebased_at)",
            (symbol, first, last, now.isoformat(), now.isoformat() if replace else None),
        )


def _mark_empty(store: BreadthStore, symbol: str, now: datetime) -> None:
    with store.conn:
        store.conn.execute(
            "INSERT INTO breadth_bar_state (symbol, synced_at, last_error) VALUES (?, ?, ?) "
            "ON CONFLICT(symbol) DO UPDATE SET synced_at = excluded.synced_at, "
            "last_error = excluded.last_error",
            (symbol, now.isoformat(), "no data"),
        )


def sync_bars(
    store: BreadthStore,
    symbols: list[str],
    now: datetime,
    *,
    fetch: Fetch = yahoo_fetch,
    chunk: int = CHUNK,
    pause: float = PAUSE_SECONDS,
    sleep: Callable[[float], None] = _time.sleep,
) -> SyncReport:
    """Bring every symbol's bars up to the last closed session. Safe to re-run.

    A symbol already at the last closed session is not asked for, so a second
    run the same evening costs nothing. New symbols are backfilled
    ``BACKFILL_DAYS``; known ones re-read ``OVERLAP_DAYS`` past their last bar.
    """
    report = SyncReport(requested=len(symbols))
    upto = last_closed_session(now)
    known = _state(store)

    batches: list[tuple[list[str], date, bool]] = []
    empty_lately = _recently_empty(store, now)
    backfill = [s for s in symbols if s not in known and s not in empty_lately]
    report.skipped_empty = sum(1 for s in symbols if s not in known and s in empty_lately)
    by_start: dict[date, list[str]] = {}
    for s in symbols:
        if s not in known:
            continue
        if known[s] >= upto:
            report.current += 1
            continue
        by_start.setdefault(known[s] - timedelta(days=OVERLAP_DAYS), []).append(s)
    for start, group in sorted(by_start.items()):
        batches += [(group[i : i + chunk], start, False) for i in range(0, len(group), chunk)]
    backfill_start = upto - timedelta(days=BACKFILL_DAYS)
    batches += [
        (backfill[i : i + chunk], backfill_start, True) for i in range(0, len(backfill), chunk)
    ]

    rebase: list[str] = []

    def run(batch: list[str], start: date, full: bool) -> bool:
        """Process one chunk; False when the source refused it."""
        result = fetch(batch, start)
        if not result and len(batch) >= RATE_LIMIT_MIN_CHUNK:
            logger.warning("breadth bars: empty chunk of %d, backing off", len(batch))
            sleep(RATE_LIMIT_BACKOFF_SECONDS)
            result = fetch(batch, start)
            if not result:
                return False
        for s in batch:
            bars, dropped = clean(result.get(s, []))
            report.dropped_bad += dropped
            bars = [b for b in bars if b.day <= upto]
            if not bars:
                report.empty.append(s)
                _mark_empty(store, s, now)
                continue
            if not full and needs_rebase(_stored_closes(store, s, start), bars):
                rebase.append(s)
                continue
            _write(store, s, bars, now, replace=full and s in rebase)
            report.fetched += 1
            report.bars_written += len(bars)
        return True

    for i, (batch, start, full) in enumerate(batches):
        if i:
            sleep(pause)
        if not run(batch, start, full):
            report.rate_limited = True
            report.remaining = sum(len(b) for b, _, _ in batches[i:])
            return report

    for i in range(0, len(rebase), chunk):
        sleep(pause)
        batch = rebase[i : i + chunk]
        if not run(batch, backfill_start, True):
            report.rate_limited = True
            report.remaining = len(rebase) - i
            return report
    report.rebased = list(rebase)
    return report
