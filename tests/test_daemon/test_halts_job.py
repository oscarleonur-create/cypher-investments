"""The trading_halts job: a failed feed is recorded, only watched names are stored."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from advisor.daemon import handlers
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import BookSnapshot, Position
from advisor.daemon.jobs import JobContext
from advisor.daemon.store import DaemonStore
from advisor.daemon.universe import Universe, WatchReason
from advisor.news import halts
from advisor.news.halts import Halt

NOW = datetime(2026, 9, 28, 10, 0, tzinfo=mc.MARKET_TZ)


@pytest.fixture
def ctx(tmp_path: Path):
    store = DaemonStore(tmp_path / "d.db")
    yield JobContext(store=store, now=NOW)
    store.close()


def pos(sym, qty):
    return Position(
        account="a", symbol=sym, underlying=sym, instrument="Equity", quantity=qty, close_price=4.0
    )


def universe(book, **kw):
    u = Universe()
    for s in book.symbols:
        u.add(s, WatchReason.HELD)
    u.add("AMZN", WatchReason.WATCHLIST)
    return u


def run(ctx):
    return asyncio.run(handlers.run_trading_halts(ctx))


def test_a_failed_feed_is_a_failed_run(ctx, monkeypatch):
    def down():
        raise TimeoutError("nasdaqtrader")

    monkeypatch.setattr(halts, "fetch", down)
    r = run(ctx)
    assert not r.ok and "halt feed failed: nasdaqtrader" in r.detail


def test_no_book_is_a_failed_run(ctx, monkeypatch):
    monkeypatch.setattr(halts, "fetch", lambda: [])
    assert not run(ctx).ok


def test_only_watched_names_are_stored_and_a_held_long_is_tier_a(ctx, monkeypatch):
    import advisor.daemon.universe as uni

    at = NOW - timedelta(minutes=10)
    feed = [
        Halt(symbol="TE", code="T12", halted_at=at),
        Halt(symbol="AMZN", code="T12", halted_at=at),
        Halt(symbol="FBGL", code="T1", halted_at=at),
        Halt(symbol="SQQQ", code="T12", halted_at=at),  # held short: informs, never tier A
    ]
    monkeypatch.setattr(halts, "fetch", lambda: feed)
    monkeypatch.setattr(uni, "build_universe", universe)
    ctx.store.save_book(BookSnapshot(net_liq=10_000, positions=[pos("TE", 50), pos("SQQQ", -5)]))

    r = run(ctx)
    assert r.ok and "4 halts in the feed, 3 new event(s)" in r.detail
    tiers = {e.symbol: e.tier.value for e in ctx.store.recent_events() if e.kind == "TRADING_HALT"}
    assert tiers == {"TE": "A", "AMZN": "B", "SQQQ": "B"}
    assert "0 new event(s)" in run(ctx).detail  # the next poll stores nothing twice
