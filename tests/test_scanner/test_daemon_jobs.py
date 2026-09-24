"""The scanner as the daemon runs it: off the event loop, in a worker thread.

A test that calls run_scan directly passed while the daemon job crashed on
its first live run, because sqlite3 connections cannot cross threads. These
tests go through the real handler and ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

from advisor.daemon import handlers
from advisor.daemon import market_calendar as mc
from advisor.daemon.jobs import JobContext
from advisor.daemon.store import DaemonStore
from advisor.scanner.models import Mover
from advisor.scanner.store import ScannerStore

NOW = datetime(2026, 9, 23, 10, 5, tzinfo=mc.MARKET_TZ)
GAP = Mover(
    symbol="AMPG",
    price=10.8,
    prev_close=10.0,
    open=10.6,
    volume=2_000_000,
    avg_volume=1_000_000,
    market_cap=500e6,
)


def _ctx(tmp_path):
    return JobContext(store=DaemonStore(tmp_path / "d.db"), now=NOW)


def test_scan_job_runs_in_a_worker_thread(tmp_path, monkeypatch):
    from advisor.scanner import scan as scan_mod

    ctx = _ctx(tmp_path)
    original = scan_mod.run_scan

    def patched(store, now, **kw):
        return original(
            store,
            now,
            fetch_movers=lambda: ([GAP], None),
            fetch_sigma=lambda syms: {},
            fetch_catalysts=lambda *a: ([], True),
        )

    monkeypatch.setattr(scan_mod, "run_scan", patched)
    result = asyncio.run(handlers.run_setup_scan(ctx))
    assert result.ok, result.detail
    assert "new A=1" in result.detail
    s = ScannerStore(tmp_path / "d.db")
    try:
        assert s.count() == 1
    finally:
        s.close()
    # Measurement only: nothing reaches the event stream.
    assert ctx.store.recent_events(limit=10) == []


def test_outcomes_job_runs_in_a_worker_thread(tmp_path, monkeypatch):
    from advisor.scanner import outcomes as out_mod

    original = out_mod.fill_outcomes
    monkeypatch.setattr(
        out_mod, "fill_outcomes", lambda store, now: original(store, now, bars_fn=lambda *a: None)
    )
    result = asyncio.run(handlers.run_scan_outcomes(_ctx(tmp_path)))
    assert result.ok and "0 pending" in result.detail


def test_screener_down_marks_the_job_failed(tmp_path, monkeypatch):
    from advisor.scanner import scan as scan_mod

    original = scan_mod.run_scan
    monkeypatch.setattr(
        scan_mod,
        "run_scan",
        lambda store, now, **kw: original(store, now, fetch_movers=lambda: ([], "HTTP 503")),
    )
    result = asyncio.run(handlers.run_setup_scan(_ctx(tmp_path)))
    assert not result.ok and "503" in result.detail
