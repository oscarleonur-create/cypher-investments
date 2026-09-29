"""One sync end to end over fake sources: each stage reports, failures do not cascade."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

from advisor.breadth import bars as B
from advisor.breadth.listings import build_directory
from advisor.breadth.run import run_sync
from advisor.breadth.store import BreadthStore, breadth_path
from advisor.daemon.market_calendar import MARKET_TZ

NOW = datetime(2026, 9, 25, 18, 30, tzinfo=MARKET_TZ)
NASDAQ = """\
Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
LIQ|Liquid Co - Common Stock|G|N|N|100|N|N
THIN|Thin Co - Common Stock|G|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
"""
OTHER = "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
SEC = {
    "0": {"cik_str": 10, "ticker": "LIQ", "title": ""},
    "1": {"cik_str": 11, "ticker": "THIN", "title": ""},
}


def _days(n):
    out, d = [], date(2026, 9, 25)
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


def fake_bars(symbols, start):
    vol = {"LIQ": 1e6, "THIN": 10.0}
    return {
        s: [B.Bar(d, 20, 20, 20, 20.0, vol[s]) for d in _days(300) if d >= start]
        for s in symbols
        if s in vol
    }


def run(tmp_path, **kw):
    db = tmp_path / "research.db"
    kw.setdefault("fetch_dir", lambda: build_directory(NASDAQ, OTHER, SEC))
    kw.setdefault("fetch_bars", fake_bars)
    kw.setdefault("fetch_frame", lambda c, u, f: (200, []))
    kw.setdefault("fetch_concept", lambda cik, c: [])
    kw.setdefault("fetch_company", lambda cik: {"sic": 3674, "sic_desc": "Semis", "name": "x"})
    kw.setdefault("fetch_index", lambda year, q: "")
    kw.setdefault("insider_fetch", {"get_text": lambda url: "", "get_bytes": lambda url: None})
    return db, run_sync(db, NOW, sleep=lambda s: None, **kw)


def test_full_run(tmp_path):
    db, s = run(tmp_path)
    assert s["ok"]
    assert s["universe"]["eligible"] == 1
    assert s["universe"]["day"] == "2026-09-25"
    assert s["facts"]["coverage_of_eligible"]["no revenue"] == 1
    # The live database gets only the rule version, nothing else.
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not any(t.startswith("breadth_") for t in tables)
    ruleset = {r[0] for r in conn.execute("SELECT ruleset FROM rule_versions")}
    assert ruleset == {"breadth.universe", "breadth.signals"}
    # Measurement ran and recorded the day; it reached nothing the user reads.
    assert s["signals"]["live"]["ok"], s["signals"]
    # Both common stocks' industries, the thin one too: it may have been
    # eligible on a replayed day. The ETF's is not read.
    assert s["signals"]["companies"]["read"] == 2
    conn.close()
    with BreadthStore(breadth_path(db)) as store:
        assert store.last_run()["ok"] == 1


def test_rerunning_a_day_rewrites_its_live_records(tmp_path):
    db, _ = run(tmp_path)
    with BreadthStore(breadth_path(db)) as store:
        store.conn.execute(
            "INSERT INTO breadth_records (id, origin, run_id, grp, symbol, day, rules, "
            "detail_json, updated_at) VALUES ('stale', 'live', 'live', 'P', 'GONE', "
            "'2026-09-25', 'old', '{}', 'x')"
        )
        store.conn.execute(
            "INSERT INTO breadth_records (id, origin, run_id, grp, symbol, day, rules, "
            "detail_json, updated_at) VALUES ('kept', 'live', 'live', 'P', 'PAST', "
            "'2026-09-24', 'old', '{}', 'x')"
        )
        store.conn.commit()
    run(tmp_path)
    with BreadthStore(breadth_path(db)) as store:
        ids = {r[0] for r in store.conn.execute("SELECT id FROM breadth_records")}
    assert "stale" not in ids and "kept" in ids


def test_directory_down_is_reported_not_raised(tmp_path):
    def down():
        raise TimeoutError("nasdaqtrader.com")

    _, s = run(tmp_path, fetch_dir=down)
    assert not s["ok"] and "symbol directory unavailable" in s["error"]


def test_refused_bars_still_snapshot_and_read_facts(tmp_path):
    names = "\n".join(f"S{i:03d}|S{i} Co - Common Stock|G|N|N|100|N|N" for i in range(200))
    header = NASDAQ.splitlines()[0]
    sec = {str(i): {"cik_str": 100 + i, "ticker": f"S{i:03d}", "title": ""} for i in range(200)}
    _, s = run(
        tmp_path,
        fetch_dir=lambda: build_directory(header + "\n" + names + "\n", OTHER, sec),
        fetch_bars=lambda symbols, start: {},
    )
    assert not s["ok"] and s["bars"]["rate_limited"]
    assert s["universe"]["excluded"] == {"no bars": 200}
    assert "facts" in s


# ── an incomplete session is not built on (2026-09-28: 156 of 5,368 names) ──

LATER = datetime(2026, 9, 28, 20, 30, tzinfo=MARKET_TZ)


def partial_bars(symbols, start):
    """Monday's close served for LIQ only, as Yahoo did at 20:30 on 2026-09-28."""
    out = fake_bars(symbols, start)
    if "LIQ" in out:
        out["LIQ"].append(B.Bar(date(2026, 9, 28), 20, 20, 20, 20.0, 1e6))
    return out


def test_a_session_the_source_has_not_finished_is_not_built(tmp_path):
    db, first = run(tmp_path)
    assert first["ok"]
    s = run_sync(
        db, LATER, sleep=lambda s: None, fetch_dir=lambda: build_directory(NASDAQ, OTHER, SEC),
        fetch_bars=partial_bars, fetch_frame=lambda c, u, f: (200, []),
        fetch_concept=lambda cik, c: [], fetch_index=lambda year, q: "",
    )  # fmt: skip
    assert not s["ok"] and "bars for 2026-09-28 incomplete: 1 names against 2" in s["error"]
    assert "universe" not in s and "signals" not in s
    with BreadthStore(breadth_path(db)) as store:
        days = {r[0] for r in store.conn.execute("SELECT DISTINCT day FROM breadth_universe")}
        assert "2026-09-28" not in days
        assert not store.last_run()["ok"]


def test_coverage_boundaries():
    assert B.complete(90, 100) and not B.complete(89, 100)
    assert B.complete(0, 0)  # the first backfill has nothing to compare with


def test_the_retry_is_a_no_op_once_the_session_is_built(tmp_path, monkeypatch):
    import asyncio

    from advisor.daemon import handlers
    from advisor.daemon.jobs import JobContext

    db, _ = run(tmp_path)  # built 2026-09-25
    calls = []

    async def sync(ctx):
        calls.append(ctx.now)
        return handlers.JobResult(job="breadth_sync", ok=True, detail="synced")

    monkeypatch.setattr(handlers, "run_breadth_sync", sync)
    retry = handlers.breadth_retry("breadth_sync_late")
    store = type("S", (), {"db_path": db})()
    done = asyncio.run(retry(JobContext(store=store, now=NOW.replace(hour=23))))
    assert done.ok and done.detail == "the session is already built" and calls == []
    # Monday night, Monday not built: it syncs.
    again = asyncio.run(retry(JobContext(store=store, now=LATER.replace(hour=23))))
    assert again.detail == "synced" and len(calls) == 1


def test_the_nightly_job_never_overlaps_the_halts_poll():
    # Jobs run one at a time; a sync of minutes inside 04:00-20:05 would delay
    # the halts poll, whose findings can be EXITs.
    from advisor.daemon.supervisor import build_registry

    job = next(j for j in build_registry() if j.name == "breadth_sync")
    t = job.trigger
    assert not t.is_due(datetime(2026, 9, 25, 20, 4, tzinfo=MARKET_TZ), None)
    assert t.is_due(datetime(2026, 9, 25, 20, 31, tzinfo=MARKET_TZ), None)
    assert t.is_due(datetime(2026, 9, 25, 23, 59, tzinfo=MARKET_TZ), None)  # laptop woke late
    # After midnight yesterday's slot is gone and today's not yet: nothing
    # can run into the 04:00 halts poll or the session.
    for hh in (0, 4, 9, 12, 20):
        assert not t.is_due(datetime(2026, 9, 28, hh, 1, tzinfo=MARKET_TZ), None)
    assert t.is_due(datetime(2026, 9, 26, 21, 0, tzinfo=MARKET_TZ), None)  # Saturday too


def test_limit_samples_common_stock_but_keeps_exclusions(tmp_path):
    _, s = run(tmp_path, limit=1)
    assert s["directory"]["common"] == 1
    assert s["universe"]["excluded"].get("etf") == 1
