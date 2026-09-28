"""The bar store: closed sessions only, refusals, splits, and re-runs."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from advisor.breadth import bars as B
from advisor.breadth.store import BreadthStore
from advisor.daemon.market_calendar import MARKET_TZ


def et(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=MARKET_TZ)


def bar(day, close, volume=1e6):
    return B.Bar(day=day, open=close, high=close, low=close, close=close, volume=volume)


def sessions(end: date, n: int) -> list[date]:
    out, d = [], end
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d -= timedelta(days=1)
    return sorted(out)


@pytest.fixture
def store(tmp_path):
    s = BreadthStore(tmp_path / "research-breadth.db")
    yield s
    s.close()


class FakeSource:
    """Serves a fixed history per symbol from any start; can refuse or go dark."""

    def __init__(self, history: dict[str, list[B.Bar]], refuse_after: int | None = None):
        self.history = history
        self.calls: list[tuple[list[str], date]] = []
        self.refuse_after = refuse_after

    def __call__(self, symbols, start):
        self.calls.append((list(symbols), start))
        if self.refuse_after is not None and len(self.calls) > self.refuse_after:
            return {}
        return {
            s: [b for b in self.history[s] if b.day >= start]
            for s in symbols
            if s in self.history and any(b.day >= start for b in self.history[s])
        }


def no_sleep(_):
    pass


class TestLastClosedSession:
    def test_mid_session_is_yesterday(self):
        assert B.last_closed_session(et(2026, 9, 24, 11, 0)) == date(2026, 9, 23)

    def test_just_after_the_bell_is_not_yet_final(self):
        assert B.last_closed_session(et(2026, 9, 24, 16, 5)) == date(2026, 9, 23)

    def test_settled_after_the_bell(self):
        assert B.last_closed_session(et(2026, 9, 24, 16, 20)) == date(2026, 9, 24)

    def test_weekend_is_friday(self):
        assert B.last_closed_session(et(2026, 9, 27, 12, 0)) == date(2026, 9, 25)

    def test_holiday_is_the_session_before(self):
        assert B.last_closed_session(et(2026, 11, 26, 20, 0)) == date(2026, 11, 25)

    def test_early_close_settles_at_13_20(self):
        assert B.last_closed_session(et(2026, 11, 27, 13, 19)) == date(2026, 11, 25)
        assert B.last_closed_session(et(2026, 11, 27, 13, 20)) == date(2026, 11, 27)

    def test_a_utc_clock_is_read_in_new_york(self):
        from datetime import timezone

        # 20:30 UTC on 2026-09-24 is 16:30 ET: settled.
        utc = datetime(2026, 9, 24, 20, 30, tzinfo=timezone.utc)
        assert B.last_closed_session(utc) == date(2026, 9, 24)


class TestClean:
    def test_bad_closes_are_dropped_and_counted(self):
        d = date(2026, 9, 24)
        raw = [
            B.Bar(d, 1, 1, 1, float("nan"), 1),
            B.Bar(d, 1, 1, 1, -3.0, 1),
            B.Bar(d, 1, 1, 1, 0.0, 1),
            B.Bar(d, None, None, None, 10.0, None),
            B.Bar(d, 1, 1, 1, 10.0, -5),
            B.Bar(d, 1, 1, 1, 10.0, 0),
        ]
        out, dropped = B.clean(raw)
        assert dropped == 3
        assert [b.volume for b in out] == [None, None, 0.0]  # zero volume is a real session


class TestRebase:
    def test_split_detected(self):
        d = date(2026, 9, 24)
        assert B.needs_rebase({d: 100.0}, [bar(d, 10.0)])

    def test_same_close_or_new_day_is_not_a_split(self):
        d = date(2026, 9, 24)
        assert not B.needs_rebase({d: 100.0}, [bar(d, 101.9)])  # inside 2%
        assert not B.needs_rebase({}, [bar(d, 10.0)])
        assert not B.needs_rebase({d: 0.0}, [bar(d, 10.0)])


NOW = et(2026, 9, 25, 18, 30)  # Friday evening, session settled


class TestSync:
    def test_backfill_then_rerun_is_free(self, store):
        days = sessions(date(2026, 9, 25), 300)
        src = FakeSource({"AAA": [bar(d, 10) for d in days], "BBB": [bar(d, 20) for d in days]})
        r1 = B.sync_bars(store, ["AAA", "BBB"], NOW, fetch=src, sleep=no_sleep)
        assert r1.fetched == 2 and r1.bars_written == 600
        assert src.calls[0][1] == date(2026, 9, 25) - timedelta(days=B.BACKFILL_DAYS)
        r2 = B.sync_bars(store, ["AAA", "BBB"], NOW, fetch=src, sleep=no_sleep)
        assert r2.current == 2 and r2.fetched == 0
        assert len(src.calls) == 1  # nothing asked the second time
        assert store.counts()["breadth_bars"] == 600

    def test_partial_session_is_never_stored(self, store):
        days = sessions(date(2026, 9, 24), 10)
        src = FakeSource({"AAA": [bar(d, 10) for d in days]})
        B.sync_bars(store, ["AAA"], et(2026, 9, 24, 11, 0), fetch=src, sleep=no_sleep)
        last = store.conn.execute("SELECT MAX(day) FROM breadth_bars").fetchone()[0]
        assert last == "2026-09-23"

    def test_incremental_reads_only_the_overlap(self, store):
        days = sessions(date(2026, 9, 24), 50)
        src = FakeSource({"AAA": [bar(d, 10) for d in days]})
        B.sync_bars(store, ["AAA"], et(2026, 9, 24, 18, 0), fetch=src, sleep=no_sleep)
        src.history["AAA"].append(bar(date(2026, 9, 25), 11))
        r = B.sync_bars(store, ["AAA"], NOW, fetch=src, sleep=no_sleep)
        assert src.calls[-1][1] == date(2026, 9, 24) - timedelta(days=B.OVERLAP_DAYS)
        assert r.fetched == 1
        assert store.conn.execute("SELECT COUNT(*) FROM breadth_bars").fetchone()[0] == 51

    def test_split_refetches_the_whole_history(self, store):
        days = sessions(date(2026, 9, 24), 50)
        src = FakeSource({"AAA": [bar(d, 100) for d in days]})
        B.sync_bars(store, ["AAA"], et(2026, 9, 24, 18, 0), fetch=src, sleep=no_sleep)
        # A 10:1 split: the source now serves every past close divided by ten.
        src.history["AAA"] = [bar(d, 10) for d in days] + [bar(date(2026, 9, 25), 10.5)]
        r = B.sync_bars(store, ["AAA"], NOW, fetch=src, sleep=no_sleep)
        assert r.rebased == ["AAA"]
        closes = {c for (c,) in store.conn.execute("SELECT close FROM breadth_bars")}
        assert closes == {10.0, 10.5}  # nothing left on the old basis
        state = store.conn.execute("SELECT rebased_at FROM breadth_bar_state").fetchone()[0]
        assert state is not None

    def test_refusal_stops_and_resumes_next_run(self, store):
        days = sessions(date(2026, 9, 25), 30)
        names = [f"S{i:03d}" for i in range(30)]
        src = FakeSource({s: [bar(d, 10) for d in days] for s in names}, refuse_after=1)
        slept = []
        r = B.sync_bars(store, names, NOW, fetch=src, chunk=10, sleep=slept.append)
        assert r.rate_limited and r.fetched == 10 and r.remaining == 20
        assert B.RATE_LIMIT_BACKOFF_SECONDS in slept  # it backed off once before giving up
        # Nothing was marked empty: a refusal is not "no data".
        assert (
            store.conn.execute(
                "SELECT COUNT(*) FROM breadth_bar_state WHERE last_error IS NOT NULL"
            ).fetchone()[0]
            == 0
        )
        src.refuse_after = None
        r2 = B.sync_bars(store, names, NOW, fetch=src, chunk=10, sleep=no_sleep)
        assert r2.current == 10 and r2.fetched == 20 and not r2.rate_limited

    def test_a_small_empty_chunk_is_not_a_refusal(self, store):
        src = FakeSource({})
        r = B.sync_bars(store, ["DEAD1", "DEAD2"], NOW, fetch=src, sleep=no_sleep)
        assert not r.rate_limited
        assert sorted(r.empty) == ["DEAD1", "DEAD2"]

    def test_empty_symbols_wait_a_week(self, store):
        src = FakeSource({})
        B.sync_bars(store, ["DEAD"], NOW, fetch=src, sleep=no_sleep)
        r = B.sync_bars(store, ["DEAD"], NOW + timedelta(days=1), fetch=src, sleep=no_sleep)
        assert r.skipped_empty == 1 and len(src.calls) == 1
        r = B.sync_bars(
            store, ["DEAD"], NOW + timedelta(days=B.EMPTY_RETRY_DAYS + 1), fetch=src, sleep=no_sleep
        )
        assert r.skipped_empty == 0 and len(src.calls) == 2

    def test_no_symbols(self, store):
        r = B.sync_bars(store, [], NOW, fetch=FakeSource({}), sleep=no_sleep)
        assert r.requested == 0 and not r.rate_limited
