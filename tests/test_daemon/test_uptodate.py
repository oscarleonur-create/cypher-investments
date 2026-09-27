"""Is the system current? Jobs against their schedule, code against main, rules against expiry."""

from __future__ import annotations

import sqlite3
from datetime import datetime, time, timedelta
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon import uptodate
from advisor.daemon.jobs import AtLeastEvery, DailyAt, EveryMinutes, Job, WindowEvery
from advisor.daemon.models import EventSource, Heartbeat
from advisor.daemon.store import DaemonStore
from advisor.daemon.uptodate import job_state, system_status

ET = mc.MARKET_TZ
MON_1100 = datetime(2026, 9, 28, 11, 0, tzinfo=ET)
SUN = datetime(2026, 9, 27, 14, 0, tzinfo=ET)


async def _noop(ctx):  # pragma: no cover
    return None


def job(trigger, name="j"):
    return Job(name, trigger, _noop)


def hb(ok=None, ran=None, error=""):
    return Heartbeat(job="j", last_ok_at=ok, last_run_at=ran or ok, last_error=error)


class TestLastDue:
    def test_daily_on_a_trading_day_after_its_slot(self):
        t = DailyAt(time(7, 0))
        assert t.last_due(MON_1100) == datetime(2026, 9, 28, 7, 0, tzinfo=ET)

    def test_daily_before_its_slot_is_the_previous_trading_day(self):
        t = DailyAt(time(16, 30))
        assert t.last_due(MON_1100) == datetime(2026, 9, 25, 16, 30, tzinfo=ET)

    def test_daily_within_the_slack_is_not_yet_due(self):
        t = DailyAt(time(10, 50))
        assert t.last_due(MON_1100) == datetime(2026, 9, 25, 10, 50, tzinfo=ET)

    def test_every_day_includes_the_weekend(self):
        t = DailyAt(time(8, 15), trading_days_only=False)
        assert t.last_due(SUN) == datetime(2026, 9, 27, 8, 15, tzinfo=ET)

    def test_daily_skips_a_holiday(self):
        """Tuesday after Labor Day, before 07:00: the slot owed is Friday's."""
        t = DailyAt(time(7, 0))
        tue = datetime(2026, 9, 8, 6, 0, tzinfo=ET)
        assert t.last_due(tue) == datetime(2026, 9, 4, 7, 0, tzinfo=ET)

    def test_session_job_on_the_weekend_owes_fridays_last_hours(self):
        t = EveryMinutes(30, during_session_only=True)
        assert t.last_due(SUN) == datetime(2026, 9, 25, 15, 0, tzinfo=ET)

    def test_session_job_in_session_owes_a_recent_run(self):
        t = EveryMinutes(15, during_session_only=True)
        assert t.last_due(MON_1100) == MON_1100 - timedelta(minutes=45)

    def test_session_job_just_after_the_open_owes_the_last_session(self):
        t = EveryMinutes(60, during_session_only=True, not_before=time(9, 45))
        at = datetime(2026, 9, 28, 10, 0, tzinfo=ET)
        assert t.last_due(at) == datetime(2026, 9, 25, 14, 0, tzinfo=ET)

    def test_session_job_after_an_early_close(self):
        t = EveryMinutes(15, during_session_only=True)
        fri = datetime(2026, 11, 27, 18, 0, tzinfo=ET)  # closed 13:00
        assert t.last_due(fri) == datetime(2026, 11, 27, 12, 30, tzinfo=ET)

    def test_always_job(self):
        t = EveryMinutes(5, during_session_only=False)
        assert t.last_due(SUN) == SUN - timedelta(minutes=25)

    def test_window_job(self):
        t = WindowEvery(time(8, 0), time(9, 25), 20)
        assert t.last_due(MON_1100) == datetime(2026, 9, 28, 8, 0, tzinfo=ET)
        assert t.last_due(SUN) == datetime(2026, 9, 25, 8, 0, tzinfo=ET)

    def test_at_least_every_has_a_day_of_slack(self):
        assert AtLeastEvery(7).last_due(SUN) == SUN - timedelta(days=8)


class TestJobState:
    def test_ok(self):
        s = job_state(job(DailyAt(time(7, 0))), hb(ok=MON_1100.replace(hour=7, minute=1)), MON_1100)
        assert s.state == "ok"

    def test_late(self):
        s = job_state(job(DailyAt(time(7, 0))), hb(ok=MON_1100 - timedelta(days=3)), MON_1100)
        assert s.state == "late"

    def test_failing_beats_late(self):
        s = job_state(
            job(DailyAt(time(7, 0))),
            hb(ok=MON_1100 - timedelta(days=3), ran=MON_1100, error="boom"),
            MON_1100,
        )
        assert s.state == "failing" and s.last_error == "boom"

    def test_never(self):
        assert job_state(job(DailyAt(time(7, 0))), hb(), MON_1100).state == "never"

    def test_utc_heartbeat(self):
        ok = MON_1100.replace(hour=7, minute=1).astimezone(mc.ZoneInfo("UTC"))
        assert job_state(job(DailyAt(time(7, 0))), hb(ok=ok), MON_1100).state == "ok"


class TestBehind:
    def test_unknown_revisions(self):
        assert uptodate.behind(None, "abc") is None
        assert uptodate.behind("unknown", "abc") is None
        assert uptodate.behind("abc", None) is None

    def test_dirty_suffix_is_stripped(self, monkeypatch):
        seen = []
        monkeypatch.setattr(uptodate, "_git", lambda *a: seen.append(a) or "3")
        assert uptodate.behind("abc123+dirty", "def456") == 3
        assert seen == [("rev-list", "--count", "abc123..def456")]

    def test_git_unavailable(self, monkeypatch):
        monkeypatch.setattr(uptodate, "_git", lambda *a: None)
        assert uptodate.behind("abc", "def") is None


@pytest.fixture
def db(tmp_path: Path):
    return tmp_path / "research.db"


class TestSystemStatus:
    def test_an_empty_store_names_what_never_ran(self, db, monkeypatch):
        monkeypatch.setattr(uptodate, "_git", lambda *a: None)
        st = system_status(db, MON_1100)
        assert not st.ok
        assert any("brief has never succeeded" in p for p in st.problems)
        assert st.code.daemon is None and st.rules.changes == []

    def test_the_daemons_revision_comes_from_its_heartbeat(self, db, monkeypatch):
        monkeypatch.setattr(uptodate, "_git", lambda *a: "cafe" if a[0] == "rev-parse" else "2")
        s = DaemonStore(db)
        s.set_watermark(EventSource.DAEMON, last_seen_ts=MON_1100, last_seen_cursor="beef")
        s.close()
        st = system_status(db, MON_1100)
        assert st.code.daemon == "beef" and st.code.daemon_behind == 2
        assert any("2 commit(s) behind main" in p for p in st.problems)

    def test_a_rule_change_near_expiry_is_flagged(self, db, monkeypatch):
        from advisor.learning import actuator
        from advisor.learning.actuator import EVIDENCE_TTL_DAYS, ChangeStore

        monkeypatch.setattr(uptodate, "_git", lambda *a: None)
        t0 = MON_1100 - timedelta(days=EVIDENCE_TTL_DAYS - 3)
        monkeypatch.setattr(actuator, "now_et", lambda: t0)
        conn = sqlite3.connect(str(db))
        store = ChangeStore(conn)
        c = store.propose("entry", "proposal.DIP_SIGMAS", 2.5)
        store.activate(c.id)
        conn.close()
        st = system_status(db, MON_1100)
        (row,) = st.rules.changes
        assert row["in_force"] and 2.9 < row["days_left"] < 3.1
        assert any(f"rule change {c.id}" in p and "expires in 3" in p for p in st.problems)

    def test_an_expired_active_change_is_reported_as_not_applying(self, db, monkeypatch):
        from advisor.learning import actuator
        from advisor.learning.actuator import EVIDENCE_TTL_DAYS, ChangeStore

        monkeypatch.setattr(uptodate, "_git", lambda *a: None)
        t0 = MON_1100 - timedelta(days=EVIDENCE_TTL_DAYS + 1)
        monkeypatch.setattr(actuator, "now_et", lambda: t0)
        conn = sqlite3.connect(str(db))
        store = ChangeStore(conn)
        c = store.propose("entry", "proposal.DIP_SIGMAS", 2.5)
        store.activate(c.id)
        conn.close()
        st = system_status(db, MON_1100)
        assert not st.rules.changes[0]["in_force"]
        assert any("past its expiry and no longer applies" in p for p in st.problems)

    def test_stale_inputs_of_the_latest_proposals(self, db, monkeypatch):
        from advisor.entry.proposal import Action, Proposal, Reason
        from advisor.entry.store import EntryStore

        monkeypatch.setattr(uptodate, "_git", lambda *a: None)
        e = EntryStore(db)
        for sym in ("AMZN", "META"):
            e.add(
                Proposal(
                    symbol=sym,
                    session=MON_1100.date(),
                    built_at=MON_1100,
                    action=Action.WAIT,
                    reasons=[Reason(text="r", source="s")],
                    features={"stale": "price,book"},
                )
            )
        e.close()
        st = system_status(db, MON_1100)
        assert st.stale_inputs == {"book": ["AMZN", "META"], "price": ["AMZN", "META"]}
        assert any("price stale on 2 name(s)" in p for p in st.problems)


def test_the_heartbeat_records_the_code_it_runs(db):
    import asyncio

    from advisor.daemon.handlers import run_heartbeat
    from advisor.daemon.jobs import JobContext
    from advisor.learning.rules import code_rev

    s = DaemonStore(db)
    asyncio.run(run_heartbeat(JobContext(store=s, now=MON_1100)))
    assert s.get_watermark(EventSource.DAEMON).last_seen_cursor == code_rev()
    s.close()
