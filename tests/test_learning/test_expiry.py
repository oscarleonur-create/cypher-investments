"""Old evidence does not decide: every rule change expires unless a sweep renews it."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest
from advisor.daemon import market_calendar as mc
from advisor.entry.proposal import current_params
from advisor.learning import actuator
from advisor.learning.actuator import (
    EVIDENCE_TTL_DAYS,
    ChangeError,
    ChangeStore,
    Status,
    active_entry_params,
    challengers,
)
from advisor.learning.sweep import FOLDS, SweepResult, Target, rejudge, revalidate

T0 = datetime(2026, 9, 1, 19, 0, tzinfo=mc.MARKET_TZ)
PARAM = "proposal.DIP_SIGMAS"
CODE = current_params().dip_sigmas


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock(monkeypatch):
    c = Clock(T0)
    monkeypatch.setattr(actuator, "now_et", c)
    return c


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(str(tmp_path / "r.db"))
    yield c
    c.close()


@pytest.fixture
def store(conn, clock):
    return ChangeStore(conn)


def active(store, clock, value=CODE + 0.5, source="sweep"):
    c = store.propose("entry", PARAM, value, source=source)
    store.activate(c.id)
    return store.get(c.id)


class TestEvidenceDate:
    def test_a_proposal_dates_its_evidence(self, store):
        c = store.propose("entry", PARAM, CODE + 0.5, source="sweep")
        assert c.evidence_at == T0.isoformat()
        assert c.expires_at == T0 + timedelta(days=EVIDENCE_TTL_DAYS)

    def test_across_the_november_dst_change_it_is_wall_time(self, store, clock):
        """Evidence from 19:00 on 1 October expires at 19:00 ET on 5 November."""
        clock.now = datetime(2026, 10, 1, 19, 0, tzinfo=mc.MARKET_TZ)
        c = store.propose("entry", PARAM, CODE + 0.5)
        assert c.expires_at == datetime(2026, 11, 5, 19, 0, tzinfo=mc.MARKET_TZ)

    def test_boundary_is_exactly_the_ttl(self, store):
        c = store.propose("entry", PARAM, CODE + 0.5)
        due = T0 + timedelta(days=EVIDENCE_TTL_DAYS)
        assert not c.expired(due - timedelta(seconds=1))
        assert c.expired(due)

    def test_a_table_from_before_expiry_is_migrated(self, tmp_path, clock):
        conn = sqlite3.connect(str(tmp_path / "old.db"))
        old_schema = actuator._SCHEMA.split("    evidence_at")[0].rstrip().rstrip(",") + "\n);"
        conn.executescript(old_schema)
        conn.execute(
            "INSERT INTO rule_changes (id, ruleset, param, value_json, previous_json, status, "
            "source, created_at) VALUES ('old1', 'entry', ?, '2.5', '2.0', 'ACTIVE', 'user', ?)",
            (PARAM, T0.isoformat()),
        )
        conn.commit()
        (c,) = ChangeStore(conn).list()
        assert c.evidence_at == T0.isoformat()
        conn.close()


class TestInForce:
    def test_an_active_change_applies_until_it_expires(self, store, conn, clock):
        c = active(store, clock)
        assert active_entry_params(conn, T0 + timedelta(days=1)).dip_sigmas == c.value
        late = T0 + timedelta(days=EVIDENCE_TTL_DAYS)
        # Not marked by anything yet, and already not applied: the code's value returns.
        assert store.get(c.id).status is Status.ACTIVE
        assert active_entry_params(conn, late).dip_sigmas == CODE

    def test_an_expired_shadow_is_no_longer_a_challenger(self, store, conn, clock):
        c = store.propose("entry", PARAM, CODE + 0.5)
        store.shadow(c.id)
        assert [cid for cid, _ in challengers(conn, "entry")] == [c.id]
        clock.now = T0 + timedelta(days=EVIDENCE_TTL_DAYS + 1)
        assert challengers(conn, "entry") == []


class TestExpireDue:
    def test_every_live_status_expires_and_the_rest_are_left(self, store, clock):
        pending = store.propose("entry", PARAM, CODE + 0.5)
        shadow = store.propose("entry", PARAM, CODE + 1.0)
        store.shadow(shadow.id)
        act = store.propose("entry", "proposal.TRADE_STOP_SIGMAS", 2.0)
        store.activate(act.id)
        rejected = store.propose("entry", PARAM, CODE + 1.5)
        store.reject(rejected.id, "no")
        clock.now = T0 + timedelta(days=EVIDENCE_TTL_DAYS)
        gone = store.expire_due()
        assert {c.id for c in gone} == {pending.id, shadow.id, act.id}
        assert all(c.status is Status.EXPIRED for c in gone)
        assert "no sweep renewed its evidence since 2026-09-01" in gone[0].note
        assert store.get(rejected.id).status is Status.REJECTED

    def test_idempotent(self, store, clock):
        store.propose("entry", PARAM, CODE + 0.5)
        clock.now = T0 + timedelta(days=EVIDENCE_TTL_DAYS)
        assert len(store.expire_due()) == 1
        assert store.expire_due() == []

    def test_nothing_due_nothing_changes(self, store, clock):
        store.propose("entry", PARAM, CODE + 0.5)
        clock.now = T0 + timedelta(days=EVIDENCE_TTL_DAYS - 1)
        assert store.expire_due() == []

    def test_an_expired_change_cannot_be_retired_or_renewed(self, store, clock):
        c = store.propose("entry", PARAM, CODE + 0.5)
        store.expire(c.id, "test")
        with pytest.raises(ChangeError):
            store.renew(c.id)
        with pytest.raises(ChangeError):
            store.activate(c.id)


class TestActivate:
    def test_a_sweep_change_with_old_evidence_is_refused_and_expired(self, store, clock):
        c = store.propose("entry", PARAM, CODE + 0.5, source="sweep")
        clock.now = T0 + timedelta(days=EVIDENCE_TTL_DAYS + 3)
        with pytest.raises(ChangeError, match="older than"):
            store.activate(c.id)
        assert store.get(c.id).status is Status.EXPIRED

    def test_a_users_change_is_dated_by_its_activation(self, store, clock):
        c = store.propose("entry", PARAM, CODE + 0.5, source="user")
        clock.now = T0 + timedelta(days=20)
        store.activate(c.id)
        assert store.get(c.id).evidence_at == clock.now.isoformat()

    def test_renewal_extends_the_life(self, store, clock):
        c = active(store, clock)
        clock.now = T0 + timedelta(days=30)
        r = store.renew(c.id, {"fresh": 1}, note="re-judged")
        assert r.expires_at == clock.now + timedelta(days=EVIDENCE_TTL_DAYS)
        assert r.evidence == {"fresh": 1} and r.note == "re-judged"


# ── The monthly re-judgment ───────────────────────────────────────────────

TARGET = Target("dip trigger", "entry", PARAM, "dip_sigmas", (1.0, 1.5, 2.0, 2.5, 3.0), 1, "t")


def recs(effect: float, days: int = 400, per_day: int = 1):
    import random

    rng = random.Random(int(effect * 1000) + 7)
    start = datetime(2025, 1, 1).date()
    return [
        (start + timedelta(days=i), effect + rng.gauss(0, 0.002))
        for i in range(days)
        for _ in range(per_day)
    ]


def result(code_effect: float, active_effect: float, active_value: float) -> SweepResult:
    records = {TARGET.name: {CODE: recs(code_effect), active_value: recs(active_effect)}}
    return SweepResult(
        records=records, sessions={}, targets=[TARGET], ran_on=datetime(2026, 9, 29).date()
    )


class TestRejudge:
    def test_a_change_that_still_beats_the_code_survives(self, store, clock):
        c = active(store, clock, value=2.5)
        v = rejudge(result(0.0, 0.02, 2.5), c)
        assert v.proposed == 2.5

    def test_a_change_no_better_than_the_code_does_not(self, store, clock):
        c = active(store, clock, value=2.5)
        v = rejudge(result(0.01, 0.01, 2.5), c)
        assert v.proposed is None

    def test_a_parameter_not_searched_is_left_to_age(self, store, clock):
        c = active(store, clock, value=2.5)
        empty = SweepResult(records={}, targets=[], ran_on=None)
        assert rejudge(empty, c) is None
        assert revalidate(empty, store).expired == []
        assert store.get(c.id).status is Status.ACTIVE

    def test_revalidate_renews_the_survivor_and_expires_the_rest(self, store, clock):
        keep = active(store, clock, value=2.5)
        clock.now = T0 + timedelta(days=28)
        r = revalidate(result(0.0, 0.02, 2.5), store)
        assert r.renewed == [keep.id] and r.expired == []
        assert store.get(keep.id).evidence_at == clock.now.isoformat()
        assert "still survived walk-forward" in store.get(keep.id).note

    def test_revalidate_expires_a_change_the_history_no_longer_supports(self, store, conn, clock):
        c = active(store, clock, value=2.5)
        r = revalidate(result(0.01, 0.0, 2.5), store)
        assert r.expired == [c.id]
        gone = store.get(c.id)
        assert gone.status is Status.EXPIRED and f"the code's {CODE} returns" in gone.note
        assert active_entry_params(conn).dip_sigmas == CODE

    def test_too_little_history_is_not_support(self, store, clock):
        """Unproven is not proven: a change the new data cannot judge expires."""
        c = active(store, clock, value=2.5)
        thin = SweepResult(
            records={TARGET.name: {CODE: recs(0.0, days=20), 2.5: recs(0.02, days=20)}},
            targets=[TARGET],
            ran_on=datetime(2026, 9, 29).date(),
        )
        r = revalidate(thin, store)
        assert r.expired == [c.id]
        assert FOLDS > 1  # the bar: at least MIN_WINS judged stretches


# ── The daemon job and its notice ─────────────────────────────────────────


class TestJob:
    def test_rule_expiry_marks_notifies_once_and_restores_the_code(self, tmp_path, clock):
        import asyncio

        from advisor.daemon.handlers import run_rule_expiry
        from advisor.daemon.jobs import JobContext
        from advisor.daemon.store import DaemonStore
        from advisor.daemon.summarize import summarize

        db = tmp_path / "research.db"
        daemon = DaemonStore(db)
        conn = sqlite3.connect(str(db))
        c = active(ChangeStore(conn), clock, value=2.5)
        clock.now = T0 + timedelta(days=EVIDENCE_TTL_DAYS + 1)
        ctx = JobContext(store=daemon, now=clock.now)

        first = asyncio.run(run_rule_expiry(ctx))
        assert first.ok and first.events_emitted == 1 and c.id in first.detail
        (event,) = [e for e in daemon.recent_events(limit=10) if e.kind == "RULE_CHANGE_EXPIRED"]
        assert event.tier.value == "B"
        assert f"the code's {CODE}" in summarize(event)
        assert active_entry_params(conn, clock.now).dip_sigmas == CODE

        again = asyncio.run(run_rule_expiry(ctx))
        assert again.events_emitted == 0 and again.detail == "nothing due"
        conn.close()
        daemon.close()

    def test_nothing_on_file(self, tmp_path, clock):
        import asyncio

        from advisor.daemon.handlers import run_rule_expiry
        from advisor.daemon.jobs import JobContext
        from advisor.daemon.store import DaemonStore

        daemon = DaemonStore(tmp_path / "research.db")
        r = asyncio.run(run_rule_expiry(JobContext(store=daemon, now=T0)))
        assert r.ok and r.detail == "nothing due"
        daemon.close()

    def test_the_job_runs_every_day_weekends_included(self):
        from advisor.daemon.supervisor import build_registry

        job = next(j for j in build_registry() if j.name == "rule_expiry")
        assert job.trigger.trading_days_only is False
