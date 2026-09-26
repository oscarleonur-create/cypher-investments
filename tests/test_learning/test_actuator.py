"""The actuator: only thresholds change, only by the user's hand, and shadows act on nothing."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime

import pytest
from advisor.daemon import market_calendar as mc
from advisor.entry.proposal import build_proposal, current_params
from advisor.entry.ruleset import entry_rules
from advisor.entry.sheet import Move, Sheet
from advisor.entry.store import EntryStore
from advisor.entry.zone import RelativeZone
from advisor.learning.actuator import (
    ENTRY,
    SESSION,
    ChangeError,
    ChangeStore,
    Status,
    active_entry_params,
    active_session_thresholds,
    challengers,
    validate,
)
from advisor.learning.rules import Origin
from advisor.learning.shadow import (
    EntryShadows,
    Memo,
    challenger_path,
    fill_shadow_outcomes,
    load_shadow,
    scan_with_shadows,
)
from advisor.scanner.detect import DEFAULT
from advisor.scanner.models import CatalystItem, Mover
from advisor.scanner.store import ScannerStore


@pytest.fixture
def db(tmp_path):
    return tmp_path / "research.db"


@pytest.fixture
def store(db):
    conn = sqlite3.connect(str(db))
    yield ChangeStore(conn)
    conn.close()


class TestValidate:
    @pytest.mark.parametrize(
        "param",
        [
            "proposal.TRADE_RISK",
            "proposal.BOOK_LIMIT",
            "proposal.MAX_TOTAL_RISK",
            "zone.WINDOW_DAYS",
        ],
    )
    def test_decided_limits_are_refused(self, param):
        with pytest.raises(ChangeError, match="decided"):
            validate(ENTRY, param, 0.5)

    def test_model_constants_are_refused(self):
        with pytest.raises(ChangeError, match="model"):
            validate(ENTRY, "sheet.SIGMA_SESSIONS", 30)

    def test_unknown_param(self):
        with pytest.raises(ChangeError, match="not a declared"):
            validate(ENTRY, "proposal.NOPE", 1.0)

    def test_a_threshold_the_code_cannot_take_yet(self):
        with pytest.raises(ChangeError, match="run time"):
            validate(ENTRY, "zone.MIN_OBSERVATIONS", 200)

    def test_rulesets_without_run_time_params(self):
        with pytest.raises(ChangeError, match="no run-time"):
            validate("scanner.premarket", "thresholds.sigma_min", 2.5)

    def test_session_count_must_be_whole(self):
        with pytest.raises(ChangeError, match="whole"):
            validate(ENTRY, "proposal.ENTRY_CONFIRM_SESSIONS", 4.5)
        assert validate(ENTRY, "proposal.ENTRY_CONFIRM_SESSIONS", 7.0) == 7

    @pytest.mark.parametrize("value", [0, -1.5])
    def test_must_be_positive(self, value):
        with pytest.raises(ChangeError, match="positive"):
            validate(ENTRY, "proposal.TRADE_STOP_SIGMAS", value)

    def test_same_value_is_not_a_change(self):
        with pytest.raises(ChangeError, match="already"):
            validate(ENTRY, "proposal.TRADE_STOP_SIGMAS", 1.5)

    @pytest.mark.parametrize("value", ["2", True, None])
    def test_must_be_a_number(self, value):
        with pytest.raises(ChangeError, match="number"):
            validate(ENTRY, "proposal.TRADE_STOP_SIGMAS", value)

    def test_scanner_threshold(self):
        assert validate(SESSION, "thresholds.sigma_min", 2.5) == 2.5


class TestLifecycle:
    def test_proposed_is_pending_and_does_nothing(self, store):
        c = store.propose(ENTRY, "proposal.TRADE_STOP_SIGMAS", 2.0)
        assert c.status is Status.PENDING and c.previous == 1.5
        assert active_entry_params(store._conn) == current_params()

    def test_activate_applies_and_retires_the_previous(self, store):
        a = store.propose(ENTRY, "proposal.TRADE_STOP_SIGMAS", 2.0)
        store.activate(a.id)
        assert active_entry_params(store._conn).trade_stop_sigmas == 2.0
        b = store.propose(ENTRY, "proposal.TRADE_STOP_SIGMAS", 2.5)
        store.activate(b.id)
        assert store.get(a.id).status is Status.RETIRED
        assert active_entry_params(store._conn).trade_stop_sigmas == 2.5

    def test_retire_is_the_rollback(self, store):
        a = store.propose(SESSION, "thresholds.sigma_min", 2.5)
        store.activate(a.id)
        assert active_session_thresholds(store._conn).sigma_min == 2.5
        store.retire(a.id, "worse live")
        assert active_session_thresholds(store._conn) == DEFAULT
        assert store.get(a.id).note == "worse live"

    def test_shadow_does_not_change_the_live_rules(self, store):
        a = store.propose(ENTRY, "proposal.DIP_SIGMAS", 2.5)
        store.shadow(a.id)
        assert active_entry_params(store._conn) == current_params()

    @pytest.mark.parametrize(
        "path,action",
        [
            ((), "retire"),  # PENDING cannot be retired
            (("reject",), "activate"),  # REJECTED cannot be activated
            (("activate",), "reject"),  # ACTIVE is retired, not rejected
            (("activate",), "shadow"),
        ],
    )
    def test_illegal_transitions(self, store, path, action):
        c = store.propose(ENTRY, "proposal.DIP_SIGMAS", 2.5)
        for step in path:
            getattr(store, step)(c.id)
        with pytest.raises(ChangeError):
            getattr(store, action)(c.id)

    def test_unknown_id(self, store):
        with pytest.raises(ChangeError, match="no change"):
            store.activate("nope")

    def test_activating_twice_is_refused_not_duplicated(self, store):
        c = store.propose(ENTRY, "proposal.DIP_SIGMAS", 2.5)
        store.activate(c.id)
        with pytest.raises(ChangeError):
            store.activate(c.id)
        assert len(store.list(status=Status.ACTIVE)) == 1


class TestChallengers:
    def test_active_rules_plus_one_change_each(self, store):
        active = store.propose(ENTRY, "proposal.TRADE_STOP_SIGMAS", 2.0)
        store.activate(active.id)
        s1 = store.propose(ENTRY, "proposal.DIP_SIGMAS", 2.5)
        s2 = store.propose(ENTRY, "proposal.TRADE_STOP_SIGMAS", 1.0)
        store.shadow(s1.id)
        store.shadow(s2.id)
        ch = dict(challengers(store._conn, ENTRY))
        assert ch[s1.id].dip_sigmas == 2.5 and ch[s1.id].trade_stop_sigmas == 2.0
        assert ch[s2.id].trade_stop_sigmas == 1.0  # its own value, over the active one

    def test_none(self, store):
        assert challengers(store._conn, SESSION) == []


NOW = datetime(2026, 9, 25, 11, 0, tzinfo=mc.MARKET_TZ)


def zone(ps=3.0, above=0):
    return RelativeZone(
        price=100.0, ps_now=ps, median=3.6, percentile=0.36, top=120.0, p25_price=90.0,
        p80_price=150.0, window_start=date(2024, 9, 25), window_end=date(2026, 9, 25),
        observations=500, sessions_above=above,
    )  # fmt: skip


def sheet(z=-2.2):
    return Sheet(
        symbol="AMZN",
        built_at=NOW,
        move=Move(price=100.0, asof=NOW.date(), day=z * 0.02, sigma=0.02, z=z),
        zone=zone(),
        zone_prev=zone(),
        candidates=["2026-09-25:C:AMZN"],
    )


class TestParamsTakeEffect:
    def test_trade_stop(self):
        p = build_proposal(sheet(), net_liq=10_000, params=current_params())
        q = build_proposal(
            sheet(), net_liq=10_000, params=current_params().__class__(
                **{**current_params().__dict__, "trade_stop_sigmas": 3.0}
            )
        )  # fmt: skip
        stop = {g.horizon: g.stop for g in p.legs}["trade"]
        wider = {g.horizon: g.stop for g in q.legs}["trade"]
        assert stop == pytest.approx(97.0) and wider == pytest.approx(94.0)

    def test_dip_trigger(self):
        base = current_params()
        loose = base.__class__(**{**base.__dict__, "dip_sigmas": 2.5})
        assert any("fell" in t for t in build_proposal(sheet(), net_liq=1e4, params=base).triggers)
        assert not any(
            "fell" in t for t in build_proposal(sheet(), net_liq=1e4, params=loose).triggers
        )

    def test_the_stamp_is_the_effective_rules(self):
        base = current_params()
        other = base.__class__(**{**base.__dict__, "dip_sigmas": 2.5})
        p = build_proposal(sheet(), net_liq=1e4, params=other)
        assert p.rules.version == entry_rules(other).version != entry_rules().version
        assert build_proposal(sheet(), net_liq=1e4).rules.version == entry_rules().version


# ── Shadow runs ───────────────────────────────────────────────────────────

SESSION_NOW = datetime(2026, 9, 23, 10, 5, tzinfo=mc.MARKET_TZ)
DIP = Mover(symbol="AAPL", name="Apple", price=94.0, prev_close=100.0, open=97.0,
            volume=30e6, avg_volume=50e6, market_cap=3e12)  # fmt: skip


class Counting:
    def __init__(self):
        self.movers = self.sigma = self.news = 0

    def memo(self):
        def movers():
            self.movers += 1
            return [DIP], None

        def sigma(syms):
            self.sigma += 1
            return {s: 0.02 for s in syms}

        def news(symbol, name, since):
            self.news += 1
            return [CatalystItem(kind="news", title="t", published_at=SESSION_NOW)], True

        return Memo(movers, sigma, news, lambda s: ([], None))


class TestScanShadows:
    def test_challengers_decide_on_the_champions_fetches(self, db, store):
        a = store.propose(SESSION, "thresholds.sigma_min", 3.5)  # AAPL is 3σ: excluded
        b = store.propose(SESSION, "thresholds.drop_min", 0.05)  # 6% drop: still in
        store.shadow(a.id)
        store.shadow(b.id)
        c = Counting()
        result, notes = scan_with_shadows(db, SESSION_NOW, memo=c.memo())
        assert c.movers == 1 and c.news == 1  # one fetch, news for the live run only
        assert len(result.new) == 1 and result.new[0].origin is Origin.LIVE
        sa = ScannerStore(challenger_path(db, a.id))
        sb = ScannerStore(challenger_path(db, b.id))
        assert sa.list() == []
        (shadowed,) = sb.list()
        assert shadowed.origin is Origin.SHADOW and not shadowed.news_checked
        assert shadowed.rules.version != result.new[0].rules.version
        sa.close()
        sb.close()
        assert len(notes) == 2

    def test_active_thresholds_run_live(self, db, store):
        a = store.propose(SESSION, "thresholds.sigma_min", 3.5)
        store.activate(a.id)
        result, notes = scan_with_shadows(db, SESSION_NOW, memo=Counting().memo())
        assert result.new == [] and notes == []

    def test_live_ledger_never_holds_a_shadow_record(self, db, store):
        a = store.propose(SESSION, "thresholds.drop_min", 0.05)
        store.shadow(a.id)
        scan_with_shadows(db, SESSION_NOW, memo=Counting().memo())
        live = ScannerStore(db)
        assert all(c.origin is Origin.LIVE for c in live.list())
        live.close()


class TestEntryShadows:
    def test_each_challenger_records_its_own_proposal(self, db, store):
        a = store.propose(ENTRY, "proposal.DIP_SIGMAS", 2.5)
        store.shadow(a.id)
        shadows = EntryShadows(db)
        shadows(sheet(), 10_000, None)
        shadows.close()
        es = EntryStore(challenger_path(db, a.id))
        (p,) = es.list()
        assert p.origin is Origin.SHADOW and not any("fell" in t for t in p.triggers)
        es.close()
        assert shadows.params == current_params()

    def test_no_challengers_records_nothing(self, db):
        shadows = EntryShadows(db)
        shadows(sheet(), 10_000, None)
        shadows.close()
        assert shadows.recorded == 0

    def test_load_and_fill(self, db, store):
        a = store.propose(ENTRY, "proposal.DIP_SIGMAS", 2.5)
        store.shadow(a.id)
        shadows = EntryShadows(db)
        shadows(sheet(), 10_000, None)
        shadows.close()
        records = load_shadow(db)
        assert records and all(r.origin == "shadow" for r in records)
        assert "proposals" in fill_shadow_outcomes(db, NOW)

    def test_nothing_on_file(self, db):
        assert fill_shadow_outcomes(db, NOW) == "no shadow records"
        assert load_shadow(db) == []


def test_a_failing_shadow_never_breaks_the_live_run(tmp_path):
    from advisor.daemon.store import DaemonStore
    from advisor.entry.run import propose_all

    daemon = DaemonStore(tmp_path / "research.db")

    def boom(*a):
        raise RuntimeError("challenger bug")

    proposals, errors = propose_all(
        daemon,
        NOW,
        symbols=["AMZN"],
        read=False,
        sheet_builder=lambda *a, **k: sheet(),
        shadow=boom,
    )
    daemon.close()
    assert len(proposals) == 1 and any("shadow failed" in e for e in errors)
