"""Every record the scanner and the entry module write carries its rules and inputs."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date, datetime

import pytest
from advisor.daemon import market_calendar as mc
from advisor.entry import proposal as proposal_mod
from advisor.entry.proposal import Proposal, build_proposal, position_stop_pct
from advisor.entry.ruleset import entry_rules
from advisor.entry.sheet import EventLine, Holding, Move, Sheet
from advisor.entry.store import EntryStore
from advisor.entry.zone import Absolute, RelativeZone
from advisor.learning.rules import Origin
from advisor.learning.store import RuleStore
from advisor.scanner import detect
from advisor.scanner.models import CatalystItem, Mover
from advisor.scanner.premarket import PremarketQuote, run_premarket_scan
from advisor.scanner.ruleset import premarket_rules, session_rules
from advisor.scanner.scan import run_scan
from advisor.scanner.store import ScannerStore

NOW = datetime(2026, 9, 23, 10, 5, tzinfo=mc.MARKET_TZ)
DIP = Mover(
    symbol="AAPL",
    name="Apple",
    price=94.0,
    prev_close=100.0,
    open=97.0,
    volume=30_000_000,
    avg_volume=50_000_000,
    market_cap=3e12,
)


def _news(symbol, name, since):
    return [CatalystItem(kind="news", title="headline", published_at=NOW)], True


@pytest.fixture
def store(tmp_path):
    s = ScannerStore(tmp_path / "research.db")
    yield s
    s.close()


def _scan(store, **kw):
    return run_scan(
        store,
        NOW,
        fetch_movers=lambda: ([DIP], None),
        fetch_sigma=lambda syms: {s: 0.02 for s in syms},
        fetch_catalysts=_news,
        fetch_peers=lambda s: ([], None),
        **kw,
    )


class TestScanner:
    def test_session_candidate_carries_the_rules_that_found_it(self, store):
        r = _scan(store)
        (c,) = r.new
        assert c.rules.version == session_rules().version
        assert c.rules.ruleset == "scanner.session"
        assert c.origin is Origin.LIVE
        assert c.volume == 30_000_000
        assert [v.version for v in RuleStore(store._conn).list()] == [c.rules.version]

    def test_the_stamp_survives_the_round_trip(self, store):
        (c,) = _scan(store).new
        back = store.get(c.id)
        assert back.rules.version == c.rules.version and back.rules.code == c.rules.code

    def test_thresholds_passed_in_are_the_ones_stamped(self, store):
        looser = replace(detect.DEFAULT, sigma_min=1.0)
        (c,) = _scan(store, thresholds=looser).new
        assert c.rules.version == session_rules(looser).version
        assert c.rules.version != session_rules().version

    def test_closed_market_writes_no_version(self, store):
        r = run_scan(store, datetime(2026, 9, 26, 10, 5, tzinfo=mc.MARKET_TZ))  # Saturday
        assert not r.ran and RuleStore(store._conn).list() == []

    def test_premarket_candidate_carries_premarket_rules(self, store):
        q = PremarketQuote(
            symbol="AMD",
            name="AMD",
            last=106.0,
            prev_close=100.0,
            premarket_volume=80_000,
            avg_volume=1_000_000,
            market_cap=200e9,
        )
        r = run_premarket_scan(
            store,
            datetime(2026, 9, 24, 8, 40, tzinfo=mc.MARKET_TZ),
            fetch_watchlist=lambda: (["AMD"], None),
            fetch_quotes=lambda s, n: {"AMD": q},
            fetch_peer_move=lambda s, n: ([], None),
            fetch_catalysts=_news,
        )
        (c,) = r.new
        assert c.rules.ruleset == "scanner.premarket"
        assert c.rules.version == premarket_rules().version


# ── Entry ─────────────────────────────────────────────────────────────────

ENTRY_NOW = datetime(2026, 9, 25, 11, 0, tzinfo=mc.MARKET_TZ)


def zone(ps=3.0, above=0, pct=0.36):
    return RelativeZone(
        price=100.0,
        ps_now=ps,
        median=3.6,
        percentile=pct,
        top=120.0,
        p25_price=90.0,
        p80_price=150.0,
        window_start=date(2024, 9, 25),
        window_end=date(2026, 9, 25),
        observations=500,
        sessions_above=above,
    )


def sheet(**kw):
    base = dict(
        symbol="AMZN",
        built_at=ENTRY_NOW,
        move=Move(price=100.0, asof=ENTRY_NOW.date(), day=-0.05, sigma=0.02, z=-2.5),
        zone=zone(),
        zone_prev=zone(),
    )
    base.update(kw)
    return Sheet(**base)


class TestProposalFeatures:
    def test_features_are_the_sheet_as_numbers(self):
        ev = EventLine(ts=ENTRY_NOW, kind="FILING_DILUTION", tier="A", text="424B5")
        s = sheet(
            events_today=[ev],
            events_week=3,
            holding=Holding(quantity=5, weight=0.07, unrealized=-0.1),
            candidates=["2026-09-25:C:AMZN", "2026-09-25:A@pre:AMZN"],
            context=Absolute(
                price=100.0,
                readings=[("generic", 0.2, 0.05), ("own", 0.3, 0.11)],
                delivered=0.11,
                consensus=0.13,
            ),
            thesis="intact",
        )
        f = build_proposal(s, net_liq=10_000).features
        assert f["move_z"] == -2.5 and f["sigma"] == 0.02 and f["price"] == 100.0
        assert f["in_zone"] is True and f["prev_in_zone"] is True
        assert f["ps"] == 3.0 and f["ps_median"] == 3.6 and f["ps_percentile"] == 0.36
        assert f["zone_distance"] == pytest.approx(3.0 / 3.6 - 1)
        assert f["setups"] == "A@pre,C"
        assert f["events_today"] == 1 and f["tier_a_today"] == 1 and f["events_week"] == 3
        assert f["held"] is True and f["weight"] == 0.07 and f["thesis"] == "intact"
        assert f["delivered_growth"] == 0.11 and f["consensus_growth"] == 0.13
        assert f["required_low"] == 0.05 and f["required_high"] == 0.11

    def test_nothing_known_is_none_not_zero(self):
        s = Sheet(symbol="NBIS", built_at=ENTRY_NOW)
        p = build_proposal(s, net_liq=None)
        assert p.action.value == "CANNOT_SAY"
        f = p.features
        for key in ("price", "move_z", "sigma", "in_zone", "ps", "zone_distance", "setups"):
            assert f[key] is None, key
        assert f["held"] is False and f["events_today"] == 0

    def test_features_do_not_restate_the_decision(self):
        f = build_proposal(sheet(), net_liq=10_000).features
        assert not {"action", "blockers", "legs", "risk", "stance"} & set(f)

    def test_every_proposal_is_stamped_whatever_its_action(self):
        for s in (sheet(), sheet(zone=zone(ps=4.0)), Sheet(symbol="X", built_at=ENTRY_NOW)):
            p = build_proposal(s, net_liq=10_000)
            assert p.rules.version == entry_rules().version, p.action

    def test_a_constant_changed_is_stamped_as_a_new_version(self, monkeypatch):
        before = build_proposal(sheet(), net_liq=10_000).rules.version
        monkeypatch.setattr(proposal_mod, "DIP_SIGMAS", 3.0)
        after = build_proposal(sheet(), net_liq=10_000).rules.version
        assert before != after


class TestLiftedConstants:
    """The literals lifted into named constants behave exactly as before."""

    def test_position_stop_is_still_two_sigma_over_two_weeks(self):
        assert position_stop_pct(0.02) == pytest.approx(2 * 0.02 * 10**0.5)
        assert position_stop_pct(0.001) == 0.08 and position_stop_pct(0.2) == 0.25

    def test_stop_basis_text_is_unchanged(self):
        p = build_proposal(sheet(), net_liq=10_000)
        (leg,) = [leg for leg in p.legs if leg.horizon == "position"]
        assert leg.stop_basis.startswith("2·σ·√10 = ")
        assert leg.stop_basis.endswith("kept in 8–25%")

    def test_cheap_quarter_boundary_is_inclusive(self):
        def position_risk(pct):
            p = build_proposal(sheet(zone=zone(pct=pct)), net_liq=10_000)
            return next(g.risk_pct for g in p.legs if g.horizon == "position")

        assert position_risk(0.25) == 0.03
        assert position_risk(0.2501) == 0.02


class TestEntryLedger:
    def test_proposal_registers_its_version(self, tmp_path):
        es = EntryStore(tmp_path / "research.db")
        p = build_proposal(sheet(), net_liq=10_000)
        assert es.add(p)
        (back,) = es.list()
        assert back.rules.version == p.rules.version and back.features == p.features
        assert [v.ruleset for v in RuleStore(es._conn).list()] == ["entry"]
        es.close()

    def test_a_pre_registry_proposal_still_loads(self):
        p = build_proposal(sheet(), net_liq=10_000)
        payload = json.loads(p.model_dump_json())
        for key in ("features", "rules", "origin"):
            payload.pop(key)
        old = Proposal.model_validate(payload)
        assert old.rules is None and old.features == {} and old.origin is Origin.LIVE

    def test_counts_span_both_ledgers(self, tmp_path):
        path = tmp_path / "research.db"
        ss, es = ScannerStore(path), EntryStore(path)
        _scan(ss)
        es.add(build_proposal(sheet(), net_liq=10_000))
        conn = sqlite3.connect(str(path))
        counts = RuleStore(conn).record_counts()
        assert counts["candidates"] == {session_rules().version: 1}
        assert counts["proposals"] == {entry_rules().version: 1}
        conn.close()
        ss.close()
        es.close()
