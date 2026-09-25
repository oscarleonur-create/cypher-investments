"""Entry proposals: triggers, legs, sizing, limits, blockers, tracking, the ledger."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from advisor.daemon import market_calendar as mc
from advisor.entry.proposal import (
    ENTRY_CONFIRM_SESSIONS,
    Action,
    build_proposal,
    position_stop_pct,
    triggers_for,
)
from advisor.entry.sheet import EventLine, Holding, Move, Sheet
from advisor.entry.store import EntryStore
from advisor.entry.track import Bar, fill_proposal_outcomes, score, sync_proposal_fills
from advisor.entry.zone import RelativeZone

NOW = datetime(2026, 9, 25, 11, 0, tzinfo=mc.MARKET_TZ)
NET_LIQ = 10_000.0


def zone(ps=3.0, median=3.6, pct=0.36, above=0):
    return RelativeZone(
        price=100.0,
        ps_now=ps,
        median=median,
        percentile=pct,
        top=120.0,
        p25_price=90.0,
        p80_price=150.0,
        window_start=date(2024, 9, 25),
        window_end=date(2026, 9, 25),
        observations=500,
        sessions_above=above,
    )


def mk(
    zone_now=None,
    zone_prev=None,
    z=0.0,
    sigma=0.02,
    events=(),
    candidates=(),
    holding=None,
    price=100.0,
):
    return Sheet(
        symbol="AMZN",
        built_at=NOW,
        move=Move(
            price=price,
            asof=NOW.date(),
            day=z * sigma if sigma else None,
            sigma=sigma,
            z=z if sigma else None,
        ),
        zone=zone_now,
        zone_prev=zone_prev,
        events_today=list(events),
        candidates=list(candidates),
        holding=holding,
    )


IN = zone()
OUT = zone(ps=4.0)
ENTERED = zone(above=ENTRY_CONFIRM_SESSIONS)


def reading(stance):
    return SimpleNamespace(stance=SimpleNamespace(value=stance), sentences=[])


class TestTriggers:
    def test_entering_after_a_real_stay_out(self):
        assert any("entered the zone" in t for t in triggers_for(mk(ENTERED, OUT)))

    def test_hovering_at_the_median_is_not_an_entry(self):
        """AMZN 3.6x vs 3.6x on 2026-09-22 would have fired every other day."""
        flap = zone(above=ENTRY_CONFIRM_SESSIONS - 1)
        assert triggers_for(mk(flap, OUT)) == []

    def test_two_sigma_dip_inside_the_zone(self):
        assert any("fell" in t for t in triggers_for(mk(IN, IN, z=-2.0)))

    def test_one_point_nine_sigma_is_not(self):
        assert triggers_for(mk(IN, IN, z=-1.99)) == []

    def test_a_dip_outside_the_zone_is_not(self):
        assert triggers_for(mk(OUT, OUT, z=-3.0)) == []


class TestActions:
    def test_in_zone_without_a_trigger(self):
        p = build_proposal(mk(IN, IN), net_liq=NET_LIQ)
        assert p.action is Action.IN_ZONE and p.legs == []

    def test_enter_with_a_sized_position_leg(self):
        p = build_proposal(mk(ENTERED, OUT), net_liq=NET_LIQ)
        (leg,) = p.legs
        assert p.action is Action.ENTER and leg.horizon == "position"
        stop_pct = position_stop_pct(0.02)
        assert leg.stop == pytest.approx(100 * (1 - stop_pct))
        assert leg.shares == int(NET_LIQ * 0.02 // (100 * stop_pct))
        assert leg.risk_pct == 0.02
        assert any("80th percentile" in x for x in leg.exit_rules)

    def test_cheapest_quarter_raises_the_risk_to_three(self):
        cheap = zone(pct=0.25, above=ENTRY_CONFIRM_SESSIONS)
        assert build_proposal(mk(cheap, OUT), net_liq=NET_LIQ).legs[0].risk_pct == 0.03

    def test_out_of_zone_no_setup_is_none(self):
        assert build_proposal(mk(OUT, OUT), net_liq=NET_LIQ).action is Action.NONE

    def test_setup_out_of_zone_is_a_trade_only(self):
        p = build_proposal(mk(OUT, OUT, candidates=["2026-09-25:A:AMZN"]), net_liq=NET_LIQ)
        assert p.action is Action.ENTER and [leg.horizon for leg in p.legs] == ["trade"]
        leg = p.legs[0]
        assert leg.stop == pytest.approx(97.0) and "next session" in leg.exit_rules[1]

    def test_setup_in_zone_gets_both_legs(self):
        p = build_proposal(mk(IN, IN, candidates=["x"]), net_liq=NET_LIQ)
        assert sorted(leg.horizon for leg in p.legs) == ["position", "trade"]
        trade = next(leg for leg in p.legs if leg.horizon == "trade")
        assert p.risk_pct == pytest.approx(0.03) and trade.risk_pct == pytest.approx(0.01)

    def test_position_using_the_whole_cap_leaves_no_trade(self):
        cheap = zone(pct=0.1)
        p = build_proposal(
            mk(cheap, cheap, candidates=["x"]), net_liq=NET_LIQ, reading=reading("CONSTRUCTIVE")
        )
        assert [leg.horizon for leg in p.legs] == ["position"]
        assert p.risk_pct == pytest.approx(0.03)
        assert any("3% cap" in g for g in p.gaps)

    def test_a_share_riskier_than_the_budget_is_still_proposed(self):
        """A $2,000 share with a 17% stop risks ~$342; 2% of $7,966 is $159."""
        p = build_proposal(mk(ENTERED, OUT, price=2000.0, sigma=0.027), net_liq=7_966.0)
        (leg,) = p.legs
        assert p.action is Action.ENTER and leg.shares == 0
        assert any("smallest position exceeds" in n for n in leg.notes)

    def test_one_share_larger_than_the_room_is_not_being_at_the_limit(self):
        """Not held, yet one $2,000 share is more than 20% of a $7,966 book."""
        p = build_proposal(mk(ENTERED, OUT, price=2000.0, sigma=0.005), net_liq=7_966.0)
        (leg,) = p.legs
        assert leg.shares == 0 and any("left under the 20%" in n for n in leg.notes)
        assert not any("already" in b for b in p.blockers)

    def test_no_zone_no_setup_cannot_say(self):
        assert build_proposal(mk(None, None), net_liq=NET_LIQ).action is Action.CANNOT_SAY

    def test_no_price_cannot_say(self):
        s = mk(IN, IN)
        s.move = None
        p = build_proposal(s, net_liq=NET_LIQ)
        assert p.action is Action.CANNOT_SAY and "no price" in p.blockers

    def test_held_name_is_add(self):
        held = Holding(quantity=2, weight=0.05, unrealized=0.1)
        assert build_proposal(mk(ENTERED, OUT, holding=held), net_liq=NET_LIQ).action is Action.ADD


class TestThesis:
    def with_thesis(self, status, broken=()):
        s = mk(ENTERED, OUT)
        s.thesis, s.thesis_broken = status, list(broken)
        return s

    def test_intact_thesis_adds_a_point_and_raises_the_cap(self):
        p = build_proposal(self.with_thesis("intact"), net_liq=NET_LIQ)
        assert p.legs[0].risk_pct == pytest.approx(0.03)
        assert any("thesis" in r.text for r in p.reasons)

    def test_intact_thesis_cap_is_four_percent(self):
        s = self.with_thesis("intact")
        s.zone = zone(pct=0.1, above=ENTRY_CONFIRM_SESSIONS)  # cheapest quarter: 3% base
        s.candidates = ["x"]
        p = build_proposal(s, net_liq=NET_LIQ, reading=reading("CONSTRUCTIVE"))
        assert p.risk_pct == pytest.approx(0.04)  # 3 + 1 + 1 capped at 4, no trade room

    def test_broken_thesis_waits_and_names_the_rule(self):
        p = build_proposal(
            self.with_thesis("broken", ["Una emisión sobre 5% rompe la tesis"]), net_liq=NET_LIQ
        )
        assert p.action is Action.WAIT and "emisión" in p.blockers[0]

    def test_no_thesis_keeps_the_three_percent_cap(self):
        s = mk(zone(pct=0.1, above=ENTRY_CONFIRM_SESSIONS), OUT, candidates=["x"])
        p = build_proposal(s, net_liq=NET_LIQ, reading=reading("CONSTRUCTIVE"))
        assert p.risk_pct == pytest.approx(0.03)


class TestLimitsAndBlockers:
    def test_tier_a_event_turns_enter_into_wait(self):
        ev = EventLine(ts=NOW, kind="FILING_DILUTION", tier="A", text="424B5")
        p = build_proposal(mk(ENTERED, OUT, events=[ev]), net_liq=NET_LIQ)
        assert p.action is Action.WAIT and "tier-A" in p.blockers[0]

    def test_at_risk_reading_turns_enter_into_wait(self):
        p = build_proposal(mk(ENTERED, OUT), net_liq=NET_LIQ, reading=reading("AT_RISK"))
        assert p.action is Action.WAIT and p.stance == "AT_RISK"

    def test_book_limit_caps_the_add(self):
        held = Holding(quantity=10, weight=0.19, unrealized=0.0)
        p = build_proposal(
            mk(zone(pct=0.1, above=9), OUT, sigma=0.01, holding=held), net_liq=NET_LIQ
        )
        (leg,) = p.legs
        assert leg.notional <= 0.01 * NET_LIQ + 1e-6 and "20% book limit" in leg.notes[0]

    def test_at_the_limit_no_add(self):
        held = Holding(quantity=10, weight=0.20, unrealized=0.0)
        p = build_proposal(mk(ENTERED, OUT, holding=held), net_liq=NET_LIQ)
        assert p.legs == [] and "20% limit" in p.blockers[0]

    def test_unknown_net_liq_still_decides_but_does_not_size(self):
        p = build_proposal(mk(ENTERED, OUT), net_liq=None)
        assert p.action is Action.ENTER and p.legs[0].shares == 0
        assert any("net liq unknown" in g for g in p.gaps)

    def test_zero_net_liq(self):
        p = build_proposal(mk(ENTERED, OUT), net_liq=0.0)
        assert p.legs[0].shares == 0 and any("net liq unknown" in g for g in p.gaps)

    def test_no_sigma_no_position_stop(self):
        p = build_proposal(mk(ENTERED, OUT, sigma=None), net_liq=NET_LIQ)
        assert p.legs == [] and any("volatility" in g for g in p.gaps)

    @pytest.mark.parametrize("sigma,expected", [(0.005, 0.08), (0.02, 0.1265), (0.09, 0.25)])
    def test_position_stop_bounds(self, sigma, expected):
        assert position_stop_pct(sigma) == pytest.approx(expected, abs=1e-3)


class TestTracking:
    def test_score_horizons_and_stops(self):
        p = build_proposal(mk(IN, IN, candidates=["x"]), net_liq=NET_LIQ)
        days = [
            date(2026, 9, 25),
            date(2026, 9, 28),
            date(2026, 9, 29),
            date(2026, 9, 30),
            date(2026, 10, 1),
            date(2026, 10, 2),
        ]
        bars = [Bar(d, 102, 96, 101) for d in days]  # the trade stop at 97 is touched
        out = score(p, bars, datetime(2026, 10, 2, 17, 0, tzinfo=mc.MARKET_TZ))
        assert out["next_close"] == pytest.approx(0.01) and out["d5"] == pytest.approx(0.01)
        assert out["trade_stop"] == 1.0
        assert "d20" not in out and "pos_stop20" not in out  # not knowable yet

    def test_nothing_before_the_next_close(self):
        p = build_proposal(mk(IN, IN), net_liq=NET_LIQ)
        bars = [Bar(date(2026, 9, 25), 101, 99, 100)]
        assert score(p, bars, NOW) == {}

    def test_no_price_no_score(self):
        s = mk(IN, IN)
        s.move = None
        assert score(build_proposal(s, net_liq=NET_LIQ), [Bar(NOW.date(), 1, 1, 1)], NOW) == {}


@pytest.fixture
def store(tmp_path: Path):
    s = EntryStore(tmp_path / "research.db")
    yield s
    s.close()


class TestLedger:
    def test_first_wins_per_action(self, store):
        p = build_proposal(mk(IN, IN), net_liq=NET_LIQ)
        assert store.add(p) and not store.add(p)
        assert len(store.list(session=NOW.date())) == 1

    def test_wait_then_enter_are_two_rows(self, store):
        ev = EventLine(ts=NOW, kind="FILING_DILUTION", tier="A", text="424B5")
        store.add(build_proposal(mk(ENTERED, OUT, events=[ev]), net_liq=NET_LIQ))
        store.add(build_proposal(mk(ENTERED, OUT), net_liq=NET_LIQ))
        assert {p.action for p in store.list()} == {Action.WAIT, Action.ENTER}

    def test_outcomes_never_overwritten(self, store):
        p = build_proposal(mk(IN, IN), net_liq=NET_LIQ)
        store.add(p)
        later = datetime(2026, 9, 28, 17, 0, tzinfo=mc.MARKET_TZ)
        bars = [Bar(date(2026, 9, 28), 110, 100, 105)]
        fill_proposal_outcomes(store, later, bars_fn=lambda *a: bars)
        fill_proposal_outcomes(store, later, bars_fn=lambda *a: [Bar(date(2026, 9, 28), 1, 1, 50)])
        assert store.list()[0].outcomes["next_close"] == pytest.approx(0.05)

    def test_same_day_is_not_scored(self, store):
        store.add(build_proposal(mk(IN, IN), net_liq=NET_LIQ))
        r = fill_proposal_outcomes(store, NOW, bars_fn=lambda *a: [])
        assert r.pending == 0

    def test_broker_fill_marks_the_proposal_taken(self, store, tmp_path):
        from advisor.scanner.journal import Fill
        from advisor.scanner.store import ScannerStore

        p = build_proposal(mk(ENTERED, OUT), net_liq=NET_LIQ)
        store.add(p)
        scanner = ScannerStore(tmp_path / "research.db")
        fill = Fill(
            underlying="AMZN",
            symbol="AMZN",
            action="Buy to Open",
            quantity=1,
            price=100.5,
            executed_at=NOW,
            account="x",
            instrument="Equity",
        )
        try:
            assert sync_proposal_fills(store, scanner, [NOW.date()], fetch=lambda a, b: [fill]) == 1
            assert sync_proposal_fills(store, scanner, [NOW.date()], fetch=lambda a, b: [fill]) == 0
            assert scanner.latest_decisions()[p.id].fill_price == 100.5
        finally:
            scanner.close()


class TestRun:
    def test_model_asked_only_where_a_decision_is_on_the_table(self, tmp_path):
        from advisor.daemon.book import BookSnapshot
        from advisor.daemon.store import DaemonStore
        from advisor.entry.run import propose_all

        daemon = DaemonStore(tmp_path / "research.db")
        daemon.save_book(BookSnapshot(net_liq=NET_LIQ))
        sheets = {"QUIET": mk(IN, IN), "GO": mk(ENTERED, OUT)}
        asked = []

        def builder(store, sym, now, scanner_store=None):
            s = sheets[sym].model_copy(update={"symbol": sym})
            return s

        def reader(sym):
            asked.append(sym)
            return SimpleNamespace(
                status=SimpleNamespace(value="OK"),
                stance=SimpleNamespace(value="CONSTRUCTIVE"),
                sentences=[],
            )

        proposals, errors = propose_all(
            daemon, NOW, symbols=["QUIET", "GO"], reader=reader, sheet_builder=builder
        )
        daemon.close()
        assert asked == ["GO"] and errors == []
        go = next(p for p in proposals if p.symbol == "GO")
        assert go.stance == "CONSTRUCTIVE" and go.legs[0].risk_pct == pytest.approx(0.03)

    def test_reading_failure_keeps_the_deterministic_proposal(self, tmp_path):
        from advisor.daemon.store import DaemonStore
        from advisor.entry.run import propose_all

        daemon = DaemonStore(tmp_path / "research.db")

        def boom(sym):
            raise RuntimeError("model down")

        proposals, errors = propose_all(
            daemon,
            NOW,
            symbols=["GO"],
            reader=boom,
            sheet_builder=lambda st, sym, now, scanner_store=None: mk(ENTERED, OUT),
        )
        daemon.close()
        assert proposals[0].action is Action.ENTER and proposals[0].stance is None
        assert "model down" in errors[0]

    def test_no_book_no_default_universe(self, tmp_path):
        from advisor.daemon.store import DaemonStore
        from advisor.entry.run import propose_all

        daemon = DaemonStore(tmp_path / "research.db")
        proposals, errors = propose_all(daemon, NOW)
        daemon.close()
        assert proposals == [] and "no book" in errors[0]
