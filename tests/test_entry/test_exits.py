"""Exit calls on held names: stop from cost, filings, thesis, rich P/S, concentration."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from advisor.daemon import market_calendar as mc
from advisor.entry.exits import exit_calls, strongest
from advisor.entry.proposal import Action, build_proposal, position_stop_pct
from advisor.entry.sheet import EventLine, Holding, Move, Sheet
from advisor.entry.store import EntryStore
from advisor.entry.zone import RelativeZone

NOW = datetime(2026, 9, 25, 16, 30, tzinfo=mc.MARKET_TZ)
NET_LIQ = 7_957.51


def zone(pct=0.5, ps=3.0, median=3.0, short=False):
    return RelativeZone(
        price=100.0,
        ps_now=ps,
        median=median,
        percentile=pct,
        top=100.0,
        p25_price=80.0,
        p80_price=130.0,
        window_start=date(2024, 9, 25),
        window_end=date(2026, 9, 25),
        observations=500,
        short=short,
    )


def held(
    price=100.0,
    cost=100.0,
    qty=10.0,
    weight=0.10,
    sigma=0.03,
    z=None,
    thesis=None,
    broken=(),
    filings=(),
    earnings=None,
):
    return Sheet(
        symbol="TE",
        built_at=NOW,
        move=Move(price=price, asof=NOW.date(), day=0.0, sigma=sigma, z=0.0),
        holding=Holding(
            quantity=qty,
            weight=weight,
            unrealized=price / cost - 1 if cost else 0.0,
            cost=cost,
        ),
        zone=z,
        thesis=thesis,
        thesis_broken=list(broken),
        filings=list(filings),
        next_earnings=earnings,
        earnings_in=5 if earnings else None,
    )


def filing(items=(), kind="FILING_OTHER", day=24):
    return EventLine(
        ts=datetime(2026, 9, day, 16, 5, tzinfo=mc.MARKET_TZ),
        kind=kind,
        tier="A",
        text="T1 Energy 8-K — Item 1.03 Bankruptcy or Receivership",
        items=list(items),
    )


class TestStopFromCost:
    """CCXI 2026-09-25: -29.5% from cost, σ 4.77%/day → stop capped at 25%: EXIT."""

    def test_past_the_stop_is_exit_with_all_shares(self):
        calls, _, _ = exit_calls(held(price=70.5, sigma=0.0477, qty=70), net_liq=NET_LIQ)
        (c,) = calls
        assert c.action == "EXIT" and c.rule == "stop" and c.shares == 70
        assert "-29.5% from your average cost $100.00" in c.why
        assert "25.0% below cost, at $75.00" in c.why
        assert c.evidence and c.would_change

    def test_exactly_at_the_stop_exits(self):
        stop = position_stop_pct(0.03)
        calls, _, _ = exit_calls(held(price=100 * (1 - stop)), net_liq=NET_LIQ)
        assert [c.action for c in calls] == ["EXIT"]

    def test_just_inside_the_stop_holds_with_its_reason(self):
        stop = position_stop_pct(0.03)
        calls, reasons, _ = exit_calls(held(price=100 * (1 - stop) + 0.01), net_liq=NET_LIQ)
        assert calls == []
        assert "volatility stop is" in reasons[0].text

    def test_a_gain_is_never_a_stop(self):
        calls, reasons, _ = exit_calls(held(price=130.0), net_liq=NET_LIQ)
        assert calls == [] and "+30.0% from your average cost" in reasons[0].text

    def test_calm_name_stop_floors_at_eight_percent(self):
        calls, _, _ = exit_calls(held(price=91.9, sigma=0.005), net_liq=NET_LIQ)
        assert [c.rule for c in calls] == ["stop"]

    def test_no_sigma_no_stop_but_a_gap(self):
        calls, reasons, gaps = exit_calls(held(price=50.0, sigma=None), net_liq=NET_LIQ)
        assert calls == [] and "position stop cannot be set" in gaps[0]
        assert "-50.0%" in reasons[0].text

    @pytest.mark.parametrize("cost", [None, 0.0])
    def test_no_cost_basis_is_a_gap(self, cost):
        calls, _, gaps = exit_calls(held(price=50.0, cost=cost), net_liq=NET_LIQ)
        assert calls == [] and "no cost basis" in gaps[0]

    def test_short_position_not_modeled(self):
        calls, _, gaps = exit_calls(held(qty=-5), net_liq=NET_LIQ)
        assert calls == [] and "longs only" in gaps[0]


class TestFilings:
    @pytest.mark.parametrize("item", ["1.03", "2.04", "3.01", "4.02"])
    def test_exit_items(self, item):
        calls, _, _ = exit_calls(held(filings=[filing([item, "9.01"])]), net_liq=NET_LIQ)
        (c,) = calls
        assert c.action == "EXIT" and c.rule == "filing" and c.shares == 10
        assert f"8-K item {item}" in c.why and "filed 2026-09-24" in c.why
        assert c.evidence[0].source.startswith("SEC EDGAR")

    def test_auditor_change_is_review_unsized(self):
        (c,) = exit_calls(held(filings=[filing(["4.01"])]), net_liq=NET_LIQ)[0]
        assert c.action == "REVIEW" and c.shares is None

    def test_foreign_filer_by_kind(self):
        """A 6-K has no item codes; the classifier's kind carries it."""
        (c,) = exit_calls(held(filings=[filing(kind="FILING_BANKRUPTCY")]), net_liq=NET_LIQ)[0]
        assert c.action == "EXIT" and "FILING_BANKRUPTCY" in c.why

    def test_the_same_item_twice_is_one_call(self):
        """An 8-K and its amendment: one call, not two."""
        twice = [filing(["1.03"], day=24), filing(["1.03"], day=25)]
        assert len(exit_calls(held(filings=twice), net_liq=NET_LIQ)[0]) == 1

    def test_ordinary_filings_do_nothing(self):
        calls, _, _ = exit_calls(held(filings=[filing(["2.02", "7.01", "8.01"])]), net_liq=NET_LIQ)
        assert calls == []


class TestReviews:
    def test_broken_thesis_is_review_not_exit(self):
        """AAOI: the user's 5% dilution rule. The user answers it (decision 2026-09-26)."""
        rule = "Any equity raise above 5% of market cap breaks the capacity-funding story"
        (c,) = exit_calls(held(thesis="broken", broken=[rule]), net_liq=NET_LIQ)[0]
        assert c.action == "REVIEW" and c.rule == "thesis" and c.shares is None
        assert rule in c.why and c.evidence[0].text == rule

    def test_rich_at_the_80th_percentile_is_review(self):
        (c,) = exit_calls(held(z=zone(pct=0.80, ps=8.1, median=3.2)), net_liq=NET_LIQ)[0]
        assert c.action == "REVIEW" and c.rule == "rich" and c.shares is None
        assert "percentile 80%" in c.why and "$130.00" in c.why

    def test_just_below_the_80th_is_not(self):
        assert exit_calls(held(z=zone(pct=0.79)), net_liq=NET_LIQ)[0] == []

    def test_short_history_is_labeled(self):
        (c,) = exit_calls(held(z=zone(pct=0.9, short=True)), net_liq=NET_LIQ)[0]
        assert "[short history]" in c.why


class TestConcentration:
    def test_spcx_trim_back_to_twenty_percent(self):
        """SPCX 2026-09-25: 11 shares at $148.68, 20.55% of $7,957.51 → sell 1."""
        s = held(price=148.68, cost=128.18, qty=11, weight=11 * 148.68 / NET_LIQ)
        (c,) = exit_calls(s, net_liq=NET_LIQ)[0]
        assert c.action == "TRIM" and c.shares == 1
        assert "20.6% of the book" in c.why and "brings it to 18.7%" in c.why

    def test_exactly_at_the_limit_is_not_a_trim(self):
        assert exit_calls(held(weight=0.20), net_liq=NET_LIQ)[0] == []

    def test_no_net_liq_no_trim(self):
        assert exit_calls(held(weight=0.5), net_liq=0)[0] == []

    def test_trim_never_sells_more_than_held(self):
        (c,) = exit_calls(held(weight=0.9, qty=1), net_liq=NET_LIQ)[0]
        assert c.shares == 1


class TestProposal:
    def test_strongest_call_wins(self):
        assert strongest([]) is None
        calls, _, _ = exit_calls(
            held(price=60.0, weight=0.3, thesis="broken", broken=["x"]), net_liq=NET_LIQ
        )
        assert {c.action for c in calls} == {"EXIT", "TRIM", "REVIEW"}
        assert strongest(calls) == "EXIT"

    def test_exit_replaces_an_add_and_drops_its_legs(self):
        """A held name in zone with a dip would be ADD; past its stop it is EXIT."""
        entered = zone(pct=0.3).model_copy(update={"sessions_above": 10})
        prev = zone(pct=0.6, ps=4.0)
        s = held(price=60.0, z=entered).model_copy(update={"zone_prev": prev})
        p = build_proposal(s, net_liq=NET_LIQ)
        assert p.action is Action.EXIT and p.legs == []
        assert p.reasons[0].source == "exit rule: stop"
        assert p.exits[0]["shares"] == 10

    def test_held_without_a_zone_is_still_judged(self):
        """SPCX has no zone (59 sessions); its concentration must still be seen."""
        s = held(price=148.68, cost=128.18, qty=11, weight=0.2055)
        p = build_proposal(s, net_liq=NET_LIQ)
        assert p.action is Action.TRIM

    def test_quiet_holding_is_hold_with_reasons(self):
        p = build_proposal(held(z=zone(pct=0.5), earnings=date(2026, 10, 2)), net_liq=NET_LIQ)
        assert p.action is Action.HOLD
        texts = " | ".join(r.text for r in p.reasons)
        assert "volatility stop" in texts and "P/S" in texts and "next results" in texts

    def test_hold_with_nothing_to_say_stays_cannot_say(self):
        s = held(cost=None, sigma=None)
        assert build_proposal(s, net_liq=NET_LIQ).action is Action.CANNOT_SAY

    def test_thesis_review_replaces_the_thesis_blocker(self):
        p = build_proposal(held(z=zone(), thesis="broken", broken=["rule"]), net_liq=NET_LIQ)
        assert p.action is Action.REVIEW
        assert not any(b.startswith("a rule of your thesis") for b in p.blockers)

    def test_not_held_names_are_untouched(self):
        s = held(z=zone()).model_copy(update={"holding": None})
        assert build_proposal(s, net_liq=NET_LIQ).action is Action.IN_ZONE


class TestRationaleIsRequired:
    def test_ledger_refuses_an_action_without_reasons(self, tmp_path):
        p = build_proposal(held(price=60.0), net_liq=NET_LIQ).model_copy(update={"reasons": []})
        store = EntryStore(tmp_path / "research.db")
        try:
            with pytest.raises(ValueError, match="without a rationale"):
                store.add(p)
        finally:
            store.close()

    def test_every_exit_call_has_why_evidence_and_what_would_change(self):
        s = held(
            price=60.0,
            weight=0.3,
            thesis="broken",
            broken=["rule"],
            z=zone(pct=0.95),
            filings=[filing(["1.03", "4.01"])],
        )
        calls, _, _ = exit_calls(s, net_liq=NET_LIQ)
        assert len(calls) == 6
        for c in calls:
            assert c.why and c.evidence and c.would_change

    def test_a_recorded_exit_round_trips(self, tmp_path):
        p = build_proposal(held(price=60.0), net_liq=NET_LIQ)
        store = EntryStore(tmp_path / "research.db")
        try:
            assert store.add(p) is True and store.add(p) is False  # once per id
            (back,) = store.list(session=p.session)
            assert back.action is Action.EXIT and back.exits[0]["rule"] == "stop"
        finally:
            store.close()


def test_intact_thesis_is_said_once():
    """NBIS 2026-09-25 printed it twice: once from the entry side, once from the exit side."""
    p = build_proposal(held(z=zone(), thesis="intact"), net_liq=NET_LIQ)
    assert p.action is Action.HOLD
    assert sum("intact" in r.text for r in p.reasons) == 1
