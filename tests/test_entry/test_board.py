"""The tracking board: the call, the risk and the close, per name."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import EQUITY, EQUITY_OPTION, BookSnapshot, Position
from advisor.entry.board import build_board, held_risk, planned_risk, timeline
from advisor.entry.proposal import Action, Leg, Proposal, Reason, position_stop_pct
from advisor.learning.trades import Book, Trade

ET = mc.MARKET_TZ
FRI = date(2026, 9, 25)


def prop(symbol="AMZN", action=Action.IN_ZONE, session=FRI, hour=10, price=100.0, **kw):
    return Proposal(
        symbol=symbol,
        session=session,
        built_at=datetime.combine(session, datetime.min.time(), tzinfo=ET).replace(hour=hour),
        action=action,
        price=price,
        reasons=[Reason(text="r", source="s")] if action.value != "IN_ZONE" else [],
        features={"sigma": 0.03, **kw.pop("features", {})},
        **kw,
    )


def pos(symbol="AAOI", qty=10.0, cost=100.0, price=90.0, instrument=EQUITY):
    return Position(
        account="A",
        symbol=symbol,
        underlying=symbol,
        instrument=instrument,
        quantity=qty,
        avg_open_price=cost,
        mark_price=price,
        close_price=price,
        multiplier=1.0 if instrument == EQUITY else 100.0,
    )


def trade(symbol="AAOI", opened=FRI, closed=None, pnl=None, qty=5.0):
    at = datetime.combine(opened, datetime.min.time(), tzinfo=ET).replace(hour=10)
    return Trade(
        account="A",
        underlying=symbol,
        symbol=symbol,
        instrument="Equity",
        direction="long",
        opened_at=at,
        closed_at=at + timedelta(days=1) if closed else None,
        entry_session=opened,
        exit_session=closed,
        quantity=qty,
        entry_price=100.0,
        exit_price=110.0 if closed else None,
        pnl=pnl,
        book=Book.HOLD,
    )


def book(*positions, net_liq=10_000.0):
    return BookSnapshot(positions=list(positions), net_liq=net_liq)


class TestHeldRisk:
    def test_the_stop_is_the_exit_rules_stop_from_cost(self):
        r = held_risk(10, 100.0, 95.0, 0.03, 10_000)
        pct = position_stop_pct(0.03)
        assert r.stop == pytest.approx(100 * (1 - pct))
        assert r.to_stop == pytest.approx(95 / r.stop - 1)
        assert r.at_risk == pytest.approx((95 - r.stop) * 10)
        assert r.at_risk_pct == pytest.approx(r.at_risk / 10_000)
        assert not r.past_stop

    def test_past_the_stop_risks_nothing_more_and_says_so(self):
        r = held_risk(10, 100.0, 50.0, 0.03, 10_000)
        assert r.past_stop and r.at_risk == 0 and r.to_stop < 0

    def test_exactly_at_the_stop_is_past_it(self):
        stop = 100 * (1 - position_stop_pct(0.03))
        assert held_risk(10, 100.0, stop, 0.03, 10_000).past_stop

    def test_no_sigma_no_stop(self):
        r = held_risk(10, 100.0, 95.0, None, 10_000)
        assert r.stop is None and "no volatility" in r.stop_basis

    def test_no_price_no_distance(self):
        r = held_risk(10, 100.0, None, 0.03, 10_000)
        assert r.stop is not None and r.to_stop is None and r.at_risk is None

    def test_zero_net_liq_no_share_of_it(self):
        r = held_risk(10, 100.0, 95.0, 0.03, 0.0)
        assert r.at_risk is not None and r.at_risk_pct is None


class TestPlannedRisk:
    def test_the_position_leg_comes_first(self):
        legs = [
            Leg(horizon="trade", entry=100, stop=97, stop_basis="t", risk_pct=0.02, shares=5,
                notional=500),
            Leg(horizon="position", entry=100, stop=90, stop_basis="p", risk_pct=0.02, shares=2,
                notional=200, target=130),
        ]  # fmt: skip
        r = planned_risk(prop(action=Action.ENTER, legs=legs))
        assert r.basis == "planned" and r.stop == 90 and r.target == 130
        assert r.at_risk == pytest.approx(20) and r.at_risk_pct == 0.02

    def test_no_legs_no_plan(self):
        assert planned_risk(prop()) is None


class TestTimeline:
    def test_the_last_call_of_each_session(self):
        ps = [
            prop(action=Action.ENTER, hour=10),
            prop(action=Action.WAIT, hour=14),
            prop(session=date(2026, 9, 24), action=Action.NONE),
        ]
        tl = timeline(ps)
        assert [(p.session, p.action) for p in tl] == [
            (date(2026, 9, 24), "NONE"),
            (FRI, "WAIT"),
        ]

    def test_only_the_last_sessions(self):
        ps = [prop(session=FRI - timedelta(days=i)) for i in range(40)]
        tl = timeline(ps, sessions=30)
        assert len(tl) == 30 and tl[-1].session == FRI


class TestBoard:
    def test_empty_everything(self):
        assert build_board([], [], None) == []

    def test_held_first_by_weight_then_watched(self):
        b = book(pos("SMALL", qty=1, price=100), pos("BIG", qty=50, price=100))
        rows = build_board([prop("AMZN")], [], b)
        assert [r.symbol for r in rows] == ["BIG", "SMALL", "AMZN"]
        assert rows[0].held and not rows[2].held

    def test_a_held_name_uses_the_newest_sigma(self):
        old = prop("AAOI", hour=10, features={"sigma": 0.10})
        new = prop("AAOI", hour=15, features={"sigma": 0.02})
        (row,) = build_board([old, new], [], book(pos()))
        assert row.risk.stop == pytest.approx(100 * (1 - position_stop_pct(0.02)))
        assert row.latest.built_at == new.built_at

    def test_a_held_name_without_proposals_still_shows(self):
        (row,) = build_board([], [], book(pos()))
        assert row.held and row.latest is None
        assert row.risk.stop is None  # no sigma yet

    def test_options_and_shorts_are_not_rows(self):
        b = book(pos("SPY", instrument=EQUITY_OPTION), pos("TSLA", qty=-5))
        assert build_board([], [], b) == []

    def test_stale_inputs_are_listed(self):
        p = prop(action=Action.WAIT, features={"stale": "price,book"}, blockers=["x"])
        (row,) = build_board([p], [], None)
        assert row.latest.stale == ["price", "book"]

    def test_trades_are_counted_and_matched_to_the_call(self):
        ps = [prop("AAOI", action=Action.ENTER, hour=10), prop("AAOI", action=Action.WAIT, hour=14)]
        ts = [
            trade(closed=date(2026, 9, 26), pnl=50.0),
            trade(opened=date(2026, 9, 24), closed=date(2026, 9, 25), pnl=-20.0),
            trade(),  # still open
        ]
        (row,) = build_board(ps, ts, book(pos()))
        assert row.realized == pytest.approx(30.0) and row.wins == 1 and row.losses == 1
        assert len(row.open_trades) == 1 and len(row.closed_trades) == 2
        # Opened on the Friday of an ENTER (later WAIT): it followed the call.
        assert row.closed_trades[0].call == "ENTER"
        assert row.closed_trades[1].call is None  # no proposal that day

    def test_option_trades_are_left_out(self):
        t = trade()
        t.instrument = "Equity Option"
        (row,) = build_board([], [t], book(pos()))
        assert row.open_trades == []

    def test_a_watched_name_shows_its_planned_risk_and_price(self):
        leg = Leg(horizon="position", entry=100, stop=90, stop_basis="p", risk_pct=0.02,
                  shares=2, notional=200)  # fmt: skip
        (row,) = build_board([prop(action=Action.ENTER, legs=[leg])], [], None)
        assert row.price == 100.0 and row.risk.basis == "planned"
