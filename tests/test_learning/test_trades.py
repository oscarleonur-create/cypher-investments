"""Round trips from broker executions: episodes, books, matching, the rule's exit."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from advisor.daemon import market_calendar as mc
from advisor.entry.track import Bar
from advisor.learning.store import TradeStore
from advisor.learning.trades import (
    Book,
    Execution,
    option_expiry,
    review,
    round_trips,
    rule_exit,
    sessions_between,
    sync_trades,
    unmatched,
)
from advisor.learning.trades import match as match_trades


def at(d: date, hh=10, mm=0, ss=0) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, ss, tzinfo=mc.MARKET_TZ)


MON, TUE, WED = date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)
FRI = date(2026, 9, 18)


def ex(action, qty, price, when, symbol="AAPL", instrument="Equity", account="A1", tx=None, u=None):
    return Execution(
        account=account,
        underlying=u or symbol.split()[0],
        symbol=symbol,
        instrument=instrument,
        action=action,
        quantity=qty,
        price=price,
        executed_at=when,
        tx_id=tx,
    )


class TestEpisodes:
    def test_a_simple_long_round_trip(self):
        (t,) = round_trips(
            [ex("Buy to Open", 10, 100, at(MON)), ex("Sell to Close", 10, 105, at(MON, 15))]
        )
        assert t.closed and t.direction == "long"
        assert t.pnl == pytest.approx(50) and t.ret == pytest.approx(0.05)
        assert t.sessions_held == 0 and t.book is Book.QUICK

    def test_scaling_in_and_out_is_one_trade(self):
        fills = [
            ex("Buy to Open", 10, 100, at(MON)),
            ex("Buy to Open", 10, 110, at(MON, 11)),
            ex("Sell to Close", 5, 120, at(TUE)),
            ex("Sell to Close", 15, 90, at(TUE, 14)),
        ]
        (t,) = round_trips(fills)
        assert t.quantity == 20 and t.executions == 4
        assert t.entry_price == pytest.approx(105)
        assert t.exit_price == pytest.approx((5 * 120 + 15 * 90) / 20)
        assert t.pnl == pytest.approx(5 * 120 + 15 * 90 - 10 * 100 - 10 * 110)

    def test_flat_then_again_is_two_trades(self):
        fills = [
            ex("Buy to Open", 1, 10, at(MON)),
            ex("Sell to Close", 1, 11, at(MON, 11)),
            ex("Buy to Open", 1, 12, at(MON, 12)),
            ex("Sell to Close", 1, 11, at(MON, 13)),
        ]
        a, b = round_trips(fills)
        assert a.pnl == pytest.approx(1) and b.pnl == pytest.approx(-1)

    def test_a_short_sale_gains_when_price_falls(self):
        (t,) = round_trips(
            [ex("Sell to Open", 2, 50, at(MON)), ex("Buy to Close", 2, 40, at(MON, 12))]
        )
        assert t.direction == "short" and t.pnl == pytest.approx(20) and t.ret == pytest.approx(0.2)

    def test_close_without_an_open_in_the_window_is_dropped(self):
        trades = round_trips([ex("Sell to Close", 5, 100, at(MON))])
        assert trades == []

    def test_still_held_is_open_with_no_outcome(self):
        (t,) = round_trips([ex("Buy to Open", 5, 100, at(MON))])
        assert not t.closed and t.pnl is None and t.ret is None and t.book is Book.UNCLASSIFIED

    def test_option_pnl_uses_the_multiplier(self):
        occ = "AAPL  261016C00200000"
        fills = [
            ex("Buy to Open", 1, 2.0, at(MON), symbol=occ, instrument="Equity Option", u="AAPL"),
            ex("Sell to Close", 1, 3.0, at(TUE), symbol=occ, instrument="Equity Option", u="AAPL"),
        ]
        (t,) = round_trips(fills)
        assert t.pnl == pytest.approx(100)

    def test_the_same_fill_twice_is_counted_once(self):
        fills = [
            ex("Buy to Open", 1, 10, at(MON), tx="1"),
            ex("Buy to Open", 1, 10, at(MON), tx="1"),
            ex("Sell to Close", 1, 11, at(MON, 12), tx="2"),
        ]
        (t,) = round_trips(fills)
        assert t.closed and t.quantity == 1

    def test_two_identical_fills_with_different_ids_are_both_real(self):
        fills = [
            ex("Sell to Close", 2, 247, at(TUE), tx="b"),
            ex("Buy to Open", 4, 240, at(MON), tx="a0"),
            ex("Sell to Close", 2, 247, at(TUE), tx="c"),
        ]
        (t,) = round_trips(fills)
        assert t.closed and t.quantity == 4

    def test_accounts_are_separate(self):
        fills = [
            ex("Buy to Open", 1, 10, at(MON), account="A1"),
            ex("Sell to Close", 1, 11, at(MON, 12), account="A2"),
        ]
        (t,) = round_trips(fills)
        assert t.account == "A1" and not t.closed

    def test_zero_quantity_is_ignored(self):
        assert round_trips([ex("Buy to Open", 0, 10, at(MON))]) == []


class TestBooks:
    @pytest.mark.parametrize(
        "exit_day,book",
        [
            (MON, Book.QUICK),  # same session
            (TUE, Book.QUICK),  # next session
            (WED, Book.UNCLASSIFIED),  # two sessions: neither rule says
            (date(2026, 9, 28), Book.UNCLASSIFIED),  # five sessions
            (date(2026, 9, 29), Book.HOLD),  # six: over a trading week
        ],
    )
    def test_boundaries(self, exit_day, book):
        (t,) = round_trips(
            [ex("Buy to Open", 1, 10, at(MON)), ex("Sell to Close", 1, 10, at(exit_day))]
        )
        assert t.book is book

    def test_friday_to_monday_is_one_session(self):
        assert sessions_between(FRI, MON) == 1

    def test_a_holiday_is_not_a_session(self):
        # Thanksgiving 2026-11-26: Wednesday to Friday is one session.
        assert sessions_between(date(2026, 11, 25), date(2026, 11, 27)) == 1

    def test_spread_legs_are_unclassified(self):
        a, b = "SPY   261016C00500000", "SPY   261016C00510000"
        fills = [
            ex("Buy to Open", 1, 5, at(MON), symbol=a, instrument="Equity Option", u="SPY"),
            ex(
                "Sell to Open",
                1,
                3,
                at(MON, 10, 0, 30),
                symbol=b,
                instrument="Equity Option",
                u="SPY",
            ),
            ex("Sell to Close", 1, 6, at(MON, 14), symbol=a, instrument="Equity Option", u="SPY"),
            ex("Buy to Close", 1, 3.5, at(MON, 14), symbol=b, instrument="Equity Option", u="SPY"),
        ]
        trades = round_trips(fills)
        assert all(t.multi_leg and t.book is Book.UNCLASSIFIED for t in trades)

    def test_options_opened_apart_are_not_a_spread(self):
        a, b = "SPY   261016C00500000", "SPY   261016C00510000"
        fills = [
            ex("Buy to Open", 1, 5, at(MON), symbol=a, instrument="Equity Option", u="SPY"),
            ex(
                "Buy to Open",
                1,
                3,
                at(MON, 10, 1, 1),
                symbol=b,
                instrument="Equity Option",
                u="SPY",
            ),
        ]
        assert not any(t.multi_leg for t in round_trips(fills))


class TestExpiry:
    def test_occ_expiry(self):
        assert option_expiry("SPY   260619C00500000") == date(2026, 6, 19)
        assert option_expiry("AAPL") is None

    def test_open_past_expiry_is_noted_not_guessed(self):
        occ = "SPY   260619C00500000"
        (t,) = round_trips(
            [
                ex(
                    "Buy to Open",
                    1,
                    5,
                    at(date(2026, 6, 1)),
                    symbol=occ,
                    instrument="Equity Option",
                    u="SPY",
                )
            ],
            today=date(2026, 9, 1),
        )
        assert not t.closed and t.pnl is None
        assert any("expiry" in n for n in t.notes)


def cand(symbol, session, cid):
    return SimpleNamespace(symbol=symbol, session=session, id=cid)


def prop(symbol, session, built, pid, stop_pct=0.03):
    leg = SimpleNamespace(horizon="trade", entry=100.0, stop=100.0 * (1 - stop_pct))
    return SimpleNamespace(
        symbol=symbol, session=session, built_at=at(session, *built), id=pid, legs=[leg]
    )


class TestMatch:
    def test_candidate_and_latest_proposal_on_the_entry_session(self):
        (t,) = round_trips([ex("Buy to Open", 1, 10, at(MON)), ex("Sell to Close", 1, 11, at(TUE))])
        match_trades(
            [t],
            [cand("AAPL", MON, "c1"), cand("AAPL", TUE, "c-other-day"), cand("MSFT", MON, "c3")],
            [prop("AAPL", MON, (9, 45), "wait"), prop("AAPL", MON, (11, 45), "enter")],
        )
        assert t.candidate_ids == ["c1"] and t.proposal_id == "enter" and t.matched

    def test_short_sales_are_not_matched_to_long_setups(self):
        (t,) = round_trips([ex("Sell to Open", 1, 10, at(MON))])
        match_trades([t], [cand("AAPL", MON, "c1")], [])
        assert not t.matched


class TestRuleExit:
    def trade(self, exit_px=104.0):
        (t,) = round_trips(
            [
                ex("Buy to Open", 1, 100, at(MON)),
                ex("Sell to Close", 1, exit_px, at(date(2026, 9, 30))),
            ]
        )
        return t

    def test_stop_touched_is_a_stop_exit(self):
        t = self.trade()
        rule_exit(
            t, prop("AAPL", MON, (10, 0), "p"), [Bar(MON, 101, 99, 100), Bar(TUE, 100, 96, 97)]
        )
        assert t.rule_exit == "stop" and t.rule_exit_price == pytest.approx(97.0)
        assert t.rule_ret == pytest.approx(-0.03)

    def test_otherwise_out_at_the_next_close(self):
        t = self.trade()
        rule_exit(
            t, prop("AAPL", MON, (10, 0), "p"), [Bar(MON, 101, 99, 100), Bar(TUE, 103, 98, 102)]
        )
        assert t.rule_exit == "time" and t.rule_ret == pytest.approx(0.02)

    def test_missing_bars_are_said_not_guessed(self):
        t = self.trade()
        rule_exit(t, prop("AAPL", MON, (10, 0), "p"), [Bar(MON, 101, 99, 100)])
        assert t.rule_exit is None and any("not knowable" in n for n in t.notes)

    def test_no_trade_leg_no_rule_exit(self):
        t = self.trade()
        p = prop("AAPL", MON, (10, 0), "p")
        p.legs = []
        rule_exit(t, p, [Bar(MON, 101, 99, 100), Bar(TUE, 103, 98, 102)])
        assert t.rule_exit is None


class TestReview:
    def test_spread_legs_have_pnl_but_no_return_stats(self):
        a, b = "SPY   261016C00500000", "SPY   261016C00510000"
        fills = [
            ex("Buy to Open", 1, 5, at(MON), symbol=a, instrument="Equity Option", u="SPY"),
            ex("Sell to Open", 1, 3, at(MON), symbol=b, instrument="Equity Option", u="SPY"),
            ex("Sell to Close", 1, 6, at(MON, 14), symbol=a, instrument="Equity Option", u="SPY"),
            ex("Buy to Close", 1, 3.5, at(MON, 14), symbol=b, instrument="Equity Option", u="SPY"),
            ex("Buy to Open", 1, 10, at(MON)),
            ex("Sell to Close", 1, 11, at(MON, 15)),
        ]
        rows = {r["book"]: r for r in review(round_trips(fills))}
        assert rows["quick"]["n"] == 1
        assert rows["spread legs"]["pnl"] == pytest.approx(100 - 50)
        assert "mean_ret" not in rows["spread legs"]

    def test_misses_count_only_after_the_system_was_looking(self):
        fills = [
            ex("Buy to Open", 1, 10, at(FRI), symbol="OLD"),
            ex("Sell to Close", 1, 11, at(FRI, 15), symbol="OLD"),
            ex("Buy to Open", 1, 10, at(TUE), symbol="NEW"),
            ex("Sell to Close", 1, 11, at(TUE, 15), symbol="NEW"),
        ]
        missed, before = unmatched(round_trips(fills), coverage_start=MON)
        assert [t.underlying for t in missed] == ["NEW"] and before == 1

    def test_no_coverage_means_no_misses(self):
        trades = round_trips(
            [ex("Buy to Open", 1, 10, at(TUE)), ex("Sell to Close", 1, 11, at(TUE, 15))]
        )
        assert unmatched(trades, None) == ([], 1)


class TestStoreAndSync:
    @pytest.fixture
    def conn(self, tmp_path):
        c = sqlite3.connect(str(tmp_path / "research.db"))
        yield c
        c.close()

    def test_upsert_is_idempotent_and_an_open_trade_closes(self, conn):
        store = TradeStore(conn)
        (open_,) = round_trips([ex("Buy to Open", 1, 10, at(MON))])
        assert store.upsert(open_) == 1 and store.upsert(open_) == 0
        (closed,) = round_trips(
            [ex("Buy to Open", 1, 10, at(MON)), ex("Sell to Close", 1, 12, at(TUE))]
        )
        assert closed.id == open_.id
        assert store.upsert(closed) == 1
        (back,) = store.list()
        assert back.closed and back.pnl == pytest.approx(2)

    def test_sync_twice_writes_once(self, conn):
        fills = [ex("Buy to Open", 1, 10, at(MON)), ex("Sell to Close", 1, 11, at(TUE))]
        store = TradeStore(conn)

        def fetch(a, b):
            return fills, None

        first = sync_trades(store, None, None, MON, WED, fetch=fetch, bars_fn=lambda *a: [])
        second = sync_trades(store, None, None, MON, WED, fetch=fetch, bars_fn=lambda *a: [])
        assert (first.written, second.written) == (1, 0)

    def test_broker_down_writes_nothing(self, conn):
        store = TradeStore(conn)
        r = sync_trades(store, None, None, MON, WED, fetch=lambda a, b: ([], "timeout"))
        assert r.error == "timeout" and store.list() == []

    def test_list_by_book_and_since(self, conn):
        store = TradeStore(conn)
        fills = [
            ex("Buy to Open", 1, 10, at(MON), symbol="A"),
            ex("Sell to Close", 1, 11, at(MON, 15), symbol="A"),
            ex("Buy to Open", 1, 10, at(WED), symbol="B"),
            ex("Sell to Close", 1, 11, at(WED + timedelta(days=14)), symbol="B"),
        ]
        for t in round_trips(fills):
            store.upsert(t)
        assert [t.underlying for t in store.list(book="quick")] == ["A"]
        assert [t.underlying for t in store.list(since=TUE)] == ["B"]
