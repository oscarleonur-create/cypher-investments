"""Today's picks on live prices, the history of picks, and evaluating one as a position."""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pytest
from advisor.breadth import picks as PK
from advisor.breadth.bars import Bar
from advisor.daemon.market_calendar import MARKET_TZ

from tests.test_breadth.test_picks import NOW, inputs, store  # noqa: F401  (fixtures)

TODAY = date(2026, 9, 28)  # a Monday; the store's last session is Friday 9/25


def live_bars(price=50.0, symbols=("AAA", "BBB", "CCC")):
    calls = []

    def fetch(syms, day):
        calls.append(list(syms))
        return {s: Bar(day, price, price, price, price, 5e5) for s in syms if s in symbols}

    return fetch, calls


def at(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=MARKET_TZ)


def test_with_live_row_adds_the_session_and_leaves_missing_names_empty(store):  # noqa: F811
    from advisor.breadth.panel import load_panel

    base = load_panel(store)
    bars = {"AAA": Bar(TODAY, 51, 52, 50, 51.5, 1e5)}
    p = PK.with_live_row(base, TODAY, bars)
    assert p.sessions[-1].date() == TODAY and len(p.sessions) == len(base.sessions) + 1
    assert p.close.iloc[-1]["AAA"] == 51.5 and p.high.iloc[-1]["AAA"] == 52
    assert np.isnan(p.close.iloc[-1]["BBB"])  # never yesterday's price carried
    assert PK.with_live_row(p, TODAY, bars) is p  # the session already there


def test_live_build_asks_only_names_that_can_qualify(store, inputs):  # noqa: F811
    fetch, calls = live_bars()
    r = PK.build_picks(store, TODAY, at(2026, 9, 28, 11), live=fetch)
    assert r["ok"] and r["provisional"] and r["day"] == "2026-09-28"
    assert calls == [["AAA"]]  # BBB has only P, CCC nothing: two families need F or I
    assert r["picks"][0]["provisional"] is True
    back = PK.latest_picks(store)
    assert back["day"] == "2026-09-28" and back["provisional"]


def test_no_live_prices_is_reported(store, inputs):  # noqa: F811
    fetch, _ = live_bars(symbols=())
    r = PK.build_picks(store, TODAY, NOW, live=fetch)
    assert not r["ok"] and "no live prices" in r["error"]


class TestRefresh:
    def test_during_the_session_it_is_provisional(self, store, inputs):  # noqa: F811
        fetch, calls = live_bars()
        r = PK.refresh_picks(store, at(2026, 9, 28, 11), live=fetch)
        assert r["provisional"] and r["day"] == "2026-09-28" and calls

    def test_after_the_close_before_the_sync_still_provisional(self, store, inputs):  # noqa: F811
        fetch, calls = live_bars()
        r = PK.refresh_picks(store, at(2026, 9, 28, 17, 50), live=fetch)
        assert r["provisional"] and calls

    def test_once_the_day_is_stored_it_is_final(self, store, inputs):  # noqa: F811
        fetch, calls = live_bars()
        r = PK.refresh_picks(store, at(2026, 9, 25, 21), live=fetch)
        assert r["ok"] and not r["provisional"] and calls == [], (r.get("error"), calls)

    def test_weekend_and_before_the_open_use_the_last_close(self, store, inputs):  # noqa: F811
        fetch, calls = live_bars()
        assert PK.refresh_picks(store, at(2026, 9, 27, 12), live=fetch)["day"] == "2026-09-25"
        assert PK.refresh_picks(store, at(2026, 9, 28, 8), live=fetch)["day"] == "2026-09-25"
        assert calls == []


def test_history_newest_first_with_moves_since(store, inputs):  # noqa: F811
    PK.build_picks(store, date(2026, 9, 25), NOW)
    fetch, _ = live_bars()
    PK.build_picks(store, TODAY, NOW, live=fetch)
    # The nightly sync then stores 9/28's bar for AAA at 55.
    with store.conn:
        store.conn.execute(
            "INSERT INTO breadth_bars (symbol, day, close, high, volume) "
            "VALUES ('AAA', '2026-09-28', 55.0, 55.0, 1e6)"
        )
    out = PK.latest_picks(store)
    assert [d["day"] for d in out["days"]] == ["2026-09-28", "2026-09-25"]
    assert out["days"][0]["provisional"] and not out["days"][1]["provisional"]
    past = PK.latest_picks(store, "2026-09-25")
    sp = past["picks"][0]["since_pick"]
    assert sp["day"] == "2026-09-28" and sp["move"] == pytest.approx(0.10)
    assert "since_pick" not in out["picks"][0]  # nothing after today yet


def test_evaluate_position_records_and_returns_the_engine_proposal(tmp_path):
    from advisor.breadth.position import evaluate_position, latest_evaluation
    from advisor.entry.proposal import Action, Proposal, Reason

    db = tmp_path / "research.db"
    seen = {}

    def runner(daemon, now, *, symbols, entry_store, read):
        seen.update(symbols=symbols, read=read)
        p = Proposal(
            symbol=symbols[0],
            session=date(2026, 9, 28),
            built_at=now,
            action=Action.ENTER,
            price=16.77,
            reasons=[Reason(text="in zone", source="test")],
        )
        entry_store.add(p)
        return [p], []

    out = evaluate_position(db, " pl ", NOW, runner=runner)
    assert seen == {"symbols": ["PL"], "read": True}
    assert out["proposal"]["action"] == "ENTER" and out["recorded"]
    again = evaluate_position(db, "PL", NOW, runner=runner)
    assert again["recorded"] is False  # the same call, same session: one record
    assert latest_evaluation(db, "pl")["action"] == "ENTER"


def test_evaluate_position_with_no_proposal(tmp_path):
    from advisor.breadth.position import evaluate_position

    out = evaluate_position(
        tmp_path / "r.db", "ZZZ", NOW, runner=lambda *a, **k: ([], ["ZZZ: sheet failed"])
    )
    assert out["proposal"] is None and out["errors"] == ["ZZZ: sheet failed"]


def test_no_evaluation_on_file(tmp_path):
    from advisor.breadth.position import latest_evaluation

    assert latest_evaluation(tmp_path / "none.db", "PL") is None


def test_the_endpoint_refuses_something_that_is_not_a_symbol():
    import asyncio

    from advisor.api.routers import breadth as router
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        asyncio.run(router.evaluate("../etc"))
