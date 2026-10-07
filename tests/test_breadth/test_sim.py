"""Picks followed in the sim: entered at the click, exited by the plan (user, 2026-10-04)."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime

import pytest
from advisor.breadth import sim as S
from advisor.daemon import market_calendar as mc

ET = mc.MARKET_TZ
FRI_NOON = datetime(2026, 10, 2, 12, 0, tzinfo=ET)


def pick(symbol="AGX", stop_pct=0.10, **kw):
    return {"symbol": symbol, "price": 383.83, "families": ["F", "P"],
            "verdict": {"action": "ENTER", "headline": "ENTER up to …"},
            "plan": {"stop_pct": stop_pct, "stage": "fresh", "group": "F+P"}, **kw}  # fmt: skip


def sim_row(entry=100.0, stop_pct=0.10, now=FRI_NOON):
    return S.new_sim(pick(stop_pct=stop_pct), "2026-10-02", entry, "TastyTrade mark", now)


class TestEntry:
    def test_entry_stop_and_window(self):
        r = sim_row()
        assert r["entry"] == 100.0 and r["stop"] == pytest.approx(90.0)
        assert r["session"] == "2026-10-02" and r["status"] == "OPEN"
        # 20 sessions after Fri 10-02: Columbus Day is not a market holiday.
        assert r["exit_by"] == "2026-10-30"
        snap = json.loads(r["pick_json"])
        assert snap["verdict"] == "ENTER" and snap["families"] == ["F", "P"]

    @pytest.mark.parametrize("now,session", [
        (datetime(2026, 10, 4, 10, 0, tzinfo=ET), "2026-10-02"),   # Sunday: Friday's close
        (datetime(2026, 10, 5, 8, 0, tzinfo=ET), "2026-10-02"),    # Monday before the bell
        (datetime(2026, 10, 5, 9, 30, tzinfo=ET), "2026-10-05"),   # the bell: Monday's session
        (datetime(2026, 11, 26, 11, 0, tzinfo=ET), "2026-11-25"),  # Thanksgiving: closed
    ])  # fmt: skip
    def test_the_entry_belongs_to_the_session_it_could_trade_in(self, now, session):
        assert sim_row(now=now)["session"] == session

    def test_holidays_are_not_sessions_in_the_window(self):
        # From 2026-11-20 the 20 sessions skip Thanksgiving (11-26) and Christmas (12-25).
        assert S.nth_session_after(date(2026, 11, 20), 20) == date(2026, 12, 21)

    def test_no_stop_estimate_is_a_time_exit_only(self):
        r = sim_row(stop_pct=None)
        assert r["stop"] is None and r["stop_pct"] is None

    @pytest.mark.parametrize("price", [None, 0.0, -5.0, float("nan")])
    def test_no_price_no_sim(self, price):
        with pytest.raises(S.SimError, match="no price"):
            sim_row(entry=price)


def bars(*rows):
    """(day, open, low, close)."""
    return list(rows)


class TestExit:
    def test_open_while_nothing_happened(self):
        r = sim_row()
        assert S.evaluate(r, bars(("2026-10-05", 101, 95, 100))) == {}

    def test_a_low_at_the_stop_exits_at_the_stop(self):
        out = S.evaluate(sim_row(), bars(("2026-10-05", 99, 94, 97), ("2026-10-06", 97, 89, 92)))
        assert out == {"status": "STOP", "exit_day": "2026-10-06", "exit_price": 90.0,
                       "ret": pytest.approx(-0.10), "held": 2}  # fmt: skip

    def test_a_gap_below_the_stop_exits_at_the_open(self):
        out = S.evaluate(sim_row(), bars(("2026-10-05", 85, 80, 82)))
        assert (
            out["status"] == "STOP"
            and out["exit_price"] == 85
            and out["ret"] == pytest.approx(-0.15)
        )

    def test_exactly_at_the_stop_is_stopped(self):
        assert S.evaluate(sim_row(), bars(("2026-10-05", 95, 90.0, 93)))["status"] == "STOP"

    def test_the_entry_session_never_counts(self):
        # Friday's low may have printed before the noon click.
        assert S.evaluate(sim_row(), bars(("2026-10-02", 95, 80, 99))) == {}

    def test_out_at_the_20th_close(self):
        r = sim_row()
        rows = [(d, 101, 96, 105) for d in ("2026-10-05", "2026-10-29", "2026-10-30")]
        out = S.evaluate(r, rows)
        assert out == {"status": "TIME", "exit_day": "2026-10-30", "exit_price": 105.0,
                       "ret": pytest.approx(0.05), "held": 3}  # fmt: skip

    def test_no_bar_on_the_last_day_closes_on_the_last_before_once_the_window_is_over(self):
        r = sim_row()
        assert S.evaluate(r, bars(("2026-10-29", 101, 96, 104))) == {}  # not over yet
        out = S.evaluate(r, bars(("2026-10-29", 101, 96, 104), ("2026-11-02", 99, 97, 98)))
        assert out["status"] == "TIME" and out["exit_day"] == "2026-10-29"

    def test_a_stop_after_the_window_does_not_count(self):
        r = sim_row()
        rows = bars(("2026-10-30", 101, 96, 103), ("2026-11-02", 80, 70, 75))
        assert S.evaluate(r, rows)["status"] == "TIME"

    def test_missing_open_and_low_read_the_close(self):
        assert S.evaluate(sim_row(), bars(("2026-10-05", None, None, 89)))["exit_price"] == 89

    def test_no_stop_runs_to_the_time_exit(self):
        r = sim_row(stop_pct=None)
        assert S.evaluate(r, bars(("2026-10-05", 50, 40, 45))) == {}
        assert S.evaluate(r, bars(("2026-10-30", 50, 40, 45)))["status"] == "TIME"


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(tmp_path / "b.db")
    c.executescript(
        "CREATE TABLE breadth_bars (symbol TEXT, day TEXT, open REAL, high REAL, low REAL, "
        "close REAL NOT NULL, volume REAL, PRIMARY KEY (symbol, day))"
    )
    yield c
    c.close()


def put_bar(conn, symbol, day, o, lo, c):
    conn.execute(
        "INSERT INTO breadth_bars VALUES (?, ?, ?, ?, ?, ?, 0)", (symbol, day, o, c, lo, c)
    )


class TestStored:
    def test_add_list_and_close_on_the_bars(self, conn):
        S.add(conn, sim_row())
        put_bar(conn, "AGX", "2026-10-05", 101, 97, 102)
        out = S.listing(conn)
        [o] = out["open"]
        assert o["last"] == 102 and o["unrealized"] == pytest.approx(0.02) and o["sessions_in"] == 1
        put_bar(conn, "AGX", "2026-10-06", 95, 88, 91)
        out = S.listing(conn)
        assert out["open"] == [] and out["closed"][0]["status"] == "STOP"
        assert out["record"] == {"n": 1, "mean": pytest.approx(-0.10), "won": 0, "stopped": 1,
                                 "held": 2.0}  # fmt: skip

    def test_one_open_sim_per_symbol(self, conn):
        S.add(conn, sim_row())
        with pytest.raises(S.SimError, match="already in the sim"):
            S.add(conn, sim_row())

    def test_a_closed_sim_can_be_followed_again(self, conn):
        S.add(conn, sim_row())
        put_bar(conn, "AGX", "2026-10-05", 85, 80, 82)
        S.update_open(conn)
        # Followed again the next day: Monday's gap belongs to the first sim, not this one.
        S.add(conn, sim_row(now=datetime(2026, 10, 6, 12, 0, tzinfo=ET)))
        assert len(S.listing(conn)["open"]) == 1

    def test_update_twice_closes_once(self, conn):
        S.add(conn, sim_row())
        put_bar(conn, "AGX", "2026-10-05", 85, 80, 82)
        assert len(S.update_open(conn)["closed"]) == 1
        assert S.update_open(conn) == {"open": 0, "closed": []}

    def test_no_bars_yet(self, conn):
        S.add(conn, sim_row())
        [o] = S.listing(conn)["open"]
        assert o["last"] is None and o["unrealized"] is None and o["sessions_in"] == 0

    def test_empty(self, conn):
        assert S.listing(conn) == {"open": [], "closed": [], "record": {"n": 0}}


class TestEndpoint:
    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        from advisor.api.app import create_app
        from advisor.api.routers import breadth as router
        from advisor.breadth.store import BreadthStore, breadth_path
        from fastapi.testclient import TestClient

        db = tmp_path / "research.db"
        monkeypatch.setattr(router, "_db", lambda: db)
        with BreadthStore(breadth_path(db)) as store:
            store.conn.execute("SELECT 1")
        picks = {"day": "2026-10-02", "picks": [pick()]}
        monkeypatch.setattr(router, "_latest", lambda day=None: picks)
        quotes = {"AGX": (384.10, "TastyTrade mark")}

        async def fake_price(symbol):
            return quotes.get(symbol, (None, "no live quote: the broker returned none"))

        monkeypatch.setattr(S, "live_price", fake_price)
        monkeypatch.setattr("advisor.daemon.market_calendar.now_et", lambda: FRI_NOON)
        with TestClient(create_app()) as c:
            c.quotes = quotes  # type: ignore[attr-defined]
            yield c

    def test_add_at_the_live_price_and_list(self, client):
        r = client.post("/api/breadth/sims", json={"symbol": "agx"})
        assert r.status_code == 200
        s = r.json()["sim"]
        assert s["entry"] == 384.10 and s["entry_source"] == "TastyTrade mark"
        assert s["stop"] == pytest.approx(384.10 * 0.9)
        assert [o["symbol"] for o in client.get("/api/breadth/sims").json()["open"]] == ["AGX"]
        assert client.post("/api/breadth/sims", json={"symbol": "AGX"}).status_code == 409

    def test_not_a_pick_is_404(self, client):
        assert client.post("/api/breadth/sims", json={"symbol": "ZZZ"}).status_code == 404

    def test_no_live_quote_is_503_and_nothing_stored(self, client):
        client.quotes.clear()
        r = client.post("/api/breadth/sims", json={"symbol": "AGX"})
        assert r.status_code == 503 and "no live quote" in r.json()["detail"]
        assert client.get("/api/breadth/sims").json()["open"] == []
