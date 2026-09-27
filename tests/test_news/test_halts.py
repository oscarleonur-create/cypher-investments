"""Exchange halts: parsed from the feed, stored once, ended once, read for an exit."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.models import EventTier
from advisor.daemon.store import DaemonStore
from advisor.entry.exits import exit_calls
from advisor.entry.proposal import Action, build_proposal
from advisor.entry.sheet import Holding, Move, Sheet
from advisor.news.halts import Halt, active_halts, ended_halts, fetch, halt_events, parse

NOW = datetime(2026, 9, 28, 10, 0, tzinfo=mc.MARKET_TZ)


def row(symbol="TE", code="T12", day="09/28/2026", clock="09:41:07.123", rdate="", rtime=""):
    return f"""<item>
      <title>{symbol}</title>
      <ndaq:HaltDate>{day}</ndaq:HaltDate>
      <ndaq:HaltTime>{clock}</ndaq:HaltTime>
      <ndaq:IssueSymbol>{symbol}</ndaq:IssueSymbol>
      <ndaq:IssueName>T1 Energy Inc.</ndaq:IssueName>
      <ndaq:Market>NYSE</ndaq:Market>
      <ndaq:ReasonCode>{code}</ndaq:ReasonCode>
      <ndaq:ResumptionDate>{rdate}</ndaq:ResumptionDate>
      <ndaq:ResumptionTradeTime>{rtime}</ndaq:ResumptionTradeTime>
    </item>"""


def feed(*rows):
    # The live feed opens with a byte-order mark.
    return (
        chr(0xFEFF) + '<?xml version="1.0" encoding="utf-8"?>'
        '<rss version="2.0" xmlns:ndaq="http://www.nasdaqtrader.com/"><channel>'
        + "".join(rows)
        + "</channel></rss>"
    )


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


class TestParse:
    def test_a_halt_is_read_in_eastern_time(self):
        (h,) = parse(feed(row()))
        assert h.symbol == "TE" and h.code == "T12" and h.market == "NYSE"
        assert h.halted_at == datetime(2026, 9, 28, 9, 41, 7, 123000, tzinfo=mc.MARKET_TZ)
        assert h.resumed_at is None and h.grade == "EXIT"

    def test_a_resumption_is_read(self):
        (h,) = parse(feed(row(rdate="09/28/2026", rtime="10:30:00")))
        assert h.resumed_at == datetime(2026, 9, 28, 10, 30, tzinfo=mc.MARKET_TZ)

    def test_a_resumption_date_with_an_unreadable_time_still_ends_it(self):
        """A format change must not leave an EXIT standing on a halt that ended."""
        (h,) = parse(feed(row(rdate="09/28/2026", rtime="soon")))
        assert h.resumed_at == datetime(2026, 9, 28, 0, 0, tzinfo=mc.MARKET_TZ)

    def test_rows_without_a_symbol_code_or_time_are_skipped(self):
        halts = parse(feed(row(symbol=""), row(code=""), row(clock=""), row(symbol="AAOI")))
        assert [h.symbol for h in halts] == ["AAOI"]

    def test_an_empty_feed_is_no_halts(self):
        assert parse(feed()) == []

    @pytest.mark.parametrize(
        "code,grade",
        [("T12", "EXIT"), ("H10", "EXIT"), ("H4", "EXIT"), ("T1", "REVIEW"), ("LUDP", None)],
    )
    def test_codes_grade(self, code, grade):
        (h,) = parse(feed(row(code=code)))
        assert h.grade == grade

    def test_a_network_failure_raises_for_the_job_to_record(self):
        def down(url):
            raise TimeoutError("nasdaqtrader")

        with pytest.raises(TimeoutError):
            fetch(get=down)


def halt(code="T12", resumed=None, symbol="TE", at=NOW - timedelta(minutes=20)):
    return Halt(symbol=symbol, market="NYSE", code=code, halted_at=at, resumed_at=resumed)


class TestEvents:
    def test_only_watched_names_become_events(self):
        events = halt_events([halt(symbol="FBGL"), halt()], {"TE": True})
        assert [e.symbol for e in events] == ["TE"]

    def test_tier_follows_the_grade_and_whether_it_is_held(self):
        held = halt_events([halt("T12"), halt("T1"), halt("LUDP")], {"TE": True})
        assert [e.tier for e in held] == [EventTier.A, EventTier.B, EventTier.C]
        watched = halt_events([halt("T12")], {"TE": False})
        assert watched[0].tier is EventTier.B

    def test_the_same_halt_polled_twice_is_stored_once(self, store):
        assert [store.emit(e) for e in halt_events([halt()], {"TE": True})] == [True]
        assert [store.emit(e) for e in halt_events([halt()], {"TE": True})] == [False]

    def test_a_resumption_seen_later_ends_the_halt_once(self, store):
        for e in halt_events([halt()], {"TE": True}):
            store.emit(e)
        (h,) = active_halts(store, "TE", NOW - timedelta(days=1))
        assert h["resumed_at"] is None
        later = halt_events([halt(resumed=NOW - timedelta(minutes=5))], {"TE": True})
        assert [store.emit(e) for e in later] == [False, True]  # halt known, resumption new
        (h,) = active_halts(store, "TE", NOW - timedelta(days=1))
        assert h["resumed_at"] == (NOW - timedelta(minutes=5)).isoformat()

    def test_two_halts_are_told_apart(self, store):
        for e in halt_events(
            [halt("T1", at=NOW - timedelta(hours=2), resumed=NOW - timedelta(hours=1)), halt()],
            {"TE": True},
        ):
            store.emit(e)
        hs = active_halts(store, "TE", NOW - timedelta(days=1))
        assert [(h["code"], h["resumed_at"] is None) for h in hs] == [("T12", True), ("T1", False)]

    def test_a_halt_still_in_force_counts_however_old(self, store):
        """The live feed lists halts from 2019 still in force."""
        old = halt(at=NOW - timedelta(days=400))
        for e in halt_events([old], {"TE": True}):
            store.emit(e)
        (h,) = active_halts(store, "TE", NOW - timedelta(days=14))
        assert h["resumed_at"] is None

    def test_an_old_halt_that_ended_is_history(self, store):
        old = halt(at=NOW - timedelta(days=40), resumed=NOW - timedelta(days=39))
        for e in halt_events([old], {"TE": True}):
            store.emit(e)
        assert active_halts(store, "TE", NOW - timedelta(days=14)) == []

    def test_a_standing_halt_is_not_buried_under_news(self, store):
        from advisor.daemon.models import Event, EventSource

        for e in halt_events([halt(at=NOW - timedelta(days=30))], {"TE": True}):
            store.emit(e)
        for k in range(600):
            store.emit(
                Event(
                    ts=NOW - timedelta(minutes=k),
                    source=EventSource.CALENDAR,
                    kind="NEWS_CONTEXT",
                    tier=EventTier.C,
                    symbol="TE",
                    dedup_key=f"n{k}",
                )
            )
        assert len(active_halts(store, "TE", NOW - timedelta(days=14))) == 1


class TestEndedUnseen:
    def test_a_halt_that_left_the_feed_ended_even_unseen(self, store):
        """The laptop slept through the resumption; the EXIT must not stand forever."""
        for e in halt_events([halt()], {"TE": True}):
            store.emit(e)
        other = halt(symbol="FBGL", code="T1")
        (e,) = ended_halts(store, [other], {"TE": True}, NOW)
        assert e.kind == "TRADING_RESUMED" and e.payload["inferred"] is True
        assert store.emit(e) is True
        (h,) = active_halts(store, "TE", NOW - timedelta(days=1))
        assert h["resumed_at"] == NOW.isoformat() and h["resumed_inferred"] is True
        assert ended_halts(store, [other], {"TE": True}, NOW) == []  # once

    def test_a_halt_still_listed_is_not_ended(self, store):
        for e in halt_events([halt()], {"TE": True}):
            store.emit(e)
        assert ended_halts(store, [halt()], {"TE": True}, NOW) == []

    def test_an_empty_feed_ends_nothing(self, store):
        """An empty or broken feed proves nothing."""
        for e in halt_events([halt()], {"TE": True}):
            store.emit(e)
        assert ended_halts(store, [], {"TE": True}, NOW) == []

    def test_an_inferred_end_says_so(self):
        s = sheet(halt("T12", resumed=NOW))
        s.halts[0]["resumed_inferred"] = True
        (c,) = exit_calls(s, net_liq=7_957.51)[0]
        assert c.action == "REVIEW" and "no longer in the exchange's halt list" in c.why


def sheet(*halts):
    return Sheet(
        symbol="TE",
        built_at=NOW,
        move=Move(price=3.78, asof=NOW.date(), day=0.0, sigma=0.069, z=0.0),
        holding=Holding(quantity=50, weight=0.024, unrealized=0.0, cost=3.80),
        halts=[
            {
                "code": h.code,
                "market": h.market,
                "halted_at": h.halted_at.isoformat(),
                "resumed_at": h.resumed_at.isoformat() if h.resumed_at else None,
                "url": "https://www.nasdaqtrader.com/rss.aspx?feed=tradehalts",
            }
            for h in halts
        ],
    )


class TestExit:
    def test_a_standing_halt_is_an_exit_on_the_exchange_s_word(self):
        (c,) = exit_calls(sheet(halt("T12")), net_liq=7_957.51)[0]
        assert c.action == "EXIT" and c.rule == "halt" and c.shares == 50
        assert "requested more information" in c.why
        assert "cannot be sold while halted: sell when trading resumes" in c.why
        assert c.evidence[0].source.startswith("Nasdaq Trader trade halts")

    def test_an_sec_suspension_is_an_exit(self):
        (c,) = exit_calls(sheet(halt("H10")), net_liq=7_957.51)[0]
        assert c.action == "EXIT" and "suspended by the SEC" in c.why

    def test_once_it_resumes_it_is_a_review(self):
        (c,) = exit_calls(sheet(halt("T12", resumed=NOW)), net_liq=7_957.51)[0]
        assert c.action == "REVIEW" and c.shares is None
        assert "trading resumed" in c.why and "is the question now" in c.why

    def test_news_pending_is_a_review_while_halted(self):
        (c,) = exit_calls(sheet(halt("T1")), net_liq=7_957.51)[0]
        assert c.action == "REVIEW" and "news pending" in c.why
        assert c.would_change == "the news released when trading resumes"

    def test_news_pending_ends_when_it_resumes(self):
        assert exit_calls(sheet(halt("T1", resumed=NOW)), net_liq=7_957.51)[0] == []

    def test_a_volatility_pause_is_not_a_call(self):
        assert exit_calls(sheet(halt("LUDP")), net_liq=7_957.51)[0] == []

    def test_one_call_per_code_newest_first(self):
        older = halt("T12", at=NOW - timedelta(days=3), resumed=NOW - timedelta(days=2))
        calls = exit_calls(sheet(halt("T12"), older), net_liq=7_957.51)[0]
        assert [c.action for c in calls] == ["EXIT"]

    def test_the_proposal_exits_with_its_rationale(self):
        p = build_proposal(sheet(halt("T12")), net_liq=7_957.51)
        assert p.action is Action.EXIT and p.reasons[0].source == "exit rule: halt"
