"""Stale inputs: which are too old to open a position on, and what they block."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import BookSnapshot
from advisor.daemon.store import DaemonStore
from advisor.entry import freshness
from advisor.entry.freshness import ADDING, BONUS, ENTRY, Stale, assess
from advisor.entry.proposal import ENTRY_CONFIRM_SESSIONS, Action, build_proposal
from advisor.entry.sheet import Holding, Move, Sheet, build_sheet
from advisor.entry.zone import RelativeZone

ET = mc.MARKET_TZ
MON = datetime(2026, 9, 28, 10, 45, tzinfo=ET)  # in session
SUN = datetime(2026, 9, 27, 14, 0, tzinfo=ET)
FRI_CLOSE = datetime(2026, 9, 25, 16, 0, tzinfo=ET)


def fresh(now=MON, **kw):
    """Every input current at ``now``; override one to make it stale."""
    args = dict(
        price_asof=mc.session_of(now),
        book_asof=now - timedelta(minutes=5),
        filings_ok=now.replace(hour=7, minute=0) if now.weekday() < 5 else FRI_CLOSE,
        held=False,
        distress_ok=now - timedelta(hours=1),
        thesis=None,
        valuation_asof=now.date(),
    )
    args.update(kw)
    return assess(now, **args)


def inputs(stale: list[Stale]) -> set[str]:
    return {s.input for s in stale}


class TestAssess:
    def test_all_current_is_empty(self):
        assert fresh() == []

    def test_price_from_the_previous_session_is_stale_in_session(self):
        out = fresh(price_asof=date(2026, 9, 25))
        assert inputs(out) == {"price"} and out[0].blocks == ENTRY
        assert "2026-09-28" in out[0].limit

    def test_fridays_bar_is_current_on_sunday(self):
        """The weekend's session is Friday's: nothing newer can exist."""
        now = SUN
        out = assess(
            now,
            price_asof=date(2026, 9, 25),
            book_asof=FRI_CLOSE + timedelta(minutes=30),
            filings_ok=datetime(2026, 9, 25, 7, 0, tzinfo=ET),
            held=False,
            distress_ok=now - timedelta(hours=2),
            thesis=None,
            valuation_asof=None,
        )
        assert out == []

    def test_book_boundary_is_one_hour_in_session(self):
        assert fresh(book_asof=MON - timedelta(minutes=60)) == []
        out = fresh(book_asof=MON - timedelta(minutes=61))
        assert inputs(out) == {"book"} and out[0].blocks == ENTRY

    def test_book_taken_before_the_close_is_stale_on_the_weekend(self):
        """A snapshot from 15:00 Friday misses the last hour of fills."""
        out = assess(
            SUN,
            price_asof=date(2026, 9, 25),
            book_asof=datetime(2026, 9, 25, 15, 0, tzinfo=ET),
            filings_ok=datetime(2026, 9, 25, 7, 0, tzinfo=ET),
            held=False,
            distress_ok=None,
            thesis=None,
            valuation_asof=None,
        )
        assert inputs(out) == {"book"}

    def test_no_book_is_stale(self):
        out = fresh(book_asof=None)
        assert inputs(out) == {"book"} and "never" in out[0].text

    def test_book_at_the_bell(self):
        """09:30 exactly is in session: the hour rule applies, not the last close."""
        bell = datetime(2026, 9, 28, 9, 30, tzinfo=ET)
        assert freshness.book_reference(bell) == bell - timedelta(minutes=60)
        pre = datetime(2026, 9, 28, 9, 29, tzinfo=ET)
        assert freshness.book_reference(pre) == FRI_CLOSE

    def test_a_brief_that_did_not_run_today_blinds_the_filing_blocker(self):
        out = fresh(filings_ok=datetime(2026, 9, 25, 7, 0, tzinfo=ET))
        assert inputs(out) == {"filings"} and out[0].blocks == ENTRY

    def test_filings_never_ingested(self):
        assert inputs(fresh(filings_ok=None)) == {"filings"}

    def test_filings_reference_skips_a_holiday(self):
        """Tuesday after Labor Day: the session before is Friday."""
        tue = date(2026, 9, 8)
        assert freshness.filings_reference(tue) == datetime(2026, 9, 4, 16, 0, tzinfo=ET)

    def test_filings_reference_on_an_early_close(self):
        """The day after Thanksgiving closes at 13:00."""
        mon = date(2026, 11, 30)
        assert freshness.filings_reference(mon) == datetime(2026, 11, 27, 13, 0, tzinfo=ET)

    def test_distress_only_counts_for_a_held_name(self):
        old = MON - timedelta(hours=25)
        assert fresh(distress_ok=old) == []
        out = fresh(distress_ok=old, held=True)
        assert inputs(out) == {"distress news"} and out[0].blocks == ADDING

    def test_distress_boundary_is_24_hours(self):
        assert fresh(distress_ok=MON - timedelta(hours=24), held=True) == []

    def test_valuation_only_matters_under_an_intact_thesis(self):
        old = MON.date() - timedelta(days=11)
        assert fresh(valuation_asof=old) == []
        assert fresh(valuation_asof=MON.date() - timedelta(days=10), thesis="intact") == []
        out = fresh(valuation_asof=old, thesis="intact")
        assert inputs(out) == {"valuation"} and out[0].blocks == BONUS
        assert inputs(fresh(valuation_asof=None, thesis="intact")) == {"valuation"}

    def test_utc_stamps_are_compared_in_et(self):
        """A heartbeat stored in UTC is the same instant, not four hours off."""
        utc = (MON - timedelta(minutes=30)).astimezone(mc.ZoneInfo("UTC"))
        assert fresh(book_asof=utc) == []


# ── What a stale input does to a proposal ────────────────────────────────


def zone(pct=0.36, above=ENTRY_CONFIRM_SESSIONS, ps=3.0):
    return RelativeZone(
        price=100.0,
        ps_now=ps,
        median=3.6,
        percentile=pct,
        top=120.0,
        p25_price=90.0,
        p80_price=150.0,
        window_start=date(2024, 9, 28),
        window_end=date(2026, 9, 28),
        observations=500,
        sessions_above=above,
    )


def entering(stale=(), holding=None, thesis=None, price=100.0):
    return Sheet(
        symbol="AMZN",
        built_at=MON,
        move=Move(price=price, asof=MON.date(), day=0.0, sigma=0.02, z=0.0),
        zone=zone(),
        zone_prev=zone(ps=4.0, pct=0.6),
        holding=holding,
        thesis=thesis,
        stale=list(stale),
    )


def stale(name, blocks=ENTRY):
    return Stale(input=name, asof="2026-09-25T07:00:00-04:00", limit="x", blocks=blocks)


class TestProposal:
    def test_current_inputs_enter(self):
        assert build_proposal(entering(), net_liq=10_000).action is Action.ENTER

    def test_a_stale_price_turns_enter_into_wait_and_says_which(self):
        p = build_proposal(entering([stale("price")]), net_liq=10_000)
        assert p.action is Action.WAIT
        assert any(b.startswith("price is stale") for b in p.blockers)
        assert p.legs  # still shown: what it would have been
        assert p.features["stale"] == "price"

    def test_every_stale_input_is_named(self):
        p = build_proposal(entering([stale("book"), stale("filings")]), net_liq=10_000)
        assert p.action is Action.WAIT
        assert sum("is stale" in b for b in p.blockers) == 2

    def test_stale_distress_does_not_block_a_new_name(self):
        p = build_proposal(entering([stale("distress news", ADDING)]), net_liq=10_000)
        assert p.action is Action.ENTER

    def test_stale_distress_blocks_an_add(self):
        held = Holding(quantity=2, weight=0.05, unrealized=0.1, cost=90.0)
        p = build_proposal(entering([stale("distress news", ADDING)], holding=held), net_liq=10_000)
        assert p.action is Action.WAIT

    def test_an_exit_goes_out_with_stale_inputs(self):
        """Past its stop from cost: EXIT, however old the book or the filings."""
        held = Holding(quantity=10, weight=0.05, unrealized=-0.4, cost=100.0)
        s = entering([stale("price"), stale("filings")], holding=held, price=60.0)
        p = build_proposal(s, net_liq=10_000)
        assert p.action is Action.EXIT and p.legs == []

    def test_quiet_in_zone_is_not_blocked(self):
        """Nothing to enter, nothing to hold back: IN_ZONE stays IN_ZONE."""
        s = entering([stale("book")])
        s.zone_prev = zone()  # already in zone yesterday: no trigger
        assert build_proposal(s, net_liq=10_000).action is Action.IN_ZONE

    def test_stale_valuation_withholds_the_thesis_bonus(self):
        s = entering([stale("valuation", BONUS)], thesis="intact")
        p = build_proposal(s, net_liq=10_000)
        assert p.action is Action.ENTER  # the bonus goes, not the entry
        assert p.legs[0].risk_pct == pytest.approx(0.02)
        assert any("no thesis bonus" in g for g in p.gaps)
        fresh_bonus = build_proposal(entering(thesis="intact"), net_liq=10_000)
        assert fresh_bonus.legs[0].risk_pct == pytest.approx(0.03)

    def test_the_guard_limits_are_in_the_rules_version(self):
        from advisor.entry.ruleset import entry_rules

        params = entry_rules().params
        assert params["freshness.BOOK_MAX_AGE_MINUTES"] == 60
        assert entry_rules().kinds["freshness.BOOK_MAX_AGE_MINUTES"].value == "decided"


# ── The sheet reads the store ────────────────────────────────────────────


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def beat(store, job, when):
    store._conn.execute(
        "INSERT OR REPLACE INTO daemon_heartbeat (job, last_run_at, last_ok_at, run_count, "
        "error_count, last_error) VALUES (?, ?, ?, 1, 0, '')",
        (job, when.isoformat(), when.isoformat()),
    )
    store._conn.commit()


def closes_to(day, n=80):
    return [(day - timedelta(days=n - 1 - i), 100.0) for i in range(n)]


def sheet_at(store, now, last_bar):
    return build_sheet(
        store,
        "AMZN",
        now,
        closes=lambda s: closes_to(last_bar),
        consensus_loader=lambda st, s: None,
        series_loader=lambda s: None,
        margin_loader=lambda s: None,
        earnings_loader=lambda s: [],
    )


class TestSheet:
    def test_an_empty_store_names_every_stale_input(self, store):
        s = sheet_at(store, MON, MON.date())
        assert inputs(s.stale) == {"book", "filings"}

    def test_a_running_daemon_is_current(self, store):
        store.save_book(BookSnapshot(as_of=MON - timedelta(minutes=10), net_liq=10_000))
        beat(store, "brief", MON.replace(hour=7, minute=0))
        beat(store, "distress_premarket", MON.replace(hour=8, minute=15))
        assert sheet_at(store, MON, MON.date()).stale == []

    def test_the_newer_of_two_distress_sweeps_counts(self, store):
        beat(store, "distress_premarket", MON - timedelta(days=3))
        beat(store, "distress_midday", MON - timedelta(hours=2))
        assert freshness.latest_ok(store, freshness.DISTRESS_JOBS) == MON - timedelta(hours=2)

    def test_yesterdays_bar_in_session_is_stale(self, store):
        """yfinance without today's row: the entry price would be Friday's close."""
        store.save_book(BookSnapshot(as_of=MON - timedelta(minutes=10), net_liq=10_000))
        beat(store, "brief", MON.replace(hour=7, minute=0))
        s = sheet_at(store, MON, date(2026, 9, 25))
        assert inputs(s.stale) == {"price"}
