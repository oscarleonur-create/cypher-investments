"""Position mechanics — above all, that thresholds are edge-triggered.

The level-vs-edge distinction is the load-bearing one here. Measured as a
level, a book held through a drawdown fires a Tier A stop alert on every
position on every sweep, forever. Live data made that failure obvious: 9 of 11
real positions tripped at once.
"""

from __future__ import annotations

from datetime import timedelta

from advisor.daemon.book import EQUITY, EQUITY_OPTION, BookSnapshot, Position
from advisor.daemon.market_calendar import now_et
from advisor.daemon.mechanics import (
    MechanicsLimits,
    crossing_events,
    diff_events,
    state_events,
)
from advisor.daemon.models import EventTier


def pos(symbol="AMD", *, qty=2, entry=100.0, price=100.0, account="A", **kw) -> Position:
    return Position(
        account=account,
        symbol=symbol,
        underlying=symbol,
        instrument=kw.pop("instrument", EQUITY),
        quantity=qty,
        multiplier=kw.pop("multiplier", 1),
        avg_open_price=entry,
        close_price=price,
        mark_price=0.0,
        **kw,
    )


def book(*positions, net_liq=10_000.0, at=None) -> BookSnapshot:
    return BookSnapshot(as_of=at or now_et(), positions=list(positions), net_liq=net_liq)


def kinds(events) -> list[str]:
    return sorted(e.kind for e in events)


class TestEdgeTriggering:
    def test_a_position_already_through_the_stop_stays_silent(self):
        """The bug live data caught: a long-standing loss is not an event."""
        underwater = book(pos(price=60.0))  # -40%, unchanged
        assert crossing_events(underwater, underwater) == []

    def test_crossing_down_through_the_stop_fires_tier_a(self):
        before = book(pos(price=97.0))  # -3%
        after = book(pos(price=90.0))  # -10%
        events = crossing_events(before, after)
        assert kinds(events) == ["STOP_BREACHED"]
        assert events[0].tier == EventTier.A
        assert events[0].payload["previous_pct"] == -0.03

    def test_falling_further_below_the_stop_does_not_refire(self):
        before = book(pos(price=90.0))  # already -10%
        after = book(pos(price=80.0))  # -20%, still below
        assert crossing_events(before, after) == []

    def test_recovering_above_and_re_breaching_fires_again(self):
        recovered = book(pos(price=95.0))  # -5%
        broken = book(pos(price=91.0))  # -9%
        assert kinds(crossing_events(recovered, broken)) == ["STOP_BREACHED"]

    def test_exactly_at_the_threshold_counts_as_crossed(self):
        before = book(pos(price=95.0))
        after = book(pos(price=92.0))  # exactly -8%
        assert kinds(crossing_events(before, after)) == ["STOP_BREACHED"]

    def test_profit_target_crossing_is_tier_b_not_an_interrupt(self):
        before = book(pos(price=120.0))  # +20%
        after = book(pos(price=126.0))  # +26%
        events = crossing_events(before, after)
        assert kinds(events) == ["PROFIT_TARGET_HIT"]
        assert events[0].tier == EventTier.B

    def test_short_position_crossing_uses_direction_adjusted_return(self):
        """Short loses when price rises."""
        before = book(pos(qty=-2, price=101.0))  # -1% for a short
        after = book(pos(qty=-2, price=110.0))  # -10% for a short
        assert kinds(crossing_events(before, after)) == ["STOP_BREACHED"]

    def test_options_are_skipped_by_equity_crossings(self):
        before = book(pos(instrument=EQUITY_OPTION, multiplier=100, price=97.0))
        after = book(pos(instrument=EQUITY_OPTION, multiplier=100, price=80.0))
        assert crossing_events(before, after) == []

    def test_a_newly_opened_position_cannot_cross(self):
        assert crossing_events(book(), book(pos(price=50.0))) == []


class TestStandingConditions:
    def test_never_emits_tier_a(self):
        """Standing conditions are digest material by construction."""
        b = book(pos(price=60.0), pos("COHR", price=60.0))
        assert all(e.tier != EventTier.A for e in state_events(b))

    def test_deep_drawdown_is_reported(self):
        events = state_events(book(pos(price=70.0)))  # -30%
        assert "DEEP_DRAWDOWN" in kinds(events)

    def test_deep_drawdown_dedups_by_week_not_by_day(self):
        """36% underwater deserves a weekly reminder, not a daily one."""
        monday = now_et()
        tuesday = monday + timedelta(days=1)
        a = state_events(book(pos(price=70.0), at=monday))[0]
        b = state_events(book(pos(price=70.0), at=tuesday))[0]
        assert a.dedup_hash() == b.dedup_hash()

    def test_concentration_uses_summed_exposure_across_accounts(self):
        b = book(
            pos(account="A", qty=15, price=100.0),
            pos(account="B", qty=10, price=100.0),
            net_liq=10_000.0,
        )  # 2500 combined = 25%
        conc = [e for e in state_events(b) if e.kind == "CONCENTRATION_WARNING"]
        assert len(conc) == 1
        assert conc[0].payload["weight"] == 0.25

    def test_no_concentration_warning_when_net_liq_is_zero(self):
        b = book(pos(qty=10, price=100.0), net_liq=0.0)
        assert [e for e in state_events(b) if e.kind == "CONCENTRATION_WARNING"] == []

    def test_empty_book_produces_nothing(self):
        assert state_events(book()) == []


class TestDiff:
    def test_first_run_reports_nothing(self):
        """No baseline must not mean 'every holding was just opened'."""
        assert diff_events(None, book(pos(), pos("COHR"))) == []

    def test_opened_and_closed(self):
        before = book(pos("AMD"))
        after = book(pos("COHR"))
        assert kinds(diff_events(before, after)) == ["POSITION_CLOSED", "POSITION_OPENED"]

    def test_size_change_is_reported(self):
        events = diff_events(book(pos(qty=10)), book(pos(qty=20)))
        assert kinds(events) == ["POSITION_SIZE_CHANGED"]
        assert events[0].payload["direction"] == "increased"

    def test_trivial_drift_is_ignored(self):
        """Fractional reinvestment must not generate an event."""
        assert diff_events(book(pos(qty=100)), book(pos(qty=102))) == []

    def test_same_symbol_in_two_accounts_tracked_separately(self):
        before = book(pos(account="A"), pos(account="B"))
        after = book(pos(account="A"))
        events = diff_events(before, after)
        assert kinds(events) == ["POSITION_CLOSED"]
        assert events[0].payload["account"] == "B"

    def test_unchanged_book_is_silent(self):
        b = book(pos(), pos("COHR"))
        assert diff_events(b, b) == []


class TestLimitsAreConfigurable:
    def test_a_tighter_stop_changes_the_trigger(self):
        limits = MechanicsLimits(equity_stop_pct=-0.03)
        before, after = book(pos(price=99.0)), book(pos(price=95.0))
        assert kinds(crossing_events(before, after, limits=limits)) == ["STOP_BREACHED"]
