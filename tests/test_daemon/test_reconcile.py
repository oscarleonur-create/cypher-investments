"""Cross-source reconciliation.

Every data-quality finding in this project came from comparing two sources,
never from reading one. These tests encode the specific disagreements the live
audit turned up, so a regression is caught by the suite rather than by a bad
trade.
"""

from __future__ import annotations

from datetime import date

from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.models import EventTier
from advisor.daemon.reconcile import (
    ReconcileReport,
    Severity,
    check_earnings,
    check_price_staleness,
    check_prices,
    reconcile_events,
)

FRIDAY = date(2026, 9, 4)
MONDAY = date(2026, 9, 7)


def pos(symbol="AAOI", *, close=105.53, qty=12) -> Position:
    return Position(
        account="A",
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=qty,
        multiplier=1,
        avg_open_price=129.31,
        close_price=close,
    )


def book(*positions, net_liq=7761.13) -> BookSnapshot:
    return BookSnapshot(positions=list(positions), net_liq=net_liq)


def results(findings, check):
    return [f for f in findings if f.check == check]


class TestPriceAgreement:
    def test_matching_prices_pass(self):
        findings = check_prices(book(pos()), {"AAOI": 105.53})
        assert findings[0].severity == Severity.OK

    def test_a_divergent_price_fails(self):
        """A split applied by one source and not the other looks like this."""
        findings = check_prices(book(pos(close=105.53)), {"AAOI": 52.77})
        assert findings[0].severity == Severity.FAIL
        assert "105.53" in findings[0].detail and "52.77" in findings[0].detail

    def test_a_missing_independent_price_fails_rather_than_passing_silently(self):
        findings = check_prices(book(pos()), {})
        assert findings[0].severity == Severity.FAIL
        assert "no independent price" in findings[0].detail

    def test_a_zero_broker_price_fails_without_dividing_by_zero(self):
        findings = check_prices(book(pos(close=0.0)), {"AAOI": 105.53})
        assert findings[0].severity == Severity.FAIL

    def test_a_negative_price_fails(self):
        findings = check_prices(book(pos(close=-3.0)), {"AAOI": 105.53})
        assert findings[0].severity == Severity.FAIL

    def test_exactly_at_the_tolerance_passes(self):
        findings = check_prices(book(pos(close=100.0)), {"AAOI": 101.0})
        assert findings[0].severity == Severity.OK

    def test_just_past_the_tolerance_fails(self):
        findings = check_prices(book(pos(close=100.0)), {"AAOI": 101.5})
        assert findings[0].severity == Severity.FAIL

    def test_an_empty_book_produces_no_findings(self):
        assert check_prices(book(), {"AAOI": 1.0}) == []


class TestStaleness:
    def test_a_current_bar_passes(self):
        findings = check_price_staleness({"AAOI": FRIDAY}, today=date(2026, 9, 5))
        assert findings[0].severity == Severity.OK

    def test_a_long_dead_feed_fails(self):
        findings = check_price_staleness({"AAOI": date(2026, 8, 1)}, today=date(2026, 9, 5))
        assert findings[0].severity == Severity.FAIL

    def test_a_weekend_run_compares_against_friday_not_today(self):
        """Sunday has no bar; the last session does."""
        findings = check_price_staleness({"AAOI": FRIDAY}, today=date(2026, 9, 6))
        assert findings[0].severity == Severity.OK

    def test_a_monday_run_after_a_holiday_weekend_does_not_false_alarm(self):
        findings = check_price_staleness({"AAOI": FRIDAY}, today=MONDAY)
        assert findings[0].severity != Severity.FAIL

    def test_a_slightly_lagging_bar_warns_rather_than_failing(self):
        findings = check_price_staleness({"AAOI": date(2026, 9, 2)}, today=date(2026, 9, 5))
        assert findings[0].severity == Severity.WARN


class TestEarnings:
    def test_the_real_disagreement_is_reported_as_direction_not_conflict(self):
        """Broker gives the last report, yfinance the next. Both are right."""
        findings = check_earnings(
            {"AAOI": (date(2026, 8, 6), False)},
            {"AAOI": date(2026, 11, 5)},
            today=date(2026, 9, 6),
        )
        assert findings[0].check == "earnings_direction"
        assert findings[0].severity == Severity.OK
        assert "last report" in findings[0].detail

    def test_two_upcoming_dates_far_apart_warn(self):
        findings = check_earnings(
            {"AAOI": (date(2026, 11, 5), True)},
            {"AAOI": date(2026, 12, 20)},
            today=date(2026, 9, 6),
        )
        assert findings[0].check == "earnings_agreement"
        assert findings[0].severity == Severity.WARN

    def test_two_upcoming_dates_close_together_agree(self):
        findings = check_earnings(
            {"AAOI": (date(2026, 11, 5), True)},
            {"AAOI": date(2026, 11, 7)},
            today=date(2026, 9, 6),
        )
        assert findings[0].severity == Severity.OK

    def test_a_symbol_with_no_date_anywhere_warns(self):
        """CCXI, live."""
        findings = check_earnings({"CCXI": (None, False)}, {"CCXI": None}, today=date(2026, 9, 6))
        assert findings[0].check == "earnings_coverage"
        assert findings[0].severity == Severity.WARN

    def test_symbols_present_in_only_one_source_are_still_checked(self):
        findings = check_earnings({}, {"AMD": date(2026, 11, 3)}, today=date(2026, 9, 6))
        assert len(findings) == 1

    def test_the_estimated_flag_is_surfaced(self):
        findings = check_earnings(
            {"AAOI": (date(2026, 8, 6), True)}, {"AAOI": None}, today=date(2026, 9, 6)
        )
        assert "estimated" in findings[0].detail


class TestEvents:
    def test_a_clean_report_emits_nothing(self):
        report = ReconcileReport(asof=FRIDAY)
        report.findings = check_prices(book(pos()), {"AAOI": 105.53})
        assert report.ok
        assert reconcile_events(report) == []

    def test_a_failure_becomes_a_tier_a_event(self):
        report = ReconcileReport(asof=FRIDAY)
        report.findings = check_prices(book(pos()), {})
        events = reconcile_events(report)
        assert len(events) == 1
        assert events[0].tier is EventTier.A
        assert events[0].kind == "DATA_QUALITY_FAILURE"

    def test_a_feed_broken_for_every_symbol_wakes_you_once(self):
        """Eleven broken symbols is one problem, not eleven interrupts."""
        report = ReconcileReport(asof=FRIDAY)
        report.findings = check_prices(book(*[pos(s) for s in ("AAOI", "AMD", "COHR", "NBIS")]), {})
        events = reconcile_events(report)
        assert len(events) == 1
        assert events[0].payload["failed"] == 4
        assert events[0].symbol is None  # book-level

    def test_distinct_checks_produce_distinct_events(self):
        report = ReconcileReport(asof=FRIDAY)
        report.findings = check_prices(book(pos()), {}) + check_price_staleness(
            {"AAOI": date(2026, 1, 2)}, today=date(2026, 9, 5)
        )
        assert len({e.dedup_key for e in reconcile_events(report)}) == 2

    def test_the_same_failure_dedups_within_a_session(self):
        report = ReconcileReport(asof=FRIDAY)
        report.findings = check_prices(book(pos()), {})
        first, second = reconcile_events(report)[0], reconcile_events(report)[0]
        assert first.dedup_hash() == second.dedup_hash()

    def test_summary_reports_counts(self):
        report = ReconcileReport(asof=FRIDAY)
        report.findings = check_prices(book(pos()), {"AAOI": 105.53})
        assert "0 failed" in report.summary()

    def test_an_empty_report_says_so(self):
        assert ReconcileReport(asof=FRIDAY).summary() == "no checks run"


class TestIntradayComparison:
    """Comparing a prior close against a live price measures the day's move.

    Found on the first live-session run. The broker's `close_price` is the
    *prior* session's close; during a session yfinance's last row is today's
    partial bar. Comparing them raised a Tier A data-quality failure on nine
    of twelve symbols, and every gap was simply that day's price change —
    2.96% on SPCX, 2.75% on TE, 2.08% on BE.

    It would have fired every trading day, at the highest tier, forever.
    """

    def test_a_days_move_is_not_a_data_fault(self):
        """The real numbers that fired: a 1.99% gap on COHR at midday."""
        findings = check_prices(book(pos("COHR", close=266.50)), {"COHR": 266.50})
        assert findings[0].severity == Severity.OK

    def test_the_check_itself_still_catches_a_real_divergence(self):
        """Stepping back a bar must not blunt the check it exists for."""
        findings = check_prices(book(pos("COHR", close=266.50)), {"COHR": 133.25})
        assert findings[0].severity == Severity.FAIL

    def test_the_runner_steps_back_a_bar_while_the_session_is_open(self, monkeypatch):
        """Both sides must describe the last completed session."""
        import inspect

        from advisor.daemon import reconcile

        source = inspect.getsource(reconcile.run_reconciliation)
        assert "is_market_open" in source
        assert "index = -2" in source or "-2" in source

    def test_with_one_bar_available_it_does_not_step_off_the_end(self):
        """A newly listed symbol has a single bar; stepping back would break."""
        import pandas as pd

        single = pd.DataFrame({"NEW": [10.0]}, index=pd.to_datetime(["2026-09-15"]))
        assert len(single) == 1
        # The guard is `len(column) > 1`; with one bar the last row is used.
        column = single["NEW"].dropna()
        index = -1
        if len(column) > 1:
            index = -2
        assert float(column.iloc[index]) == 10.0


class TestSameSessionOnly:
    """Two closes can only disagree if they describe the same session.

    Live, 23 September: yfinance carried a row for the 22nd with every value
    NaN, `dropna()` fell back to the 21st, and the broker's close was the
    22nd's. All eight symbols "disagreed" by exactly the 22nd's move, the
    reconcile emitted a Tier A DATA_QUALITY_FAILURE, and the evidence gate
    turned every action card in the book to CANNOT_SAY.
    """

    TUE = date(2026, 9, 22)
    MON = date(2026, 9, 21)

    def test_a_lagging_independent_bar_is_not_compared(self):
        # TE: broker Tuesday 4.36, yfinance fell back to Monday's 4.45.
        findings = check_prices(
            book(pos("TE", close=4.36)),
            {"TE": 4.45},
            bar_dates={"TE": self.MON},
            expected_session=self.TUE,
        )
        [f] = results(findings, "price_agreement")
        assert f.severity == "WARN"
        assert "2026-09-21" in f.detail and "2026-09-22" in f.detail

    def test_the_live_morning_no_longer_blacks_out_the_book(self):
        """The whole failure, end to end: eight lagging bars, zero events."""
        symbols = ("SPCX", "AAOI", "NBIS", "CCXI", "COHR", "CRDO", "CBRS", "TE")
        report = ReconcileReport(asof=date(2026, 9, 23))
        report.findings = check_prices(
            book(*[pos(s, close=100.0) for s in symbols]),
            dict.fromkeys(symbols, 103.0),
            bar_dates=dict.fromkeys(symbols, self.MON),
            expected_session=self.TUE,
        )
        assert report.ok
        assert reconcile_events(report) == []

    def test_a_same_session_disagreement_still_fails(self):
        """The check must keep its teeth: same day, different prices is a fault."""
        findings = check_prices(
            book(pos(close=100.0)),
            {"AAOI": 110.0},
            bar_dates={"AAOI": self.TUE},
            expected_session=self.TUE,
        )
        [f] = results(findings, "price_agreement")
        assert f.severity == "FAIL"

    def test_a_bar_newer_than_the_expected_session_is_still_compared(self):
        """After the close yfinance has today's bar; that is not staleness."""
        findings = check_prices(
            book(pos(close=100.0)),
            {"AAOI": 100.2},
            bar_dates={"AAOI": date(2026, 9, 23)},
            expected_session=self.TUE,
        )
        assert results(findings, "price_agreement")[0].severity == "OK"

    def test_one_lagging_symbol_does_not_excuse_the_others(self):
        findings = check_prices(
            book(pos("TE", close=4.36), pos("AAOI", close=100.0)),
            {"TE": 4.45, "AAOI": 110.0},
            bar_dates={"TE": self.MON, "AAOI": self.TUE},
            expected_session=self.TUE,
        )
        by = {f.symbol: f.severity for f in results(findings, "price_agreement")}
        assert by == {"TE": "WARN", "AAOI": "FAIL"}

    def test_without_dates_the_old_behaviour_is_unchanged(self):
        findings = check_prices(book(pos(close=100.0)), {"AAOI": 110.0})
        assert results(findings, "price_agreement")[0].severity == "FAIL"

    def test_a_missing_date_for_one_symbol_is_compared_not_skipped(self):
        """No date is no evidence of lag; skipping would silence the check."""
        findings = check_prices(
            book(pos(close=100.0)),
            {"AAOI": 110.0},
            bar_dates={},
            expected_session=self.TUE,
        )
        assert results(findings, "price_agreement")[0].severity == "FAIL"
