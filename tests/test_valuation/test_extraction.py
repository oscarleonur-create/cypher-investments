"""XBRL extraction, and the traps that make it dangerous.

Two live failures shaped every rule in here:

- SPCX's Q2 revenue appears sixteen times in the same period, once per
  segment. A read that took the first row would have reported $461M of
  products revenue as the company's $7,814M total — wrong by 17x, silently.
- A first version took the largest undimensioned `LongTermDebt` value and
  picked up "Proceeds from debt and other financing obligations", a
  cash-flow line of $51.8bn, as SPCX's $39.4bn balance-sheet debt.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from advisor.valuation.fundamentals import (
    _balance_rows,
    _duration_value,
    _frame,
    _share_total,
    _total_debt,
)


class FakeQuery:
    def __init__(self, frames: dict[str, pd.DataFrame], concept: str | None = None):
        self._frames, self._concept = frames, concept

    def by_concept(self, concept: str):
        return FakeQuery(self._frames, concept)

    def to_dataframe(self):
        return self._frames.get(self._concept)


class FakeXbrl:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = frames

    def query(self):
        return FakeQuery(self._frames)


# The real shape of SPCX's revenue facts, reduced to what matters.
REVENUE = pd.DataFrame(
    {
        "numeric_value": [461e6, 7_353e6, 7_814e6, 962e6, 4_291e6, 2_561e6],
        "is_dimensioned": [True, True, False, True, True, True],
        "period_start": ["2026-04-01"] * 6,
        "period_end": ["2026-06-30"] * 6,
    }
)

# The real shape of the debt facts: cash-flow lines and balance lines share
# the concept, and only the dated balance rows are debt.
DEBT = pd.DataFrame(
    {
        "numeric_value": [51_812e6, 39_396e6, 2_525e6, 36_839e6, 928e6, 21_968e6],
        "is_dimensioned": [False] * 6,
        "period_key": [
            "duration_2026-01-01_2026-06-30",
            "duration_2026-01-01_2026-06-30",
            "instant_2026-06-30",
            "instant_2026-06-30",
            "instant_2025-12-31",
            "instant_2025-12-31",
        ],
        "statement_type": [
            "CashFlow",
            "CashFlow",
            "BalanceSheet",
            "BalanceSheet",
            "BalanceSheet",
            "BalanceSheet",
        ],
    }
)

SHARES = pd.DataFrame(
    {"numeric_value": [7_696_293_669.0, 5_485_486_276.0], "is_dimensioned": [True, True]}
)


class TestDimensionedFacts:
    def test_only_the_undimensioned_revenue_is_the_total(self):
        """Fifteen slices and one total; the total is $7,814M."""
        value = _duration_value(
            FakeXbrl({"Revenues": REVENUE}),
            ("Revenues",),
            date(2026, 4, 1),
            date(2026, 6, 30),
        )
        assert value == pytest.approx(7_814e6)

    def test_a_segment_figure_is_never_returned_as_the_total(self):
        value = _duration_value(
            FakeXbrl({"Revenues": REVENUE}), ("Revenues",), date(2026, 4, 1), date(2026, 6, 30)
        )
        for slice_value in (461e6, 962e6, 4_291e6, 2_561e6):
            assert value != slice_value

    def test_a_frame_with_no_undimensioned_rows_yields_nothing(self):
        only_slices = REVENUE[REVENUE["is_dimensioned"]]
        assert _frame(FakeXbrl({"R": only_slices}), "R") is None

    def test_a_missing_concept_yields_nothing(self):
        assert _frame(FakeXbrl({}), "Nope") is None

    def test_a_frame_without_numeric_values_does_not_raise(self):
        odd = pd.DataFrame({"is_dimensioned": [False], "label": ["x"]})
        assert _frame(FakeXbrl({"R": odd}), "R") is None


class TestBalanceFacts:
    def test_the_cash_flow_line_is_not_mistaken_for_debt(self):
        """The exact bug: $51.8bn of proceeds read as $39.4bn of debt."""
        debt, mentioned = _total_debt(FakeXbrl({"LongTermDebt": DEBT}), date(2026, 6, 30))
        assert debt == pytest.approx(39_364e6)
        assert debt != pytest.approx(51_812e6)
        assert mentioned is True

    def test_debt_is_current_plus_non_current(self):
        debt, _ = _total_debt(FakeXbrl({"LongTermDebt": DEBT}), date(2026, 6, 30))
        assert debt == pytest.approx(2_525e6 + 36_839e6)

    def test_the_prior_period_balance_is_not_used(self):
        debt, _ = _total_debt(FakeXbrl({"LongTermDebt": DEBT}), date(2025, 12, 31))
        assert debt == pytest.approx(928e6 + 21_968e6)

    def test_an_instant_with_no_facts_yields_nothing(self):
        assert (
            _balance_rows(FakeXbrl({"LongTermDebt": DEBT}), "LongTermDebt", date(2024, 1, 1))
            is None
        )

    def test_a_filing_that_never_mentions_debt_reports_so(self):
        """CRDO has no debt; that is different from debt it could not parse."""
        debt, mentioned = _total_debt(FakeXbrl({}), date(2026, 6, 30))
        assert debt is None
        assert mentioned is False

    def test_debt_mentioned_but_unparseable_is_distinguished_from_absent(self):
        only_cashflow = DEBT[DEBT["statement_type"] == "CashFlow"]
        debt, mentioned = _total_debt(FakeXbrl({"LongTermDebt": only_cashflow}), date(2026, 6, 30))
        assert debt is None
        assert mentioned is True  # so the caller must not assume zero

    def test_the_statement_filter_is_dropped_when_it_would_leave_nothing(self):
        """AAOI tags cash without a BalanceSheet statement_type."""
        untyped = pd.DataFrame(
            {
                "numeric_value": [500e6],
                "is_dimensioned": [False],
                "period_key": ["instant_2026-06-30"],
                "statement_type": ["Unknown"],
            }
        )
        rows = _balance_rows(FakeXbrl({"Cash": untyped}), "Cash", date(2026, 6, 30))
        assert rows is not None and len(rows) == 1


class TestShareCounts:
    def test_classes_are_summed_because_classes_are_the_dimension(self):
        """The one place the undimensioned rule is deliberately not applied."""
        assert _share_total(
            FakeXbrl({"EntityCommonStockSharesOutstanding": SHARES})
        ) == pytest.approx(13_181_779_945.0)

    def test_a_single_class_issuer_still_works(self):
        one = pd.DataFrame({"numeric_value": [84_569_237.0], "is_dimensioned": [False]})
        assert _share_total(FakeXbrl({"EntityCommonStockSharesOutstanding": one})) == pytest.approx(
            84_569_237.0
        )

    def test_taking_one_class_alone_would_understate_the_company(self):
        total = _share_total(FakeXbrl({"EntityCommonStockSharesOutstanding": SHARES}))
        assert total > 7_696_293_669.0
        assert total > 5_485_486_276.0

    def test_a_missing_share_concept_yields_nothing(self):
        assert _share_total(FakeXbrl({})) is None
