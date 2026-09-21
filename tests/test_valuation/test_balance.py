"""Interim balance figures, proved against the periodic filing we already hold.

`interim.py` takes an income statement only where a sentence's own arithmetic
reconciles. A balance sheet offers no such prose — it is two flattened columns
with the note number sitting between the label and the figures:

    Cash and cash equivalents 4 3,678.1 8,042.1

The proof comes from outside. Nebius's 20-F gives $3,678.1M of cash at 31
December 2025, and that is the left-hand column. When the prior column
reproduces a figure we independently hold, the columns were read in the right
order and the right-hand one can be trusted.

Mixing halves is not hypothetical: between these two columns Nebius raised
$5.75bn of convertibles, taking cash from $3.7bn to $8.0bn and debt from
$4.1bn to $8.5bn. A June revenue over a December balance sheet would move
enterprise value by billions, in the flattering direction.
"""

from __future__ import annotations

from datetime import date

import pytest
from advisor.valuation.balance import confirm_balance
from advisor.valuation.models import Fundamentals

# Verbatim from nbis-20260812xex99d2.htm, flattened as the reader sees it.
HEADER = "As of December 31, June 30, Notes 2025 2026 ASSETS "
CASH = "Cash and cash equivalents 4 3,678.1 8,042.1 "
DEBT = (
    "Accounts payable, accrued and other 4 1,210.1 1,301.0 liabilities "
    "Debt, current 12 24.5 46.7 Income and non-income taxes 17.7 59.0 payable "
    "Operating lease liabilities, 8 760.5 1,510.4 non-current "
    "Debt, non-current 12 4,103.2 8,499.0 Deferred revenue, non-current 1,302.0 4,995.8 "
)
# The issued block comes first and carries a non-nil Class C; only the
# outstanding block belongs in a market capitalisation.
SHARES = (
    "Ordinary shares: par value (Class A EUR0.01, Class B EUR0.10 and Class C EUR0.09); "
    "shares issued (Class A: 288,489,061 and 288,585,891, respectively, Class B: 14 8.4 "
    "8.4 33,551,883 and 33,455,053, respectively, and Class C: 2,146,791 and 2,243,621, "
    "respectively); shares outstanding (Class A: 219,465,088 and 238,400,165, "
    "respectively, Class B: 33,551,883 and 33,455,053, respectively, and Class C: nil) "
    "Treasury shares at cost (Class A: 69,023,973 and 50,185,726, respectively) "
)
# The convertible notes mature in 2030, and that date sits in a footnote.
FOOTNOTE = (
    "The 2030 Notes will mature on October 31, 2030, unless earlier converted, "
    "redeemed or repurchased. "
)

BALANCE_SHEET = HEADER + CASH + DEBT + SHARES + FOOTNOTE


def known(
    *,
    period_end: date = date(2025, 12, 31),
    cash: float | None = 3_678_100_000.0,
    total_debt: float | None = 4_127_700_000.0,
    shares: float | None = 253_016_971.0,
) -> Fundamentals:
    return Fundamentals(
        symbol="NBIS",
        source_accession="0001104659-26-052948",
        period_end=period_end,
        fiscal_period="FY",
        revenue=529_800_000.0,
        cash=cash,
        total_debt=total_debt,
        shares_outstanding=shares,
    )


# --- the happy path, which is the real filing -------------------------------


def test_the_current_column_is_taken_when_the_prior_one_is_proved():
    b = confirm_balance(BALANCE_SHEET, known())
    assert b.period_end == date(2026, 6, 30)
    assert b.cash == 8_042_100_000.0
    assert b.total_debt == 46_700_000.0 + 8_499_000_000.0
    assert b.shares_outstanding == 238_400_165 + 33_455_053
    assert sorted(b.confirmed) == ["cash", "debt", "shares"]
    assert b.complete


def test_debt_is_the_sum_of_its_two_lines():
    """The 20-F's $4,127.7M is 24.5 + 4,103.2; either line alone proves nothing."""
    b = confirm_balance(BALANCE_SHEET, known())
    assert b.total_debt == 8_545_700_000.0


def test_outstanding_shares_win_over_issued():
    """Issued is 288,585,891 Class A; outstanding is 238,400,165. The
    difference is 50m treasury shares, and counting them would overstate the
    market capitalisation by a fifth."""
    b = confirm_balance(BALANCE_SHEET, known())
    assert b.shares_outstanding == 271_855_218
    assert b.shares_outstanding < 288_585_891


def test_a_nil_share_class_contributes_nothing():
    b = confirm_balance(BALANCE_SHEET, known())
    assert b.shares_outstanding == 238_400_165 + 33_455_053


# --- the proof failing ------------------------------------------------------


@pytest.mark.parametrize(
    "field,value,concept",
    [
        ("cash", 1_000_000_000.0, "cash"),
        ("total_debt", 1_000_000_000.0, "debt"),
        ("shares", 1.0, "shares"),
    ],
)
def test_a_concept_whose_prior_column_disagrees_is_dropped(field, value, concept):
    b = confirm_balance(BALANCE_SHEET, known(**{field: value}))
    assert (
        getattr(
            b,
            "cash"
            if concept == "cash"
            else "total_debt"
            if concept == "debt"
            else "shares_outstanding",
        )
        is None
    )
    assert any(r.startswith(concept) for r in b.unconfirmed)
    assert not b.complete


def test_one_failed_concept_does_not_sink_the_others():
    b = confirm_balance(BALANCE_SHEET, known(cash=1_000_000_000.0))
    assert b.cash is None
    assert b.total_debt is not None and b.shares_outstanding is not None


@pytest.mark.parametrize("field", ["cash", "total_debt", "shares"])
def test_nothing_known_for_the_prior_period_proves_nothing(field):
    b = confirm_balance(BALANCE_SHEET, known(**{field: None}))
    assert not b.complete
    assert any("nothing known" in r for r in b.unconfirmed)


def test_a_missing_line_is_reported_not_guessed():
    b = confirm_balance(BALANCE_SHEET.replace(CASH, ""), known())
    assert b.cash is None
    assert any("no two-column line" in r for r in b.unconfirmed)


def test_only_one_debt_line_proves_nothing():
    b = confirm_balance(BALANCE_SHEET.replace("Debt, non-current 12 4,103.2 8,499.0 ", ""), known())
    assert b.total_debt is None


# --- dating the columns -----------------------------------------------------


def test_the_balance_is_dated_by_its_header_not_by_a_footnote():
    """Taking the newest date anywhere read 2030-10-31 — the maturity of the
    convertible notes."""
    b = confirm_balance(BALANCE_SHEET, known())
    assert b.period_end == date(2026, 6, 30)


def test_a_prior_column_dated_differently_from_the_known_filing_is_refused():
    b = confirm_balance(BALANCE_SHEET, known(period_end=date(2025, 9, 30)))
    assert not b.complete
    assert any("prior column is dated" in r for r in b.unconfirmed)


def test_columns_in_the_other_order_are_refused():
    """Some filers put the current period on the left. Then the prior column
    is not the period we hold, and the figures beside it are not the ones
    being proved — so nothing is taken."""
    reversed_header = BALANCE_SHEET.replace(
        "As of December 31, June 30, Notes 2025 2026",
        "As of June 30, December 31, Notes 2026 2025",
    )
    b = confirm_balance(reversed_header, known())
    assert not b.complete
    assert any("prior column is dated" in r for r in b.unconfirmed)


def test_a_header_naming_one_date_twice_is_refused():
    same = BALANCE_SHEET.replace(
        "As of December 31, June 30, Notes 2025 2026",
        "As of December 31, December 31, Notes 2025 2025",
    )
    b = confirm_balance(same, known())
    assert not b.complete
    assert any("not after" in r for r in b.unconfirmed)


def test_no_header_means_no_reading():
    b = confirm_balance(CASH + DEBT + SHARES, known())
    assert not b.complete
    assert any("no balance-sheet header" in r for r in b.unconfirmed)


def test_empty_text_is_refused_rather_than_failing():
    b = confirm_balance("", known())
    assert not b.complete and b.cash is None
