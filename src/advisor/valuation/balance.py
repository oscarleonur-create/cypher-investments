"""Interim balance-sheet figures, proved against the filing we already have.

`interim.py` reads the income statement out of a 6-K by taking only what the
exhibit states twice. The balance sheet has no such prose: it is a plain
two-column table, prior period and current, flattened into the text with the
note number sitting between the label and the figures.

    Cash and cash equivalents 4 3,678.1 8,042.1
    Debt, current 12 24.5 46.7
    Debt, non-current 12 4,103.2 8,499.0

Nothing inside that confirms it. The confirmation comes from outside: **the
prior column is a period we have already read from XBRL**. Nebius's 20-F gives
cash of $3,678.1M and debt of $4,127.7M at 31 December 2025, and 253,016,971
shares outstanding. Every one of those is the left-hand column above. When the
left column reproduces a figure we independently hold, the columns were read
in the right order and the right-hand one can be trusted. When it does not,
the concept is dropped.

This is the same standard `interim.py` applies, with the second statement
coming from a different document rather than a different sentence. A figure
that cannot be proved this way is absent, never estimated.
"""

from __future__ import annotations

import logging
import re
from datetime import date

from pydantic import BaseModel, Field

from advisor.valuation.interim import _normalise, _number
from advisor.valuation.models import Fundamentals

logger = logging.getLogger(__name__)

# Balance figures are stated in millions to one decimal, so the prior column
# and an XBRL fact agree to within a rounding of the smaller unit.
MATCH_TOLERANCE_USD = 100_000.0

# Share counts are exact integers in both places; they either match or the
# columns were misread.
SHARE_TOLERANCE = 1.0

_MILLIONS = 1e6


class InterimBalance(BaseModel):
    """What the interim balance sheet proved, and what it could not."""

    period_end: date | None = None
    cash: float | None = None
    total_debt: float | None = None
    shares_outstanding: float | None = None
    confirmed: list[str] = Field(default_factory=list)
    unconfirmed: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return (
            self.cash is not None
            and self.total_debt is not None
            and self.shares_outstanding is not None
            and self.period_end is not None
        )


def _two_column(text: str, label: str) -> tuple[float, float] | None:
    """The prior and current figures on one balance-sheet line.

    A note number can sit between the label and its figures — "Cash and cash
    equivalents 4 3,678.1 8,042.1" — and a note is an integer while a balance
    figure always carries its decimal. That is what tells them apart.
    """
    pattern = re.compile(
        re.escape(label) + r"\s*(?:\d{1,2}(?:,\s*\d{1,2})*\s+)?" r"(-?[\d,]+\.\d)\s+(-?[\d,]+\.\d)",
        re.I,
    )
    m = pattern.search(text)
    if m is None:
        return None
    return _number(m.group(1)) * _MILLIONS, _number(m.group(2)) * _MILLIONS


def _share_classes(text: str) -> tuple[float, float] | None:
    """Ordinary shares outstanding, summed across classes.

    The cover of a foreign issuer's interim statements gives issued and
    outstanding separately, and the two differ by treasury: Nebius reports
    288,585,891 Class A issued against 238,400,165 outstanding. Only the
    outstanding count belongs in a market capitalisation.
    """
    window = re.search(r"outstanding\s*\((?P<body>[^)]{20,400})\)", text, re.I)
    if window is None:
        return None
    prior_total = current_total = 0.0
    found = False
    for m in re.finditer(
        r"Class\s+[ABC]\s*:\s*([\d,]{5,}|nil)\s+and\s+([\d,]{5,}|nil)", window.group("body"), re.I
    ):
        left, right = m.group(1), m.group(2)
        if left.lower() == "nil" and right.lower() == "nil":
            continue
        prior_total += 0.0 if left.lower() == "nil" else _number(left)
        current_total += 0.0 if right.lower() == "nil" else _number(right)
        found = True
    return (prior_total, current_total) if found else None


_MONTHS = (
    "January February March April May June July August September October " "November December"
).split()


# "As of December 31, June 30, Notes 2025 2026" — the two column headings and
# then the two years, in the same order.
_BALANCE_HEADER = re.compile(
    r"As\s+of\s+([A-Z][a-z]+)\s+(\d{1,2}),\s*([A-Z][a-z]+)\s+(\d{1,2}),"
    r"[^\d]{0,40}(\d{4})\s+(\d{4})",
)


def _column_dates(text: str) -> tuple[date, date] | None:
    """The two dates the balance sheet's own header names.

    Taking the newest date anywhere in the document read 2030-10-31 — the
    maturity of the convertible notes, sitting in a footnote. A balance sheet
    is dated by its header and by nothing else.
    """
    m = _BALANCE_HEADER.search(text)
    if m is None:
        return None
    try:
        prior = date(int(m.group(5)), _MONTHS.index(m.group(1)) + 1, int(m.group(2)))
        current = date(int(m.group(6)), _MONTHS.index(m.group(3)) + 1, int(m.group(4)))
    except ValueError:
        return None
    return prior, current


def _matches(observed: float, known: float, *, tolerance: float) -> bool:
    return abs(observed - known) <= tolerance


def confirm_balance(text: str, known: Fundamentals) -> InterimBalance:
    """Take the current column only where the prior column reproduces ``known``."""
    body = _normalise(text)
    out = InterimBalance()
    dates = _column_dates(body)
    if dates is None:
        out.unconfirmed.append("no balance-sheet header naming its two columns")
        return out
    prior_date, current_date = dates
    # The header's left column must be the period we already hold, or these
    # are not the columns the figures below are being checked against.
    if prior_date != known.period_end:
        out.unconfirmed.append(
            f"the prior column is dated {prior_date}, not the known {known.period_end}"
        )
        return out
    if current_date <= known.period_end:
        # Nothing newer to read; refuse rather than restate what we have.
        out.unconfirmed.append(f"the statements end {current_date}, not after {known.period_end}")
        return out
    out.period_end = current_date

    cash = _two_column(body, "Cash and cash equivalents")
    if cash is None:
        out.unconfirmed.append("cash: no two-column line found")
    elif known.cash is None:
        out.unconfirmed.append("cash: nothing known for the prior period to check it against")
    elif not _matches(cash[0], known.cash, tolerance=MATCH_TOLERANCE_USD):
        out.unconfirmed.append(
            f"cash: prior column {cash[0]:,.0f} is not the known {known.cash:,.0f}"
        )
    else:
        out.cash = cash[1]
        out.confirmed.append("cash")

    current = _two_column(body, "Debt, current")
    non_current = _two_column(body, "Debt, non-current")
    if current is None or non_current is None:
        out.unconfirmed.append("debt: current and non-current lines not both found")
    elif known.total_debt is None:
        out.unconfirmed.append("debt: nothing known for the prior period to check it against")
    else:
        prior_total = current[0] + non_current[0]
        if not _matches(prior_total, known.total_debt, tolerance=MATCH_TOLERANCE_USD):
            out.unconfirmed.append(
                f"debt: prior columns sum to {prior_total:,.0f}, "
                f"not the known {known.total_debt:,.0f}"
            )
        else:
            out.total_debt = current[1] + non_current[1]
            out.confirmed.append("debt")

    shares = _share_classes(body)
    if shares is None:
        out.unconfirmed.append("shares: no outstanding-by-class figures found")
    elif known.shares_outstanding is None:
        out.unconfirmed.append("shares: nothing known for the prior period to check it against")
    elif not _matches(shares[0], known.shares_outstanding, tolerance=SHARE_TOLERANCE):
        out.unconfirmed.append(
            f"shares: prior column {shares[0]:,.0f} is not the known "
            f"{known.shares_outstanding:,.0f}"
        )
    else:
        out.shares_outstanding = shares[1]
        out.confirmed.append("shares")

    return out
