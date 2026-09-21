"""When a newer 6-K replaces the periodic filing, and when it must not.

Nebius files a 20-F once a year. Its newest periodic filing was 263 days old
and priced a 14% position at 107x revenue against a real 26x — a run-rate of
$0.53bn where the last reported quarter annualises to $2.33bn.

Replacing it is only safe when the whole set comes from the same filing. The
counter-example is in the same data: between December and June, Nebius's cash
went $3.7bn to $8.0bn and its debt $4.1bn to $8.5bn on the back of a $5.75bn
convertible. A June revenue over a December balance sheet would have moved
enterprise value by billions, in the direction that flatters.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from advisor.valuation import fundamentals as fx
from advisor.valuation.models import Fundamentals, ValuationSnapshot


def periodic(*, period_end: date, complete: bool = True) -> Fundamentals:
    return Fundamentals(
        symbol="NBIS",
        source_accession="0001104659-26-052948",
        period_end=period_end,
        fiscal_period="FY",
        revenue=529_800_000.0,
        cash=3_678_100_000.0,
        total_debt=4_127_700_000.0,
        shares_outstanding=253_016_971.0,
        missing=[] if complete else ["revenue"],
    )


def interim(*, period_end: date) -> Fundamentals:
    return Fundamentals(
        symbol="NBIS",
        source_accession="0001104659-26-094844",
        period_end=period_end,
        fiscal_period="Q",
        revenue=582_300_000.0,
        cash=8_042_100_000.0,
        total_debt=8_545_700_000.0,
        shares_outstanding=271_855_218.0,
    )


STALE = date.today() - timedelta(days=ValuationSnapshot.STALE_AFTER_DAYS + 1)
FRESH = date.today() - timedelta(days=ValuationSnapshot.STALE_AFTER_DAYS - 1)


def test_a_stale_periodic_filing_is_replaced(monkeypatch):
    newer = interim(period_end=STALE + timedelta(days=90))
    monkeypatch.setattr(fx, "interim_fundamentals", lambda s, k: newer)
    assert fx._prefer_interim("NBIS", periodic(period_end=STALE)) is newer


def test_a_fresh_periodic_filing_is_never_looked_past(monkeypatch):
    """The guard is a round trip per symbol, so it must not fire for a filer
    whose last report is recent — every 10-Q filer in the book."""

    def explode(symbol, known):  # pragma: no cover — must not be reached
        raise AssertionError("EDGAR was queried for a fresh filing")

    monkeypatch.setattr(fx, "interim_fundamentals", explode)
    held = periodic(period_end=FRESH)
    assert fx._prefer_interim("NBIS", held) is held


def test_exactly_at_the_staleness_boundary_is_not_stale(monkeypatch):
    def explode(symbol, known):  # pragma: no cover
        raise AssertionError("queried at the boundary")

    monkeypatch.setattr(fx, "interim_fundamentals", explode)
    at_boundary = date.today() - timedelta(days=ValuationSnapshot.STALE_AFTER_DAYS)
    held = periodic(period_end=at_boundary)
    assert fx._prefer_interim("NBIS", held) is held


def test_nothing_newer_found_keeps_the_periodic_filing(monkeypatch):
    monkeypatch.setattr(fx, "interim_fundamentals", lambda s, k: None)
    held = periodic(period_end=STALE)
    assert fx._prefer_interim("NBIS", held) is held


def test_an_incomplete_interim_never_replaces(monkeypatch):
    """Half a set is the mixing case: a fresh revenue over a stale balance
    sheet is worse than an honestly old valuation."""
    half = interim(period_end=STALE + timedelta(days=90))
    half.missing = ["cash"]
    monkeypatch.setattr(fx, "interim_fundamentals", lambda s, k: half)
    held = periodic(period_end=STALE)
    assert fx._prefer_interim("NBIS", held) is held


def test_an_interim_no_newer_than_the_periodic_filing_is_ignored(monkeypatch):
    same = interim(period_end=STALE)
    monkeypatch.setattr(fx, "interim_fundamentals", lambda s, k: same)
    held = periodic(period_end=STALE)
    assert fx._prefer_interim("NBIS", held) is held


def test_an_incomplete_periodic_filing_is_returned_untouched(monkeypatch):
    def explode(symbol, known):  # pragma: no cover
        raise AssertionError("queried for an incomplete filing")

    monkeypatch.setattr(fx, "interim_fundamentals", explode)
    held = periodic(period_end=STALE, complete=False)
    assert fx._prefer_interim("NBIS", held) is held


def test_a_failure_reaching_edgar_leaves_the_valuation_standing(monkeypatch):
    """A network error must cost the refresh, never the whole valuation."""

    def boom(symbol, known):
        raise RuntimeError("connection reset by peer")

    monkeypatch.setattr(fx, "interim_fundamentals", boom)
    held = periodic(period_end=STALE)
    assert fx._prefer_interim("NBIS", held) is held


@pytest.mark.parametrize("days_newer", [1, 90, 365])
def test_any_genuinely_newer_complete_interim_wins(monkeypatch, days_newer):
    newer = interim(period_end=STALE + timedelta(days=days_newer))
    monkeypatch.setattr(fx, "interim_fundamentals", lambda s, k: newer)
    assert fx._prefer_interim("NBIS", periodic(period_end=STALE)) is newer
