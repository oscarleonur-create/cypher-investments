"""Required growth shown as a range across margins, not a single fragile number."""

from __future__ import annotations

from pathlib import Path

import pytest
from advisor.daemon.store import DaemonStore
from advisor.story.scorecard import build_scorecard

from tests.test_story.test_scorecard import CONSENSUS, valuation


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def rows(store, **margins):
    snap = valuation().model_copy(update=margins)
    store.save_valuation(snap)
    card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: CONSENSUS)
    return {r.label: r for r in card.expectations}


def test_no_own_margins_keeps_the_single_generic_number(store):
    r = rows(store)
    assert r["Price requires"].value == "25.4%/yr × 10y"
    assert r["If FY2027 holds"].value == "12.8%/yr"
    assert "unavailable" in r["Price requires"].detail


def test_a_lower_own_margin_opens_a_range_above_the_generic(store):
    r = rows(store, margin_trailing=0.10, margin_trailing_label="trailing 4 quarters")
    low, _, high = r["Price requires"].value.partition(" to ")
    assert low.startswith("25.") and "/yr × 10y" in high
    assert "10.0% FCF margin (own trailing 4 quarters)" in r["Price requires"].detail
    assert r["If FY2027 holds"].value.startswith("12.8% to ")


def test_burning_cash_margin_is_named_not_used(store):
    r = rows(store, margin_median=-0.45, margin_median_label="FY2023–FY2025 median")
    assert "burning cash" in r["Price requires"].detail
    assert " to " not in r["Price requires"].value
