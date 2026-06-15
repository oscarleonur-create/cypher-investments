"""Tests for the consolidated fair-price synthesis + watchlist store."""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

from advisor.research.models import AnalystConsensus, ResearchReport
from advisor.research.store import ResearchStore
from advisor.research.valuation.fair_price import build_fair_price


def _report_with_dcf(consensus_target: float | None = 70.0) -> ResearchReport:
    """A report with a realistic DCF (mocked yfinance) + optional consensus."""
    from advisor.research.valuation.dcf import build_dcf

    fake_info = {
        "currentPrice": 50.0,
        "sharesOutstanding": 100e6,
        "totalDebt": 500e6,
        "totalCash": 200e6,
        "beta": 1.2,
        "totalRevenue": 1000e6,
        "freeCashflow": 150e6,
        "revenueGrowth": 0.10,
        "enterpriseToEbitda": 15.0,
    }
    with (
        patch("yfinance.Ticker") as mock_ticker,
        patch("advisor.research.valuation.dcf._risk_free_rate", return_value=0.043),
    ):
        mock_ticker.return_value.info = fake_info
        dcf = build_dcf("TEST", statements=None)

    consensus = (
        AnalystConsensus(symbol="TEST", n_analysts=12, target_price_mean=consensus_target)
        if consensus_target is not None
        else None
    )
    return ResearchReport(symbol="TEST", as_of=date.today(), dcf=dcf, consensus=consensus)


def test_fair_price_blends_methods_within_range():
    report = _report_with_dcf()
    fp = build_fair_price(report)

    assert fp is not None
    assert fp.symbol == "TEST"
    assert fp.current_price == 50.0
    # Headline sits inside the reported range.
    assert fp.low <= fp.fair_price <= fp.high
    # At least DCF base + consensus contributed.
    names = {m.name for m in fp.methods}
    assert "dcf_base" in names and "analyst_target" in names
    # Weights renormalize to ~1 over the present methods.
    assert abs(sum(m.weight for m in fp.methods) - 1.0) < 1e-6
    # Upside is consistent with the blend.
    assert abs(fp.upside_pct - (fp.fair_price / fp.current_price - 1.0)) < 1e-6


def test_fair_price_handles_single_method():
    # No consensus, no multiples → DCF base + (maybe) Bayesian only; still valid.
    report = _report_with_dcf(consensus_target=None)
    fp = build_fair_price(report)
    assert fp is not None
    assert fp.fair_price > 0
    assert abs(sum(m.weight for m in fp.methods) - 1.0) < 1e-6


def test_fair_price_none_without_any_valuation():
    report = ResearchReport(symbol="TEST", as_of=date.today())
    assert build_fair_price(report) is None


def test_watchlist_round_trip(tmp_path):
    store = ResearchStore(tmp_path / "research.db")

    assert store.load_watchlist() == []

    store.add_to_watchlist("nvda", note="AI compute")
    store.add_to_watchlist("amd")
    wl = store.load_watchlist()
    syms = {row["symbol"] for row in wl}
    assert syms == {"NVDA", "AMD"}
    nvda = next(r for r in wl if r["symbol"] == "NVDA")
    assert nvda["note"] == "AI compute"

    # Re-adding updates the note rather than duplicating the row.
    store.add_to_watchlist("NVDA", note="updated")
    wl = store.load_watchlist()
    assert len([r for r in wl if r["symbol"] == "NVDA"]) == 1
    assert next(r for r in wl if r["symbol"] == "NVDA")["note"] == "updated"

    store.remove_from_watchlist("NVDA")
    assert {r["symbol"] for r in store.load_watchlist()} == {"AMD"}
    store.close()
