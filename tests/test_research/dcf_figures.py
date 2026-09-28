"""Figures for DCF tests: the numbers the old tests fed through a mocked
``yfinance.info``, now passed straight to the engine, with no network."""

from __future__ import annotations

from datetime import date

from advisor.valuation.figures import Figures
from advisor.valuation.models import OwnMargin


def figures(**overrides) -> Figures:
    """$50, 100M shares, $300M net debt, $1bn revenue growing 10%, 15% FCF."""
    base = dict(
        symbol="TEST",
        price=50.0,
        price_source="test",
        shares=100e6,
        net_cash=200e6 - 500e6,
        balance_asof=date(2026, 6, 30),
        source_accession="0000000000-26-000001",
        revenue_base=1000e6,
        revenue_base_label="trailing 12 months to 2026-06-30 (test)",
        revenue_growth=0.10,
        revenue_growth_label="test",
        start_margin=0.15,
        start_margin_label="FCF, test",
        capex_intensity=0.05,
        margins=[
            OwnMargin(kind="fcf", value=0.15, label="FCF, test"),
            OwnMargin(kind="median", value=0.12, label="median, test"),
            OwnMargin(kind="nopat", value=0.18, label="NOPAT, test"),
        ],
    )
    base.update(overrides)
    return Figures(**base)
