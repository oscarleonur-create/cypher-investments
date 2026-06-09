"""Tests for the price-history + fundamentals assembly."""

from __future__ import annotations

from datetime import date

from advisor.research.models import (
    IncomeStatementPeriod,
    PriceBar,
    RatioBundle,
    RatioPeriod,
    ResearchReport,
    StatementBundle,
)
from advisor.research.price_history import build_price_history


def _report() -> ResearchReport:
    """Two annual periods: FY2022 (EPS 2.0) and FY2023 (EPS 4.0)."""
    income = [
        IncomeStatementPeriod(
            period_end=date(2022, 12, 31), fiscal_year=2022, revenue=1000.0, eps_diluted=2.0
        ),
        IncomeStatementPeriod(
            period_end=date(2023, 12, 31), fiscal_year=2023, revenue=1500.0, eps_diluted=4.0
        ),
    ]
    ratios = RatioBundle(
        symbol="TEST",
        periods=[
            RatioPeriod(period_end=date(2022, 12, 31), fiscal_year=2022, net_margin=0.10),
            RatioPeriod(period_end=date(2023, 12, 31), fiscal_year=2023, net_margin=0.15),
        ],
    )
    return ResearchReport(
        symbol="TEST",
        as_of=date.today(),
        statements=StatementBundle(symbol="TEST", income=income),
        ratios=ratios,
    )


def _bars() -> list[PriceBar]:
    return [
        PriceBar(date=date(2022, 6, 1), close=20.0),  # before any reported EPS
        PriceBar(date=date(2023, 1, 15), close=40.0),  # after FY2022 (EPS 2.0) → PE 20
        PriceBar(date=date(2024, 1, 15), close=80.0),  # after FY2023 (EPS 4.0) → PE 20
    ]


def test_pe_series_steps_with_trailing_eps():
    res = build_price_history("TEST", _report(), bars=_bars())
    pe = {p.date: p.pe for p in res.pe_series}
    assert pe[date(2022, 6, 1)] is None  # no EPS reported yet
    assert pe[date(2023, 1, 15)] == 20.0  # 40 / 2.0
    assert pe[date(2024, 1, 15)] == 20.0  # 80 / 4.0


def test_pe_null_when_eps_non_positive():
    rep = _report()
    rep.statements.income[1].eps_diluted = -1.0  # FY2023 swings to a loss
    res = build_price_history("TEST", rep, bars=_bars())
    pe = {p.date: p.pe for p in res.pe_series}
    assert pe[date(2024, 1, 15)] is None  # negative EPS → no P/E


def test_earnings_markers_and_yoy():
    res = build_price_history("TEST", _report(), bars=_bars())
    assert [m.date for m in res.earnings] == [date(2022, 12, 31), date(2023, 12, 31)]
    by_date = {m.date: m for m in res.earnings}
    assert by_date[date(2022, 12, 31)].yoy_eps_growth is None  # no prior period
    assert by_date[date(2023, 12, 31)].yoy_eps_growth == 1.0  # (4 - 2) / 2
    assert by_date[date(2023, 12, 31)].revenue == 1500.0


def test_fundamentals_carry_net_margin():
    res = build_price_history("TEST", _report(), bars=_bars())
    fy = {f.fiscal_year: f for f in res.fundamentals}
    assert fy[2023].net_margin == 0.15
    assert fy[2023].eps == 4.0
    assert res.bars == sorted(res.bars, key=lambda b: b.date)
