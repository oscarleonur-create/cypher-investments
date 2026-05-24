"""Synthetic statement fixtures for ratio + red-flag testing.

Numbers chosen so the resulting ratios are easy to verify by hand. Five
fiscal years FY2024 → FY2020, newest-first (matching bundle convention).
"""

from __future__ import annotations

from datetime import date

import pytest
from advisor.research.models import (
    BalanceSheetPeriod,
    CashFlowPeriod,
    IncomeStatementPeriod,
    StatementBundle,
)


def _income(year: int, **overrides) -> IncomeStatementPeriod:
    base = dict(
        period_end=date(year, 12, 31),
        fiscal_year=year,
        revenue=1000.0,
        cost_of_revenue=600.0,
        gross_profit=400.0,
        operating_expenses=200.0,
        operating_income=200.0,
        interest_expense=20.0,
        pretax_income=180.0,
        income_tax=36.0,
        net_income=144.0,
        eps_diluted=1.44,
        shares_basic=100.0,
        shares_diluted=100.0,
        ebitda=240.0,
    )
    base.update(overrides)
    return IncomeStatementPeriod(**base)


def _balance(year: int, **overrides) -> BalanceSheetPeriod:
    base = dict(
        period_end=date(year, 12, 31),
        fiscal_year=year,
        cash_and_equivalents=200.0,
        accounts_receivable=100.0,
        inventory=80.0,
        current_assets=500.0,
        total_assets=2000.0,
        accounts_payable=80.0,
        short_term_debt=50.0,
        current_liabilities=300.0,
        long_term_debt=450.0,
        total_liabilities=900.0,
        total_equity=1100.0,
        shares_outstanding=100.0,
    )
    base.update(overrides)
    return BalanceSheetPeriod(**base)


def _cashflow(year: int, **overrides) -> CashFlowPeriod:
    base = dict(
        period_end=date(year, 12, 31),
        fiscal_year=year,
        operating_cash_flow=200.0,
        capex=-50.0,
        free_cash_flow=150.0,
        investing_cash_flow=-60.0,
        financing_cash_flow=-40.0,
        dividends_paid=-30.0,
        share_repurchases=-10.0,
    )
    base.update(overrides)
    return CashFlowPeriod(**base)


@pytest.fixture
def clean_bundle() -> StatementBundle:
    """Flat 5-yr bundle with stable per-period values — no red flags."""
    years = [2024, 2023, 2022, 2021, 2020]
    return StatementBundle(
        symbol="TEST",
        income=[_income(y) for y in years],
        balance=[_balance(y) for y in years],
        cashflow=[_cashflow(y) for y in years],
        source="test",
    )


@pytest.fixture
def income_factory():
    return _income


@pytest.fixture
def balance_factory():
    return _balance


@pytest.fixture
def cashflow_factory():
    return _cashflow
