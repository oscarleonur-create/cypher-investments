"""Pydantic models for the fundamental research engine.

Phase 1 ships the financial-statement and ratio side of the blueprint.
Later phases will populate the ecosystem / valuation / industry / thesis
sections — their fields are reserved here as `Optional` so the report shape
is stable across phases and existing JSON payloads stay forward-compatible.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ── Filing references ────────────────────────────────────────────────────────


class FormType(StrEnum):
    K10 = "10-K"
    Q10 = "10-Q"
    K8 = "8-K"
    DEF14A = "DEF 14A"
    F13 = "13F-HR"
    FORM4 = "4"


class FilingRef(BaseModel):
    """Lightweight pointer to an SEC filing — enough to cite, not to embed."""

    accession_number: str
    form: FormType
    filing_date: date
    period_of_report: date | None = None
    url: str = ""


# ── Financial statements (5-yr trend) ────────────────────────────────────────


class StatementPeriod(BaseModel):
    """Single fiscal-period observation for one line item set."""

    period_end: date
    fiscal_year: int
    is_ttm: bool = False
    source_filing: str | None = None  # accession number, if from EDGAR


class IncomeStatementPeriod(StatementPeriod):
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_expenses: float | None = None
    operating_income: float | None = None
    interest_expense: float | None = None
    pretax_income: float | None = None
    income_tax: float | None = None
    net_income: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None
    shares_basic: float | None = None
    shares_diluted: float | None = None
    ebitda: float | None = None  # approximation; provider-derived if available


class BalanceSheetPeriod(StatementPeriod):
    cash_and_equivalents: float | None = None
    short_term_investments: float | None = None
    accounts_receivable: float | None = None
    inventory: float | None = None
    current_assets: float | None = None
    goodwill: float | None = None
    intangibles: float | None = None
    total_assets: float | None = None
    accounts_payable: float | None = None
    short_term_debt: float | None = None
    current_liabilities: float | None = None
    long_term_debt: float | None = None
    total_liabilities: float | None = None
    total_equity: float | None = None
    shares_outstanding: float | None = None


class CashFlowPeriod(StatementPeriod):
    operating_cash_flow: float | None = None
    capex: float | None = (
        None  # stored as a negative outflow when from EDGAR; tests/ratios assume signed
    )
    free_cash_flow: float | None = None
    investing_cash_flow: float | None = None
    financing_cash_flow: float | None = None
    dividends_paid: float | None = None
    share_repurchases: float | None = None
    net_change_in_cash: float | None = None


class StatementBundle(BaseModel):
    """5-yr (or whatever the provider returned) financial statement trend."""

    symbol: str
    currency: str = "USD"
    income: list[IncomeStatementPeriod] = Field(default_factory=list)
    balance: list[BalanceSheetPeriod] = Field(default_factory=list)
    cashflow: list[CashFlowPeriod] = Field(default_factory=list)
    source: str = "edgar"  # "edgar" or "yfinance"
    fetched_at: datetime = Field(default_factory=datetime.now)

    def latest_income(self) -> IncomeStatementPeriod | None:
        return self.income[0] if self.income else None

    def latest_balance(self) -> BalanceSheetPeriod | None:
        return self.balance[0] if self.balance else None

    def latest_cashflow(self) -> CashFlowPeriod | None:
        return self.cashflow[0] if self.cashflow else None


# ── Ratios ──────────────────────────────────────────────────────────────────


class RatioPeriod(BaseModel):
    """Per-period ratio set (matches a fiscal year or TTM)."""

    period_end: date
    fiscal_year: int
    is_ttm: bool = False

    # Profitability
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    roa: float | None = None  # net income / avg total assets
    roe: float | None = None  # net income / avg equity
    roic: float | None = None  # NOPAT / invested capital

    # Liquidity & leverage
    current_ratio: float | None = None
    quick_ratio: float | None = None
    debt_to_equity: float | None = None
    debt_to_ebitda: float | None = None
    interest_coverage: float | None = None  # EBIT / interest

    # Efficiency
    asset_turnover: float | None = None
    inventory_turns: float | None = None
    dso: float | None = None  # days sales outstanding

    # Cash quality
    fcf_margin: float | None = None
    capex_intensity: float | None = None  # |capex| / revenue
    fcf_to_net_income: float | None = None  # FCF / net income (quality check)


class RatioBundle(BaseModel):
    symbol: str
    periods: list[RatioPeriod] = Field(default_factory=list)
    share_count_cagr_3y: float | None = None  # share count CAGR; positive = dilution
    fetched_at: datetime = Field(default_factory=datetime.now)

    def latest(self) -> RatioPeriod | None:
        return self.periods[0] if self.periods else None


# ── Red flags ───────────────────────────────────────────────────────────────


class RedFlagSeverity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RedFlag(BaseModel):
    code: str  # stable identifier, e.g. "DSO_DRIFT"
    title: str
    severity: RedFlagSeverity
    detail: str
    period_end: date | None = None  # which period triggered


class RedFlagList(BaseModel):
    symbol: str
    flags: list[RedFlag] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)

    @property
    def high_severity_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == RedFlagSeverity.HIGH)


# ── Research report (Phase 1 skeleton — later phases fill in more sections) ──


class ResearchReport(BaseModel):
    """Top-level report. Sections map 1:1 to the analyst blueprint headings."""

    symbol: str
    as_of: date
    created_at: datetime = Field(default_factory=datetime.now)

    # § Business (Phase 3) — one-sentence model, segments, revenue drivers
    business_model: str | None = None

    # § Ecosystem (Phase 2) — customers, suppliers, competitors, holders, insiders, board
    # placeholder: typed structs added in Phase 2

    # § Industry (Phase 3) — Porter's 5, moat, share narrative

    # § Financials (Phase 1) — fully populated below
    statements: StatementBundle | None = None
    ratios: RatioBundle | None = None
    red_flags: RedFlagList | None = None

    # § Valuation (Phase 2) — multiples, DCF, reverse-DCF
    # § Catalysts & Risks (Phase 2)
    # § Thesis (Phase 3) — bull/base/bear targets + probabilities

    filings: list[FilingRef] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # free-form caveats from each layer
