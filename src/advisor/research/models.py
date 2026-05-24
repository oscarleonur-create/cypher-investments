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


# ── Peers & multiples (Phase 2) ─────────────────────────────────────────────


class PeerSnapshot(BaseModel):
    """Market data + multiples for one company (subject or peer)."""

    symbol: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    market_cap: float | None = None  # USD
    enterprise_value: float | None = None
    price: float | None = None
    revenue_ttm: float | None = None
    ebitda_ttm: float | None = None
    fcf_ttm: float | None = None
    net_income_ttm: float | None = None
    eps_ttm: float | None = None
    revenue_growth_yoy: float | None = None
    gross_margin: float | None = None
    operating_margin: float | None = None
    # Multiples (trailing)
    ev_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    pe_trailing: float | None = None
    pe_forward: float | None = None
    p_to_fcf: float | None = None
    peg: float | None = None
    beta: float | None = None
    fetched_at: datetime = Field(default_factory=datetime.now)


class HistoricalMultiple(BaseModel):
    period_end: date
    fiscal_year: int
    ev_to_sales: float | None = None
    ev_to_ebitda: float | None = None
    pe: float | None = None
    p_to_fcf: float | None = None


class MultiplesResult(BaseModel):
    subject: PeerSnapshot
    peers: list[PeerSnapshot] = Field(default_factory=list)
    own_history: list[HistoricalMultiple] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)

    @property
    def all_snapshots(self) -> list[PeerSnapshot]:
        return [self.subject, *self.peers]


# ── DCF valuation (Phase 2) ──────────────────────────────────────────────────


class DcfAssumptions(BaseModel):
    """Explicit assumption set for one DCF scenario."""

    scenario: str  # "base" | "bull" | "bear"
    revenue_growth_yr1_3: float  # CAGR for years 1-3
    revenue_growth_yr4_10: float  # CAGR fade for years 4-10
    target_fcf_margin: float  # long-run FCF / revenue
    capex_intensity: float  # capex / revenue
    terminal_growth_rate: float = 0.025
    terminal_exit_multiple: float | None = None  # EV/EBITDA; None → Gordon only
    wacc: float = 0.09


class DcfScenario(BaseModel):
    assumptions: DcfAssumptions
    projected_fcf: list[float] = Field(default_factory=list)  # year 1 … year 10
    terminal_value_gordon: float | None = None
    terminal_value_exit: float | None = None
    enterprise_value: float
    equity_value: float
    implied_price: float
    upside_pct: float  # vs current price


class DcfResult(BaseModel):
    symbol: str
    current_price: float
    shares_outstanding: float
    net_debt: float
    wacc: float
    risk_free_rate: float
    equity_risk_premium: float = 0.055
    beta: float | None = None
    base: DcfScenario | None = None
    bull: DcfScenario | None = None
    bear: DcfScenario | None = None
    implied_growth_rate: float | None = None  # reverse-DCF result
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Ecosystem (Phase 2) ──────────────────────────────────────────────────────


class InstitutionalHolder(BaseModel):
    name: str
    shares: float | None = None
    pct_held: float | None = None
    value_usd: float | None = None
    date_reported: date | None = None


class InsiderTransaction(BaseModel):
    insider_name: str
    title: str = ""
    transaction_type: str  # "Purchase" | "Sale" | "Option Exercise" | etc.
    shares: float | None = None
    price_per_share: float | None = None
    value_usd: float | None = None
    transaction_date: date | None = None


class InsiderSummary(BaseModel):
    symbol: str
    transactions: list[InsiderTransaction] = Field(default_factory=list)
    net_buying_usd: float = 0.0  # positive = net buying, negative = net selling
    c_suite_buying: bool = False
    lookback_days: int = 365
    fetched_at: datetime = Field(default_factory=datetime.now)


class HolderSummary(BaseModel):
    symbol: str
    top_holders: list[InstitutionalHolder] = Field(default_factory=list)
    pct_institutional: float | None = None
    pct_insider: float | None = None
    fetched_at: datetime = Field(default_factory=datetime.now)


class CustomerEntry(BaseModel):
    name: str
    concentration_pct: float | None = None  # revenue concentration if disclosed
    note: str = ""


class SupplierEntry(BaseModel):
    name: str
    category: str = ""  # e.g. "sole-source", "key input"
    note: str = ""


class EcosystemResult(BaseModel):
    symbol: str
    holders: HolderSummary | None = None
    insiders: InsiderSummary | None = None
    top_customers: list[CustomerEntry] = Field(default_factory=list)
    top_suppliers: list[SupplierEntry] = Field(default_factory=list)
    concentration_risk: str = ""  # free-text summary
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Catalysts & Risks (Phase 2) ──────────────────────────────────────────────


class CatalystType(StrEnum):
    EARNINGS = "EARNINGS"
    PRODUCT_LAUNCH = "PRODUCT_LAUNCH"
    REGULATORY = "REGULATORY"
    CONTRACT = "CONTRACT"
    MACRO = "MACRO"
    OTHER = "OTHER"


class CatalystItem(BaseModel):
    description: str
    catalyst_type: CatalystType = CatalystType.OTHER
    expected_date: str = ""  # YYYY-MM-DD or quarter string
    is_near_term: bool = True  # within 90 days
    source: str = ""


class RiskSeverity(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class RiskItem(BaseModel):
    description: str
    category: str = ""  # "competitive" | "macro" | "regulatory" | "execution" | "financial"
    probability: float | None = None  # 0-1
    impact: float | None = None  # 0-1
    severity: RiskSeverity = RiskSeverity.MEDIUM
    source: str = ""

    @property
    def risk_score(self) -> float | None:
        if self.probability is None or self.impact is None:
            return None
        return self.probability * self.impact


class CatalystRiskResult(BaseModel):
    symbol: str
    catalysts: list[CatalystItem] = Field(default_factory=list)
    risks: list[RiskItem] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)

    @property
    def high_risks(self) -> list[RiskItem]:
        return [r for r in self.risks if r.severity == RiskSeverity.HIGH]


# ── Industry analysis (Phase 3) ─────────────────────────────────────────────


class MoatType(StrEnum):
    NETWORK_EFFECT = "network_effect"
    COST_ADVANTAGE = "cost_advantage"
    SWITCHING_COSTS = "switching_costs"
    INTANGIBLE_ASSETS = "intangible_assets"
    EFFICIENT_SCALE = "efficient_scale"
    NONE = "none"


class PortersForceStrength(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PortersForces(BaseModel):
    competitive_rivalry: PortersForceStrength = PortersForceStrength.MEDIUM
    supplier_power: PortersForceStrength = PortersForceStrength.MEDIUM
    buyer_power: PortersForceStrength = PortersForceStrength.MEDIUM
    threat_of_new_entrants: PortersForceStrength = PortersForceStrength.MEDIUM
    threat_of_substitutes: PortersForceStrength = PortersForceStrength.MEDIUM
    rationale: str = ""


class IndustryAnalysis(BaseModel):
    symbol: str
    sector: str = ""
    industry: str = ""
    porters_forces: PortersForces | None = None
    moat_type: MoatType = MoatType.NONE
    moat_description: str = ""
    market_share_narrative: str = ""
    key_competitors: list[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Earnings call transcripts (Phase 3) ─────────────────────────────────────


class TranscriptTone(StrEnum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class TranscriptSummary(BaseModel):
    quarter: str  # e.g. "Q1 2025"
    earnings_date: str = ""
    tone: TranscriptTone = TranscriptTone.NEUTRAL
    key_topics: list[str] = Field(default_factory=list)
    management_guidance: str = ""
    analyst_concerns: str = ""
    highlight_quote: str = ""


class TranscriptAnalysis(BaseModel):
    symbol: str
    summaries: list[TranscriptSummary] = Field(default_factory=list)
    tone_trend: str = ""
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Market data snapshot (Phase 3) ───────────────────────────────────────────


class MarketDataSnapshot(BaseModel):
    symbol: str
    short_interest_pct: float | None = None  # % of float
    days_to_cover: float | None = None
    float_shares: float | None = None
    shares_outstanding: float | None = None
    avg_volume_10d: float | None = None
    avg_volume_90d: float | None = None
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Thesis (Phase 3) ─────────────────────────────────────────────────────────


class ThesisScenario(BaseModel):
    scenario: str  # "bull" | "base" | "bear"
    probability: float  # 0-1
    target_price: float | None = None
    upside_pct: float | None = None
    description: str = ""
    key_assumptions: list[str] = Field(default_factory=list)
    what_proves_wrong: list[str] = Field(default_factory=list)


class ThesisResult(BaseModel):
    symbol: str
    current_price: float | None = None
    bull: ThesisScenario | None = None
    base: ThesisScenario | None = None
    bear: ThesisScenario | None = None
    summary: str = ""
    conviction: str = "MEDIUM"  # "HIGH" | "MEDIUM" | "LOW"
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Research report (full Phase 1 + 2 + 3 model) ────────────────────────────


class ResearchReport(BaseModel):
    """Top-level report. Sections map 1:1 to the analyst blueprint headings."""

    symbol: str
    as_of: date
    created_at: datetime = Field(default_factory=datetime.now)

    # § Business (Phase 3) — one-sentence model, segments, revenue drivers
    business_model: str | None = None

    # § Ecosystem (Phase 2)
    ecosystem: EcosystemResult | None = None

    # § Industry (Phase 3) — Porter's 5, moat, share narrative

    # § Financials (Phase 1)
    statements: StatementBundle | None = None
    ratios: RatioBundle | None = None
    red_flags: RedFlagList | None = None

    # § Valuation (Phase 2)
    multiples: MultiplesResult | None = None
    dcf: DcfResult | None = None

    # § Catalysts & Risks (Phase 2)
    catalyst_risk: CatalystRiskResult | None = None

    # § Thesis (Phase 3) — bull/base/bear targets + probabilities

    # § Industry (Phase 3)
    industry: IndustryAnalysis | None = None

    # § Transcripts (Phase 3)
    transcripts: TranscriptAnalysis | None = None

    # § Market Data (Phase 3)
    market_data: MarketDataSnapshot | None = None

    # § Thesis (Phase 3)
    thesis: ThesisResult | None = None

    filings: list[FilingRef] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
