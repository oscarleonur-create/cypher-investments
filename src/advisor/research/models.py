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
    # Inputs to the scenario engine — persisted so the dashboard's what-if
    # sliders can replay the base scenario exactly without re-fetching.
    base_revenue: float | None = None
    seed_fcf: float | None = None
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Fair price (consolidated valuation) ──────────────────────────────────────
#
# A single headline fair value blended from whichever valuation methods produced
# a usable estimate (DCF base, peer multiples, Bayesian median, analyst target).


class FairPriceMethod(BaseModel):
    """One method's contribution to the blended fair price."""

    name: str  # "dcf_base" | "multiples" | "bayesian_median" | "analyst_target"
    label: str
    estimate: float
    weight: float  # renormalized so the present methods sum to 1.0


class FairPriceResult(BaseModel):
    """Consolidated fair-value estimate with a range and per-method breakdown."""

    symbol: str
    current_price: float
    fair_price: float  # weighted blend of the methods below
    low: float
    high: float
    upside_pct: float  # fair_price / current_price − 1
    methods: list[FairPriceMethod] = Field(default_factory=list)
    confidence: str = "LOW"  # "HIGH" | "MEDIUM" | "LOW"
    note: str = ""
    as_of: datetime = Field(default_factory=datetime.now)


# ── Bayesian pricing (posterior fair-value distribution) ─────────────────────
#
# Treats the DCF drivers as prior distributions, lets research signals update
# them (precision-weighted Normal conjugate update), and Monte-Carlos the
# implied price through the same projection as dcf.compute_dcf_scenario. The
# three groups below are the "connections that affect pricing" surfaced — and
# made adjustable — in the frontend's what-if panel.


class CatalystScenario(BaseModel):
    """One discrete scenario in the scenario × catalyst EV table."""

    label: str
    probability: float
    target_price: float
    upside_pct: float
    supporting_catalysts: list[str] = Field(default_factory=list)
    invalidating_catalysts: list[str] = Field(default_factory=list)


class PriorDriver(BaseModel):
    """One valuation driver modelled as a prior distribution (mean + uncertainty)."""

    key: str  # e.g. "revenue_growth_yr1_3" | "target_fcf_margin" | "wacc"
    label: str
    mean: float  # prior mean (seeded from the base DCF scenario)
    std: float  # prior uncertainty, 1σ (seeded from the bull/bear spread)
    min: float  # slider / clamp bound (matches dcf.py ranges)
    max: float
    unit: str = "pct"  # "pct" | "x" | "ratio"


class EvidenceSignal(BaseModel):
    """A research signal that nudges a driver's prior toward a posterior.

    ``observed`` is the driver value the signal implies (Normal observation);
    when ``observed`` is None the signal only inflates the driver's uncertainty
    (e.g. high option-implied vol → wider fair-value distribution).
    """

    key: str  # "analyst_target" | "reverse_dcf" | "options_iv" | "mgmt_quality" | "red_flags"
    label: str
    target_driver: str  # which PriorDriver.key it informs
    observed: float | None = None
    precision: float = 1.0  # base precision relative to the prior
    weight: float = 1.0  # 0..1 user-adjustable confidence on this signal
    note: str = ""


class EcosystemFactor(BaseModel):
    """A toggleable ecosystem node (customer/supplier/holder/peer) that perturbs a driver."""

    key: str
    label: str
    kind: str  # "customer" | "supplier" | "holder" | "peer"
    driver: str  # which PriorDriver.key it perturbs when active
    active: bool = False
    mean_delta: float = 0.0  # additive shift to the driver mean when active
    std_delta: float = 0.0  # additive shift to the driver uncertainty when active
    note: str = ""


class HistogramBin(BaseModel):
    x: float  # bin centre (implied price)
    count: int


class BayesianOverrides(BaseModel):
    """User adjustments from the what-if sliders. All fields optional."""

    driver_mean: dict[str, float] = Field(default_factory=dict)
    driver_std: dict[str, float] = Field(default_factory=dict)
    evidence_weight: dict[str, float] = Field(default_factory=dict)
    ecosystem_active: dict[str, bool] = Field(default_factory=dict)
    n_draws: int | None = None


class BayesianPriceResult(BaseModel):
    """Posterior distribution over fair value + the connections that shaped it."""

    symbol: str
    current_price: float
    n_draws: int = 0

    # The three groups of adjustable "connections" surfaced in the UI.
    drivers: list[PriorDriver] = Field(default_factory=list)
    evidence: list[EvidenceSignal] = Field(default_factory=list)
    ecosystem: list[EcosystemFactor] = Field(default_factory=list)

    # Posterior fair-value summary.
    mean_price: float = 0.0
    median_price: float = 0.0
    p5: float = 0.0
    p25: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    prob_undervalued: float = 0.0  # P(fair value > current price)
    expected_upside_pct: float = 0.0  # median / current − 1
    histogram: list[HistogramBin] = Field(default_factory=list)

    # Honesty flags: a revenue-growth DCF can't value a pre-revenue or
    # deeply cash-burning company, so the posterior collapses to ~net-cash or
    # zero. When meaningful is False the UI shows `note` instead of fake stats.
    meaningful: bool = True
    note: str = ""

    # Discrete scenario × catalyst EV table (bull/base/bear with catalyst attribution).
    catalyst_scenarios: list[CatalystScenario] = Field(default_factory=list)

    as_of: datetime = Field(default_factory=datetime.now)


# ── Price history + fundamentals overlay ─────────────────────────────────────


class PriceBar(BaseModel):
    """One daily price observation for the chart."""

    date: date
    close: float
    volume: float | None = None


class EarningsMarker(BaseModel):
    """A reported-earnings point plotted on the price line.

    Shows actuals (revenue / diluted EPS) plus YoY EPS growth — we don't store
    historical consensus estimates, so this is not a literal beat/miss.
    """

    date: date
    revenue: float | None = None
    eps: float | None = None
    yoy_eps_growth: float | None = None  # vs the same period a year earlier


class PePoint(BaseModel):
    """Trailing P/E aligned to a price bar (null where EPS ≤ 0)."""

    date: date
    pe: float | None = None


class FundamentalPoint(BaseModel):
    """A fiscal period's headline figures for the revenue/EPS sub-panel."""

    date: date
    fiscal_year: int
    revenue: float | None = None
    eps: float | None = None
    net_margin: float | None = None


class PriceHistoryResult(BaseModel):
    """Daily price series + the fundamentals overlaid on the ticker chart."""

    symbol: str
    bars: list[PriceBar] = Field(default_factory=list)
    earnings: list[EarningsMarker] = Field(default_factory=list)
    pe_series: list[PePoint] = Field(default_factory=list)
    fundamentals: list[FundamentalPoint] = Field(default_factory=list)
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
    probability: float | None = None  # 0–1 estimated probability of occurring
    direction: str = "neutral"  # "bullish" | "bearish" | "mixed" | "neutral"
    price_impact_pct: float | None = None  # % price move magnitude if fires (positive = up)


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


# ── Company network (Phase 3) ────────────────────────────────────────────────


class RelationshipType(StrEnum):
    PEER = "peer"
    CUSTOMER = "customer"
    SUPPLIER = "supplier"
    TOP_HOLDER = "top_holder"
    COMPETITOR = "competitor"


class NetworkNode(BaseModel):
    """One connected company in the subject's ecosystem network."""

    symbol: str
    name: str = ""
    relationship: RelationshipType
    note: str = ""  # e.g. "~15% revenue", "Top-3 holder"
    snapshot: PeerSnapshot | None = None


class CompanyNetwork(BaseModel):
    subject_symbol: str
    subject_snapshot: PeerSnapshot | None = None
    nodes: list[NetworkNode] = Field(default_factory=list)
    as_of: date = Field(default_factory=date.today)

    @property
    def by_relationship(self) -> dict[str, list[NetworkNode]]:
        result: dict[str, list[NetworkNode]] = {}
        for node in self.nodes:
            result.setdefault(node.relationship.value, []).append(node)
        return result


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
    moat_strength: float | None = None  # 0–10; 0=none, 5=narrow, 8+=wide
    competitive_advantages: list[str] = Field(default_factory=list)
    competitive_position: str = ""  # "leader" | "challenger" | "niche" | "laggard"
    market_share_narrative: str = ""
    key_competitors: list[str] = Field(default_factory=list)
    sector_tailwinds: list[str] = Field(default_factory=list)
    sector_headwinds: list[str] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Earnings call transcripts (Phase 3) ─────────────────────────────────────


class TranscriptTone(StrEnum):
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"


class TranscriptSource(BaseModel):
    """A web source the transcript analysis was built from (link to the call)."""

    url: str
    title: str = ""


class TranscriptSummary(BaseModel):
    quarter: str  # e.g. "Q1 2025"
    earnings_date: str = ""
    tone: TranscriptTone = TranscriptTone.NEUTRAL
    key_topics: list[str] = Field(default_factory=list)
    management_guidance: str = ""
    analyst_concerns: str = ""
    highlight_quote: str = ""
    source_url: str = ""  # link to the full transcript for this quarter, if known


class TranscriptAnalysis(BaseModel):
    symbol: str
    summaries: list[TranscriptSummary] = Field(default_factory=list)
    tone_trend: str = ""
    sources: list[TranscriptSource] = Field(default_factory=list)
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


# ── Analyst Consensus (Phase 4 — Variant Perception) ────────────────────────


class AnalystConsensus(BaseModel):
    """Sell-side consensus pulled from yfinance analyst data."""

    symbol: str
    n_analysts: int = 0
    recommendation_mean: float | None = None  # 1=Strong Buy … 5=Strong Sell
    recommendation_key: str = ""  # "buy" | "hold" | "sell" | …
    target_price_low: float | None = None
    target_price_mean: float | None = None
    target_price_median: float | None = None
    target_price_high: float | None = None
    current_price: float | None = None
    consensus_upside_pct: float | None = None  # (mean_target − price) / price

    # Forward estimates
    eps_estimate_current_year: float | None = None
    eps_estimate_next_year: float | None = None
    revenue_estimate_current_year: float | None = None
    revenue_estimate_next_year: float | None = None

    # Estimate revision momentum (up/down counts in last 30 days)
    eps_revisions_up_30d: int = 0
    eps_revisions_down_30d: int = 0
    revision_trend: str = "flat"  # "improving" | "deteriorating" | "flat"

    # Rating changes in last 30 days
    upgrades_30d: int = 0
    downgrades_30d: int = 0

    fetched_at: datetime = Field(default_factory=datetime.now)


class MispricingType(StrEnum):
    SENTIMENT_DRIVEN = "sentiment_driven"
    ESTIMATE_REVISION = "estimate_revision"
    MULTIPLE_EXPANSION = "multiple_expansion"
    CATALYST_BLIND = "catalyst_blind"
    NONE = "none"


class VariantPerceptionResult(BaseModel):
    """Our view vs. market consensus — the core of hedge-fund alpha articulation."""

    symbol: str

    # Market's embedded assumptions
    market_implied_growth: float | None = None  # from reverse-DCF
    consensus_target_price: float | None = None  # sell-side mean target
    consensus_recommendation: str = ""

    # Our model's base case
    our_base_price: float | None = None
    our_base_growth: float | None = None

    # The delta — where we differ
    price_vs_consensus_pct: float | None = None  # our_base / consensus_mean − 1
    our_edge: str = ""  # LLM: why we see it differently
    consensus_misses: list[str] = Field(default_factory=list)
    our_key_insight: str = ""  # 1-sentence edge articulation

    mispricing_type: MispricingType = MispricingType.NONE
    conviction: str = "MEDIUM"  # HIGH | MEDIUM | LOW

    fetched_at: datetime = Field(default_factory=datetime.now)


# ── KPI Tracker (Phase 6) ─────────────────────────────────────────────────────


class KpiStatus(StrEnum):
    ON_TRACK = "on_track"
    CAUTION = "caution"
    BREACHED = "breached"
    UNKNOWN = "unknown"


class KpiDefinition(BaseModel):
    """One measurable KPI that validates or invalidates the investment thesis."""

    metric_name: str
    description: str = ""
    measurement_source: str  # e.g. "yfinance:revenueGrowth" or "ratio:roic"
    bull_threshold: float | None = None  # above this = bullish signal
    bear_threshold: float | None = None  # below this = thesis challenged
    current_value: float | None = None
    previous_value: float | None = None  # last check, for trend
    status: KpiStatus = KpiStatus.UNKNOWN
    checked_at: datetime | None = None
    unit: str = "ratio"  # "ratio" | "pct" | "usd" | "x" | "days"


class ThesisMonitorResult(BaseModel):
    """Snapshot of all KPIs at a point in time — thesis health dashboard."""

    symbol: str
    kpis: list[KpiDefinition] = Field(default_factory=list)
    alerts: list[str] = Field(default_factory=list)  # human-readable breach messages
    thesis_status: str = "ON_TRACK"  # "ON_TRACK" | "CAUTION" | "INVALIDATED"
    checked_at: datetime = Field(default_factory=datetime.now)


# ── Management Quality (Phase 5) ─────────────────────────────────────────────


class ManagementQualityScore(BaseModel):
    """Capital allocation quality + insider behaviour composite (0–100)."""

    symbol: str

    # Components (each 0–max_points, documented inline)
    roic_trend: str = "stable"  # "improving" | "stable" | "declining"
    roic_points: int = 0  # 0–25
    insider_ownership_pct: float | None = None
    insider_ownership_points: int = 0  # 0–20
    c_suite_buying: bool = False
    c_suite_buying_points: int = 0  # 0–15
    net_insider_buying: bool = True  # net buying direction
    net_insider_buying_points: int = 0  # 0–10
    compensation_risk_score: float | None = None  # ISS 1–10 from yfinance
    compensation_points: int = 0  # 0–15
    capital_return_points: int = 0  # 0–15 (buybacks+dividends consistency)

    composite_score: float = 50.0  # 0–100
    quality_tier: str = "MEDIUM"  # "HIGH" | "MEDIUM" | "LOW"

    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Investment Memo (Phase 5) ─────────────────────────────────────────────────


class InvestmentMemo(BaseModel):
    """IC-presentation format — readable in under 90 seconds."""

    symbol: str
    company_name: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    # §0 Executive Summary
    executive_summary: str = ""

    # §1 Our Edge
    key_insight: str = ""
    mispricing_type: str = ""

    # §2 Business Quality
    moat_type: str = ""
    moat_description: str = ""
    management_quality_score: float | None = None
    management_quality_tier: str = ""

    # §3 Valuation snapshot
    current_price: float | None = None
    bear_target: float | None = None
    base_target: float | None = None
    bull_target: float | None = None
    bear_upside: float | None = None
    base_upside: float | None = None
    bull_upside: float | None = None
    consensus_target: float | None = None
    consensus_upside: float | None = None
    n_analysts: int = 0

    # §4 Top catalysts
    key_catalysts: list[str] = Field(default_factory=list)

    # §5 Downside-first
    max_loss_scenario: str = ""  # LLM: if fully wrong, what happens
    bear_case_deep_dive: str = ""

    # §6 Position sizing (conviction-based tiers)
    conviction: str = "MEDIUM"
    position_size_pct_low: float = 0.0
    position_size_pct_high: float = 0.0
    sizing_rationale: str = ""

    # §7 Exit triggers
    exit_triggers: list[str] = Field(default_factory=list)


# ── Options flow (live market data) ──────────────────────────────────────────


class UnusualOptionTrade(BaseModel):
    """Single contract flagged for unusually high volume relative to open interest."""

    contract_symbol: str
    expiration: str
    strike: float
    option_type: str  # "call" | "put"
    volume: int
    open_interest: int
    volume_oi_ratio: float
    implied_volatility: float
    last_price: float
    estimated_premium: float  # volume * last_price * 100 (notional)


class OptionsExpirationSummary(BaseModel):
    """Per-expiration aggregate across all strikes."""

    expiration: str
    call_volume: int = 0
    put_volume: int = 0
    call_oi: int = 0
    put_oi: int = 0
    pcr_volume: float | None = None  # put/call ratio by volume


class OptionsFlowResult(BaseModel):
    """Aggregated options market snapshot — pulled live from yfinance."""

    symbol: str
    current_price: float | None = None

    # Aggregate flow
    put_call_ratio: float | None = None  # total put vol / total call vol
    total_call_volume: int = 0
    total_put_volume: int = 0
    total_call_oi: int = 0
    total_put_oi: int = 0

    # Pricing signals
    max_pain_price: float | None = None  # nearest expiry max-pain strike
    atm_iv: float | None = None  # avg IV of strikes ±5% of current price

    # Notable flow
    unusual_activity: list[UnusualOptionTrade] = Field(default_factory=list)

    # Per-expiration breakdown (nearest N expirations)
    expirations: list[OptionsExpirationSummary] = Field(default_factory=list)
    nearest_expiry: str = ""

    fetched_at: datetime = Field(default_factory=datetime.now)


# ── X / Twitter social sentiment (optional enrichment) ──────────────────────


class XPost(BaseModel):
    """Single X/Twitter post extracted from Tavily search results."""

    text: str
    sentiment: str = "neutral"  # "bullish" | "bearish" | "neutral"
    themes: list[str] = Field(default_factory=list)
    url: str = ""


class XSentimentResult(BaseModel):
    """Aggregated X.com social sentiment for a ticker."""

    symbol: str
    score: float = 50.0  # 0-100 composite (50 = neutral)
    positive_pct: float = 50.0  # 0-100
    top_bullish_themes: list[str] = Field(default_factory=list)
    top_bearish_themes: list[str] = Field(default_factory=list)
    key_posts: list[XPost] = Field(default_factory=list)
    post_count: int = 0
    as_of: datetime = Field(default_factory=datetime.now)


# ── Deep Research (white-paper, cited brief) ─────────────────────────────────


class SourceType(StrEnum):
    SEC_FILING = "sec_filing"  # 10-K / 10-Q / 8-K
    NEWS = "news"
    WEBSITE = "website"  # company's own site / IR / product pages
    OTHER = "other"


class Reference(BaseModel):
    """A numbered bibliography entry the brief's claims cite by id."""

    id: int  # 1-based footnote number
    title: str = ""
    url: str = ""
    source_type: SourceType = SourceType.OTHER
    published_date: str = ""  # ISO when known (filing date / article date)
    detail: str = ""  # e.g. "10-K, Item 1. Business" or publisher


class CustomerUseCase(BaseModel):
    customer: str
    use_case: str = ""  # what they use the product FOR
    program: str = ""  # specific program/product, e.g. "USS Gerald R. Ford"
    citation_ids: list[int] = Field(default_factory=list)


class SupplyChainPosition(BaseModel):
    market_share_pct: float | None = None
    share_basis: str = ""  # e.g. "near-term semi-grade AlSiC (AGM presentation)"
    geographic_note: str = ""  # e.g. "majority of production in East Asia"
    global_players: list[str] = Field(default_factory=list)  # Denka, Sumitomo, BYD, JFC
    sole_source: bool | None = None
    position_note: str = ""  # e.g. "primary US/Western hedge"
    citation_ids: list[int] = Field(default_factory=list)


class RecentDevelopment(BaseModel):
    date: str = ""
    headline: str
    amount_usd: float | None = None
    citation_ids: list[int] = Field(default_factory=list)


class FilingQuote(BaseModel):
    quote: str  # verbatim from filing text
    form: str = ""  # 10-K / 10-Q
    filing_date: str = ""
    accession_number: str = ""
    url: str = ""
    citation_id: int | None = None


class SecondOrderThesis(BaseModel):
    thesis: str = ""
    analogs: list[str] = Field(default_factory=list)  # "InP→telecom", "Toto→memory"
    is_speculative: bool = True  # always true; UI badges it "Speculative"
    citation_ids: list[int] = Field(default_factory=list)


class DeepResearch(BaseModel):
    """White-paper-style, citation-backed business brief (SEC + news + website)."""

    symbol: str
    abstract: str = ""  # white-paper abstract / TLDR
    what_they_do: str = ""
    customers: list[CustomerUseCase] = Field(default_factory=list)
    supply_chain: SupplyChainPosition | None = None
    recent_developments: list[RecentDevelopment] = Field(default_factory=list)
    management_quotes: list[FilingQuote] = Field(default_factory=list)
    second_order_thesis: SecondOrderThesis | None = None
    references: list[Reference] = Field(default_factory=list)  # the bibliography
    fetched_at: datetime = Field(default_factory=datetime.now)


# ── Position recommendation (symbol-scoped, position-aware BUY/SELL advisory) ──


class RecommendedAction(StrEnum):
    BUY = "BUY"
    SELL = "SELL"
    INCREASE = "INCREASE"
    DECREASE = "DECREASE"
    HOLD = "HOLD"


class PositionContext(BaseModel):
    """Real, Python-computed position facts — never LLM-generated."""

    has_position: bool = False
    equity_quantity: float = 0.0
    average_open_price: float = 0.0
    mark: float = 0.0
    unrealized_pnl_pct: float | None = None
    option_legs: list[str] = Field(default_factory=list)
    accounts: list[str] = Field(default_factory=list)
    error: str = ""  # set on a best-effort account-data fetch failure


class ActionRecommendation(BaseModel):
    symbol: str
    action: RecommendedAction = RecommendedAction.HOLD
    conviction: str = "LOW"  # "HIGH" | "MEDIUM" | "LOW"
    reasoning: str = ""
    key_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    position: PositionContext = Field(default_factory=PositionContext)
    generated_at: datetime = Field(default_factory=datetime.now)


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
    fair_price: FairPriceResult | None = None  # consolidated headline fair value

    # § Catalysts & Risks (Phase 2)
    catalyst_risk: CatalystRiskResult | None = None

    # § Thesis (Phase 3) — bull/base/bear targets + probabilities

    # § Company Network (Phase 3)
    network: CompanyNetwork | None = None

    # § Industry (Phase 3)
    industry: IndustryAnalysis | None = None

    # § Transcripts (Phase 3)
    transcripts: TranscriptAnalysis | None = None

    # § Market Data (Phase 3)
    market_data: MarketDataSnapshot | None = None

    # § Thesis (Phase 3)
    thesis: ThesisResult | None = None

    # § Variant Perception (Phase 4)
    consensus: AnalystConsensus | None = None
    variant_perception: VariantPerceptionResult | None = None

    # § Management Quality (Phase 5)
    management_quality: ManagementQualityScore | None = None

    # § Investment Memo (Phase 5)
    investment_memo: InvestmentMemo | None = None

    # § X / Twitter social sentiment (optional)
    x_sentiment: XSentimentResult | None = None

    # § Options flow (live)
    options_flow: OptionsFlowResult | None = None

    # § KPI Monitor (Phase 6)
    kpi_monitor: ThesisMonitorResult | None = None

    filings: list[FilingRef] = Field(default_factory=list)

    # § Deep Research (white-paper, cited brief — SEC + news + website)
    deep_research: DeepResearch | None = None

    notes: list[str] = Field(default_factory=list)
