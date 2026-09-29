// TS mirrors of the Pydantic fields the UI actually reads. Not exhaustive —
// research panels access nested objects defensively.

export interface ResearchSummary {
  thesis_status: "ON_TRACK" | "CAUTION" | "INVALIDATED" | "UNKNOWN";
  conviction: string | null;
  attention: "HIGH" | "MEDIUM" | "LOW";
  next_earnings_date: string | null;
  has_report: boolean;
  kpi_alerts: string[];
  sector: string | null;
}

export interface Holding {
  symbol: string;
  quantity: number;
  average_open_price: number;
  multiplier: number;
  close_price: number;
  mark_price: number;
  accounts: string[];
  research: ResearchSummary | null;
}

export interface Balances {
  net_liq: number;
  cash: number;
  buying_power: number;
  accounts: string[];
}

export interface HoldingsResponse {
  holdings: Holding[];
  balances: Balances;
  symbols: string[];
}

export interface Quote {
  symbol: string;
  bid: number;
  ask: number;
  mid: number;
  ts: string;
}

export interface PositionReview {
  symbol: string;
  company_name: string;
  accounts: string[];
  thesis_status: string;
  kpi_alerts: string[];
  conviction: string | null;
  report_was_built: boolean;
  has_report: boolean;
  near_term_catalysts: string[];
  next_earnings_date: string | null;
  attention: string;
  error: string | null;
}

export interface PortfolioReview {
  generated_at: string;
  account_numbers: string[];
  positions: PositionReview[];
}

// ── Market context (VIX + regime) ─────────────────────────────────────────────

export interface VixPoint {
  date: string;
  vix: number;
}

export interface VixSnapshot {
  current: number;
  sma20: number;
  percentile_1y: number;
  history: VixPoint[];
}

export interface Regime {
  date: string;
  regime_name: string;
  label: string; // Calm | Normal | Stressed
  vix: number;
  regime_prob: number[];
  spy_vol: number;
}

export interface MarketContext {
  vix: VixSnapshot | null;
  regime: Regime | null;
}

// ── Sector rotation ───────────────────────────────────────────────────────────

export interface SectorMomentum {
  etf: string;
  etf_return_1m: number | null;
  etf_return_3m: number | null;
  rel_1m: number | null;
  rel_3m: number | null;
  leading: boolean;
}

export interface RotationResponse {
  rotation: Record<string, SectorMomentum>;
  weights: Record<string, number>;
}

export interface Job {
  id: string;
  kind: string;
  target: string;
  status: "running" | "done" | "error";
  message: string;
  error: string | null;
  started_at: string;
  finished_at: string | null;
}

export interface FilingRef {
  accession_number: string;
  form: string; // "10-K" | "10-Q" | "8-K" | "DEF 14A" | "13F-HR" | "4"
  filing_date: string;
  period_of_report: string | null;
  url: string;
}

export interface TranscriptSource {
  url: string;
  title: string;
}

export interface TranscriptSummary {
  quarter: string;
  earnings_date: string;
  tone: "bullish" | "neutral" | "bearish";
  key_topics: string[];
  management_guidance: string;
  analyst_concerns: string;
  highlight_quote: string;
  source_url: string;
}

export interface TranscriptAnalysis {
  symbol: string;
  summaries: TranscriptSummary[];
  tone_trend: string;
  sources: TranscriptSource[];
  fetched_at: string;
}

// ── Deep Research (white-paper, cited brief) ──────────────────────────────────

export type SourceType = "sec_filing" | "news" | "website" | "other";

export interface Reference {
  id: number;
  title: string;
  url: string;
  source_type: SourceType;
  published_date: string;
  detail: string;
}

export interface CustomerUseCase {
  customer: string;
  use_case: string;
  program: string;
  citation_ids: number[];
}

export interface SupplyChainPosition {
  market_share_pct: number | null;
  share_basis: string;
  geographic_note: string;
  global_players: string[];
  sole_source: boolean | null;
  position_note: string;
  citation_ids: number[];
}

export interface RecentDevelopment {
  date: string;
  headline: string;
  amount_usd: number | null;
  citation_ids: number[];
}

export interface FilingQuote {
  quote: string;
  form: string;
  filing_date: string;
  accession_number: string;
  url: string;
  citation_id: number | null;
}

export interface SecondOrderThesis {
  thesis: string;
  analogs: string[];
  is_speculative: boolean;
  citation_ids: number[];
}

export interface DeepResearch {
  symbol: string;
  abstract: string;
  what_they_do: string;
  customers: CustomerUseCase[];
  supply_chain: SupplyChainPosition | null;
  recent_developments: RecentDevelopment[];
  management_quotes: FilingQuote[];
  second_order_thesis: SecondOrderThesis | null;
  references: Reference[];
  fetched_at: string;
}

// ResearchReport is large + deeply nested; the panels read it defensively.
export type ResearchReport = Record<string, any>;

// ── Position recommendation (symbol-scoped BUY/SELL/HOLD advisory) ────────────

export type RecommendedAction = "BUY" | "SELL" | "INCREASE" | "DECREASE" | "HOLD";

export interface PositionContext {
  has_position: boolean;
  equity_quantity: number;
  average_open_price: number;
  mark: number;
  unrealized_pnl_pct: number | null;
  option_legs: string[];
  accounts: string[];
  error: string;
}

export interface ActionRecommendation {
  symbol: string;
  action: RecommendedAction;
  conviction: string; // "HIGH" | "MEDIUM" | "LOW"
  reasoning: string;
  key_factors: string[];
  risks: string[];
  position: PositionContext;
  generated_at: string;
}

// ── Price history + fundamentals overlay ──────────────────────────────────────

export interface PriceBar {
  date: string;
  close: number;
  volume: number | null;
}

export interface EarningsMarker {
  date: string;
  revenue: number | null;
  eps: number | null;
  yoy_eps_growth: number | null;
}

export interface PePoint {
  date: string;
  pe: number | null;
}

export interface FundamentalPoint {
  date: string;
  fiscal_year: number;
  revenue: number | null;
  eps: number | null;
  net_margin: number | null;
}

export interface PriceHistoryResult {
  symbol: string;
  bars: PriceBar[];
  earnings: EarningsMarker[];
  pe_series: PePoint[];
  fundamentals: FundamentalPoint[];
  fetched_at: string;
}

// Slider adjustments POSTed back to recompute the posterior. All optional.
// ── Watchlist ─────────────────────────────────────────────────────────────────

export interface WatchlistSummary {
  has_report: boolean;
  thesis_status: string;
  attention: "HIGH" | "MEDIUM" | "LOW";
  conviction: string | null;
  current_price: number | null;
  kpi_alerts: string[];
}

export interface WatchlistItem {
  symbol: string;
  note: string;
  added_at: string;
  research: WatchlistSummary | null;
}

export interface WatchlistResponse {
  watchlist: WatchlistItem[];
}

// ── Investment theses (long-form research write-ups) ──────────────────────────

export type Conviction = "HIGH" | "MEDIUM" | "LOW";
export type ThesisStatus = "DRAFT" | "ACTIVE" | "ARCHIVED";

// List-row shape (GET /api/theses) — no markdown body.
export interface ThesisSummary {
  id: string;
  symbol: string; // "" = thematic
  title: string;
  tags: string[];
  conviction: Conviction;
  status: ThesisStatus;
  created_at: string;
  updated_at: string;
}

// Full document (GET /api/theses/:id) — includes the markdown body.
export interface Thesis extends ThesisSummary {
  content: string;
}

// Body for create/update (POST/PUT /api/theses).
export interface ThesisInput {
  symbol: string;
  title: string;
  content: string;
  tags: string[];
  conviction: Conviction;
  status: ThesisStatus;
}

// ── Research agent (interactive chat) ─────────────────────────────────────────

// One SSE event streamed from POST /api/research/:symbol/chat.
export type ChatEvent =
  | { type: "meta"; conversation_id: string }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; ok: boolean }
  | { type: "token"; text: string }
  | { type: "done"; text: string }
  | { type: "error"; message: string };

export interface ToolEvent {
  name: string;
  ok?: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  tools?: string[]; // tool names used (assistant turns)
}

export interface ConversationSummary {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface Conversation {
  id: string;
  symbol: string;
  title: string;
  messages: ChatMessage[];
  created_at: string;
  updated_at: string;
}

// ── Portfolio performance (deposit-adjusted returns) ──────────────────────────

export interface PeriodReturn {
  label: string;
  start_date: string | null;
  end_date: string;
  return_pct: number | null;
  v_start: number | null;
  v_end: number | null;
  cf_net: number | null;
}

export interface EquityPoint {
  date: string;
  value: number;
  is_deposit: boolean;
  deposit_amount: number | null;
}

export interface PerformanceResponse {
  periods: PeriodReturn[];
  equity_curve: EquityPoint[];
  snapshot_count: number;
  cash_flow_count: number;
}

// ── Daemon: the always-on advisor's own state ─────────────────────────────────

export type EventTier = "A" | "B" | "C";
export type SourceTier = "PRIMARY" | "BROKER" | "AGGREGATOR" | "UNTAGGED";

export interface DaemonJob {
  name: string;
  description: string;
  last_run_at: string | null;
  last_ok_at: string | null;
  run_count: number;
  error_count: number;
  last_error: string;
}

export interface DaemonWatermark {
  source: string;
  last_seen_ts: string | null;
  updated_at: string | null;
}

export interface DaemonStatus {
  now: string;
  market_open: boolean;
  trading_day: boolean;
  jobs: DaemonJob[];
  watermarks: DaemonWatermark[];
  event_counts: Record<string, number>;
}

export interface DaemonEvent {
  id: string;
  ts: string;
  source: string;
  kind: string;
  tier: EventTier;
  symbol: string | null;
  payload: Record<string, unknown>;
  /** One numeric line saying what this means, computed server-side so the two
   *  readers cannot drift apart. Empty when the payload carries nothing. */
  summary: string;
}

export interface FactorExposureRow {
  factor: string;
  net_loading: number;
  concentration: number;
  contributors: { symbol: string; share: number }[];
}

export interface BookExposure {
  asof: string;
  net_liq: number;
  covered_weight: number;
  uncovered: string[];
  factors: FactorExposureRow[];
}

export interface SourceItem {
  tier: SourceTier;
  provider: string;
  url: string;
  title: string;
  published_at: string;
  symbol: string;
  doc_type: string | null;
  item_codes: string[];
  accession: string | null;
  match: string;
  confidence: number;
  /** The filer's or publisher's own opening words; null when unreadable. */
  lead: string | null;
}

export interface ReconcileFinding {
  check: string;
  severity: "OK" | "WARN" | "FAIL";
  symbol: string | null;
  detail: string;
}

export interface ReconcileReport {
  asof: string;
  ok: boolean;
  summary: string;
  findings: ReconcileFinding[];
}

export interface FactorLoadingRow {
  factor: string;
  loading: number;
  tstat: number;
  material: boolean;
}

export interface SymbolDaemonDetail {
  symbol: string;
  sensitivity: {
    asof: string;
    n_obs: number;
    r2: number;
    resid_vol: number;
    loadings: FactorLoadingRow[];
  } | null;
  contribution: {
    factor: string;
    loading: number;
    contribution: number;
    book_total: number;
  }[];
  timeline: {
    published_at: string;
    tier: SourceTier;
    provider: string;
    title: string;
    url: string;
    doc_type: string | null;
    item_codes: string[];
    match: string;
    /** The filer's or publisher's own opening words; null when unreadable. */
    lead: string | null;
  }[];
  events: DaemonEvent[];
}

export interface JobRunResult {
  job: string;
  ok: boolean;
  detail: string;
  events_emitted: number;
  duration_ms: number;
  ran_at: string;
}

// ── Story: one event, assembled ───────────────────────────────────────────────

export type Confidence = "MEASURED" | "ESTIMATED" | "UNAVAILABLE";
export type Verdict =
  | "NOT_MARKET"
  | "UNEXPLAINED"
  | "PARTLY_SPECIFIC"
  | "CONSISTENT"
  | "UNKNOWN";

export interface Story {
  symbol: string;
  headline: string;
  unavailable: string[];
  assembled_at: string;
  anchor: {
    event_id: string;
    kind: string;
    tier: EventTier;
    symbol: string;
    occurred_at: string;
    ingested_at: string;
    headline: string;
    source: string;
    url: string | null;
    quote: string | null;
    facts: Record<string, number | string | string[]>;
  };
  position: {
    confidence: Confidence;
    quantity: number | null;
    avg_open_price: number | null;
    weight_of_net_liq: number | null;
    net_liq: number | null;
    snapshot_asof: string | null;
    covers_event: boolean;
    note: string;
  };
  reaction: {
    confidence: Confidence;
    session: string | null;
    before: number | null;
    after: number | null;
    pct_move: number | null;
    dollars: number | null;
    pct_of_book: number | null;
    priced_next_session: boolean;
    note: string;
  };
  attribution: {
    confidence: Confidence;
    verdict: Verdict;
    actual_return: number | null;
    expected_return: number | null;
    residual_z: number | null;
    resid_vol: number | null;
    r2: number | null;
    note: string;
  };
  corroboration: {
    window_days: number;
    items: { tier: SourceTier; title: string; url: string; published_at: string }[];
    primary_count: number;
    aggregator_count: number;
    untagged_count: number;
  };
  thesis: {
    confidence: Confidence;
    exists: boolean;
    title: string | null;
    conviction: string | null;
    status: string | null;
    note: string;
  };
}

// ── Structured thesis claims ──────────────────────────────────────────────────

export type ClaimKind =
  | "DRIVER"
  | "INVALIDATION"
  | "KPI"
  | "MACRO_DRIVER"
  | "CATALYST"
  | "RISK";

export interface ThesisClaim {
  id: string | null;
  kind: ClaimKind;
  text: string;
  monitored: boolean;
  trigger_description: string;
  trigger: {
    event_kinds: string[];
    field: string | null;
    comparator: string;
    threshold: number | null;
    factor: string | null;
  };
  due: string | null;
}

export interface StructuredThesis {
  title: string;
  conviction: string | null;
  status: string | null;
  substantive: boolean;
  prose_note: string;
  coverage: number;
  claims: ThesisClaim[];
}

export interface ThesisCoverageRow {
  symbol: string;
  weight: number;
  written: boolean;
  claims: number;
  monitored: number;
}

export interface ThesisCoverage {
  rows: ThesisCoverageRow[];
  uncovered_count: number;
  uncovered_weight: number;
}

// ── Action cards ──────────────────────────────────────────────────────────────

export type ActionKind =
  | "REVIEW_NOW"
  | "REVIEW"
  | "WRITE_THESIS"
  | "HOLD"
  | "CANNOT_SAY";

export interface EvidenceItem {
  name: string;
  asof: string | null;
  detail: string;
  ok: boolean;
  blocking: boolean;
}

export interface ClaimVerdict {
  text: string;
  kind: string;
  status: "BROKEN" | "STANDING" | "INTACT" | "UNTESTED" | "UNREACHABLE";
  note: string;
  /** Both carried so a decision can be recorded against this rule and later
   *  compared to where the number has gone. Acknowledging a 14% weight is not
   *  acknowledging a 25% one. */
  claim_id: string | null;
  observed: number | null;
}

/** Named for the loop step, not "Verdict" — `Verdict` is already the story
 *  layer's attribution of a price move. */
export type DecisionVerdict = "ACKNOWLEDGED" | "ACTED" | "DISMISSED" | "THESIS_REVISED";

export interface DecidedRef {
  subject_kind: string;
  subject_id: string;
  text: string;
  verdict: DecisionVerdict;
  note: string;
  reason: string;
  decided_at: string | null;
}

export interface ActionCard {
  symbol: string;
  assembled_at: string;
  action: ActionKind;
  headline: string;
  because: string[];
  deadline: string | null;
  position_note: string;
  triggers: { kind: string; tier: EventTier; when: string; detail: string; url: string | null }[];
  claims: ClaimVerdict[];
  evidence: { items: EvidenceItem[] };
  rationale: Rationale | Record<string, never>;
  what_would_sharpen_this: string[];
  /** Answered already, and still answered — shown, never dropped. */
  decided: DecidedRef[];
  /** Answered, and no longer answered: the number moved past where the call
   *  was made, or an action had time to show and did not. */
  reopened: DecidedRef[];
}

export type Bearing = "SUPPORTS" | "AGAINST" | "CONTEXT" | "BLIND";
export type ActionSource = "YOUR_RULE" | "ARITHMETIC" | "NONE";

export interface ReasonStep {
  label: string;
  fact: string;
  bearing: Bearing;
}

export interface ProposedAction {
  source: ActionSource;
  text: string;
  size: string;
  quoted_from: string;
}

export interface Rationale {
  steps: ReasonStep[];
  proposed: ProposedAction | null;
}

export type ReadingStance = "CONSTRUCTIVE" | "NEUTRAL" | "CAUTIOUS" | "AT_RISK";

export interface ReadingFact {
  id: string;
  kind: "POSITION" | "VALUATION" | "SCORECARD" | "EVENT" | "NEWS" | "CLAIM";
  text: string;
  date: string | null;
  url: string | null;
  source: string | null;
}

export interface ReadingSentence {
  text: string;
  facts: string[];
}

export interface ScorecardRow {
  label: string;
  value: string;
  detail: string;
  source: string;
  status: "OK" | "TRIPPED" | "UNCHECKABLE" | null;
  claim: string | null;
}

/** Deterministic numbers: what the price requires, what the business
 *  delivers, what analysts expect, and the holder's lines. */
export interface Scorecard {
  symbol: string;
  expectations: ScorecardRow[];
  thresholds: ScorecardRow[];
}

/** Model-written, shown only when every number traces to a cited fact. */
export interface TickerReading {
  symbol: string;
  status: "OK" | "REJECTED" | "NO_FACTS" | "UNAVAILABLE";
  stance: ReadingStance | null;
  sentences: ReadingSentence[];
  facts: ReadingFact[];
  facts_hash: string;
  model: string | null;
  problems: string[];
  generated_at: string;
  window_days: number;
  scorecard: Scorecard | null;
}

export type AngleStatus = "SUGGESTED" | "CONFIRMED" | "REJECTED";

/** A product or segment the holder confirmed stands for part of a company. */
export interface WatchAngle {
  symbol: string;
  term: string;
  status: AngleStatus;
  source: string | null;
  updated_at: string;
}

// ── Tracking: entry, risk and close per name; whether the system is current ──

export interface TrackPoint {
  session: string;
  action: string;
  price: number | null;
}

export interface TrackTrade {
  opened: string;
  closed: string | null;
  quantity: number;
  entry_price: number;
  exit_price: number | null;
  pnl: number | null;
  ret: number | null;
  book: string;
  /** The system's call the session the trade was opened (ENTER/ADD = followed). */
  call: string | null;
}

export interface TrackRisk {
  basis: "held" | "planned";
  entry: number;
  price: number | null;
  stop: number | null;
  stop_basis: string;
  target: number | null;
  shares: number;
  to_stop: number | null;
  at_risk: number | null;
  at_risk_pct: number | null;
  past_stop: boolean;
}

export interface TrackLatest {
  session: string;
  built_at: string;
  action: string;
  price: number | null;
  blockers: string[];
  reasons: string[];
  stale: string[];
  rules: string | null;
  legs: Record<string, unknown>[];
}

/** One news item judged by the news agent: context only, never an action. */
export interface TrackNews {
  published_at: string;
  title: string;
  provider: string;
  url: string | null;
  about_company: boolean;
  direction: "POSITIVE" | "NEGATIVE" | "MIXED" | "NEUTRAL";
  materiality: "HIGH" | "MEDIUM" | "LOW";
  event_type: string;
  novelty: string;
  basis: string;
  why: string;
  thesis: string[];
  what: string;
  magnitude: string;
  watch: string;
  market_read: "MOVED_WITH" | "MOVED_AGAINST" | "MOVED" | "QUIET" | "UNKNOWN";
  market: string | null;
  /** article: the publisher's page · feed: the feed's own text · headline: nothing more */
  read_from: "article" | "feed" | "headline";
}

/** The news agent's synthesis of a name's week. */
export interface TrackNewsWeek {
  day: string;
  net: "POSITIVE" | "NEGATIVE" | "MIXED" | "NEUTRAL";
  headline: string;
  text: string;
  thesis: string;
  watch: string[];
  items: number;
}

export interface TrackRow {
  news: TrackNews[];
  news_week: TrackNewsWeek | null;
  news_summary: Partial<
    Record<
      "items" | "about" | "positive" | "negative" | "mixed" | "high" | "against_thesis" | "for_thesis",
      number
    >
  >;
  symbol: string;
  held: boolean;
  quantity: number;
  cost: number | null;
  price: number | null;
  weight: number | null;
  unrealized: number | null;
  unrealized_pct: number | null;
  latest: TrackLatest | null;
  risk: TrackRisk | null;
  timeline: TrackPoint[];
  open_trades: TrackTrade[];
  closed_trades: TrackTrade[];
  realized: number;
  wins: number;
  losses: number;
}

export interface TrackJob {
  name: string;
  description: string;
  schedule: string;
  state: "ok" | "late" | "failing" | "never" | "idle";
  last_ok_at: string | null;
  last_run_at: string | null;
  due_since: string | null;
  last_error: string;
}

export interface TrackRuleChange {
  id: string;
  ruleset: string;
  param: string;
  value: number;
  previous: number;
  status: string;
  source: string;
  note: string;
  evidence_at: string | null;
  expires_at: string;
  days_left: number;
  in_force: boolean;
}

export interface SystemStatus {
  now: string;
  ok: boolean;
  problems: string[];
  code: {
    main: string | null;
    daemon: string | null;
    daemon_behind: number | null;
    api: string | null;
    api_behind: number | null;
    dirty: string[];
  };
  jobs: TrackJob[];
  rules: {
    code_version: string | null;
    proposals_version: string | null;
    proposals_session: string | null;
    changes: TrackRuleChange[];
    expired_recently: TrackRuleChange[];
  };
  stale_inputs: Record<string, string[]>;
}

// ── Price range (one valuation engine, four cases + rationale) ────────────────

export type PriceCaseName = "bear" | "base" | "bull" | "market";

export interface PriceCase {
  name: PriceCaseName;
  value_per_share: number;
  upside: number;
  growth: number | null;
  held_years: number | null;
  margin: number | null;
  margin_label: string;
}

/** The company in a few sourced bullets (``story.overview``). */
export interface OverviewBullet {
  topic: string;
  text: string;
  source: string;
  tone: "pos" | "neg" | "warn" | "neutral";
}

export interface CompanyOverview {
  symbol: string;
  as_of: string;
  bullets: OverviewBullet[];
  gaps: string[];
}

export interface PriceRange {
  symbol: string;
  asof: string;
  price: number;
  price_source: string;
  live: boolean;
  stale: boolean;
  period_end: string;
  market_cap: number;
  net_cash: number | null;
  enterprise_value: number;
  revenue_base: number | null;
  revenue_base_label: string;
  ev_to_revenue: number | null;
  growth: number | null;
  growth_label: string;
  own_margins: { kind: string; value: number; label: string }[];
  cases: PriceCase[];
  refused: string | null;
  requires: { margin: number; label: string; growth: number }[];
  verdict: "above_bull" | "above_base" | "above_bear" | "below_bear" | null;
  rationale: string[];
  assumptions: string;
  notes: string[];
}

// ── Breadth picks ─────────────────────────────────────────────────────────

export interface PickReason {
  family: "F" | "P" | "I" | "";
  text: string;
  source: string;
}

export interface Pick {
  rank: number;
  symbol: string;
  name: string | null;
  sector: string | null;
  held: boolean;
  families: ("F" | "I" | "P")[];
  since: string;
  sessions_since: number;
  price: number;
  move_since: number | null;
  return_20d: number | null;
  reasons: PickReason[];
  invalidates: string[];
  provisional?: boolean;
  asof?: string;
  /** For a past day's pick: its move from that day's price to the latest close. */
  since_pick?: { day: string; close: number; move: number } | null;
}

export interface EntryLeg {
  horizon: "trade" | "position" | string;
  entry: number;
  stop: number;
  stop_basis: string;
  risk_pct: number;
  shares: number;
  notional: number;
  exit_rules: string[];
  notes: string[];
  target?: number | null;
}

/** What depth a name already has (``/api/depth/status``). */
export interface DepthStatus {
  news_judged: number;
  news_material: number;
  news_last: string | null;
  report_at: string | null;
  deep_research: boolean;
}

/** One news item as the news agent judged it (``news.judge.Judgment``). */
export interface NewsJudgment {
  key: string;
  symbol: string;
  published_at: string;
  title: string;
  provider: string;
  tier: string;
  url: string | null;
  about_company: boolean;
  event_type: string;
  direction: "POSITIVE" | "NEGATIVE" | "MIXED" | "NEUTRAL";
  materiality: "HIGH" | "MEDIUM" | "LOW";
  novelty: string;
  quote: string;
  why: string;
  what: string;
  magnitude: string;
  watch: string;
  market_read: string;
  read_from: string;
  problems: string[];
}

export interface NewsSummary {
  symbol: string;
  day: string;
  net: string;
  headline: string;
  text: string;
  thesis: string;
  watch: string[];
  items: number;
  generated_at: string;
}

/** The entry engine's proposal for one name (``entry.proposal.Proposal``). */
export interface EntryProposal {
  symbol: string;
  session: string;
  built_at: string;
  action: string;
  price: number | null;
  triggers: string[];
  reasons: { text: string; source: string }[];
  blockers: string[];
  legs: EntryLeg[];
  stance: string | null;
  reading: string[];
  gaps: string[];
  net_liq: number | null;
}

export interface PickCell {
  group: string;
  horizon: string;
  n: number;
  windows?: number;
  excess?: number;
  ci?: [number, number] | null;
  excess_trend?: number | null;
  ci_trend?: [number, number] | null;
  beat?: number;
  tail?: number | null;
  verdict: string;
}

export interface ReplayWindow {
  years: number;
  run_id: string;
  from: string;
  to: string;
  cells: PickCell[];
  cells_tested: number;
}

export interface PicksResponse {
  day: string | null;
  days: { day: string; count: number; provisional: boolean }[];
  provisional?: boolean;
  built_at?: string | null;
  rules?: string | null;
  picks: Pick[];
  /** The latest replay of each window, longest first. */
  track_record: ReplayWindow[];
  live_records?: Record<string, number>;
  caveats: string[];
}
