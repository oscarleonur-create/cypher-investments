// TS mirrors of the Pydantic fields the UI actually reads. Not exhaustive —
// research panels access nested objects defensively.

export interface ResearchSummary {
  thesis_status: "ON_TRACK" | "CAUTION" | "INVALIDATED" | "UNKNOWN";
  conviction: string | null;
  attention: "HIGH" | "MEDIUM" | "LOW";
  next_earnings_date: string | null;
  base_upside: number | null;
  has_report: boolean;
  kpi_alerts: string[];
  sector: string | null;
  bayes_upside: number | null;
  bayes_prob_undervalued: number | null;
  analyst_target: number | null;
  analyst_upside: number | null;
  analyst_n: number | null;
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
  base_target: number | null;
  base_upside: number | null;
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

// ── Bayesian pricing (what-if posterior) ──────────────────────────────────────

export interface PriorDriver {
  key: string;
  label: string;
  mean: number;
  std: number;
  min: number;
  max: number;
  unit: "pct" | "x" | "ratio";
}

export interface EvidenceSignal {
  key: string;
  label: string;
  target_driver: string;
  observed: number | null;
  precision: number;
  weight: number;
  note: string;
}

export interface EcosystemFactor {
  key: string;
  label: string;
  kind: "customer" | "supplier" | "holder" | "peer";
  driver: string;
  active: boolean;
  mean_delta: number;
  std_delta: number;
  note: string;
}

export interface HistogramBin {
  x: number;
  count: number;
}

export interface CatalystScenario {
  label: string;
  probability: number;
  target_price: number;
  upside_pct: number;
  supporting_catalysts: string[];
  invalidating_catalysts: string[];
}

export interface BayesianPriceResult {
  symbol: string;
  current_price: number;
  n_draws: number;
  drivers: PriorDriver[];
  evidence: EvidenceSignal[];
  ecosystem: EcosystemFactor[];
  mean_price: number;
  median_price: number;
  p5: number;
  p25: number;
  p75: number;
  p95: number;
  prob_undervalued: number;
  expected_upside_pct: number;
  histogram: HistogramBin[];
  meaningful: boolean;
  note: string;
  catalyst_scenarios: CatalystScenario[];
  as_of: string;
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
export interface BayesianOverrides {
  driver_mean?: Record<string, number>;
  driver_std?: Record<string, number>;
  evidence_weight?: Record<string, number>;
  ecosystem_active?: Record<string, boolean>;
  n_draws?: number;
}

// ── Fair price (consolidated valuation) ───────────────────────────────────────

export interface FairPriceMethod {
  name: string; // dcf_base | multiples | bayesian_median | analyst_target
  label: string;
  estimate: number;
  weight: number;
}

export interface FairPriceResult {
  symbol: string;
  current_price: number;
  fair_price: number;
  low: number;
  high: number;
  upside_pct: number;
  methods: FairPriceMethod[];
  confidence: "HIGH" | "MEDIUM" | "LOW";
  note: string;
  as_of: string;
}

// ── Watchlist ─────────────────────────────────────────────────────────────────

export interface WatchlistSummary {
  has_report: boolean;
  thesis_status: string;
  attention: "HIGH" | "MEDIUM" | "LOW";
  conviction: string | null;
  base_upside: number | null;
  current_price: number | null;
  fair_price: number | null;
  fair_upside: number | null;
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

// ── Scalping scanner ──────────────────────────────────────────────────────────

export interface ScalpStrategyInfo {
  name: string;
  label: string;
  description: string;
  defaults: Record<string, number>;
}

export interface ScalpSignal {
  symbol: string;
  strategy: string;
  action: "LONG" | "SHORT" | "FLAT";
  reason: string;
  price: number;
  entry: number;
  stop: number;
  target: number;
  score: number;
  technical_score: number | null;
  bar_time: string;
  // catalyst context
  rvol: number | null;
  gap_pct: number | null;
  earnings_today: boolean;
  days_to_earnings: number | null;
  headlines: string[];
  sentiment_score: number | null;
  catalyst_note: string;
  // risk gate verdict
  risk_approved: boolean | null;
  risk_quantity: number | null;
  risk_note: string;
}

export interface ScalpScanResult {
  scanned_at: string;
  interval: string;
  symbols_scanned: number;
  source: string; // "tastytrade" | "yfinance"
  min_rvol: number;
  gated_out: number;
  signals: ScalpSignal[];
  errors: string[];
}

// ── Swing trade scanner ───────────────────────────────────────────────────────

export interface SwingStrategyInfo {
  name: string;
  label: string;
  description: string;
  typical_hold_days: string;
}

export interface SwingSignal {
  symbol: string;
  strategy: string;
  verdict: "ENTER" | "CAUTION" | "PASS";
  reasoning: string;
  suggested_hold_days: number;
  // technical
  technical_signal: string;
  technical_price: number;
  technical_bullish: boolean;
  volume_ratio: number;
  // sentiment
  sentiment_bullish: boolean;
  sentiment_pct: number;
  sentiment_headlines: string[];
  // fundamental
  fundamental_clear: boolean;
  earnings_within_7_days: boolean;
  earnings_date: string | null;
  insider_buying: boolean;
  // ml
  ml_available: boolean;
  ml_signal: string;
  ml_win_prob: number | null;
  scanned_at: string;
}

export interface SwingScanResult {
  scanned_at: string;
  strategy: string;
  symbols_scanned: number;
  signals: SwingSignal[];
  errors: string[];
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

export interface ScalpCandle {
  t: number; // epoch ms
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  volume: number | null;
  vwap: number | null;
}

export interface ScalpPreview {
  symbol: string;
  interval: string;
  source: string;
  candles: ScalpCandle[];
  signal: ScalpSignal | null;
}
