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

// ResearchReport is large + deeply nested; the panels read it defensively.
export type ResearchReport = Record<string, any>;
