import type {
  BayesianOverrides,
  BayesianPriceResult,
  ChatEvent,
  Conversation,
  ConversationSummary,
  HoldingsResponse,
  Job,
  MarketContext,
  PortfolioReview,
  PriceHistoryResult,
  RotationResponse,
  ResearchReport,
  Thesis,
  ThesisInput,
  ThesisSummary,
  WatchlistResponse,
} from "./types";

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post<T>(url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function put<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function del<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: "DELETE" });
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  holdings: () => get<HoldingsResponse>("/api/portfolio/holdings"),
  review: () =>
    get<{ review: PortfolioReview | null; fetched_at: string | null }>("/api/portfolio/review"),
  refreshReview: (rebuild = false) =>
    post<{ job_id: string }>(`/api/portfolio/review/refresh?rebuild_uncovered=${rebuild}`),
  market: () => get<MarketContext>("/api/portfolio/market"),
  rotation: () => get<RotationResponse>("/api/portfolio/rotation"),
  research: (symbol: string) => get<ResearchReport>(`/api/research/${symbol}`),
  priceHistory: (symbol: string) =>
    get<{ result: PriceHistoryResult; fetched_at: string }>(
      `/api/research/${symbol}/price-history`
    ),
  refreshResearch: (symbol: string) => post<{ job_id: string }>(`/api/research/${symbol}/refresh`),
  job: (id: string) => get<Job>(`/api/jobs/${id}`),
  bayesian: (symbol: string) =>
    get<{ result: BayesianPriceResult; fetched_at: string }>(`/api/portfolio/bayesian/${symbol}`),
  recomputeBayesian: (symbol: string, overrides: BayesianOverrides) =>
    post<{ result: BayesianPriceResult }>(`/api/portfolio/bayesian/${symbol}`, overrides),
  conversations: (symbol: string) =>
    get<{ conversations: ConversationSummary[] }>(`/api/research/${symbol}/conversations`),
  conversation: (id: string) => get<Conversation>(`/api/agent/conversations/${id}`),
  watchlist: () => get<WatchlistResponse>("/api/watchlist"),
  addWatchlist: (symbol: string, note = "") =>
    post<{ symbol: string; job_id: string }>("/api/watchlist", { symbol, note }),
  removeWatchlist: (symbol: string) =>
    del<{ symbol: string; removed: boolean }>(`/api/watchlist/${symbol}`),
  theses: (symbol?: string) =>
    get<{ theses: ThesisSummary[] }>(
      symbol ? `/api/theses?symbol=${encodeURIComponent(symbol)}` : "/api/theses"
    ),
  thesis: (id: string) => get<Thesis>(`/api/theses/${id}`),
  createThesis: (body: ThesisInput) => post<Thesis>("/api/theses", body),
  updateThesis: (id: string, body: ThesisInput) => put<Thesis>(`/api/theses/${id}`, body),
  deleteThesis: (id: string) =>
    del<{ id: string; removed: boolean }>(`/api/theses/${id}`),
};

/**
 * Stream a research-agent chat turn. POSTs the message and yields parsed SSE
 * events as they arrive (EventSource can't POST, so we read the body stream).
 */
export async function* agentChat(
  symbol: string,
  message: string,
  conversationId?: string,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`/api/research/${symbol}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, conversation_id: conversationId }),
  });
  if (!res.ok || !res.body) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() ?? ""; // keep the trailing partial frame
    for (const frame of frames) {
      const line = frame.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        yield JSON.parse(line.slice(6)) as ChatEvent;
      } catch {
        // ignore malformed frame
      }
    }
  }
}
