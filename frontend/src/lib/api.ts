import type { HoldingsResponse, Job, PortfolioReview, ResearchReport } from "./types";

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function post<T>(url: string): Promise<T> {
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  holdings: () => get<HoldingsResponse>("/api/portfolio/holdings"),
  review: () =>
    get<{ review: PortfolioReview | null; fetched_at: string | null }>("/api/portfolio/review"),
  refreshReview: (rebuild = false) =>
    post<{ job_id: string }>(`/api/portfolio/review/refresh?rebuild_uncovered=${rebuild}`),
  research: (symbol: string) => get<ResearchReport>(`/api/research/${symbol}`),
  refreshResearch: (symbol: string) => post<{ job_id: string }>(`/api/research/${symbol}/refresh`),
  job: (id: string) => get<Job>(`/api/jobs/${id}`),
};
