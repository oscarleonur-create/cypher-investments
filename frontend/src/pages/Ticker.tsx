import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { ResearchReport } from "@/lib/types";
import type { QuotesState } from "@/lib/useQuotes";
import { useJob } from "@/lib/useJob";
import { cn, fmtEt, fmtNum, fmtPct, fmtShares } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Stat } from "@/components/common";
import { AgentPanel } from "@/components/AgentPanel";
import { TickerDaemonPanels } from "@/components/tickerDaemon";
import { TickerNow } from "@/components/TickerNow";
import { ReadingCard } from "@/components/ReadingCard";
import { OverviewCard } from "@/components/OverviewCard";
import { PriceRangeCard } from "@/components/PriceRangeCard";
import { AnglesPanel } from "@/components/AnglesPanel";
import { StoryCard } from "@/components/StoryCard";
import { ClaimsPanel } from "@/components/ClaimsPanel";
import {
  CatalystsPanel,
  DeepResearchPanel,
  EcosystemPanel,
  FilingsPanel,
  FinancialsPanel,
  KpiPanel,
  MemoPanel,
  MoatPanel,
  OptionsFlowPanel,
  PriceChartPanel,
  RatiosPanel,
  RecommendationPanel,
  SentimentPanel,
  ThesisPanel,
  TranscriptsPanel,
} from "@/components/panels";

const TABS = [
  { id: "research", label: "Research" },
  { id: "financials", label: "Financials" },
  { id: "rules", label: "Thesis & rules" },
  { id: "ask", label: "Ask" },
] as const;
type Tab = (typeof TABS)[number]["id"];

/** Days since ``iso``, for saying how old a read is. */
function age(iso?: string | null): number | null {
  if (!iso) return null;
  const t = new Date(iso).getTime();
  return Number.isNaN(t) ? null : Math.floor((Date.now() - t) / 86_400_000);
}

function AgeLine({ label, iso }: { label: string; iso?: string | null }) {
  const d = age(iso);
  if (d == null) return <span className="text-xs text-muted">{label}: never built</span>;
  return (
    <span className={cn("text-xs", d > 7 ? "text-warn" : "text-muted")}>
      {label} {fmtEt(iso)} · {d === 0 ? "today" : `${d} day${d === 1 ? "" : "s"} old`}
    </span>
  );
}

/** The deep research with its date, and a refresh of just that part. */
function DeepResearchBlock({ sym, r, onDone }: { sym: string; r: ResearchReport; onDone: () => void }) {
  const job = useJob(onDone);
  const run = async () => job.start((await api.refreshDeepResearch(sym)).job_id);
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-3">
        <AgeLine label="Deep research from" iso={r.deep_research?.fetched_at} />
        <Button size="sm" variant="outline" onClick={run} disabled={job.running}>
          <RefreshCw className={cn("mr-1.5 h-3.5 w-3.5", job.running && "animate-spin")} />
          {job.running ? "Refreshing…" : "Refresh deep research"}
        </Button>
        {job.job && job.job.status !== "done" && (
          <span className={cn("text-xs", job.job.status === "error" ? "text-neg" : "text-muted")}>
            {job.job.status === "error" ? job.job.error : job.job.message}
          </span>
        )}
      </div>
      <DeepResearchPanel r={r} />
    </div>
  );
}

export default function Ticker({ quotes }: { quotes: QuotesState }) {
  const { symbol = "" } = useParams();
  const sym = symbol.toUpperCase();
  const [tab, setTab] = useState<Tab>("research");

  const { data, isLoading, error, refetch } = useQuery<ResearchReport>({
    queryKey: ["research", sym],
    queryFn: () => api.research(sym),
    retry: false,
  });

  const job = useJob(() => refetch());
  const rebuild = async () => {
    const { job_id } = await api.refreshResearch(sym);
    job.start(job_id);
  };

  const q = quotes.quotes[sym];
  const md = data?.market_data;

  const rebuildButton = (
    <Button onClick={rebuild} disabled={job.running} variant="outline" size="sm">
      <RefreshCw className={cn("mr-1.5 h-3.5 w-3.5", job.running && "animate-spin")} />
      {job.running ? "Rebuilding…" : data ? "Rebuild full report" : "Build research now"}
    </Button>
  );

  return (
    <div className="space-y-5">
      {/* Header */}
      <div>
        <Link to="/" className="inline-flex items-center gap-1 text-sm text-muted hover:text-text">
          <ArrowLeft className="h-4 w-4" /> Portfolio
        </Link>
        <div className="mt-1 flex items-baseline gap-3">
          <h1 className="text-2xl font-bold">{sym}</h1>
          {q?.mid ? <span className="text-xl tnum">{fmtNum(q.mid)}</span> : null}
        </div>
        {data?.business_model && (
          <p className="mt-1 max-w-2xl text-sm text-muted">{data.business_model}</p>
        )}
      </div>

      {/* Freshest first: where the company stands, then this week's news and filings. */}
      <OverviewCard symbol={sym} />
      <TickerNow symbol={sym} />

      <PriceChartPanel symbol={sym} quotes={quotes} />

      {/* What the price asks and what the facts imply: both from the daemon's
          store, current on every read (not the cached report). */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <PriceRangeCard symbol={sym} />
        <ReadingCard symbol={sym} />
      </div>

      {/* Everything else, one tab at a time. */}
      <div className="flex flex-wrap gap-1 border-b border-border">
        {TABS.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={cn(
              "-mb-px border-b-2 px-3 py-2 text-sm font-medium",
              tab === t.id
                ? "border-accent text-text"
                : "border-transparent text-muted hover:text-text"
            )}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "research" && (
        <div className="space-y-4">
          <div className="flex flex-wrap items-center gap-3">
            <AgeLine label="Report built" iso={data?.created_at} />
            {rebuildButton}
            {job.job && job.job.status !== "done" && (
              <span className={cn("text-xs", job.job.status === "error" ? "text-neg" : "text-muted")}>
                {job.job.status === "error"
                  ? `Rebuild failed: ${job.job.error}`
                  : `Rebuilding (minutes)… ${job.job.message}`}
              </span>
            )}
          </div>
          {isLoading ? (
            <Card className="p-6 text-sm text-muted">Loading research…</Card>
          ) : error ? (
            <Card className="p-6 text-sm text-muted">{(error as Error).message}</Card>
          ) : data ? (
            <>
              <DeepResearchBlock sym={sym} r={data} onDone={() => refetch()} />
              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div className="space-y-4">
                  <MemoPanel r={data} />
                  <CatalystsPanel r={data} />
                  <MoatPanel r={data} />
                  <TranscriptsPanel r={data} />
                </div>
                <div className="space-y-4">
                  <ThesisPanel r={data} />
                  <EcosystemPanel r={data} />
                  <KpiPanel r={data} />
                  <SentimentPanel r={data} />
                  <OptionsFlowPanel r={data} />
                  <FilingsPanel r={data} />
                </div>
              </div>
            </>
          ) : null}
        </div>
      )}

      {tab === "financials" && (
        <div className="space-y-4">
          {md && (
            <Card className="p-4">
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
                {/* A share count, not dollars. */}
                <Stat label="Float" value={fmtShares(md.float_shares)} />
                {/* Stored as a percentage already (12.07), not a fraction. */}
                <Stat label="Short %" value={fmtPct(md.short_interest_pct, { scale: false })} />
                <Stat label="Avg vol 10d" value={fmtNum(md.avg_volume_10d, 0)} />
                <Stat
                  label="Days to cover"
                  value={md.days_to_cover ? fmtNum(md.days_to_cover) : "—"}
                />
              </div>
            </Card>
          )}
          {data ? (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <FinancialsPanel r={data} />
              <RatiosPanel r={data} />
            </div>
          ) : (
            <Card className="p-6 text-sm text-muted">No report yet. {rebuildButton}</Card>
          )}
          <TickerDaemonPanels symbol={sym} parts={["factors"]} />
        </div>
      )}

      {tab === "rules" && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div className="space-y-4">
            <ClaimsPanel symbol={sym} />
            <AnglesPanel symbol={sym} />
          </div>
          <div className="space-y-4">
            <StoryCard symbol={sym} />
            <TickerDaemonPanels symbol={sym} parts={["events"]} />
          </div>
        </div>
      )}

      {tab === "ask" && (
        <div className="space-y-4">
          <AgentPanel symbol={sym} />
          <RecommendationPanel symbol={sym} />
        </div>
      )}
    </div>
  );
}
