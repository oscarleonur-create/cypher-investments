import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { ResearchReport } from "@/lib/types";
import type { QuotesState } from "@/lib/useQuotes";
import { useJob } from "@/lib/useJob";
import { cn, fmtNum, fmtPct, fmtUsd } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Stat } from "@/components/common";
import { AgentPanel } from "@/components/AgentPanel";
import {
  BayesianPricingPanel,
  CatalystsPanel,
  DeepResearchPanel,
  EcosystemPanel,
  FairPricePanel,
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
  ValuationPanel,
} from "@/components/panels";

export default function Ticker({ quotes }: { quotes: QuotesState }) {
  const { symbol = "" } = useParams();
  const sym = symbol.toUpperCase();

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

  return (
    <div className="space-y-5">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <Link to="/" className="inline-flex items-center gap-1 text-sm text-muted hover:text-text">
            <ArrowLeft className="h-4 w-4" /> Portfolio
          </Link>
          <div className="mt-1 flex items-baseline gap-3">
            <h1 className="text-2xl font-bold">{sym}</h1>
            {q?.mid ? (
              <span className="text-xl tnum">{fmtNum(q.mid)}</span>
            ) : null}
          </div>
          {data?.business_model && (
            <p className="mt-1 max-w-2xl text-sm text-muted">{data.business_model}</p>
          )}
        </div>
        <Button onClick={rebuild} disabled={job.running} variant="outline" size="sm">
          <RefreshCw className={cn("h-4 w-4", job.running && "animate-spin")} />
          Rebuild research
        </Button>
      </div>

      {job.job && job.job.status !== "done" && (
        <div className="text-xs text-muted">
          {job.job.status === "error"
            ? `Rebuild failed: ${job.job.error}`
            : `Rebuilding… ${job.job.message}`}
        </div>
      )}

      {/* Market data strip */}
      {md && (
        <Card className="p-4">
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Float" value={fmtUsd(md.float_shares)} />
            <Stat label="Short %" value={fmtPct(md.short_interest_pct)} />
            <Stat label="Avg vol 10d" value={fmtNum(md.avg_volume_10d, 0)} />
            <Stat
              label="Days to cover"
              value={md.days_to_cover ? fmtNum(md.days_to_cover) : "—"}
            />
          </div>
        </Card>
      )}

      {/* Live price chart + fundamentals overlay; renders nothing until a
          report is cached for this ticker (the endpoint 404s without one). */}
      <PriceChartPanel symbol={sym} quotes={quotes} />

      {/* Interactive, tool-using research agent — always available on a ticker
          page (it can work even before a full report has been built). */}
      <AgentPanel symbol={sym} />

      {isLoading ? (
        <Card className="p-6 text-sm text-muted">Loading research…</Card>
      ) : error ? (
        <Card className="p-6">
          <div className="text-sm text-muted">
            {(error as Error).message}
          </div>
          <Button onClick={rebuild} disabled={job.running} className="mt-3" size="sm">
            <RefreshCw className={cn("h-4 w-4", job.running && "animate-spin")} />
            Build research now
          </Button>
        </Card>
      ) : data ? (
        <div className="space-y-4">
          {/* Headline white-paper brief — full width */}
          <DeepResearchPanel r={data} />
          {/* On-demand BUY/SELL/INCREASE/DECREASE/HOLD recommendation, checks
              the real position + cached research. Decision support only. */}
          <RecommendationPanel symbol={sym} />
          {/* Portfolio-only Bayesian what-if; renders nothing for non-holdings. */}
          <BayesianPricingPanel symbol={sym} />
          <div className="grid gap-4 lg:grid-cols-2">
          <div className="space-y-4">
            <FairPricePanel r={data} />
            <MemoPanel r={data} />
            <ThesisPanel r={data} />
            <ValuationPanel r={data} />
            <FinancialsPanel r={data} />
            <RatiosPanel r={data} />
            <TranscriptsPanel r={data} />
          </div>
          <div className="space-y-4">
            <KpiPanel r={data} />
            <CatalystsPanel r={data} />
            <MoatPanel r={data} />
            <EcosystemPanel r={data} />
            <OptionsFlowPanel r={data} />
            <SentimentPanel r={data} />
            <FilingsPanel r={data} />
          </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
