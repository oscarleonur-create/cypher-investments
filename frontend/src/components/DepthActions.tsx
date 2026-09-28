import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ArrowRight, BookOpen, ChevronDown, ChevronRight, Newspaper } from "lucide-react";
import { api } from "@/lib/api";
import type { DepthStatus, NewsJudgment } from "@/lib/types";
import { useJob } from "@/lib/useJob";
import { cn, fmtEt } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

const DIRECTION: Record<string, "pos" | "neg" | "warn" | "muted"> = {
  POSITIVE: "pos",
  NEGATIVE: "neg",
  MIXED: "warn",
  NEUTRAL: "muted",
};

/** Depth for many names: one query for the whole page, shared by every row. */
export function useDepthStatus(symbols: string[]) {
  return useQuery({
    queryKey: ["depth", "status", symbols.join(",")],
    queryFn: () => api.depthStatus(symbols),
    enabled: symbols.length > 0,
    refetchInterval: 60_000,
  });
}

function when(iso: string | null): string {
  return iso ? fmtEt(iso) : "never";
}

function JobLine({ job }: { job: ReturnType<typeof useJob>["job"] }) {
  if (!job) return null;
  return (
    <span className={cn("text-xs", job.status === "error" ? "text-neg" : "text-muted")}>
      {job.status === "error" ? job.error || job.message : job.message}
    </span>
  );
}

function Judgment({ j }: { j: NewsJudgment }) {
  return (
    <li className="space-y-1 border-b border-border/50 pb-2 last:border-0">
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="text-muted tnum">{j.published_at.slice(0, 10)}</span>
        {j.about_company ? (
          <>
            <Badge variant={DIRECTION[j.direction] ?? "muted"}>{j.direction.toLowerCase()}</Badge>
            <Badge variant={j.materiality === "HIGH" ? "neg" : j.materiality === "MEDIUM" ? "warn" : "muted"}>
              {j.materiality.toLowerCase()}
            </Badge>
            <Badge variant="muted">{j.novelty.toLowerCase().replace("_", " ")}</Badge>
          </>
        ) : (
          <Badge variant="muted">not about the company</Badge>
        )}
        <span className="text-muted">{j.provider}</span>
      </div>
      <div className="text-sm font-medium">
        {j.url ? (
          <a href={j.url} target="_blank" rel="noreferrer" className="hover:text-accent">
            {j.title}
          </a>
        ) : (
          j.title
        )}
      </div>
      {j.what && <div className="text-sm">{j.what}</div>}
      {j.why && <div className="text-sm text-muted">Why it matters: {j.why}</div>}
      {j.watch && <div className="text-xs text-muted">Watch: {j.watch}</div>}
      {j.quote && <div className="border-l-2 border-border pl-2 text-xs italic text-muted">“{j.quote}”</div>}
      {j.problems.length > 0 && (
        <div className="text-xs text-warn">Removed by the number check: {j.problems.join("; ")}</div>
      )}
    </li>
  );
}

function NewsPanel({ symbol }: { symbol: string }) {
  const q = useQuery({ queryKey: ["depth", "news", symbol], queryFn: () => api.newsJudgments(symbol) });
  if (q.isLoading) return <div className="text-sm text-muted">Loading news…</div>;
  const d = q.data;
  if (!d || (!d.judgments.length && !d.summary)) {
    return (
      <div className="text-sm text-muted">
        No judged news in the last two weeks. Run the news agent to pull and judge it.
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {d.summary && (
        <div className="rounded-md border border-border bg-panel-2/40 p-3 text-sm">
          <div className="mb-1 flex items-center gap-2">
            <Badge variant={DIRECTION[d.summary.net] ?? "muted"}>{d.summary.net.toLowerCase()}</Badge>
            <span className="font-medium">{d.summary.headline}</span>
            <span className="ml-auto text-xs text-muted">week of {d.summary.day}</span>
          </div>
          <div>{d.summary.text}</div>
          {d.summary.watch.length > 0 && (
            <div className="mt-1 text-xs text-muted">Watch: {d.summary.watch.join("; ")}</div>
          )}
        </div>
      )}
      <ul className="space-y-2">
        {d.judgments.map((j) => (
          <Judgment key={j.key} j={j} />
        ))}
      </ul>
      <div className="text-xs text-muted">
        Context and measurement only: a judgment changes no action. Each must quote its item; a
        number not in the item or its context voids the judgment.
      </div>
    </div>
  );
}

/** News agent and deep research for one name, with what it already has. */
export function DepthActions({ symbol, status }: { symbol: string; status?: DepthStatus }) {
  const qc = useQueryClient();
  const [open, setOpen] = useState(false);
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["depth"] });
    qc.invalidateQueries({ queryKey: ["research", symbol] });
  };
  const news = useJob(() => {
    refresh();
    setOpen(true);
  });
  const research = useJob(refresh);
  const runNews = async () => news.start((await api.runNewsAgent([symbol])).job_id);
  const runResearch = async () => research.start((await api.runDeepResearch([symbol])).job_id);

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" size="sm" onClick={runNews} disabled={news.running}>
          <Newspaper className="mr-1.5 h-3.5 w-3.5" />
          {news.running ? "Reading news…" : "Run news agent"}
        </Button>
        <Button variant="outline" size="sm" onClick={runResearch} disabled={research.running}>
          <BookOpen className="mr-1.5 h-3.5 w-3.5" />
          {research.running ? "Researching…" : status?.report_at ? "Rebuild deep research" : "Run deep research"}
        </Button>
        <button
          onClick={() => setOpen((o) => !o)}
          className="inline-flex items-center text-sm text-muted hover:text-text"
        >
          {open ? <ChevronDown className="mr-0.5 h-3.5 w-3.5" /> : <ChevronRight className="mr-0.5 h-3.5 w-3.5" />}
          News{status ? ` (${status.news_judged}${status.news_material ? `, ${status.news_material} material` : ""})` : ""}
        </button>
        <Link to={`/ticker/${symbol}`} className="inline-flex items-center text-sm text-muted hover:text-accent">
          {status?.report_at ? `Report ${when(status.report_at)}` : "Full research"}
          <ArrowRight className="ml-1 h-3.5 w-3.5" />
        </Link>
        <JobLine job={news.job} />
        <JobLine job={research.job} />
      </div>
      {open && <NewsPanel symbol={symbol} />}
    </div>
  );
}

/** Run either action for every name on the page, after saying what it costs. */
export function DepthBulk({ symbols, what }: { symbols: string[]; what: string }) {
  const qc = useQueryClient();
  const [confirm, setConfirm] = useState<"news" | "research" | null>(null);
  const job = useJob(() => qc.invalidateQueries({ queryKey: ["depth"] }));
  const n = symbols.length;
  if (!n) return null;
  const go = async () => {
    const start = confirm === "news" ? api.runNewsAgent : api.runDeepResearch;
    setConfirm(null);
    job.start((await start(symbols)).job_id);
  };
  return (
    <div className="flex flex-wrap items-center gap-2 text-sm">
      <span className="text-muted">For all {n} {what}:</span>
      {confirm ? (
        <>
          <span className="text-warn">
            {confirm === "news"
              ? `${n} Tavily searches, about 2 min per name.`
              : `Several searches and model calls per name; minutes each.`}
          </span>
          <Button size="sm" onClick={go}>
            Run for {n}
          </Button>
          <Button size="sm" variant="outline" onClick={() => setConfirm(null)}>
            Cancel
          </Button>
        </>
      ) : (
        <>
          <Button size="sm" variant="outline" onClick={() => setConfirm("news")} disabled={job.running}>
            <Newspaper className="mr-1.5 h-3.5 w-3.5" />
            News agent
          </Button>
          <Button size="sm" variant="outline" onClick={() => setConfirm("research")} disabled={job.running}>
            <BookOpen className="mr-1.5 h-3.5 w-3.5" />
            Deep research
          </Button>
        </>
      )}
      <JobLine job={job.job} />
    </div>
  );
}
