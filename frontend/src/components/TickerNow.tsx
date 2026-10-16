import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { NewsJudgment } from "@/lib/types";
import { fmtEt } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Section } from "@/components/common";
import { DIRECTION, Judgment } from "@/components/DepthActions";
import { Lead, SourceBadge } from "@/components/daemon";

const SHOWN = 5;
const RANK: Record<string, number> = { HIGH: 0, MEDIUM: 1, LOW: 2 };

/** What matters, newest and most material first; the rest is one click away. Pure. */
export function material(js: NewsJudgment[]): { top: NewsJudgment[]; rest: NewsJudgment[] } {
  const about = js.filter((j) => j.about_company && j.materiality !== "LOW");
  const sorted = [...about].sort(
    (a, b) =>
      (RANK[a.materiality] ?? 3) - (RANK[b.materiality] ?? 3) ||
      b.published_at.localeCompare(a.published_at)
  );
  const top = sorted.slice(0, SHOWN);
  const keys = new Set(top.map((j) => j.key));
  return { top, rest: js.filter((j) => !keys.has(j.key)) };
}

function Toggle({ open, onClick, label }: { open: boolean; onClick: () => void; label: string }) {
  return (
    <button onClick={onClick} className="text-xs text-muted hover:text-text">
      {open ? "▾" : "▸"} {label}
    </button>
  );
}

/** The freshest of a name, first: the news agent's week, the material news, the filings. */
export function TickerNow({ symbol }: { symbol: string }) {
  const news = useQuery({
    queryKey: ["depth", "news", symbol],
    queryFn: () => api.newsJudgments(symbol),
  });
  const daemon = useQuery({
    queryKey: ["daemon", "symbol", symbol],
    queryFn: () => api.daemonSymbol(symbol),
  });
  const [allNews, setAllNews] = useState(false);
  const [allFeed, setAllFeed] = useState(false);

  const summary = news.data?.summary;
  const { top, rest } = material(news.data?.judgments ?? []);
  const timeline = daemon.data?.timeline ?? [];
  const filings = timeline.filter((r) => r.tier === "PRIMARY").slice(0, SHOWN);
  const empty = !summary && !top.length && !filings.length && !timeline.length;
  if (news.isLoading || daemon.isLoading) return null;

  return (
    <Section title="This week" empty={empty}>
      <div className="space-y-4">
        {summary && (
          <div className="rounded-md border border-border bg-panel-2/40 p-3 text-sm">
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <Badge variant={DIRECTION[summary.net] ?? "muted"}>{summary.net.toLowerCase()}</Badge>
              <span className="font-medium">{summary.headline}</span>
              <span className="ml-auto text-xs text-muted">news agent · {summary.day}</span>
            </div>
            <div>{summary.text}</div>
            {summary.watch.length > 0 && (
              <div className="mt-1 text-xs text-muted">Watch: {summary.watch.join("; ")}</div>
            )}
          </div>
        )}

        {top.length > 0 && (
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">
              Material news
            </div>
            <ul className="space-y-2">
              {top.map((j) => (
                <Judgment key={j.key} j={j} />
              ))}
            </ul>
          </div>
        )}
        {rest.length > 0 && (
          <div className="space-y-2">
            <Toggle
              open={allNews}
              onClick={() => setAllNews((o) => !o)}
              label={`All judged news (${rest.length} more)`}
            />
            {allNews && (
              <ul className="space-y-2">
                {rest.map((j) => (
                  <Judgment key={j.key} j={j} />
                ))}
              </ul>
            )}
          </div>
        )}

        {filings.length > 0 && (
          <div>
            <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">Filings</div>
            {filings.map((row) => (
              <div key={row.url + row.published_at} className="flex items-start gap-3 py-1.5">
                <SourceBadge tier={row.tier} match={row.match} />
                <div className="min-w-0 flex-1">
                  <a href={row.url} target="_blank" rel="noreferrer" className="text-sm hover:text-accent">
                    {row.title}
                  </a>
                  <div className="text-xs text-muted tnum">
                    {fmtEt(row.published_at)}
                    {row.doc_type && ` · ${row.doc_type}`}
                    {row.item_codes.length > 0 && ` [${row.item_codes.join(", ")}]`}
                  </div>
                  {row.lead && <Lead text={row.lead} />}
                </div>
              </div>
            ))}
          </div>
        )}
        {timeline.length > 0 && (
          <div className="space-y-2">
            <Toggle
              open={allFeed}
              onClick={() => setAllFeed((o) => !o)}
              label={`Everything archived (${timeline.length} filings & news, unjudged included)`}
            />
            {allFeed && (
              <div>
                {timeline.map((row) => (
                  <div key={row.url + row.published_at} className="flex items-start gap-3 py-1">
                    <SourceBadge tier={row.tier} match={row.match} />
                    <div className="min-w-0 flex-1 text-sm">
                      <a href={row.url} target="_blank" rel="noreferrer" className="hover:text-accent">
                        {row.title}
                      </a>
                      <span className="ml-2 text-xs text-muted tnum">{fmtEt(row.published_at)}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </Section>
  );
}
