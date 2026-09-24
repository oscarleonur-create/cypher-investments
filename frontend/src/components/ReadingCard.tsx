import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type {
  ReadingFact,
  ReadingStance,
  Scorecard,
  ScorecardRow,
  TickerReading,
} from "@/lib/types";
import { cn, fmtEt } from "@/lib/utils";
import { Card } from "@/components/ui/card";

const STANCE: Record<ReadingStance, { label: string; tone: string }> = {
  CONSTRUCTIVE: { label: "Constructive", tone: "bg-pos/15 text-pos" },
  NEUTRAL: { label: "Neutral", tone: "bg-panel-2 text-muted" },
  CAUTIOUS: { label: "Cautious", tone: "bg-warn/15 text-warn" },
  AT_RISK: { label: "At risk", tone: "bg-neg/15 text-neg" },
};

const MARK: Record<string, { glyph: string; tone: string; title: string }> = {
  OK: { glyph: "✓", tone: "text-pos", title: "within the line" },
  TRIPPED: { glyph: "✗", tone: "text-neg", title: "past the line" },
  UNCHECKABLE: { glyph: "?", tone: "text-muted", title: "nothing can check this" },
};

function ScoreRow({ row }: { row: ScorecardRow }) {
  const mark = row.status ? MARK[row.status] : null;
  return (
    <div
      className="grid grid-cols-[16px_minmax(0,11rem)_minmax(0,7rem)_1fr] items-baseline gap-2 py-0.5"
      title={row.claim ? `Your claim: ${row.claim}` : row.source}
    >
      <span className={cn("text-xs", mark?.tone)} aria-label={mark?.title}>
        {mark?.glyph ?? ""}
      </span>
      <span className="truncate text-xs text-muted">{row.label}</span>
      <span className="text-sm font-medium tnum">{row.value}</span>
      <span className="min-w-0 text-xs text-muted">
        {row.detail}
        {row.source && <span className="opacity-60"> · {row.source}</span>}
      </span>
    </div>
  );
}

/** The numbers first. Read from stored filings, valuation, consensus and
 *  claims — no model writes any of this. */
function ScorecardTable({ card }: { card: Scorecard }) {
  return (
    <div className="mb-3 space-y-3">
      {card.expectations.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">Expectations</div>
          {card.expectations.map((r) => (
            <ScoreRow key={r.label} row={r} />
          ))}
        </div>
      )}
      {card.thresholds.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">
            Your thresholds
          </div>
          {card.thresholds.map((r) => (
            <ScoreRow key={r.label} row={r} />
          ))}
        </div>
      )}
    </div>
  );
}

/** A citation chip: hover shows the fact the sentence rests on. */
function Cite({ fact }: { fact: ReadingFact | undefined; id: string }) {
  if (!fact) return null;
  return (
    <span
      title={`${fact.date ? `${fact.date} · ` : ""}${fact.text}`}
      className="ml-1 cursor-help rounded bg-panel-2 px-1 align-[1px] text-[10px] text-muted tnum"
    >
      {fact.id}
    </span>
  );
}

/** What the recent facts on this name imply together. Model-written, and
 *  shown only when every number in it traces to a fact it cites; otherwise
 *  the facts are shown and the prose is not. */
export function ReadingCard({ symbol }: { symbol: string }) {
  const client = useQueryClient();
  const key = ["daemon", "reading", symbol];
  const { data, isLoading } = useQuery({
    queryKey: key,
    queryFn: () => api.daemonReading(symbol),
    enabled: Boolean(symbol),
    staleTime: 5 * 60_000,
  });
  const [refreshing, setRefreshing] = useState(false);
  const [showFacts, setShowFacts] = useState(false);

  if (isLoading || !data) return null;
  const card = data.scorecard;
  const hasNumbers = Boolean(card && (card.expectations.length || card.thresholds.length));
  if (data.status === "NO_FACTS" && !hasNumbers) return null;

  const refresh = async () => {
    setRefreshing(true);
    try {
      client.setQueryData<TickerReading>(key, await api.daemonReading(symbol, true));
    } finally {
      setRefreshing(false);
    }
  };
  const byId = Object.fromEntries(data.facts.map((f) => [f.id, f]));
  const stance = data.stance ? STANCE[data.stance] : null;

  return (
    <Card className="p-4">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-muted">Reading</h3>
        {stance && (
          <span className={cn("rounded px-2 py-0.5 text-xs font-medium", stance.tone)}>
            {stance.label}
          </span>
        )}
        <span className="text-xs text-muted">
          {data.facts.length} facts · last {data.window_days}d · {fmtEt(data.generated_at)}
          {data.model && ` · ${data.model}`}
        </span>
        <button
          onClick={refresh}
          disabled={refreshing}
          title="Write the reading again from the current facts"
          className="ml-auto text-muted hover:text-text disabled:opacity-50"
        >
          <RefreshCw className={cn("h-3.5 w-3.5", refreshing && "animate-spin")} />
        </button>
      </div>

      {card && hasNumbers && <ScorecardTable card={card} />}

      {data.status === "OK" && (
        <div className="space-y-2 border-t border-border/50 pt-3 text-sm leading-relaxed">
          {data.sentences.map((s, i) => (
            <p key={i}>
              {s.text}
              {s.facts.map((id) => (
                <Cite key={id} id={id} fact={byId[id]} />
              ))}
            </p>
          ))}
        </div>
      )}
      {data.status === "REJECTED" && (
        <div className="text-sm text-warn">
          No reading: the model's draft cited numbers its facts do not contain, twice. The
          facts are below.
        </div>
      )}
      {data.status === "UNAVAILABLE" && (
        <div className="text-sm text-muted">
          No reading: {data.problems[0] ?? "the model is unavailable"}.
        </div>
      )}

      {data.status === "OK" && (
        <button
          onClick={() => setShowFacts((v) => !v)}
          className="mt-3 text-xs text-accent hover:underline"
        >
          {showFacts ? "Hide facts" : "Show the facts it was written from"}
        </button>
      )}
      {(showFacts || data.status === "REJECTED" || data.status === "UNAVAILABLE") && (
        <ol className="mt-2 space-y-1 text-xs text-muted">
          {data.facts.map((f) => (
            <li key={f.id} className="flex gap-2">
              <span className="w-7 shrink-0 tnum">{f.id}</span>
              <span className="min-w-0">
                {f.date && <span className="tnum">{f.date} · </span>}
                {f.url ? (
                  <a href={f.url} target="_blank" rel="noreferrer" className="hover:text-text">
                    {f.text}
                  </a>
                ) : (
                  f.text
                )}
              </span>
            </li>
          ))}
        </ol>
      )}
    </Card>
  );
}
