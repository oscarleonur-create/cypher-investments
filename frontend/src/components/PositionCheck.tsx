import { useQuery } from "@tanstack/react-query";
import { Scale } from "lucide-react";
import { api } from "@/lib/api";
import type { EntryProposal } from "@/lib/types";
import { useJob } from "@/lib/useJob";
import { cn, fmtEt, fmtNum, fmtPct, fmtUsd } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

/** What each entry action means, in the engine's own terms — never "buy". */
const ACTION: Record<string, { tone: "pos" | "warn" | "muted" | "neg" | "accent"; means: string }> = {
  ENTER: { tone: "pos", means: "a leg qualifies under your entry rules — sizing and stop below" },
  ADD: { tone: "pos", means: "held; the position leg qualifies within the book limit" },
  WAIT: { tone: "warn", means: "a leg would qualify, but something must be read first" },
  IN_ZONE: { tone: "accent", means: "the price is inside its zone; nothing triggered today" },
  NONE: { tone: "muted", means: "outside its zone and no setup: no position proposed" },
  CANNOT_SAY: { tone: "muted", means: "no price, or neither a zone nor a setup to judge by" },
  HOLD: { tone: "muted", means: "held; no exit call" },
  TRIM: { tone: "warn", means: "held; above the book limit" },
  REVIEW: { tone: "warn", means: "held; something needs your answer" },
  EXIT: { tone: "neg", means: "held; past its stop or a filing ends the case" },
};

function Evaluation({ p }: { p: EntryProposal }) {
  const a = ACTION[p.action] ?? { tone: "muted" as const, means: "" };
  return (
    <div className="space-y-3 rounded-md border border-border bg-panel-2/40 p-3">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Badge variant={a.tone}>{p.action}</Badge>
        <span className="text-muted">{a.means}</span>
        <span className="ml-auto text-xs text-muted">
          session {p.session} · {fmtEt(p.built_at)}
        </span>
      </div>

      {p.reasons.length > 0 && (
        <ul className="space-y-1 text-sm">
          {p.reasons.map((r, i) => (
            <li key={i}>
              {r.text} <span className="text-xs text-muted">— {r.source}</span>
            </li>
          ))}
        </ul>
      )}

      {p.legs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border text-xs text-muted">
                <th className="px-2 py-1 text-left font-medium">Leg</th>
                <th className="px-2 py-1 text-right font-medium">Entry</th>
                <th className="px-2 py-1 text-right font-medium">Stop</th>
                <th className="px-2 py-1 text-right font-medium">Risk</th>
                <th className="px-2 py-1 text-right font-medium">Shares</th>
                <th className="px-2 py-1 text-right font-medium">Notional</th>
              </tr>
            </thead>
            <tbody>
              {p.legs.map((l) => (
                <tr key={l.horizon} className="border-b border-border/50 last:border-0 align-top">
                  <td className="px-2 py-1.5">
                    <div className="font-medium">{l.horizon}</div>
                    <div className="text-xs text-muted">{l.stop_basis}</div>
                  </td>
                  <td className="px-2 py-1.5 text-right tnum">{fmtUsd(l.entry)}</td>
                  <td className="px-2 py-1.5 text-right tnum text-neg">{fmtUsd(l.stop)}</td>
                  <td className="px-2 py-1.5 text-right tnum">{fmtPct(l.risk_pct)} of net liq</td>
                  <td className="px-2 py-1.5 text-right tnum">{fmtNum(l.shares, 0)}</td>
                  <td className="px-2 py-1.5 text-right tnum">{fmtUsd(l.notional)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {p.reading.length > 0 && (
        <div className="text-sm">
          <div className="mb-1 text-xs uppercase tracking-wide text-muted">
            Model reading{p.stance ? ` · ${p.stance}` : ""}
          </div>
          <ul className="list-disc space-y-0.5 pl-5">
            {p.reading.map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
        </div>
      )}

      {p.blockers.length > 0 && (
        <div className="text-sm text-warn">Blocked: {p.blockers.join("; ")}</div>
      )}
      {p.gaps.length > 0 && (
        <div className="text-xs text-muted">What it could not see: {p.gaps.join("; ")}.</div>
      )}
      <div className="text-xs text-muted">
        Recorded in the proposals ledger and judged by what the price does next. No
        order is placed.
      </div>
    </div>
  );
}

/** From a pick to a position decision: the daemon's entry engine, run for this name. */
export function PositionCheck({ symbol }: { symbol: string }) {
  const latest = useQuery({
    queryKey: ["breadth", "evaluation", symbol],
    queryFn: () => api.positionEvaluation(symbol),
  });
  const job = useJob(() => latest.refetch());
  const run = async () => {
    const { job_id } = await api.evaluatePosition(symbol);
    job.start(job_id);
  };
  const p = latest.data?.proposal ?? null;

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="outline" onClick={run} disabled={job.running}>
          <Scale className="mr-1.5 h-3.5 w-3.5" />
          {job.running ? "Evaluating…" : p ? "Re-evaluate position" : "Evaluate position"}
        </Button>
        {job.job && (
          <span
            className={cn(
              "text-xs",
              job.job.status === "error" ? "text-neg" : "text-muted"
            )}
          >
            {job.job.status === "error" ? job.job.error || job.job.message : job.job.message}
          </span>
        )}
      </div>
      {p && <Evaluation p={p} />}
    </div>
  );
}
