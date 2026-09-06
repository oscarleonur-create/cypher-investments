import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, Play, RefreshCw, ShieldCheck } from "lucide-react";
import { api } from "@/lib/api";
import type { EventTier, ReconcileReport } from "@/lib/types";
import { cn, fmtEt, fmtNum, fmtUsd } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Section, Stat } from "@/components/common";
import { EventRow, ExposureBars, FindingRow, SourceRow } from "@/components/daemon";

const TIERS: { key: EventTier | "ALL"; label: string; hint: string }[] = [
  { key: "ALL", label: "All", hint: "Everything in the stream" },
  { key: "A", label: "Act", hint: "Actionable with a deadline" },
  { key: "B", label: "Digest", hint: "Worth reading today" },
  { key: "C", label: "Context", hint: "Logged only" },
];

export default function Signals() {
  const qc = useQueryClient();
  const [tier, setTier] = useState<EventTier | "ALL">("ALL");

  // Store-backed: instant, safe to poll.
  const status = useQuery({
    queryKey: ["daemon", "status"],
    queryFn: api.daemonStatus,
    refetchInterval: 60_000,
  });
  const events = useQuery({
    queryKey: ["daemon", "events", tier],
    queryFn: () => api.daemonEvents({ limit: 60, tier: tier === "ALL" ? undefined : tier }),
    refetchInterval: 60_000,
  });
  const exposure = useQuery({ queryKey: ["daemon", "exposure"], queryFn: api.daemonExposure });
  const sources = useQuery({
    queryKey: ["daemon", "sources"],
    queryFn: () => api.daemonSources(undefined, 25),
  });

  // Network-bound: user-triggered only.
  const reconcile = useMutation<ReconcileReport>({ mutationFn: api.daemonReconcile });
  const runJob = useMutation({
    mutationFn: (job: string) => api.daemonRun(job),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["daemon"] }),
  });

  const counts = status.data?.event_counts ?? {};
  const staleJobs = (status.data?.jobs ?? []).filter((j) => j.error_count > 0);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-4">
          <h1 className="text-lg font-semibold">Signals</h1>
          {status.data && (
            <span className="text-xs text-muted">
              {fmtEt(status.data.now, { withYear: true })} ·{" "}
              {status.data.market_open ? (
                <span className="text-pos">market open</span>
              ) : (
                "market closed"
              )}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            onClick={() => runJob.mutate("brief")}
            disabled={runJob.isPending}
          >
            <Play className="mr-1.5 h-3.5 w-3.5" />
            {runJob.isPending ? "Running…" : "Run brief"}
          </Button>
          <Button onClick={() => reconcile.mutate()} disabled={reconcile.isPending}>
            <ShieldCheck className="mr-1.5 h-3.5 w-3.5" />
            {reconcile.isPending ? "Checking…" : "Check data"}
          </Button>
        </div>
      </div>

      {runJob.data && (
        <div className="rounded border border-border bg-panel px-3 py-2 text-xs text-muted">
          <span className={runJob.data.ok ? "text-pos" : "text-neg"}>
            {runJob.data.ok ? "ok" : "failed"}
          </span>{" "}
          {runJob.data.job} ({runJob.data.duration_ms}ms) — {runJob.data.detail}
        </div>
      )}

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat label="Act" value={counts.A ?? 0} sub="interrupts" />
        <Stat label="Digest" value={counts.B ?? 0} sub="worth reading" />
        <Stat label="Context" value={counts.C ?? 0} sub="logged only" />
        <Stat
          label="Jobs failing"
          value={staleJobs.length}
          sub={staleJobs.length ? staleJobs.map((j) => j.name).join(", ") : "all healthy"}
          className={staleJobs.length ? "text-neg" : undefined}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-[3fr_2fr]">
        <div className="space-y-4">
          <Section
            title="Event stream"
            empty={!events.data?.events.length}
            right={
              <div className="flex gap-1">
                {TIERS.map((t) => (
                  <button
                    key={t.key}
                    title={t.hint}
                    onClick={() => setTier(t.key)}
                    className={cn(
                      "rounded px-2 py-0.5 text-xs",
                      tier === t.key ? "bg-panel-2 text-text" : "text-muted hover:text-text"
                    )}
                  >
                    {t.label}
                  </button>
                ))}
              </div>
            }
          >
            <div>
              {events.data?.events.map((e) => (
                <EventRow key={e.id} event={e} />
              ))}
            </div>
          </Section>

          {reconcile.data && (
            <Section
              title="Data quality"
              right={
                <span
                  className={cn("text-xs", reconcile.data.ok ? "text-pos" : "text-neg")}
                >
                  {reconcile.data.summary}
                </span>
              }
            >
              <div>
                {[...reconcile.data.findings]
                  .sort((a, b) =>
                    a.severity === b.severity ? 0 : a.severity === "FAIL" ? -1 : 1
                  )
                  .map((f, i) => (
                    <FindingRow key={`${f.check}-${f.symbol}-${i}`} finding={f} />
                  ))}
              </div>
            </Section>
          )}
          {reconcile.isError && (
            <div className="flex items-center gap-2 rounded border border-neg/40 bg-neg/10 px-3 py-2 text-xs text-neg">
              <AlertTriangle className="h-3.5 w-3.5" />
              {(reconcile.error as Error).message}
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Section
            title="Factor exposure"
            empty={!exposure.data?.exposure}
            right={
              <button
                title="Re-estimate factor loadings (weekly job)"
                onClick={() => runJob.mutate("macro_refresh")}
                className="text-muted hover:text-text"
              >
                <RefreshCw className="h-3.5 w-3.5" />
              </button>
            }
          >
            {exposure.data?.exposure && (
              <div className="space-y-3">
                <div className="text-xs text-muted">
                  {exposure.data.exposure.asof} · net liq{" "}
                  {fmtUsd(exposure.data.exposure.net_liq)} ·{" "}
                  {(exposure.data.exposure.covered_weight * 100).toFixed(0)}% of notional
                  covered
                </div>
                <ExposureBars exposure={exposure.data.exposure} />
                {exposure.data.exposure.uncovered.length > 0 && (
                  <div className="text-xs text-warn">
                    No estimate for {exposure.data.exposure.uncovered.join(", ")} — reported
                    as uncovered, not as zero exposure.
                  </div>
                )}
                <p className="text-xs text-muted">{exposure.data.caveat}</p>
              </div>
            )}
          </Section>

          <Section title="Recently read" empty={!sources.data?.items.length}>
            <div>
              {sources.data?.items.map((item) => (
                <SourceRow key={`${item.accession ?? item.url}`} item={item} />
              ))}
            </div>
          </Section>

          <Section title="Jobs" empty={!status.data?.jobs.length}>
            <div className="space-y-1">
              {status.data?.jobs.map((j) => (
                <div key={j.name} className="flex items-baseline justify-between gap-2 text-xs">
                  <span className={j.error_count ? "text-neg" : ""}>{j.name}</span>
                  <span className="text-muted tnum">
                    {j.last_ok_at ? fmtEt(j.last_ok_at) : "never"}
                    {j.error_count > 0 && ` · ${fmtNum(j.error_count, 0)} err`}
                  </span>
                </div>
              ))}
            </div>
          </Section>
        </div>
      </div>
    </div>
  );
}
