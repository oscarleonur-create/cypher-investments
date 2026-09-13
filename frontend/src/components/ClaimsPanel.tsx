import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Plus } from "lucide-react";
import { api } from "@/lib/api";
import type { ClaimKind } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Section } from "@/components/common";

/** Ready-made claims for the shapes the event stream can actually check.
 *  Typing a claim has to be faster than not typing one — the evidence being
 *  that this repo's prose thesis template sat unfilled on both theses for
 *  months while doing nothing for anyone. */
const PRESETS: {
  label: string;
  kind: ClaimKind;
  text: string;
  event_kinds: string[];
  field?: string;
  comparator?: string;
  threshold?: number;
}[] = [
  {
    label: "Dilution above 5%",
    kind: "INVALIDATION",
    text: "An equity raise above 5% of market cap breaks the story",
    event_kinds: ["FILING_DILUTION"],
    field: "dilution_pct",
    comparator: "ABOVE",
    threshold: 0.05,
  },
  {
    label: "Restatement or auditor change",
    kind: "INVALIDATION",
    text: "A restatement or auditor change ends this immediately",
    event_kinds: ["FILING_RESTATEMENT", "FILING_AUDITOR_CHANGE"],
  },
  {
    label: "Late filing",
    kind: "INVALIDATION",
    text: "A late periodic filing means the numbers cannot be trusted",
    event_kinds: ["FILING_LATE_FILING"],
  },
  {
    label: "Unexplained 2σ move",
    kind: "KPI",
    text: "A move macro cannot explain means something changed I do not know about",
    event_kinds: ["RESIDUAL_DIVERGENCE"],
    field: "residual_z",
    comparator: "ABOVE",
    threshold: 2,
  },
];

const KIND_TONE: Record<ClaimKind, string> = {
  INVALIDATION: "text-neg",
  DRIVER: "text-pos",
  KPI: "text-accent",
  MACRO_DRIVER: "text-accent",
  CATALYST: "text-warn",
  RISK: "text-muted",
};

export function ClaimsPanel({ symbol }: { symbol: string }) {
  const qc = useQueryClient();
  const [text, setText] = useState("");
  const [kind, setKind] = useState<ClaimKind>("INVALIDATION");

  const { data } = useQuery({
    queryKey: ["daemon", "thesis", symbol],
    queryFn: () => api.daemonThesis(symbol),
    enabled: Boolean(symbol),
  });

  const add = useMutation({
    mutationFn: (body: Parameters<typeof api.daemonAddClaim>[1]) =>
      api.daemonAddClaim(symbol, body),
    onSuccess: () => {
      setText("");
      qc.invalidateQueries({ queryKey: ["daemon", "thesis", symbol] });
      qc.invalidateQueries({ queryKey: ["daemon", "story", symbol] });
    },
  });

  const thesis = data?.thesis;
  const claims = thesis?.claims ?? [];

  return (
    <Section
      title="Thesis claims"
      right={
        claims.length > 0 ? (
          <span className="text-xs text-muted">
            {claims.filter((c) => c.monitored).length} of {claims.length} machine-checked
          </span>
        ) : undefined
      }
    >
      <div className="space-y-3">
        {thesis && !thesis.substantive && thesis.prose_note && (
          <div className="rounded border border-warn/40 bg-warn/10 px-2 py-1.5 text-xs text-warn">
            {thesis.prose_note}
          </div>
        )}

        {claims.length > 0 && (
          <div className="space-y-1.5">
            {claims.map((claim) => (
              <div key={claim.id} className="border-b border-border/40 pb-1.5 last:border-0">
                <div className="flex items-start gap-2">
                  <span
                    className={cn(
                      "shrink-0 text-[10px] font-semibold uppercase",
                      KIND_TONE[claim.kind]
                    )}
                  >
                    {claim.kind.replace("_", " ")}
                  </span>
                  <span className="min-w-0 text-sm">{claim.text}</span>
                </div>
                <div
                  className={cn(
                    "mt-0.5 pl-[76px] text-xs",
                    claim.monitored ? "text-muted" : "text-warn"
                  )}
                >
                  {claim.monitored
                    ? `checked by: ${claim.trigger_description}`
                    : "recorded, but no event will ever check this"}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="space-y-2">
          <div className="flex flex-wrap gap-1">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                onClick={() => add.mutate(preset)}
                disabled={add.isPending}
                className="rounded border border-border px-2 py-0.5 text-xs text-muted hover:border-accent hover:text-accent"
                title={preset.text}
              >
                + {preset.label}
              </button>
            ))}
          </div>

          <div className="flex gap-2">
            <select
              value={kind}
              onChange={(e) => setKind(e.target.value as ClaimKind)}
              className="rounded border border-border bg-panel-2 px-2 py-1 text-xs"
            >
              {(["INVALIDATION", "DRIVER", "KPI", "RISK", "CATALYST"] as ClaimKind[]).map(
                (k) => (
                  <option key={k} value={k}>
                    {k.toLowerCase()}
                  </option>
                )
              )}
            </select>
            <input
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && text.trim()) add.mutate({ kind, text: text.trim() });
              }}
              placeholder="What would tell you this is wrong?"
              className="min-w-0 flex-1 rounded border border-border bg-panel-2 px-2 py-1 text-sm"
            />
            <Button
              variant="outline"
              onClick={() => text.trim() && add.mutate({ kind, text: text.trim() })}
              disabled={!text.trim() || add.isPending}
            >
              <Plus className="h-3.5 w-3.5" />
            </Button>
          </div>
          <p className="text-xs text-muted">
            A claim typed here is recorded. One added from a preset is also{" "}
            <em>checked against every matching filing</em> — that is the difference
            between a note and a monitor.
          </p>
        </div>
      </div>
    </Section>
  );
}
