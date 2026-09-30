import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { Pick, PickCell, PickPlan, PicksResponse, ReplayWindow } from "@/lib/types";
import { useJob } from "@/lib/useJob";
import { cn, fmtEt, fmtPct, fmtUsd, pnlColor } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Section, Th } from "@/components/common";
import { PositionCheck } from "@/components/PositionCheck";
import { DepthActions, DepthBulk, useDepthStatus } from "@/components/DepthActions";
import type { DepthStatus } from "@/lib/types";

/** What each family is, in the words the page uses. */
const FAMILY: Record<string, { label: string; hint: string; variant: "pos" | "accent" | "warn" }> = {
  F: { label: "F · revenue accelerating", hint: "fundamentals, from the 10-Q", variant: "pos" },
  P: { label: "P · price strength", hint: "momentum, 52-week high, breakout", variant: "accent" },
  I: { label: "I · insiders buying", hint: "open-market Form 4 purchases", variant: "warn" },
};

const HORIZON: Record<string, string> = { d20: "20 sessions", d60: "60 sessions", d120: "120 sessions" };

/** A percentage for an interval end: a rounding "-0.0%" reads as a sign, so it is 0.0%. */
function end(x: number): string {
  return Math.abs(x) < 0.0005 ? "0.0%" : fmtPct(x, { sign: true });
}

function ci(c: [number, number] | null | undefined): string {
  return c ? `${end(c[0])} … ${end(c[1])}` : "—";
}

function cell(w: ReplayWindow, group: string, horizon: string): PickCell | undefined {
  return w.cells.find((c) => c.group === group && c.horizon === horizon);
}

/** The honest headline: what the list rests on, before any name. Every window, not the best. */
function Evidence({ data }: { data: PicksResponse }) {
  const live = Object.values(data.live_records ?? {}).reduce((a, b) => a + b, 0);
  const lines = data.track_record
    .map((w) => ({ w, c: cell(w, "2+", "d20") }))
    .filter((x): x is { w: ReplayWindow; c: PickCell } => !!x.c && x.c.n > 0);
  return (
    <div className="flex gap-3 rounded-md border border-warn/30 bg-warn/10 px-4 py-3 text-sm">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warn" />
      <div className="space-y-1">
        <div className="font-medium text-text">Not proven. Ordered by evidence, not by expected return.</div>
        <div className="text-muted">
          Each name has two or more measured families agreeing on the last close. Over the next 20
          sessions, past names in this group beat same-day peers of their industry and size by:
        </div>
        {lines.length > 0 ? (
          <ul className="space-y-0.5 text-muted">
            {lines.map(({ w, c }) => (
              <li key={w.run_id}>
                {w.years}-year replay ({w.from} → {w.to}):{" "}
                <span className={pnlColor(c.excess)}>{fmtPct(c.excess, { sign: true })}</span>, 95%
                interval {ci(c.ci)} — <span className="text-text">{c.verdict}</span>
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-muted">No replay on file yet.</div>
        )}
        <div className="text-muted">
          Names that failed since are missing from the history, which flatters these. Live record:{" "}
          {live} records so far — too few for any verdict.
        </div>
      </div>
    </div>
  );
}

const STAGE: Record<string, { label: string; variant: "pos" | "warn" | "muted" }> = {
  fresh: { label: "measured entry", variant: "pos" },
  late: { label: "late: not measured", variant: "warn" },
  past: { label: "window over", variant: "muted" },
};

/** Enter at what price, because of what, expecting what: every number from the pick or the replay. */
function EntryPlan({ plan, price }: { plan: PickPlan; price: number }) {
  if (!plan.ok) {
    return <div className="text-xs text-muted">No entry plan: {plan.gap}</div>;
  }
  const stage = STAGE[plan.stage ?? ""] ?? STAGE.past;
  const m = plan.measured;
  const fresh = plan.stage === "fresh";
  return (
    <div className="space-y-2 rounded-md border border-border bg-panel-2/40 p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-medium">Entry plan</span>
        <Badge variant={stage.variant}>{stage.label}</Badge>
        {plan.group && <span className="text-xs text-muted">record of group {plan.group}</span>}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">{fresh ? "Enter near" : "If entered now"}</div>
          <div className="font-semibold tnum">{fmtUsd(plan.entry)}</div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">Target</div>
          <div className="font-semibold tnum text-pos">
            {plan.target != null ? fmtUsd(plan.target) : "—"}
            {plan.target_pct != null && (
              <span className="ml-1 text-xs font-normal">({fmtPct(plan.target_pct, { sign: true })})</span>
            )}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">Stop</div>
          <div className="font-semibold tnum text-neg">
            {plan.stop != null ? fmtUsd(plan.stop) : "—"}
            {plan.stop_pct != null && (
              <span className="ml-1 text-xs font-normal">({fmtPct(-plan.stop_pct, { sign: true })})</span>
            )}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">Size</div>
          <div className="font-semibold tnum">
            {plan.size?.shares != null ? `${plan.size.shares} sh · ${fmtUsd(plan.size.notional)}` : "—"}
          </div>
        </div>
        <div>
          <div className="text-xs uppercase tracking-wide text-muted">Else out on</div>
          <div className="font-semibold tnum">{plan.exit_on ?? plan.review_on ?? "—"}</div>
        </div>
      </div>

      {plan.because && (
        <div>
          <span className="font-medium">Because </span>
          {plan.because}.
        </div>
      )}

      {plan.expects && plan.expects.length > 0 && (
        <div>
          <span className="font-medium">For it to stand: </span>
          {plan.expects.join("; ")}.
        </div>
      )}

      {plan.expect && plan.expect.length > 0 && (
        <div className="space-y-1">
          <div className="font-medium">
            What we expect — only what was measured, over {plan.horizon_sessions} sessions:
          </div>
          <ul className="space-y-1 text-muted">
            {plan.expect.map((e) => (
              <li key={e.years}>
                {e.text}
                {e.price_mean != null && m?.price != null && (
                  <span className="text-text">
                    {" "}
                    For this name from {fmtUsd(m.price)} on {m.day}: the average is{" "}
                    <span className="tnum">{fmtUsd(e.price_mean)}</span>
                    {e.price_tail != null && (
                      <>
                        , the typical worst drop <span className="tnum">{fmtUsd(e.price_tail)}</span>
                      </>
                    )}
                    {!fresh && (
                      <>
                        ; it is at <span className="tnum">{fmtUsd(price)}</span>
                      </>
                    )}
                    .
                  </span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {plan.replayed && plan.replayed.length > 0 && (
        <div className="space-y-1">
          <div className="font-medium">The plan as shown, replayed — +5% target, −2.5% stop, this entry delay:</div>
          <ul className="space-y-1 text-muted">
            {plan.replayed.map((r) => (
              <li key={r.years}>
                {r.text}{" "}
                <span className={cn("text-text", pnlColor(r.ret))}>
                  Per trade: {fmtPct(r.ret, { sign: true })} ({r.raw.verdict}).
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {plan.timing && <div className={cn(fresh ? "text-muted" : "text-warn")}>{plan.timing}</div>}

      <div className="text-xs text-muted">
        {plan.stop_basis && <>Stop: {plan.stop_basis}. </>}
        {plan.size?.note && <>Size: {plan.size.note}. </>}
        The replays are past names, not a forecast for this one: check the zone, results date and news with Evaluate
        position before acting.
      </div>
    </div>
  );
}

function PickCard({ p, depth }: { p: Pick; depth?: DepthStatus }) {
  return (
    <Card>
      <CardContent className="space-y-3 pt-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm text-muted tnum">#{p.rank}</span>
              <Link to={`/ticker/${p.symbol}`} className="text-lg font-semibold hover:text-accent">
                {p.symbol}
              </Link>
              {p.held && <Badge variant="muted">held</Badge>}
              {p.families.map((f) => (
                <Badge key={f} variant={FAMILY[f].variant} title={FAMILY[f].hint}>
                  {FAMILY[f].label}
                </Badge>
              ))}
            </div>
            <div className="truncate text-xs text-muted">
              {p.name ?? "—"} · {p.sector ?? "sector n/a"}
            </div>
          </div>
          <div className="flex gap-5 text-right">
            <div>
              <div className="text-xs uppercase tracking-wide text-muted">Close</div>
              <div className="font-semibold tnum">{fmtUsd(p.price)}</div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-muted">Since {p.since}</div>
              <div className={cn("font-semibold tnum", pnlColor(p.move_since))}>
                {fmtPct(p.move_since, { sign: true })}
              </div>
            </div>
            <div>
              <div className="text-xs uppercase tracking-wide text-muted">20 sessions</div>
              <div className={cn("font-semibold tnum", pnlColor(p.return_20d))}>
                {fmtPct(p.return_20d, { sign: true })}
              </div>
            </div>
            {p.since_pick && (
              <div>
                <div className="text-xs uppercase tracking-wide text-muted">
                  Since pick → {p.since_pick.day}
                </div>
                <div className={cn("font-semibold tnum", pnlColor(p.since_pick.move))}>
                  {fmtPct(p.since_pick.move, { sign: true })}
                </div>
              </div>
            )}
          </div>
        </div>

        {p.plan && <EntryPlan plan={p.plan} price={p.price} />}

        <ul className="space-y-1.5">
          {p.reasons.map((r, i) => (
            <li key={i} className="flex gap-2 text-sm">
              <span className="w-4 shrink-0 font-mono text-xs text-muted">{r.family}</span>
              <span>
                {r.text} <span className="text-xs text-muted">— {r.source}</span>
              </span>
            </li>
          ))}
        </ul>

        {p.invalidates.length > 0 && (
          <div className="text-xs text-muted">
            <span className="font-medium text-text">Would undo it:</span> {p.invalidates.join("; ")}.
          </div>
        )}

        <div className="space-y-3 border-t border-border/60 pt-3">
          <DepthActions symbol={p.symbol} status={depth} />
          <PositionCheck symbol={p.symbol} />
        </div>
      </CardContent>
    </Card>
  );
}

function TrackRecord({ data }: { data: PicksResponse }) {
  const [pick, setPick] = useState(0);
  const w = data.track_record[Math.min(pick, data.track_record.length - 1)];
  const groups = ["2+", "F+P", "F", "I"];
  const rows = w
    ? groups.flatMap((g) =>
        ["d20", "d60", "d120"].map((h) => cell(w, g, h)).filter((c): c is PickCell => !!c && c.n > 0)
      )
    : [];
  const windows = (
    <div className="flex gap-1">
      {data.track_record.map((r, i) => (
        <button
          key={r.run_id}
          onClick={() => setPick(i)}
          className={cn(
            "rounded-md px-2 py-1 text-xs",
            r === w ? "bg-panel-2 text-text" : "text-muted hover:text-text"
          )}
        >
          {r.years}-year window
        </button>
      ))}
    </div>
  );
  return (
    <Section
      title="Track record — what each group did next, in the replay"
      empty={!rows.length}
      right={data.track_record.length > 1 ? windows : undefined}
    >
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <Th>Group</Th>
              <Th>Horizon</Th>
              <Th className="text-right">Records</Th>
              <Th className="text-right">vs peers</Th>
              <Th className="text-right">95% interval</Th>
              <Th className="text-right">vs peers with the same trend</Th>
              <Th className="text-right">Typical worst drop</Th>
              <Th>Verdict</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((c) => (
              <tr key={`${c.group}-${c.horizon}`} className="border-b border-border/50 last:border-0">
                <td className="px-2 py-1.5 font-medium">{c.group}</td>
                <td className="px-2 py-1.5 text-muted">{HORIZON[c.horizon] ?? c.horizon}</td>
                <td className="px-2 py-1.5 text-right tnum">{c.n.toLocaleString()}</td>
                <td className={cn("px-2 py-1.5 text-right tnum", pnlColor(c.excess))}>
                  {fmtPct(c.excess, { sign: true })}
                </td>
                <td className="px-2 py-1.5 text-right tnum text-muted">{ci(c.ci)}</td>
                <td className={cn("px-2 py-1.5 text-right tnum", pnlColor(c.excess_trend))}>
                  {fmtPct(c.excess_trend, { sign: true })}
                </td>
                <td className="px-2 py-1.5 text-right tnum text-neg">{fmtPct(c.tail, { sign: true })}</td>
                <td className="px-2 py-1.5">
                  <Badge variant={c.verdict === "EDGE" ? "pos" : c.verdict === "NEGATIVE" ? "neg" : "muted"}>
                    {c.verdict}
                  </Badge>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {w && (
        <p className="mt-3 text-xs text-muted">
          {w.from} → {w.to}. {w.cells_tested} cells were tested in that replay: at 95%, about{" "}
          {(w.cells_tested * 0.025).toFixed(1)} would clear zero by chance alone.
          Peers are ten names drawn the same day from the same industry and size; "same trend" peers
          also moved alike over the prior 60 sessions.
        </p>
      )}
    </Section>
  );
}

/** Today first; earlier days are the history, each pick with how it has moved since. */
function DayPicker({
  data,
  selected,
  onPick,
}: {
  data: PicksResponse;
  selected: string | null;
  onPick: (day: string | null) => void;
}) {
  if (data.days.length < 2) return null;
  return (
    <div className="flex flex-wrap items-center gap-1 text-xs">
      <span className="mr-1 text-muted">History:</span>
      {data.days.map((d, i) => {
        const active = (selected ?? data.days[0].day) === d.day;
        return (
          <button
            key={d.day}
            onClick={() => onPick(i === 0 ? null : d.day)}
            className={cn(
              "rounded-md px-2 py-1",
              active ? "bg-panel-2 text-text" : "text-muted hover:text-text"
            )}
          >
            {i === 0 ? "Latest" : d.day}
            {i === 0 && <span className="ml-1 text-muted">({d.day})</span>}
            {d.provisional && <span className="ml-1 text-warn">·prov</span>}
          </button>
        );
      })}
    </div>
  );
}

export default function Picks() {
  const qc = useQueryClient();
  const [day, setDay] = useState<string | null>(null);
  const q = useQuery({
    queryKey: ["breadth", "picks", day],
    queryFn: () => api.breadthPicks(day ?? undefined),
    refetchInterval: 300_000,
  });
  const data = q.data;
  const job = useJob(() => {
    setDay(null);
    qc.invalidateQueries({ queryKey: ["breadth", "picks"] });
  });
  const refresh = async () => {
    const { job_id } = await api.refreshPicks();
    job.start(job_id);
  };
  const latest = data?.days[0]?.day;
  const isLatest = !!data?.day && data.day === latest;
  const symbols = (data?.picks ?? []).map((p) => p.symbol);
  const depth = useDepthStatus(symbols);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex flex-wrap items-baseline gap-3">
          <h1 className="text-lg font-semibold">Picks</h1>
          {data?.day && (
            <span className="text-xs text-muted">
              {isLatest ? "latest" : "history"} · {data.day}
              {data.provisional ? (
                <span className="text-warn"> · provisional, live prices</span>
              ) : (
                <> · close</>
              )}
              {data.built_at && <> · built {fmtEt(data.built_at, { withYear: true })}</>}
              {data.rules && <> · rules {data.rules}</>}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {job.job && (
            <span className={cn("text-xs", job.job.status === "error" ? "text-neg" : "text-muted")}>
              {job.job.status === "error" ? job.job.error || job.job.message : job.job.message}
            </span>
          )}
          <Button variant="outline" onClick={refresh} disabled={job.running}>
            <RefreshCw className={cn("mr-1.5 h-3.5 w-3.5", job.running && "animate-spin")} />
            {job.running ? "Building…" : "Refresh now"}
          </Button>
        </div>
      </div>

      {data && <DayPicker data={data} selected={day} onPick={setDay} />}
      {data && <DepthBulk symbols={symbols} what="picks" />}

      {data?.provisional && isLatest && (
        <div className="text-xs text-muted">
          Provisional: built on today's prices so far. F and I rest on filings through last
          night (EDGAR publishes its indexes at the end of the day), and a breakout counts the
          session's volume so far. The 20:30 ET run makes today's list final.
        </div>
      )}

      {q.isLoading && <div className="text-sm text-muted">Loading…</div>}
      {q.error && <div className="text-sm text-neg">{String((q.error as Error).message)}</div>}

      {data && (
        <>
          <Evidence data={data} />
          {data.picks.length === 0 ? (
            <Card>
              <CardContent className="pt-4 text-sm text-muted">
                No name has two families agreeing on the last close, or the picks have not been
                built yet (they are built nightly at 20:30 ET, or with{" "}
                <code>advisor breadth picks --build</code>).
              </CardContent>
            </Card>
          ) : (
            <div className="space-y-3">
              {data.picks.map((p) => (
                <PickCard key={p.symbol} p={p} depth={depth.data?.status[p.symbol]} />
              ))}
            </div>
          )}
          <TrackRecord data={data} />
          {data.caveats.length > 0 && (
            <Section title="What the measurement cannot see">
              <ul className="list-disc space-y-1 pl-5 text-sm text-muted">
                {data.caveats.map((c) => (
                  <li key={c}>{c}</li>
                ))}
              </ul>
            </Section>
          )}
        </>
      )}
    </div>
  );
}
