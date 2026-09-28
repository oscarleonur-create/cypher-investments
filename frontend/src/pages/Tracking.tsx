import { Fragment, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronRight, Clock } from "lucide-react";
import { api } from "@/lib/api";
import type { SystemStatus, TrackPoint, TrackRisk, TrackRow, TrackTrade } from "@/lib/types";
import type { QuotesState } from "@/lib/useQuotes";
import { cn, fmtEt, fmtMoney, fmtNum, fmtPct, pnlColor } from "@/lib/utils";
import { Stat, Th } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";

// What each call asks of the user, as a colour: act, sell, answer, wait, nothing.
const ACTION_STYLE: Record<string, { variant: string; dot: string }> = {
  ENTER: { variant: "accent", dot: "bg-accent" },
  ADD: { variant: "accent", dot: "bg-accent" },
  EXIT: { variant: "neg", dot: "bg-neg" },
  TRIM: { variant: "warn", dot: "bg-warn" },
  REVIEW: { variant: "warn", dot: "bg-warn" },
  WAIT: { variant: "warn", dot: "bg-warn/50" },
  HOLD: { variant: "default", dot: "bg-muted/60" },
  IN_ZONE: { variant: "pos", dot: "bg-pos/60" },
  NONE: { variant: "muted", dot: "bg-border" },
  CANNOT_SAY: { variant: "muted", dot: "bg-border" },
};

function ActionBadge({ action }: { action?: string | null }) {
  if (!action) return <span className="text-xs text-muted">no call yet</span>;
  const s = ACTION_STYLE[action] || ACTION_STYLE.NONE;
  return <Badge variant={s.variant as any}>{action.replace(/_/g, " ")}</Badge>;
}

export default function Tracking({ quotes }: { quotes: QuotesState }) {
  const status = useQuery<SystemStatus>({
    queryKey: ["tracking-status"],
    queryFn: api.trackingStatus,
    refetchInterval: 60000,
  });
  const board = useQuery({
    queryKey: ["tracking-board"],
    queryFn: api.trackingBoard,
    refetchInterval: 60000,
  });

  const rows = useMemo(
    () => (board.data?.rows || []).map((r) => withLivePrice(r, quotes)),
    [board.data, quotes]
  );

  const totals = useMemo(() => {
    let atRisk = 0;
    let atRiskPct = 0;
    let open = 0;
    let realized = 0;
    let past = 0;
    for (const r of rows) {
      if (r.held) {
        atRisk += r.risk?.at_risk || 0;
        atRiskPct += r.risk?.at_risk_pct || 0;
        open += r.unrealized || 0;
        if (r.risk?.past_stop) past += 1;
      }
      realized += r.realized;
    }
    return { atRisk, atRiskPct, open, realized, past, held: rows.filter((r) => r.held).length };
  }, [rows]);

  return (
    <div className="space-y-5">
      <SystemCard q={status} />

      <Card className="p-4">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-5">
          <Stat label="Held" value={totals.held} sub={`${rows.length - totals.held} watched`} />
          <Stat
            label="At risk to stops"
            value={fmtMoney(totals.atRisk)}
            sub={`${fmtPct(totals.atRiskPct)} of net liq`}
          />
          <Stat
            label="Past their stop"
            value={<span className={totals.past ? "text-neg" : ""}>{totals.past}</span>}
            sub="an EXIT is on the table"
          />
          <Stat
            label="Open P&L"
            value={<span className={pnlColor(totals.open)}>{fmtMoney(totals.open, true)}</span>}
          />
          <Stat
            label="Realized (on file)"
            value={
              <span className={pnlColor(totals.realized)}>{fmtMoney(totals.realized, true)}</span>
            }
            sub="equity trades, all books"
          />
        </div>
      </Card>

      <Card className="overflow-hidden">
        {board.isLoading ? (
          <div className="p-6 text-sm text-muted">Loading the board…</div>
        ) : board.error ? (
          <div className="p-6 text-sm text-neg">{(board.error as Error).message}</div>
        ) : rows.length === 0 ? (
          <div className="p-6 text-sm text-muted">
            Nothing held and no proposals recorded yet. The daemon records proposals hourly in
            session from 09:45 ET.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="border-b border-border bg-panel-2/40">
                <tr>
                  <Th></Th>
                  <Th>Symbol</Th>
                  <Th>Call</Th>
                  <Th>Stop · cost · price · target</Th>
                  <Th className="text-right">To stop</Th>
                  <Th className="text-right">At risk</Th>
                  <Th className="text-right">Open P&L</Th>
                  <Th className="text-right">Realized</Th>
                  <Th>Last 30 sessions</Th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <BoardRow key={r.symbol} r={r} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

/** The live quote replaces the stored price, and every number that hangs on it. */
function withLivePrice(r: TrackRow, q: QuotesState): TrackRow {
  const live = q.quotes[r.symbol]?.mid || 0;
  if (!(live > 0)) return r;
  const out: TrackRow = { ...r, price: live };
  if (r.held && r.cost) {
    out.unrealized = (live - r.cost) * r.quantity;
    out.unrealized_pct = live / r.cost - 1;
  }
  if (r.risk && r.risk.stop) {
    const risk: TrackRisk = { ...r.risk, price: live };
    risk.to_stop = live / r.risk.stop - 1;
    risk.past_stop = live <= r.risk.stop;
    if (r.risk.basis === "held") {
      const perShare = Math.max(live - r.risk.stop, 0);
      const old = r.risk.at_risk || 0;
      risk.at_risk = perShare * r.risk.shares;
      if (r.risk.at_risk_pct != null && old > 0) {
        risk.at_risk_pct = (r.risk.at_risk_pct * (risk.at_risk || 0)) / old;
      }
    }
    out.risk = risk;
  }
  return out;
}

// ── The system: is anything old deciding? ─────────────────────────────────

function SystemCard({ q }: { q: ReturnType<typeof useQuery<SystemStatus>> }) {
  const [open, setOpen] = useState(false);
  const s = q.data;
  if (q.isLoading) return <Card className="p-4 text-sm text-muted">Checking the system…</Card>;
  if (q.error || !s)
    return (
      <Card className="p-4 text-sm text-neg">
        System status unavailable: {(q.error as Error)?.message}
      </Card>
    );
  const bad = s.jobs.filter((j) => j.state !== "ok" && j.state !== "idle");
  return (
    <Card className={cn("p-4", s.ok ? "" : "border-warn/40")}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="flex items-start gap-2">
          {s.ok ? (
            <CheckCircle2 className="mt-0.5 h-5 w-5 text-pos" />
          ) : (
            <AlertTriangle className="mt-0.5 h-5 w-5 text-warn" />
          )}
          <div>
            <div className="text-sm font-semibold">
              {s.ok ? "Everything deciding is current" : `${s.problems.length} thing(s) not current`}
            </div>
            <div className="text-xs text-muted">
              Checked {fmtEt(s.now)} · stale inputs block entries, never exits · rule changes
              expire unless a sweep renews them
            </div>
          </div>
        </div>
        <button
          onClick={() => setOpen(!open)}
          className="flex items-center gap-1 text-xs text-muted hover:text-text"
        >
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          details
        </button>
      </div>

      {!s.ok && (
        <ul className="mt-3 space-y-1 text-sm">
          {s.problems.map((p) => (
            <li key={p} className="flex gap-2">
              <span className="text-warn">•</span>
              <span>{p}</span>
            </li>
          ))}
        </ul>
      )}

      {open && (
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-3">
          <div>
            <div className="mb-1 text-xs uppercase tracking-wide text-muted">Code</div>
            <CodeLine label="main (last fetch)" rev={s.code.main} />
            <CodeLine label="daemon" rev={s.code.daemon} behind={s.code.daemon_behind} />
            <CodeLine label="this API" rev={s.code.api} behind={s.code.api_behind} />
          </div>
          <div>
            <div className="mb-1 text-xs uppercase tracking-wide text-muted">Rules</div>
            <div className="text-xs text-muted">
              code declares <span className="font-mono text-text">{s.rules.code_version}</span>
              {s.rules.proposals_version && (
                <>
                  {" "}
                  · proposals of {s.rules.proposals_session} ran{" "}
                  <span className="font-mono text-text">{s.rules.proposals_version}</span>
                </>
              )}
            </div>
            {s.rules.changes.length === 0 ? (
              <div className="mt-1 text-xs text-muted">No rule changes: the code's values run.</div>
            ) : (
              s.rules.changes.map((c) => <RuleLine key={c.id} c={c} />)
            )}
            {s.rules.expired_recently.map((c) => (
              <div key={c.id} className="mt-1 text-xs text-muted">
                <span className="font-mono">{c.param}</span> {c.value} expired → {c.previous}:{" "}
                {c.note}
              </div>
            ))}
          </div>
          <div>
            <div className="mb-1 text-xs uppercase tracking-wide text-muted">
              Jobs {bad.length ? `· ${bad.length} not ok` : "· all ok"}
            </div>
            <div className="flex flex-wrap gap-1">
              {s.jobs.map((j) => (
                <span
                  key={j.name}
                  title={`${j.schedule}\nlast ok ${fmtEt(j.last_ok_at)}\nowed by ${fmtEt(
                    j.due_since
                  )}${j.last_error ? `\n${j.last_error}` : ""}`}
                  className={cn(
                    "rounded border px-1.5 py-0.5 text-[11px]",
                    j.state === "ok" && "border-pos/30 text-pos",
                    j.state === "idle" && "border-border text-muted",
                    j.state === "late" && "border-warn/40 text-warn",
                    (j.state === "failing" || j.state === "never") && "border-neg/40 text-neg"
                  )}
                >
                  {j.name}
                </span>
              ))}
            </div>
            {Object.keys(s.stale_inputs).length > 0 && (
              <div className="mt-2 text-xs">
                <span className="text-muted">stale in the latest proposals: </span>
                {Object.entries(s.stale_inputs).map(([k, v]) => (
                  <span key={k} className="mr-2 text-warn" title={v.join(", ")}>
                    {k} ({v.length})
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}

function CodeLine({ label, rev, behind }: { label: string; rev: string | null; behind?: number | null }) {
  return (
    <div className="flex items-baseline justify-between gap-2 text-xs">
      <span className="text-muted">{label}</span>
      <span className="font-mono">
        {rev || "—"}
        {behind != null && (
          <span className={cn("ml-2", behind > 0 ? "text-warn" : "text-pos")}>
            {behind > 0 ? `${behind} behind` : "current"}
          </span>
        )}
      </span>
    </div>
  );
}

function RuleLine({ c }: { c: SystemStatus["rules"]["changes"][number] }) {
  const ttl = 35;
  const left = Math.max(0, Math.min(1, c.days_left / ttl));
  return (
    <div className="mt-1.5 text-xs">
      <div className="flex items-baseline justify-between gap-2">
        <span>
          <span className="font-mono">{c.param}</span> {c.previous} → {c.value}{" "}
          <span className="text-muted">({c.status.toLowerCase()})</span>
        </span>
        <span className={cn(c.days_left < 7 ? "text-warn" : "text-muted")}>
          <Clock className="mr-0.5 inline h-3 w-3" />
          {c.in_force || c.status !== "ACTIVE" ? `${Math.max(0, c.days_left).toFixed(0)}d` : "expired"}
        </span>
      </div>
      <div className="mt-0.5 h-1 rounded bg-border">
        <div
          className={cn("h-1 rounded", c.days_left < 7 ? "bg-warn" : "bg-accent")}
          style={{ width: `${left * 100}%` }}
        />
      </div>
    </div>
  );
}

// ── One name ──────────────────────────────────────────────────────────────

function BoardRow({ r }: { r: TrackRow }) {
  const [open, setOpen] = useState(false);
  const k = r.risk;
  const stale = r.latest?.stale || [];
  return (
    <Fragment>
      <tr
        className="cursor-pointer border-b border-border/40 hover:bg-panel-2/40"
        onClick={() => setOpen(!open)}
      >
        <td className="px-2 py-2 text-muted">
          {open ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </td>
        <td className="px-2 py-2">
          <div className="flex items-center gap-2">
            <Link
              to={`/ticker/${r.symbol}`}
              onClick={(e) => e.stopPropagation()}
              className="font-semibold hover:underline"
            >
              {r.symbol}
            </Link>
            <span className="text-[10px] uppercase text-muted">
              {r.held ? `held ${fmtPct(r.weight)}` : "watch"}
            </span>
          </div>
          <NewsTally s={r.news_summary} />
        </td>
        <td className="px-2 py-2">
          <div className="flex items-center gap-1.5">
            <ActionBadge action={r.latest?.action} />
            {stale.length > 0 && (
              <span title={`stale: ${stale.join(", ")}`}>
                <AlertTriangle className="h-3.5 w-3.5 text-warn" />
              </span>
            )}
          </div>
          {r.latest && <div className="mt-0.5 text-[11px] text-muted">{r.latest.session}</div>}
        </td>
        <td className="px-2 py-2">
          <RiskLadder k={k} />
        </td>
        <td
          className={cn(
            "px-2 py-2 text-right text-sm tnum",
            k?.past_stop ? "text-neg" : k?.to_stop != null && k.to_stop < 0.05 ? "text-warn" : ""
          )}
        >
          {k?.to_stop == null ? "—" : k.past_stop ? "past" : fmtPct(k.to_stop)}
        </td>
        <td className="px-2 py-2 text-right text-sm tnum">
          {k?.at_risk == null ? "—" : fmtMoney(k.at_risk)}
          {k?.at_risk_pct != null && (
            <div className="text-[11px] text-muted">
              {fmtPct(k.at_risk_pct)} {k.basis === "planned" ? "planned" : ""}
            </div>
          )}
        </td>
        <td className={cn("px-2 py-2 text-right text-sm tnum", pnlColor(r.unrealized))}>
          {r.held ? fmtMoney(r.unrealized, true) : "—"}
          {r.held && r.unrealized_pct != null && (
            <div className="text-[11px]">{fmtPct(r.unrealized_pct, { sign: true })}</div>
          )}
        </td>
        <td className={cn("px-2 py-2 text-right text-sm tnum", pnlColor(r.realized))}>
          {r.wins + r.losses ? fmtMoney(r.realized, true) : "—"}
          {r.wins + r.losses > 0 && (
            <div className="text-[11px] text-muted">
              {r.wins}W / {r.losses}L
            </div>
          )}
        </td>
        <td className="px-2 py-2">
          <Timeline points={r.timeline} />
        </td>
      </tr>
      {open && (
        <tr className="border-b border-border/40 bg-panel-2/20">
          <td></td>
          <td colSpan={8} className="px-2 py-3">
            <RowDetail r={r} />
          </td>
        </tr>
      )}
    </Fragment>
  );
}

/** Stop, cost (or planned entry), price and target on one scale. */
function RiskLadder({ k }: { k: TrackRisk | null }) {
  if (!k || k.stop == null) {
    return <span className="text-xs text-muted">{k?.stop_basis || "no stop to show"}</span>;
  }
  const marks = [k.stop, k.entry, k.price, k.target].filter(
    (v): v is number => v != null && v > 0
  );
  const lo = Math.min(...marks) * 0.97;
  const hi = Math.max(...marks) * 1.03;
  const W = 170;
  const x = (v: number) => ((v - lo) / (hi - lo || 1)) * W;
  const price = k.price;
  const up = price != null && price >= k.entry;
  return (
    <svg width={W} height={22} className="overflow-visible" role="img">
      <title>
        {`stop ${fmtNum(k.stop)} · ${k.basis === "held" ? "cost" : "entry"} ${fmtNum(k.entry)}` +
          (price != null ? ` · price ${fmtNum(price)}` : "") +
          (k.target != null ? ` · target ${fmtNum(k.target)}` : "") +
          `\n${k.stop_basis}`}
      </title>
      <rect x={0} y={9} width={x(k.stop)} height={4} className="fill-neg/40" rx={1} />
      <rect x={x(k.stop)} y={10} width={W - x(k.stop)} height={2} className="fill-border" />
      <line x1={x(k.stop)} x2={x(k.stop)} y1={4} y2={18} className="stroke-neg" strokeWidth={2} />
      <line
        x1={x(k.entry)}
        x2={x(k.entry)}
        y1={5}
        y2={17}
        className="stroke-muted"
        strokeWidth={2}
        strokeDasharray={k.basis === "planned" ? "2 2" : undefined}
      />
      {k.target != null && (
        <line x1={x(k.target)} x2={x(k.target)} y1={4} y2={18} className="stroke-pos" strokeWidth={2} />
      )}
      {price != null && (
        <circle
          cx={x(price)}
          cy={11}
          r={4.5}
          className={cn(k.past_stop ? "fill-neg" : up ? "fill-pos" : "fill-warn", "stroke-bg")}
          strokeWidth={1.5}
        />
      )}
    </svg>
  );
}

function Timeline({ points }: { points: TrackPoint[] }) {
  if (points.length === 0) return <span className="text-xs text-muted">—</span>;
  return (
    <div className="flex items-center gap-[2px]">
      {points.map((p) => (
        <span
          key={p.session}
          title={`${p.session}: ${p.action}${p.price ? ` at ${fmtNum(p.price)}` : ""}`}
          className={cn("h-3 w-1.5 rounded-sm", (ACTION_STYLE[p.action] || ACTION_STYLE.NONE).dot)}
        />
      ))}
    </div>
  );
}

function RowDetail({ r }: { r: TrackRow }) {
  const l = r.latest;
  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <div className="space-y-2 text-sm">
        {l ? (
          <>
            <div className="text-xs text-muted">
              Latest call {fmtEt(l.built_at)} · rules{" "}
              <span className="font-mono">{l.rules || "unstamped"}</span>
            </div>
            {l.blockers.length > 0 && (
              <div>
                <div className="text-xs uppercase tracking-wide text-muted">Blocked by</div>
                <ul className="list-disc pl-5">
                  {l.blockers.map((b) => (
                    <li key={b} className={b.includes("is stale") ? "text-warn" : ""}>
                      {b}
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {l.reasons.length > 0 && (
              <div>
                <div className="text-xs uppercase tracking-wide text-muted">Why</div>
                <ul className="list-disc pl-5">
                  {l.reasons.map((x) => (
                    <li key={x}>{x}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        ) : (
          <div className="text-xs text-muted">No proposal recorded for this name yet.</div>
        )}
        {r.risk && (
          <div className="text-xs text-muted">
            Stop {fmtNum(r.risk.stop)} — {r.risk.stop_basis}
            {r.held && r.cost != null && ` · ${fmtNum(r.quantity)} sh at ${fmtNum(r.cost)}`}
          </div>
        )}
      </div>
      <div>
        <div className="text-xs uppercase tracking-wide text-muted">Trades</div>
        {r.open_trades.length + r.closed_trades.length === 0 ? (
          <div className="text-xs text-muted">None on file.</div>
        ) : (
          <table className="mt-1 w-full text-xs">
            <thead>
              <tr className="text-muted">
                <th className="text-left font-normal">opened</th>
                <th className="text-left font-normal">closed</th>
                <th className="text-right font-normal">qty</th>
                <th className="text-right font-normal">in → out</th>
                <th className="text-right font-normal">P&L</th>
                <th className="text-left font-normal pl-2">book</th>
                <th className="text-left font-normal">call</th>
              </tr>
            </thead>
            <tbody>
              {[...r.open_trades, ...r.closed_trades].map((t) => (
                <TradeRow key={`${t.opened}-${t.closed}-${t.quantity}-${t.entry_price}`} t={t} />
              ))}
            </tbody>
          </table>
        )}
      </div>
      {(r.news_week || r.news.length > 0) && (
        <div className="space-y-3 border-t border-border/40 pt-3 lg:col-span-2">
          <NewsWeek w={r.news_week} />
          <NewsList news={r.news} />
        </div>
      )}
    </div>
  );
}

function TradeRow({ t }: { t: TrackTrade }) {
  const followed = t.call === "ENTER" || t.call === "ADD";
  return (
    <tr className="border-t border-border/30 tnum">
      <td>{t.opened}</td>
      <td>{t.closed || <span className="text-accent">open</span>}</td>
      <td className="text-right">{fmtNum(t.quantity)}</td>
      <td className="text-right">
        {fmtNum(t.entry_price)} → {t.exit_price != null ? fmtNum(t.exit_price) : "…"}
      </td>
      <td className={cn("text-right", pnlColor(t.pnl))}>
        {t.pnl != null ? fmtMoney(t.pnl, true) : "—"}
      </td>
      <td className="pl-2 text-muted">{t.book}</td>
      <td className={followed ? "text-accent" : "text-muted"}>
        {t.call ? (followed ? `followed ${t.call}` : `against ${t.call}`) : "no call"}
      </td>
    </tr>
  );
}

// ── The news agent: context only, measured before it may decide anything ───

const DIR_STYLE: Record<string, string> = {
  POSITIVE: "text-pos",
  NEGATIVE: "text-neg",
  MIXED: "text-warn",
  NEUTRAL: "text-muted",
};

/** Material calls of the last week: ▲ positive, ▼ negative, and thesis hits. */
function NewsTally({ s }: { s: TrackRow["news_summary"] }) {
  if (!s || !s.items) return null;
  const parts = [
    s.positive ? (
      <span key="p" className="text-pos">
        ▲{s.positive}
      </span>
    ) : null,
    s.negative ? (
      <span key="n" className="text-neg">
        ▼{s.negative}
      </span>
    ) : null,
    s.mixed ? (
      <span key="m" className="text-warn">
        ◆{s.mixed}
      </span>
    ) : null,
    s.against_thesis ? (
      <span
        key="a"
        className="text-neg"
        title="items the agent reads as evidence against your thesis"
      >
        ✕thesis {s.against_thesis}
      </span>
    ) : null,
  ].filter(Boolean);
  return (
    <div
      className="mt-0.5 flex items-center gap-1.5 text-[11px]"
      title={`news, last 7 days: ${s.items} item(s), ${s.about ?? 0} about the company; material calls shown`}
    >
      <span className="text-muted">news</span>
      {parts.length ? parts : <span className="text-muted">{s.items} low</span>}
    </div>
  );
}

const MATERIALITY = ["HIGH", "MEDIUM", "LOW"];

/** The agent's synthesis of the name's week: what changed, what is noise, what to watch. */
function NewsWeek({ w }: { w: TrackRow["news_week"] }) {
  if (!w) return null;
  return (
    <div className="rounded-md border border-border/60 bg-panel-2/40 p-3 text-sm">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="text-xs uppercase tracking-wide text-muted">The week in news</span>
        <span className={cn("text-xs font-medium", DIR_STYLE[w.net])}>{w.net.toLowerCase()}</span>
        <span className="text-xs text-muted">
          {w.day} · from {w.items} item(s) · context only, not advice
        </span>
      </div>
      <div className="mt-1 font-medium">{w.headline}</div>
      <p className="mt-1 text-muted">{w.text}</p>
      {w.thesis && (
        <p className="mt-1">
          <span className="text-muted">Thesis / position: </span>
          {w.thesis}
        </p>
      )}
      {w.watch.length > 0 && (
        <ul className="mt-1 list-disc pl-5 text-xs text-muted">
          {w.watch.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

const READ_LABEL: Record<string, string> = {
  article: "read the article",
  feed: "read the feed text",
  headline: "headline only",
};

const MARKET_LABEL: Record<string, string> = {
  MOVED_WITH: "market moved with it",
  MOVED_AGAINST: "market moved against it",
  MOVED: "market moved",
  QUIET: "market quiet",
  UNKNOWN: "not traded yet",
};

function NewsList({ news }: { news: TrackRow["news"] }) {
  const [showOff, setShowOff] = useState(false);
  if (!news || news.length === 0) return null;
  const about = [...news]
    .filter((n) => n.about_company)
    .sort(
      (a, b) =>
        MATERIALITY.indexOf(a.materiality) - MATERIALITY.indexOf(b.materiality) ||
        b.published_at.localeCompare(a.published_at)
    );
  const off = news.filter((n) => !n.about_company);
  return (
    <div>
      <div className="text-xs uppercase tracking-wide text-muted">
        Each item · news agent (context only, being measured)
      </div>
      <ul className="mt-1 space-y-2.5">
        {about.map((n) => (
          <li key={`${n.published_at}-${n.title}`} className="text-xs">
            <div className="flex flex-wrap items-baseline gap-1.5">
              <span className={cn("font-medium", DIR_STYLE[n.direction])}>
                {n.direction.toLowerCase()} {n.materiality.toLowerCase()}
              </span>
              <span className="text-muted">
                {n.event_type.toLowerCase()} · {n.novelty.toLowerCase().replace("_", " ")} ·{" "}
                {n.basis.toLowerCase()} ·{" "}
                <span title={n.market ?? ""}>{MARKET_LABEL[n.market_read] ?? n.market_read}</span>{" "}
                ·{" "}
                <span className={n.read_from === "headline" ? "text-warn" : ""}>
                  {READ_LABEL[n.read_from] ?? n.read_from}
                </span>
              </span>
              {n.thesis.map((t) => (
                <span key={t} className={t.startsWith("against") ? "text-neg" : "text-pos"}>
                  {t} thesis
                </span>
              ))}
            </div>
            <div>
              {n.url ? (
                <a href={n.url} target="_blank" rel="noreferrer" className="hover:underline">
                  {n.title}
                </a>
              ) : (
                n.title
              )}{" "}
              <span className="text-muted">
                — {n.provider}, {fmtEt(n.published_at)}
              </span>
            </div>
            <dl className="mt-0.5 grid grid-cols-[4.5rem_1fr] gap-x-2 text-muted">
              {(
                [
                  ["what", n.what],
                  ["size", n.magnitude],
                  ["why", n.why],
                  ["watch", n.watch],
                ] as const
              )
                .filter(([, v]) => v)
                .map(([k, v]) => (
                  <Fragment key={k}>
                    <dt className="uppercase tracking-wide">{k}</dt>
                    <dd className={k === "why" ? "text-text" : ""}>{v}</dd>
                  </Fragment>
                ))}
            </dl>
          </li>
        ))}
      </ul>
      {off.length > 0 && (
        <button
          onClick={() => setShowOff(!showOff)}
          className="mt-2 text-xs text-muted hover:text-text"
        >
          {showOff ? "hide" : "show"} {off.length} off-topic item(s)
        </button>
      )}
      {showOff && (
        <ul className="mt-1 space-y-0.5 text-xs text-muted">
          {off.map((n) => (
            <li key={`${n.published_at}-${n.title}`}>
              off-topic — {n.title} ({n.provider})
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
