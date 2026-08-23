import { useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Play, TrendingUp, AlertTriangle, MinusCircle, ChevronDown, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import { UNIVERSES } from "@/lib/constants";
import type { SwingScanResult, SwingSignal, SwingStrategyInfo } from "@/lib/types";
import { useJob } from "@/lib/useJob";
import { cn, fmtNum, fmtPct } from "@/lib/utils";
import { Th } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Slider, Toggle } from "@/components/ui/slider";

type VerdictFilter = "ENTER" | "CAUTION" | "ALL";

function verdictVariant(v: string): "pos" | "warn" | "muted" {
  return v === "ENTER" ? "pos" : v === "CAUTION" ? "warn" : "muted";
}

function VerdictIcon({ v }: { v: string }) {
  if (v === "ENTER") return <TrendingUp className="h-3.5 w-3.5 text-pos" />;
  if (v === "CAUTION") return <AlertTriangle className="h-3.5 w-3.5 text-warn" />;
  return <MinusCircle className="h-3.5 w-3.5 text-muted" />;
}

export default function SwingTrade() {
  const strategies = useQuery({
    queryKey: ["swing-strategies"],
    queryFn: api.swingStrategies,
  });
  const all = strategies.data?.strategies ?? [];

  const [universe, setUniverse] = useState("semiconductors");
  const [symbols, setSymbols] = useState("");
  const [strategy, setStrategy] = useState("momentum_breakout");
  const [maxSymbols, setMaxSymbols] = useState(25);
  const [includeMl, setIncludeMl] = useState(true);
  const [filter, setFilter] = useState<VerdictFilter>("CAUTION");
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const signalsQuery = useQuery<{ result: SwingScanResult | null }>({
    queryKey: ["swing-signals", jobId],
    queryFn: () => api.swingSignals(jobId ?? undefined),
    enabled: !!jobId,
  });
  const scanJob = useJob(() => signalsQuery.refetch());

  const run = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const { job_id } = await api.swingRun({
        universe,
        symbols: universe === "custom" ? symbols.split(/[\s,]+/).filter(Boolean) : [],
        strategy,
        max_symbols: maxSymbols,
        include_ml: includeMl,
      });
      setJobId(job_id);
      scanJob.start(job_id);
    } catch (err) {
      alert((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const result = signalsQuery.data?.result ?? null;

  const visible = (result?.signals ?? []).filter((s) => {
    if (filter === "ENTER") return s.verdict === "ENTER";
    if (filter === "CAUTION") return s.verdict === "ENTER" || s.verdict === "CAUTION";
    return true;
  });

  const enterCount = result?.signals.filter((s) => s.verdict === "ENTER").length ?? 0;
  const cautionCount = result?.signals.filter((s) => s.verdict === "CAUTION").length ?? 0;

  return (
    <div className="space-y-5">
      {/* ── Controls ────────────────────────────────────────────── */}
      <Card className="space-y-4 p-4">
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <Label>Universe</Label>
            <div className="flex overflow-hidden rounded-md border border-border">
              {UNIVERSES.map((u) => (
                <button
                  key={u.key}
                  onClick={() => setUniverse(u.key)}
                  className={cn(
                    "flex-1 px-3 py-1.5 text-sm transition-colors",
                    u.key === universe
                      ? "bg-accent text-white"
                      : "text-muted hover:text-text"
                  )}
                >
                  {u.label}
                </button>
              ))}
            </div>
            {universe === "custom" && (
              <input
                value={symbols}
                onChange={(e) => setSymbols(e.target.value.toUpperCase())}
                placeholder="AAPL MSFT NVDA…"
                className="mt-2 h-9 w-full rounded-lg border border-border bg-transparent px-3 text-sm uppercase outline-none focus:border-accent"
              />
            )}
          </div>

          <div>
            <Label>Strategy</Label>
            <div className="flex flex-col gap-1">
              {all.map((s) => (
                <StrategyOption
                  key={s.name}
                  s={s}
                  active={s.name === strategy}
                  onPick={() => setStrategy(s.name)}
                />
              ))}
              {all.length === 0 && (
                <div className="text-xs text-muted">Loading strategies…</div>
              )}
            </div>
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <Label>
              Max symbols{" "}
              <span className="text-text tnum">{maxSymbols}</span>
            </Label>
            <Slider
              value={maxSymbols}
              min={5}
              max={50}
              step={5}
              onChange={setMaxSymbols}
            />
          </div>
          <label className="flex items-center gap-2 self-end text-sm text-muted">
            <Toggle checked={includeMl} onChange={setIncludeMl} />
            ML signal layer (swing pattern model)
          </label>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={run} disabled={busy || scanJob.running}>
            <Play className="h-4 w-4" />
            {scanJob.running ? "Scanning…" : "Scan"}
          </Button>

          {scanJob.job && (
            <span
              className={cn(
                "text-xs",
                scanJob.job.status === "error" ? "text-neg" : "text-muted"
              )}
            >
              {scanJob.job.status === "error"
                ? `Failed: ${scanJob.job.error}`
                : scanJob.job.message}
            </span>
          )}

          {result && (
            <span className="ml-auto text-xs text-muted">
              <span className="text-pos font-medium">{enterCount} ENTER</span>
              {" · "}
              <span className="text-warn font-medium">{cautionCount} CAUTION</span>
              {" · "}
              {result.symbols_scanned} scanned
            </span>
          )}
        </div>
      </Card>

      {/* ── Results ─────────────────────────────────────────────── */}
      {result && (
        <>
          {/* Verdict filter */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-muted">Show:</span>
            {(["ENTER", "CAUTION", "ALL"] as VerdictFilter[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  "rounded-full border px-3 py-1 text-xs transition-colors",
                  f === filter
                    ? "border-accent bg-accent/15 text-text"
                    : "border-border text-muted hover:text-text"
                )}
              >
                {f === "ALL" ? "All" : f === "CAUTION" ? "ENTER + CAUTION" : "ENTER only"}
              </button>
            ))}
            <span className="ml-auto text-xs text-muted">
              {visible.length} result{visible.length !== 1 ? "s" : ""}
            </span>
          </div>

          <ResultsTable signals={visible} />

          {result.errors.length > 0 && (
            <div className="rounded-lg border border-border/40 px-4 py-2 text-xs text-muted">
              {result.errors.length} symbol{result.errors.length !== 1 ? "s" : ""} failed:{" "}
              {result.errors.slice(0, 3).join(" · ")}
              {result.errors.length > 3 && ` · +${result.errors.length - 3} more`}
            </div>
          )}
        </>
      )}

      {!result && !scanJob.running && (
        <Card className="p-6 text-sm text-muted">
          Pick a universe and strategy, then <strong>Scan</strong> for swing trade setups.
          Expect ~30–90 s depending on universe size — each symbol runs full
          technical + sentiment + fundamental checks.
        </Card>
      )}
    </div>
  );
}

// ── Strategy option pill ──────────────────────────────────────────────────────

function StrategyOption({
  s,
  active,
  onPick,
}: {
  s: SwingStrategyInfo;
  active: boolean;
  onPick: () => void;
}) {
  return (
    <button
      onClick={onPick}
      className={cn(
        "flex items-center justify-between rounded-md border px-3 py-1.5 text-left transition-colors",
        active
          ? "border-accent bg-accent/5 text-text"
          : "border-border text-muted hover:text-text"
      )}
    >
      <span className="text-sm font-medium">{s.label}</span>
      <span className="text-[10px] text-muted">{s.typical_hold_days}</span>
    </button>
  );
}

// ── Results table ─────────────────────────────────────────────────────────────

function ResultsTable({ signals }: { signals: SwingSignal[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);

  if (signals.length === 0) {
    return (
      <Card className="p-6 text-sm text-muted">
        No setups match the current filter.
      </Card>
    );
  }

  return (
    <Card className="overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full border-collapse">
          <thead className="border-b border-border bg-panel-2/40">
            <tr>
              <Th />
              <Th>Symbol</Th>
              <Th>Verdict</Th>
              <Th className="text-right">Price</Th>
              <Th>Signal</Th>
              <Th className="text-right">Sentiment</Th>
              <Th className="text-right">Hold</Th>
              <Th>ML</Th>
              <Th>Flags</Th>
            </tr>
          </thead>
          <tbody>
            {signals.map((s) => {
              const key = s.symbol;
              const open = expanded === key;
              return (
                <>
                  <tr
                    key={key}
                    onClick={() => setExpanded(open ? null : key)}
                    className="cursor-pointer border-b border-border/40 hover:bg-panel-2/40"
                  >
                    <td className="w-6 px-2 py-2 text-muted">
                      {open ? (
                        <ChevronDown className="h-3.5 w-3.5" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5" />
                      )}
                    </td>
                    <td className="px-2 py-2">
                      <Link
                        to={`/ticker/${s.symbol}`}
                        className="font-semibold hover:text-accent"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {s.symbol}
                      </Link>
                    </td>
                    <td className="px-2 py-2">
                      <div className="flex items-center gap-1.5">
                        <VerdictIcon v={s.verdict} />
                        <Badge variant={verdictVariant(s.verdict) as any}>
                          {s.verdict}
                        </Badge>
                      </div>
                    </td>
                    <td className="px-2 py-2 text-right tnum">
                      {fmtNum(s.technical_price)}
                    </td>
                    <td className="px-2 py-2 text-sm text-muted">
                      {s.technical_signal}
                    </td>
                    <td
                      className={cn(
                        "px-2 py-2 text-right tnum",
                        s.sentiment_bullish ? "text-pos" : "text-muted"
                      )}
                    >
                      {fmtNum(s.sentiment_pct, 0)}%
                    </td>
                    <td className="px-2 py-2 text-right tnum text-muted">
                      {s.suggested_hold_days > 0 ? `${s.suggested_hold_days}d` : "—"}
                    </td>
                    <td className="px-2 py-2">
                      {s.ml_available ? (
                        <span
                          className={cn(
                            "text-xs tnum",
                            s.ml_signal === "BUY"
                              ? "text-pos"
                              : s.ml_signal === "SELL"
                              ? "text-neg"
                              : "text-muted"
                          )}
                        >
                          {s.ml_signal}
                          {s.ml_win_prob != null &&
                            ` ${fmtPct(s.ml_win_prob)}`}
                        </span>
                      ) : (
                        <span className="text-xs text-muted">—</span>
                      )}
                    </td>
                    <td className="px-2 py-2">
                      <div className="flex flex-wrap items-center gap-1">
                        {s.insider_buying && (
                          <Badge variant="accent">insider buy</Badge>
                        )}
                        {s.earnings_within_7_days && (
                          <Badge variant="warn">earnings &lt;7d</Badge>
                        )}
                        {s.volume_ratio >= 2 && (
                          <Badge variant="pos">
                            {fmtNum(s.volume_ratio, 1)}× vol
                          </Badge>
                        )}
                      </div>
                    </td>
                  </tr>
                  {open && (
                    <tr
                      key={`${key}-detail`}
                      className="border-b border-border/40 bg-panel-2/20"
                    >
                      <td colSpan={9} className="px-4 pb-3 pt-2">
                        <DetailRow s={s} />
                      </td>
                    </tr>
                  )}
                </>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// ── Expanded detail ───────────────────────────────────────────────────────────

function DetailRow({ s }: { s: SwingSignal }) {
  return (
    <div className="space-y-2">
      <p className="text-sm text-muted leading-relaxed">{s.reasoning}</p>

      {s.sentiment_headlines.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">
            Recent headlines
          </div>
          <ul className="space-y-0.5">
            {s.sentiment_headlines.map((h, i) => (
              <li key={i} className="truncate text-xs text-muted">
                • {h}
              </li>
            ))}
          </ul>
        </div>
      )}

      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted">
        <span>
          Technical:{" "}
          <span className={s.technical_bullish ? "text-pos" : "text-muted"}>
            {s.technical_signal}
          </span>
        </span>
        <span>
          Sentiment:{" "}
          <span className={s.sentiment_bullish ? "text-pos" : "text-muted"}>
            {fmtNum(s.sentiment_pct, 0)}%{" "}
            {s.sentiment_bullish ? "bullish" : "bearish"}
          </span>
        </span>
        <span>
          Fundamentals:{" "}
          <span className={s.fundamental_clear ? "text-pos" : "text-warn"}>
            {s.fundamental_clear
              ? "clear"
              : s.earnings_within_7_days
              ? "earnings &lt;7d"
              : "caution"}
          </span>
        </span>
        {s.earnings_date && (
          <span>Earnings: {s.earnings_date}</span>
        )}
        {s.ml_available && s.ml_win_prob != null && (
          <span>
            ML win prob:{" "}
            <span className="text-text">{fmtPct(s.ml_win_prob)}</span>
          </span>
        )}
      </div>
    </div>
  );
}

function Label({ children }: { children: ReactNode }) {
  return (
    <div className="mb-1.5 text-xs uppercase tracking-wide text-muted">
      {children}
    </div>
  );
}
