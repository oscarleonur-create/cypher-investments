import { useMemo, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Activity, Play } from "lucide-react";
import { api } from "@/lib/api";
import type { ScalpPreview, ScalpScanResult, ScalpSignal, ScalpStrategyInfo } from "@/lib/types";
import { useJob } from "@/lib/useJob";
import { cn, fmtNum, fmtPct, pnlColor } from "@/lib/utils";
import { Th } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Slider, Toggle } from "@/components/ui/slider";

const INTERVALS = ["1m", "5m", "15m"] as const;
const UNIVERSES = [
  { key: "semiconductors", label: "Semiconductors" },
  { key: "sp500", label: "S&P 500" },
  { key: "custom", label: "Custom" },
] as const;

const chartAxis = { stroke: "#8b97ad", fontSize: 11 };
const grid = "#232b3b";

function actionVariant(a: string) {
  return a === "LONG" ? "pos" : a === "SHORT" ? "neg" : "muted";
}

export default function Scalping() {
  const [tab, setTab] = useState<"scanner" | "strategies">("scanner");
  return (
    <div className="space-y-5">
      <div className="flex items-center gap-2">
        <div className="flex overflow-hidden rounded-lg border border-border">
          {(["scanner", "strategies"] as const).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                "px-4 py-1.5 text-sm font-medium capitalize transition-colors",
                t === tab ? "bg-panel-2 text-text" : "text-muted hover:text-text"
              )}
            >
              {t}
            </button>
          ))}
        </div>
        <span className="text-xs text-muted">Intraday equity scalping</span>
      </div>

      {tab === "scanner" ? <ScannerTab /> : <StrategiesTab />}
    </div>
  );
}

// ── Scanner ───────────────────────────────────────────────────────────────────

function ScannerTab() {
  const strategies = useQuery({ queryKey: ["scalp-strategies"], queryFn: api.scalpStrategies });
  const all = strategies.data?.strategies || [];

  const [universe, setUniverse] = useState("semiconductors");
  const [symbols, setSymbols] = useState("");
  const [interval, setInterval] = useState<string>("5m");
  const [selected, setSelected] = useState<string[]>([]);
  const [minRvol, setMinRvol] = useState(1.5);
  const [enrichLlm, setEnrichLlm] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const signals = useQuery<{ result: ScalpScanResult | null }>({
    queryKey: ["scalp-signals", jobId],
    queryFn: () => api.scalpSignals(jobId || undefined),
    enabled: !!jobId,
  });
  const scanJob = useJob(() => signals.refetch());

  const toggle = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));

  const run = async () => {
    if (busy) return;
    setBusy(true);
    try {
      const { job_id } = await api.scalpRun({
        universe,
        symbols: universe === "custom" ? symbols.split(/[\s,]+/).filter(Boolean) : [],
        interval,
        strategies: selected,
        min_rvol: minRvol,
        enrich_llm: enrichLlm,
      });
      setJobId(job_id);
      scanJob.start(job_id);
    } catch (err) {
      alert((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const result = signals.data?.result || null;
  const [active, setActive] = useState<ScalpSignal | null>(null);

  return (
    <div className="space-y-5">
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
                    u.key === universe ? "bg-accent text-white" : "text-muted hover:text-text"
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
                placeholder="SPY QQQ NVDA…"
                className="mt-2 h-9 w-full rounded-lg border border-border bg-transparent px-3 text-sm uppercase outline-none focus:border-accent"
              />
            )}
          </div>

          <div>
            <Label>Interval</Label>
            <div className="flex overflow-hidden rounded-md border border-border">
              {INTERVALS.map((iv) => (
                <button
                  key={iv}
                  onClick={() => setInterval(iv)}
                  className={cn(
                    "flex-1 px-3 py-1.5 text-sm transition-colors",
                    iv === interval ? "bg-accent text-white" : "text-muted hover:text-text"
                  )}
                >
                  {iv}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div>
          <Label>Strategies {selected.length === 0 && <span className="text-muted">(all)</span>}</Label>
          <div className="flex flex-wrap gap-2">
            {all.map((s) => (
              <button
                key={s.name}
                onClick={() => toggle(s.name)}
                className={cn(
                  "rounded-full border px-3 py-1 text-xs transition-colors",
                  selected.includes(s.name)
                    ? "border-accent bg-accent/15 text-text"
                    : "border-border text-muted hover:text-text"
                )}
                title={s.description}
              >
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <Label>
              RVOL gate{" "}
              <span className="text-text tnum">{minRvol.toFixed(1)}×</span>{" "}
              <span className="text-muted">
                — drop setups below {minRvol.toFixed(1)}× relative volume
              </span>
            </Label>
            <Slider value={minRvol} min={0} max={5} step={0.1} onChange={setMinRvol} />
          </div>
          <label className="flex items-center gap-2 self-end text-sm text-muted">
            <Toggle checked={enrichLlm} onChange={setEnrichLlm} />
            Deep catalyst (LLM news sentiment — slower)
          </label>
        </div>

        <div className="flex items-center gap-3">
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
              {result.signals.length} signals · {result.symbols_scanned} scanned
              {result.gated_out > 0 ? ` · ${result.gated_out} gated (low RVOL)` : ""} ·{" "}
              <Badge variant={result.source === "tastytrade" ? "pos" : "warn"}>
                {result.source === "tastytrade" ? "live" : "delayed"}
              </Badge>
            </span>
          )}
        </div>
      </Card>

      <Card className="overflow-hidden">
        {!result ? (
          <div className="p-6 text-sm text-muted">
            Pick a universe and strategies, then <strong>Scan</strong> for live setups.
          </div>
        ) : result.signals.length === 0 ? (
          <div className="p-6 text-sm text-muted">
            No setups on the latest bars. Markets may be closed or quiet — try another interval.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="border-b border-border bg-panel-2/40">
                <tr>
                  <Th>Symbol</Th>
                  <Th>Setup</Th>
                  <Th>Side</Th>
                  <Th className="text-right">Entry</Th>
                  <Th className="text-right">Stop</Th>
                  <Th className="text-right">Target</Th>
                  <Th className="text-right">RVOL</Th>
                  <Th className="text-right">Gap</Th>
                  <Th className="text-right">Score</Th>
                  <Th>Catalyst</Th>
                </tr>
              </thead>
              <tbody>
                {result.signals.map((s, i) => (
                  <tr
                    key={`${s.symbol}-${s.strategy}-${i}`}
                    onClick={() => setActive(s)}
                    className={cn(
                      "cursor-pointer border-b border-border/40 hover:bg-panel-2/40",
                      active === s && "bg-panel-2/60"
                    )}
                  >
                    <td className="px-2 py-2 font-semibold">{s.symbol}</td>
                    <td className="px-2 py-2 text-sm text-muted">{s.strategy}</td>
                    <td className="px-2 py-2">
                      <Badge variant={actionVariant(s.action) as any}>{s.action}</Badge>
                    </td>
                    <td className="px-2 py-2 text-right tnum">{fmtNum(s.entry)}</td>
                    <td className="px-2 py-2 text-right tnum text-neg">{fmtNum(s.stop)}</td>
                    <td className="px-2 py-2 text-right tnum text-pos">{fmtNum(s.target)}</td>
                    <td
                      className={cn(
                        "px-2 py-2 text-right tnum",
                        s.rvol != null && s.rvol >= 2 ? "text-pos" : "text-muted"
                      )}
                    >
                      {s.rvol != null ? `${fmtNum(s.rvol, 1)}×` : "—"}
                    </td>
                    <td className={cn("px-2 py-2 text-right tnum", pnlColor(s.gap_pct))}>
                      {s.gap_pct != null ? fmtPct(s.gap_pct, { sign: true }) : "—"}
                    </td>
                    <td className="px-2 py-2 text-right tnum">{fmtNum(s.score, 0)}</td>
                    <td className="px-2 py-2 text-xs">
                      <div className="flex flex-wrap items-center gap-1">
                        {s.earnings_today && <Badge variant="warn">earnings</Badge>}
                        {s.headlines.length > 0 && (
                          <Badge variant="accent">{s.headlines.length} news</Badge>
                        )}
                        <span className="text-muted">{s.catalyst_note || s.reason}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {active && (
        <SignalChart symbol={active.symbol} interval={result?.interval || interval} signal={active} />
      )}
    </div>
  );
}

// ── Strategies ────────────────────────────────────────────────────────────────

function StrategiesTab() {
  const strategies = useQuery({ queryKey: ["scalp-strategies"], queryFn: api.scalpStrategies });
  const all = strategies.data?.strategies || [];

  const [symbol, setSymbol] = useState("SPY");
  const [interval, setInterval] = useState("5m");
  const [strategy, setStrategy] = useState("vwap_reversion");

  return (
    <div className="space-y-5">
      <div className="grid gap-3 md:grid-cols-3">
        {all.map((s) => (
          <StrategyCard key={s.name} s={s} active={s.name === strategy} onPick={() => setStrategy(s.name)} />
        ))}
      </div>

      <Card className="space-y-3 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <Label>Preview symbol</Label>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              className="h-9 w-32 rounded-lg border border-border bg-transparent px-3 text-sm uppercase outline-none focus:border-accent"
            />
          </div>
          <div>
            <Label>Interval</Label>
            <div className="flex overflow-hidden rounded-md border border-border">
              {INTERVALS.map((iv) => (
                <button
                  key={iv}
                  onClick={() => setInterval(iv)}
                  className={cn(
                    "px-3 py-1.5 text-sm transition-colors",
                    iv === interval ? "bg-accent text-white" : "text-muted hover:text-text"
                  )}
                >
                  {iv}
                </button>
              ))}
            </div>
          </div>
          <span className="text-xs text-muted">
            Showing <strong>{all.find((s) => s.name === strategy)?.label || strategy}</strong> on{" "}
            {symbol}
          </span>
        </div>
      </Card>

      <PreviewChart symbol={symbol} interval={interval} strategy={strategy} />
    </div>
  );
}

function StrategyCard({
  s,
  active,
  onPick,
}: {
  s: ScalpStrategyInfo;
  active: boolean;
  onPick: () => void;
}) {
  return (
    <button
      onClick={onPick}
      className={cn(
        "rounded-lg border p-3 text-left transition-colors",
        active ? "border-accent bg-accent/5" : "border-border hover:border-border/80"
      )}
    >
      <div className="mb-1 flex items-center gap-2">
        <Activity className="h-4 w-4 text-accent" />
        <span className="font-medium">{s.label}</span>
      </div>
      <p className="mb-2 text-xs text-muted">{s.description}</p>
      <div className="flex flex-wrap gap-1">
        {Object.entries(s.defaults).map(([k, v]) => (
          <span key={k} className="rounded bg-panel-2 px-1.5 py-0.5 text-[10px] text-muted tnum">
            {k}={v}
          </span>
        ))}
      </div>
    </button>
  );
}

// ── Charts ────────────────────────────────────────────────────────────────────

function SignalChart({
  symbol,
  interval,
  signal,
}: {
  symbol: string;
  interval: string;
  signal: ScalpSignal;
}) {
  const { data, isLoading } = useQuery<ScalpPreview>({
    queryKey: ["scalp-preview", symbol, interval, signal.strategy],
    queryFn: () => api.scalpPreview(symbol, interval, signal.strategy),
  });
  return <CandleChart title={`${symbol} · ${signal.strategy}`} data={data} loading={isLoading} signal={signal} />;
}

function PreviewChart({
  symbol,
  interval,
  strategy,
}: {
  symbol: string;
  interval: string;
  strategy: string;
}) {
  const { data, isLoading } = useQuery<ScalpPreview>({
    queryKey: ["scalp-preview", symbol, interval, strategy],
    queryFn: () => api.scalpPreview(symbol, interval, strategy),
    enabled: symbol.length > 0,
  });
  return (
    <CandleChart
      title={`${symbol} · ${strategy}`}
      data={data}
      loading={isLoading}
      signal={data?.signal || null}
    />
  );
}

function CandleChart({
  title,
  data,
  loading,
  signal,
}: {
  title: string;
  data?: ScalpPreview;
  loading: boolean;
  signal: ScalpSignal | null;
}) {
  const points = useMemo(
    () => (data?.candles || []).map((c) => ({ t: c.t, close: c.close, vwap: c.vwap })),
    [data]
  );

  return (
    <Card className="p-4">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-sm font-medium">{title}</div>
        {signal ? (
          <Badge variant={actionVariant(signal.action) as any}>
            {signal.action} · score {fmtNum(signal.score, 0)}
          </Badge>
        ) : (
          <span className="text-xs text-muted">no active setup</span>
        )}
      </div>

      {loading ? (
        <div className="py-12 text-center text-sm text-muted">Loading candles…</div>
      ) : points.length === 0 ? (
        <div className="py-12 text-center text-sm text-muted">
          No intraday candles{data ? ` (${data.source})` : ""}. Market may be closed.
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={320}>
          <ComposedChart data={points} margin={{ top: 6, right: 8, left: 8, bottom: 0 }}>
            <defs>
              <linearGradient id="scalpfill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#5b8def" stopOpacity={0.3} />
                <stop offset="100%" stopColor="#5b8def" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke={grid} vertical={false} />
            <XAxis
              type="number"
              dataKey="t"
              scale="time"
              domain={["dataMin", "dataMax"]}
              {...chartAxis}
              tickLine={false}
              tickFormatter={(t: number) =>
                new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
              }
            />
            <YAxis
              {...chartAxis}
              tickLine={false}
              width={52}
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => fmtNum(v, 0)}
            />
            <Tooltip
              contentStyle={{ background: "#121722", border: `1px solid ${grid}` }}
              labelFormatter={(t: any) => new Date(t).toLocaleString()}
              formatter={(v: any, n: any) => [fmtNum(v), n === "vwap" ? "VWAP" : "Price"]}
            />
            <Area
              type="monotone"
              dataKey="close"
              stroke="#5b8def"
              fill="url(#scalpfill)"
              strokeWidth={1.5}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="vwap"
              stroke="#f5a524"
              strokeWidth={1.25}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
            {signal && (
              <>
                <ReferenceLine
                  y={signal.entry}
                  stroke="#8b97ad"
                  strokeDasharray="4 3"
                  label={{ value: "entry", fill: "#8b97ad", fontSize: 10, position: "right" }}
                />
                <ReferenceLine
                  y={signal.stop}
                  stroke="#ef4444"
                  strokeDasharray="4 3"
                  label={{ value: "stop", fill: "#ef4444", fontSize: 10, position: "right" }}
                />
                <ReferenceLine
                  y={signal.target}
                  stroke="#19c37d"
                  strokeDasharray="4 3"
                  label={{ value: "target", fill: "#19c37d", fontSize: 10, position: "right" }}
                />
              </>
            )}
          </ComposedChart>
        </ResponsiveContainer>
      )}

      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        <span className="text-amber-400">● VWAP</span>
        <span className="text-blue-400">● Price</span>
        {signal?.rvol != null && <span>RVOL {fmtNum(signal.rvol, 1)}×</span>}
        {signal?.gap_pct != null && (
          <span className={pnlColor(signal.gap_pct)}>gap {fmtPct(signal.gap_pct, { sign: true })}</span>
        )}
        {signal?.earnings_today && <span className="text-warn">earnings today</span>}
        {signal?.sentiment_score != null && (
          <span>news sentiment {fmtNum(signal.sentiment_score, 0)}/100</span>
        )}
        {signal && <span>{signal.reason}</span>}
      </div>

      {signal && signal.headlines.length > 0 && (
        <div className="mt-2 border-t border-border/40 pt-2">
          <div className="mb-1 text-[10px] uppercase text-muted">Recent headlines</div>
          <ul className="space-y-0.5">
            {signal.headlines.slice(0, 5).map((h, i) => (
              <li key={i} className="truncate text-xs text-muted" title={h}>
                • {h}
              </li>
            ))}
          </ul>
        </div>
      )}
    </Card>
  );
}

function Label({ children }: { children: ReactNode }) {
  return <div className="mb-1.5 text-xs uppercase tracking-wide text-muted">{children}</div>;
}
