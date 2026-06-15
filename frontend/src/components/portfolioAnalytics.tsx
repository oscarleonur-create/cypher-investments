import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import type { MarketContext, ResearchSummary, RotationResponse } from "@/lib/types";
import { fmtNum, fmtPct, fmtUpside } from "@/lib/utils";
import { Badge } from "./ui/badge";
import { Section, Stat, Td, Th } from "./common";

const chartAxis = { stroke: "#8b97ad", fontSize: 11 };
const grid = "#232b3b";
const tooltipStyle = { background: "#121722", border: `1px solid ${grid}`, color: "#c9d2e0" };
// recharts defaults the tooltip label to black; force a light colour.
const tooltipText = { color: "#c9d2e0" };

// Discrete colourway for pie wedges.
const WEDGE = [
  "#60a5fa",
  "#19c37d",
  "#f59e0b",
  "#ef4444",
  "#a78bfa",
  "#22d3ee",
  "#f472b6",
  "#84cc16",
  "#fb923c",
  "#38bdf8",
];

// Minimal row shape this module needs (a subset of Portfolio's Row).
export interface AnalyticsRow {
  symbol: string;
  mktValue: number;
  research: ResearchSummary | null;
}

// ── Concentration ─────────────────────────────────────────────────────────────

export function ConcentrationPanel({ rows }: { rows: AnalyticsRow[] }) {
  const { byHolding, bySector, metrics } = useMemo(() => {
    const valued = rows.filter((r) => Math.abs(r.mktValue) > 0);
    const total = valued.reduce((s, r) => s + Math.abs(r.mktValue), 0);
    if (total <= 0) {
      return { byHolding: [], bySector: [], metrics: null };
    }

    const byHolding = valued
      .map((r) => ({ name: r.symbol, value: Math.abs(r.mktValue) / total }))
      .sort((a, b) => b.value - a.value);

    const sectorMap = new Map<string, number>();
    for (const r of valued) {
      const sector = r.research?.sector || "Unknown";
      sectorMap.set(sector, (sectorMap.get(sector) || 0) + Math.abs(r.mktValue) / total);
    }
    const bySector = [...sectorMap.entries()]
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);

    const weights = byHolding.map((d) => d.value);
    const hhi = weights.reduce((s, w) => s + w * w, 0);
    const metrics = {
      n: weights.length,
      top: weights[0] ?? 0,
      top5: weights.slice(0, 5).reduce((s, w) => s + w, 0),
      effN: hhi > 0 ? 1 / hhi : 0,
    };

    return { byHolding, bySector, metrics };
  }, [rows]);

  return (
    <Section title="Concentration" empty={!metrics}>
      {metrics && (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Stat label="Holdings" value={metrics.n} />
            <Stat label="Top holding" value={fmtPct(metrics.top, { scale: true })} />
            <Stat label="Top 5" value={fmtPct(metrics.top5, { scale: true })} />
            <Stat
              label="Effective N"
              value={fmtNum(metrics.effN, 1)}
              sub="1 / HHI — diversification"
            />
          </div>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <WeightPie title="By holding" data={byHolding} />
            <WeightPie title="By sector" data={bySector} />
          </div>
        </>
      )}
    </Section>
  );
}

const RADIAN = Math.PI / 180;

// Light, leader-lined labels — only on slices large enough not to collide.
// Smaller slices stay readable via the tooltip.
function renderWedgeLabel(p: any) {
  if (p.percent < 0.04) return null;
  const r = p.outerRadius + 14;
  const x = p.cx + r * Math.cos(-p.midAngle * RADIAN);
  const y = p.cy + r * Math.sin(-p.midAngle * RADIAN);
  const anchor = x >= p.cx ? "start" : "end";
  return (
    <text
      x={x}
      y={y}
      fill="#c9d2e0"
      fontSize={11}
      textAnchor={anchor}
      dominantBaseline="central"
    >
      {`${p.name} ${(p.percent * 100).toFixed(0)}%`}
    </text>
  );
}

function WeightPie({ title, data }: { title: string; data: { name: string; value: number }[] }) {
  return (
    <div>
      <div className="mb-1 text-center text-xs uppercase text-muted">{title}</div>
      <ResponsiveContainer width="100%" height={240}>
        <PieChart margin={{ top: 8, right: 56, bottom: 8, left: 56 }}>
          <Pie
            data={data}
            dataKey="value"
            nameKey="name"
            innerRadius={48}
            outerRadius={76}
            paddingAngle={1}
            label={renderWedgeLabel}
            labelLine={{ stroke: "#5b6679" }}
            isAnimationActive={false}
          >
            {data.map((_, i) => (
              <Cell key={i} fill={WEDGE[i % WEDGE.length]} stroke="#0e131c" />
            ))}
          </Pie>
          <Tooltip
            contentStyle={tooltipStyle}
            labelStyle={tooltipText}
            itemStyle={tooltipText}
            formatter={(v: any) => fmtPct(v, { scale: true })}
          />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

// ── Valuation risk ──────────────────────────────────────────────────────────

type Flag = "OVERPRICED" | "PRICED_IN" | "UPSIDE" | "UNKNOWN";

function valuationFlag(upside: number | null | undefined): Flag {
  if (upside == null) return "UNKNOWN";
  if (upside < 0) return "OVERPRICED";
  if (upside < 0.05) return "PRICED_IN";
  return "UPSIDE";
}

const FLAG_META: Record<Flag, { label: string; variant: "pos" | "warn" | "neg" | "muted" }> = {
  OVERPRICED: { label: "Overpriced", variant: "neg" },
  PRICED_IN: { label: "Priced in", variant: "warn" },
  UPSIDE: { label: "Upside", variant: "pos" },
  UNKNOWN: { label: "No data", variant: "muted" },
};

export function ValuationRiskPanel({ rows }: { rows: AnalyticsRow[] }) {
  const data = useMemo(() => {
    return rows
      .map((r) => {
        const analyst = r.research?.analyst_upside ?? null;
        const base = r.research?.base_upside ?? null;
        const bayes = r.research?.bayes_upside ?? null;
        // Analyst mean target is the headline measure; DCF/Bayes are fallbacks.
        const upside = analyst ?? base ?? bayes;
        return {
          symbol: r.symbol,
          analyst,
          analystTarget: r.research?.analyst_target ?? null,
          analystN: r.research?.analyst_n ?? null,
          base,
          bayes,
          prob: r.research?.bayes_prob_undervalued ?? null,
          upside,
          flag: valuationFlag(upside),
        };
      })
      .sort((a, b) => {
        if (a.upside == null) return 1;
        if (b.upside == null) return -1;
        return a.upside - b.upside; // most overpriced first
      });
  }, [rows]);

  const priced = data.filter((d) => d.upside != null);
  const counts = {
    over: priced.filter((d) => d.flag === "OVERPRICED").length,
    in: priced.filter((d) => d.flag === "PRICED_IN").length,
    none: data.filter((d) => d.upside == null).length,
  };
  const chartData = priced.map((d) => ({ symbol: d.symbol, upside: (d.upside as number) * 100 }));

  return (
    <Section title="Valuation risk" empty={data.length === 0}>
      <div className="mb-3 grid grid-cols-3 gap-3">
        <Stat label="Overpriced" value={<span className="text-neg">{counts.over}</span>} />
        <Stat label="Priced in" value={<span className="text-warn">{counts.in}</span>} />
        <Stat label="No data" value={counts.none} />
      </div>

      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={Math.max(160, chartData.length * 26 + 30)}>
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 4, right: 28, left: 8, bottom: 0 }}
          >
            <CartesianGrid stroke={grid} horizontal={false} />
            <XAxis
              type="number"
              {...chartAxis}
              tickLine={false}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
            />
            <YAxis type="category" dataKey="symbol" {...chartAxis} tickLine={false} width={52} />
            <Tooltip
              contentStyle={tooltipStyle}
              labelStyle={tooltipText}
              itemStyle={tooltipText}
              formatter={(v: any) => fmtPct(v / 100, { sign: true })}
            />
            <ReferenceLine x={0} stroke="#8b97ad" />
            <Bar dataKey="upside" radius={[0, 3, 3, 0]}>
              {chartData.map((d, i) => (
                <Cell key={i} fill={d.upside >= 0 ? "#19c37d" : "#ef4444"} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}

      <div className="mt-3 overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <Th>Ticker</Th>
              <Th>Flag</Th>
              <Th className="text-right">Analyst target</Th>
              <Th className="text-right">Analyst ▲</Th>
              <Th className="text-right">DCF ▲</Th>
              <Th className="text-right">Bayes ▲</Th>
            </tr>
          </thead>
          <tbody>
            {data.map((d) => {
              const meta = FLAG_META[d.flag];
              return (
                <tr key={d.symbol} className="border-t border-border/40">
                  <Td className="font-semibold">{d.symbol}</Td>
                  <Td>
                    <Badge variant={meta.variant as any}>{meta.label}</Badge>
                  </Td>
                  <Td className="text-right">
                    {d.analystTarget == null ? (
                      "—"
                    ) : (
                      <>
                        {fmtNum(d.analystTarget)}
                        {d.analystN ? (
                          <span className="ml-1 text-[10px] text-muted">{d.analystN}a</span>
                        ) : null}
                      </>
                    )}
                  </Td>
                  <Td className="text-right">{fmtUpside(d.analyst)}</Td>
                  <Td className="text-right">{fmtUpside(d.base)}</Td>
                  <Td className="text-right">{fmtUpside(d.bayes)}</Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="mt-2 text-xs text-muted">
        Upside uses the <span className="font-medium">analyst mean price target</span>, falling back
        to the DCF base case then the Bayesian posterior. Names with no coverage or research show as{" "}
        <span className="font-medium">No data</span>.
      </div>
    </Section>
  );
}

// ── Sector rotation ───────────────────────────────────────────────────────────

export function SectorRotationPanel() {
  const { data, isLoading } = useQuery<RotationResponse>({
    queryKey: ["portfolio-rotation"],
    queryFn: api.rotation,
    staleTime: 15 * 60 * 1000,
  });

  const sectors = data ? Object.keys(data.rotation) : [];
  const chartData = sectors.map((s) => ({
    sector: s,
    weight: (data!.weights[s] || 0) * 100,
    rel3m: data!.rotation[s].rel_3m != null ? (data!.rotation[s].rel_3m as number) * 100 : 0,
  }));

  return (
    <Section
      title="Sector rotation"
      empty={!isLoading && sectors.length === 0}
    >
      {isLoading ? (
        <div className="text-sm text-muted">Fetching sector momentum…</div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
              <CartesianGrid stroke={grid} vertical={false} />
              <XAxis dataKey="sector" {...chartAxis} tickLine={false} interval={0} angle={-12} dy={8} height={48} />
              <YAxis {...chartAxis} tickLine={false} width={40} tickFormatter={(v) => `${v}%`} />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={tooltipText}
                itemStyle={tooltipText}
                formatter={(v: any, n: any) =>
                  n === "weight" ? fmtPct(v / 100) : fmtPct(v / 100, { sign: true })
                }
              />
              <Bar dataKey="weight" fill="#3b82f6" radius={[3, 3, 0, 0]} name="weight" />
            </BarChart>
          </ResponsiveContainer>

          <div className="mt-3 overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr>
                  <Th>Sector</Th>
                  <Th>ETF</Th>
                  <Th className="text-right">Weight</Th>
                  <Th className="text-right">ETF 3-mo</Th>
                  <Th className="text-right">vs SPY 3m</Th>
                  <Th className="text-right">vs SPY 1m</Th>
                  <Th>Trend</Th>
                </tr>
              </thead>
              <tbody>
                {sectors.map((s) => {
                  const r = data!.rotation[s];
                  return (
                    <tr key={s} className="border-t border-border/40">
                      <Td>{s}</Td>
                      <Td className="text-muted">{r.etf}</Td>
                      <Td className="text-right">{fmtPct(data!.weights[s] || 0)}</Td>
                      <Td className="text-right">{fmtPct(r.etf_return_3m, { sign: true })}</Td>
                      <Td className="text-right">{fmtPct(r.rel_3m, { sign: true })}</Td>
                      <Td className="text-right">{fmtPct(r.rel_1m, { sign: true })}</Td>
                      <Td>
                        <Badge variant={r.leading ? "pos" : "neg"}>
                          {r.leading ? "Leading" : "Lagging"}
                        </Badge>
                      </Td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="mt-2 text-xs text-muted">
            Positive <span className="font-medium">vs SPY</span> means the sector is outperforming —
            watch for being overweight a lagging sector.
          </div>
        </>
      )}
    </Section>
  );
}

// ── Market (VIX) ──────────────────────────────────────────────────────────────

function vixVariant(vix: number): { label: string; variant: "pos" | "warn" | "neg" } {
  if (vix < 15) return { label: "Calm", variant: "pos" };
  if (vix < 25) return { label: "Elevated", variant: "warn" };
  return { label: "Stressed", variant: "neg" };
}

export function MarketPanel() {
  const { data, isLoading } = useQuery<MarketContext>({
    queryKey: ["portfolio-market"],
    queryFn: api.market,
    staleTime: 15 * 60 * 1000,
  });

  const vix = data?.vix;
  const regime = data?.regime;

  return (
    <Section title="Market — VIX" empty={!isLoading && !vix}>
      {isLoading ? (
        <div className="text-sm text-muted">Fetching VIX…</div>
      ) : vix ? (
        <>
          <div className="mb-3 flex flex-wrap items-center gap-4">
            <div>
              <div className="text-xs uppercase tracking-wide text-muted">VIX</div>
              <div className="flex items-baseline gap-2">
                <span className="text-3xl font-semibold tnum">{fmtNum(vix.current, 1)}</span>
                <Badge variant={vixVariant(vix.current).variant as any}>
                  {regime?.label ?? vixVariant(vix.current).label}
                </Badge>
              </div>
            </div>
            <Stat label="20-day avg" value={fmtNum(vix.sma20, 1)} />
            <Stat
              label="1y percentile"
              value={fmtPct(vix.percentile_1y, { scale: true })}
              sub="where today sits in the past year"
            />
          </div>

          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={vix.history} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
              <defs>
                <linearGradient id="vixFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#60a5fa" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#60a5fa" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={grid} vertical={false} />
              <XAxis dataKey="date" {...chartAxis} tickLine={false} minTickGap={48} />
              <YAxis {...chartAxis} tickLine={false} width={32} />
              <Tooltip
                contentStyle={tooltipStyle}
                labelStyle={tooltipText}
                itemStyle={tooltipText}
                formatter={(v: any) => fmtNum(v, 1)}
              />
              <ReferenceLine y={vix.sma20} stroke="#8b97ad" strokeDasharray="4 4" />
              <Area
                type="monotone"
                dataKey="vix"
                stroke="#60a5fa"
                strokeWidth={1.5}
                fill="url(#vixFill)"
              />
            </AreaChart>
          </ResponsiveContainer>

          {regime ? (
            <div className="mt-2 text-xs text-muted">
              HMM regime read {regime.date} · model VIX {fmtNum(regime.vix, 1)}.
            </div>
          ) : (
            <div className="mt-2 text-xs text-muted">
              No trained regime model on disk — showing VIX only. Train one with{" "}
              <code>advisor ml regime</code> to enable the calm/normal/stressed read.
            </div>
          )}
        </>
      ) : null}
    </Section>
  );
}
