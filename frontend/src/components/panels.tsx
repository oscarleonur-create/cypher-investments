import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import type {
  ResearchReport,
} from "@/lib/types";
import type { QuotesState } from "@/lib/useQuotes";
import { cn, fmtNum, fmtPct, fmtUsd, pnlColor } from "@/lib/utils";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Toggle } from "./ui/slider";
import { KV, Section, Td, Th, ThesisBadge } from "./common";

const chartAxis = { stroke: "#8b97ad", fontSize: 11 };
const grid = "#232b3b";

function sev(s?: string) {
  const v = (s || "").toUpperCase();
  return v === "HIGH" ? "neg" : v === "MEDIUM" ? "warn" : "muted";
}

// yfinance often returns pct_held = null but value_usd / shares populated.
function holderValue(h: any): string {
  if (h?.value_usd) return fmtUsd(h.value_usd);
  if (h?.pct_held != null) return fmtPct(h.pct_held);
  if (h?.shares) return `${fmtNum(h.shares, 0)} sh`;
  return "—";
}

// ── Thesis & Edge ───────────────────────────────────────────────────────────

export function ThesisPanel({ r }: { r: ResearchReport }) {
  const t = r.thesis;
  const vp = r.variant_perception;
  const c = r.consensus;
  return (
    <Section title="Thesis & Edge" empty={!t && !vp && !c}>
      {t?.summary && <p className="text-sm mb-3">{t.summary}</p>}
      <div className="flex flex-wrap gap-2 mb-3">
        {t?.conviction && <Badge variant="accent">Conviction: {t.conviction}</Badge>}
        {vp?.mispricing_type && vp.mispricing_type !== "none" && (
          <Badge variant="muted">{String(vp.mispricing_type).replace(/_/g, " ")}</Badge>
        )}
      </div>

      {vp?.our_key_insight && (
        <div className="text-sm mb-2">
          <span className="text-muted">Edge: </span>
          {vp.our_key_insight}
        </div>
      )}
      {c?.recommendation_key && (
        <div className="mt-2">
          <KV k={`Sell-side (${c.n_analysts} analysts)`} v={c.recommendation_key} />
        </div>
      )}
    </Section>
  );
}

// ── Financials ──────────────────────────────────────────────────────────────

export function FinancialsPanel({ r }: { r: ResearchReport }) {
  const s = r.statements;
  const income = s?.income || [];
  const cashflow = s?.cashflow || [];
  const data = income
    .map((p: any, i: number) => ({
      fy: p.fiscal_year,
      revenue: p.revenue,
      net_income: p.net_income,
      fcf: cashflow[i]?.free_cash_flow,
    }))
    .reverse();

  return (
    <Section title="Financials" empty={income.length === 0}>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis dataKey="fy" {...chartAxis} tickLine={false} />
          <YAxis {...chartAxis} tickLine={false} width={52} tickFormatter={(v) => fmtUsd(v)} />
          <Tooltip
            contentStyle={{ background: "#121722", border: `1px solid ${grid}` }}
            formatter={(v: any) => fmtUsd(v)}
          />
          <Bar dataKey="revenue" fill="#3b82f6" radius={[3, 3, 0, 0]} />
          <Bar dataKey="net_income" fill="#19c37d" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
      <div className="overflow-x-auto mt-3">
        <table className="w-full">
          <thead>
            <tr>
              <Th>FY</Th>
              <Th className="text-right">Revenue</Th>
              <Th className="text-right">Op income</Th>
              <Th className="text-right">Net income</Th>
              <Th className="text-right">EPS</Th>
              <Th className="text-right">FCF</Th>
            </tr>
          </thead>
          <tbody>
            {income.map((p: any, i: number) => (
              <tr key={p.fiscal_year} className="border-t border-border/40">
                <Td>{p.fiscal_year}</Td>
                <Td className="text-right">{fmtUsd(p.revenue)}</Td>
                <Td className="text-right">{fmtUsd(p.operating_income)}</Td>
                <Td className="text-right">{fmtUsd(p.net_income)}</Td>
                <Td className="text-right">{p.eps_diluted ? fmtNum(p.eps_diluted) : "—"}</Td>
                <Td className="text-right">{fmtUsd(cashflow[i]?.free_cash_flow)}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

// ── Ratios + red flags ──────────────────────────────────────────────────────

export function RatiosPanel({ r }: { r: ResearchReport }) {
  const periods = r.ratios?.periods || [];
  const flags = r.red_flags?.flags || [];
  return (
    <Section title="Ratios & Quality" empty={periods.length === 0 && flags.length === 0}>
      {periods.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <Th>FY</Th>
                <Th className="text-right">Gross</Th>
                <Th className="text-right">Op</Th>
                <Th className="text-right">Net</Th>
                <Th className="text-right">ROE</Th>
                <Th className="text-right">ROIC</Th>
                <Th className="text-right">D/E</Th>
                <Th className="text-right">FCF mgn</Th>
              </tr>
            </thead>
            <tbody>
              {periods.map((p: any) => (
                <tr key={p.fiscal_year} className="border-t border-border/40">
                  <Td>{p.fiscal_year}</Td>
                  <Td className="text-right">{fmtPct(p.gross_margin)}</Td>
                  <Td className="text-right">{fmtPct(p.operating_margin)}</Td>
                  <Td className="text-right">{fmtPct(p.net_margin)}</Td>
                  <Td className="text-right">{fmtPct(p.roe)}</Td>
                  <Td className="text-right">{fmtPct(p.roic)}</Td>
                  <Td className="text-right">{p.debt_to_equity ? fmtNum(p.debt_to_equity) : "—"}</Td>
                  <Td className="text-right">{fmtPct(p.fcf_margin)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {flags.length > 0 && (
        <div className="mt-3 space-y-1">
          {flags.map((f: any, i: number) => (
            <div key={i} className="flex items-start gap-2 text-sm">
              <Badge variant={sev(f.severity) as any}>{f.severity}</Badge>
              <span>
                <span className="font-medium">{f.title}</span>{" "}
                <span className="text-muted">{f.detail}</span>
              </span>
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

// ── Ecosystem ───────────────────────────────────────────────────────────────

// Classify a raw SEC transaction_type into a Buy / Sell side + colour.
function insiderSide(type?: string): { label: string; variant: "pos" | "neg" | "muted" } {
  const t = (type || "").toLowerCase();
  if (t.includes("purchase") || t.includes("buy") || t.includes("acqui"))
    return { label: "Buy", variant: "pos" };
  if (t.includes("sale") || t.includes("sell") || t.includes("dispos"))
    return { label: "Sell", variant: "neg" };
  return { label: type || "—", variant: "muted" };
}

// "2026-05-20" → "May 20 '26" (compact, sorts already done server-side).
function fmtInsiderDate(d?: string): string {
  if (!d) return "—";
  const dt = new Date(d + "T00:00:00");
  if (Number.isNaN(dt.getTime())) return d;
  const mon = dt.toLocaleString("en-US", { month: "short" });
  return `${mon} ${dt.getDate()} '${String(dt.getFullYear()).slice(2)}`;
}

/** Dated, buy/sell-coded table of insider (Form 4) transactions. */
function InsiderTransactions({ txns }: { txns?: any[] }) {
  const [expanded, setExpanded] = React.useState(false);
  const all = txns || [];
  if (all.length === 0) {
    return (
      <div className="mt-3 text-xs text-muted">No insider transactions on file.</div>
    );
  }
  const rows = expanded ? all : all.slice(0, 8);
  return (
    <div className="mt-3">
      <div className="mb-1 flex items-center justify-between">
        <div className="text-xs uppercase text-muted">Insider transactions</div>
        <div className="text-[10px] text-muted">{all.length} on file</div>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr>
              <Th>Date</Th>
              <Th>Insider</Th>
              <Th className="text-center">Side</Th>
              <Th className="text-right">Shares</Th>
              <Th className="text-right">Value</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((t: any, i: number) => {
              const side = insiderSide(t.transaction_type);
              return (
                <tr key={i} className="border-t border-border/40">
                  <Td className="whitespace-nowrap text-muted">{fmtInsiderDate(t.transaction_date)}</Td>
                  <Td>
                    <div className="leading-tight">{t.insider_name}</div>
                    {t.title && <div className="text-[10px] text-muted">{t.title}</div>}
                  </Td>
                  <Td className="text-center">
                    <Badge variant={side.variant as any}>{side.label}</Badge>
                  </Td>
                  <Td className="text-right">{t.shares != null ? fmtNum(t.shares, 0) : "—"}</Td>
                  <Td className="text-right">{t.value_usd ? fmtUsd(t.value_usd) : "—"}</Td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {all.length > 8 && (
        <button
          className="mt-1 text-xs text-accent hover:underline"
          onClick={() => setExpanded((v) => !v)}
        >
          {expanded ? "Show less" : `Show all ${all.length}`}
        </button>
      )}
    </div>
  );
}

export function EcosystemPanel({ r }: { r: ResearchReport }) {
  const e = r.ecosystem;
  const holders = e?.holders;
  const insiders = e?.insiders;
  return (
    <Section title="Ecosystem" empty={!e}>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <div>
          <div className="text-xs uppercase text-muted mb-1">Ownership</div>
          <KV k="Institutional" v={fmtPct(holders?.pct_institutional)} />
          <KV k="Insider" v={fmtPct(holders?.pct_insider)} />
          <div className="mt-2 mb-1 text-[10px] uppercase text-muted">Top holders (value)</div>
          {(holders?.top_holders || []).slice(0, 6).map((h: any, i: number) => (
            <KV key={i} k={h.name} v={holderValue(h)} />
          ))}
        </div>
        <div>
          <div className="text-xs uppercase text-muted mb-1">Insider activity</div>
          <KV
            k="Net buying"
            v={
              <span className={pnlColor(insiders?.net_buying_usd)}>
                {fmtUsd(insiders?.net_buying_usd)}
              </span>
            }
          />
          <KV k="C-suite buying" v={insiders?.c_suite_buying ? "Yes" : "No"} />
          {insiders?.lookback_days && (
            <KV k="Lookback" v={`${insiders.lookback_days}d`} />
          )}
        </div>
      </div>

      <InsiderTransactions txns={insiders?.transactions} />

      {(e?.top_customers?.length || e?.top_suppliers?.length) > 0 && (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 mt-3">
          {e?.top_customers?.length > 0 && (
            <div>
              <div className="text-xs uppercase text-muted mb-1">Key customers</div>
              {e.top_customers.map((c: any, i: number) => (
                <KV key={i} k={c.name} v={c.note || "—"} />
              ))}
            </div>
          )}
          {e?.top_suppliers?.length > 0 && (
            <div>
              <div className="text-xs uppercase text-muted mb-1">Key suppliers</div>
              {e.top_suppliers.map((c: any, i: number) => (
                <KV key={i} k={c.name} v={c.category || "—"} />
              ))}
            </div>
          )}
        </div>
      )}
    </Section>
  );
}

// ── Competitive / Moat ──────────────────────────────────────────────────────

export function MoatPanel({ r }: { r: ResearchReport }) {
  const ind = r.industry;
  const pf = ind?.porters_forces;
  return (
    <Section title="Competitive & Moat" empty={!ind}>
      <div className="flex flex-wrap gap-2 mb-2">
        {ind?.moat_type && (
          <Badge variant="accent">Moat: {String(ind.moat_type).replace(/_/g, " ")}</Badge>
        )}
        {ind?.competitive_position && (
          <Badge variant="muted">{ind.competitive_position}</Badge>
        )}
        {ind?.moat_strength != null && <Badge variant="muted">Strength {ind.moat_strength}/10</Badge>}
      </div>
      {ind?.moat_description && <p className="text-sm mb-2">{ind.moat_description}</p>}
      {pf && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2 my-2">
          {[
            ["Rivalry", pf.competitive_rivalry],
            ["Suppliers", pf.supplier_power],
            ["Buyers", pf.buyer_power],
            ["Entrants", pf.threat_of_new_entrants],
            ["Substitutes", pf.threat_of_substitutes],
          ].map(([k, v]) => (
            <div key={k as string} className="text-center">
              <div className="text-[10px] uppercase text-muted">{k}</div>
              <Badge variant={sev(v as string) as any}>{v as string}</Badge>
            </div>
          ))}
        </div>
      )}
      {ind?.key_competitors?.length > 0 && (
        <div className="text-sm">
          <span className="text-muted">Competitors: </span>
          {ind.key_competitors.join(", ")}
        </div>
      )}
    </Section>
  );
}

// ── Catalysts & Risks ───────────────────────────────────────────────────────

function catalystProbColor(p: number | undefined): string {
  if (p == null) return "text-muted";
  if (p >= 0.7) return "text-pos";
  if (p >= 0.4) return "text-warn";
  return "text-muted";
}

function directionIcon(d: string | undefined): string {
  if (d === "bullish") return "▲";
  if (d === "bearish") return "▼";
  if (d === "mixed") return "↔";
  return "";
}

function directionColor(d: string | undefined): string {
  if (d === "bullish") return "text-pos";
  if (d === "bearish") return "text-neg";
  return "text-muted";
}

export function CatalystsPanel({ r }: { r: ResearchReport }) {
  const cr = r.catalyst_risk;
  const catalysts = cr?.catalysts || [];
  const risks = cr?.risks || [];
  return (
    <Section title="Catalysts & Risks" empty={catalysts.length === 0 && risks.length === 0}>
      {catalysts.length > 0 && (
        <div className="mb-3 space-y-1.5">
          {catalysts.map((c: any, i: number) => (
            <div key={i} className="flex items-start gap-2 text-sm">
              <Badge variant={c.is_near_term ? "warn" : "muted"}>{c.catalyst_type}</Badge>
              <span className="text-muted tnum shrink-0">{c.expected_date}</span>
              {c.direction && c.direction !== "neutral" && (
                <span className={`shrink-0 font-mono text-xs ${directionColor(c.direction)}`}>
                  {directionIcon(c.direction)}
                </span>
              )}
              <span className="min-w-0 flex-1">{c.description}</span>
              {c.probability != null && (
                <span className={`shrink-0 tnum text-xs font-medium ${catalystProbColor(c.probability)}`}>
                  {Math.round(c.probability * 100)}%
                </span>
              )}
              {c.price_impact_pct != null && (
                <span className="shrink-0 tnum text-xs text-muted">
                  ±{c.price_impact_pct.toFixed(0)}%
                </span>
              )}
            </div>
          ))}
        </div>
      )}
      {risks.length > 0 && (
        <div className="space-y-1">
          {risks.map((rk: any, i: number) => (
            <div key={i} className="flex items-start gap-2 text-sm">
              <Badge variant={sev(rk.severity) as any}>{rk.severity}</Badge>
              <span className="text-muted">{rk.category}</span>
              <span>{rk.description}</span>
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

// ── KPI Monitor ─────────────────────────────────────────────────────────────

export function KpiPanel({ r }: { r: ResearchReport }) {
  const km = r.kpi_monitor;
  const kpis = km?.kpis || [];
  const kc = (s: string) =>
    s === "on_track" ? "pos" : s === "caution" ? "warn" : s === "breached" ? "neg" : "muted";
  return (
    <Section
      title="KPI Monitor"
      empty={!km}
      right={km && <ThesisBadge status={km.thesis_status} />}
    >
      <div className="space-y-2">
        {kpis.map((k: any, i: number) => (
          <div key={i} className="flex items-center justify-between gap-3 text-sm">
            <div className="min-w-0">
              <div className="font-medium truncate">{k.metric_name}</div>
              <div className="text-xs text-muted truncate">{k.description}</div>
            </div>
            <div className="flex items-center gap-3 shrink-0 tnum">
              <span>{fmtKpi(k.current_value, k.unit)}</span>
              <Badge variant={kc(k.status) as any}>{String(k.status).replace(/_/g, " ")}</Badge>
            </div>
          </div>
        ))}
      </div>
      {km?.alerts?.length > 0 && (
        <div className="mt-3 space-y-1">
          {km.alerts.map((a: string, i: number) => (
            <div key={i} className={`text-sm ${a.startsWith("⚠") ? "text-neg" : "text-warn"}`}>
              {a}
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

function fmtKpi(v: number | null, unit: string): string {
  if (v == null) return "—";
  if (unit === "pct") return fmtPct(v);
  if (unit === "x" || unit === "ratio") return `${fmtNum(v)}x`;
  if (unit === "usd") return fmtUsd(v);
  return fmtNum(v, 3);
}

// ── Options Flow ────────────────────────────────────────────────────────────

export function OptionsFlowPanel({ r }: { r: ResearchReport }) {
  const f = r.options_flow;
  const unusual = f?.unusual_activity || [];
  return (
    <Section title="Options Flow" empty={!f}>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-3">
        <KV k="Put/Call" v={f?.put_call_ratio ? fmtNum(f.put_call_ratio) : "—"} />
        <KV k="Max pain" v={f?.max_pain_price ? fmtNum(f.max_pain_price) : "—"} />
        <KV k="ATM IV" v={fmtPct(f?.atm_iv)} />
        <KV k="Next expiry" v={f?.nearest_expiry || "—"} />
      </div>
      {unusual.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <Th>Contract</Th>
                <Th className="text-right">Strike</Th>
                <Th className="text-right">Vol</Th>
                <Th className="text-right">Vol/OI</Th>
                <Th className="text-right">IV</Th>
              </tr>
            </thead>
            <tbody>
              {unusual.slice(0, 8).map((u: any, i: number) => (
                <tr key={i} className="border-t border-border/40">
                  <Td>
                    <Badge variant={u.option_type === "call" ? "pos" : "neg"}>
                      {u.option_type}
                    </Badge>{" "}
                    {u.expiration}
                  </Td>
                  <Td className="text-right">{fmtNum(u.strike)}</Td>
                  <Td className="text-right">{fmtNum(u.volume, 0)}</Td>
                  <Td className="text-right">{fmtNum(u.volume_oi_ratio)}</Td>
                  <Td className="text-right">{fmtPct(u.implied_volatility)}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Section>
  );
}

// ── X Sentiment ─────────────────────────────────────────────────────────────

export function SentimentPanel({ r }: { r: ResearchReport }) {
  const x = r.x_sentiment;
  return (
    <Section title="Social Sentiment (X)" empty={!x}>
      <div className="grid grid-cols-2 gap-3 mb-3">
        <KV k="Score" v={x ? `${fmtNum(x.score, 0)}/100` : "—"} />
        <KV k="Positive" v={x ? `${fmtNum(x.positive_pct, 0)}%` : "—"} />
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        {x?.top_bullish_themes?.length > 0 && (
          <div>
            <div className="text-xs uppercase text-pos mb-1">Bullish</div>
            {x.top_bullish_themes.map((t: string, i: number) => (
              <div key={i} className="text-sm">
                • {t}
              </div>
            ))}
          </div>
        )}
        {x?.top_bearish_themes?.length > 0 && (
          <div>
            <div className="text-xs uppercase text-neg mb-1">Bearish</div>
            {x.top_bearish_themes.map((t: string, i: number) => (
              <div key={i} className="text-sm">
                • {t}
              </div>
            ))}
          </div>
        )}
      </div>
    </Section>
  );
}

// ── Earnings-call transcripts ────────────────────────────────────────────────

function toneVariant(t?: string) {
  const v = (t || "").toLowerCase();
  return v === "bullish" ? "pos" : v === "bearish" ? "neg" : "muted";
}

export function TranscriptsPanel({ r }: { r: ResearchReport }) {
  const t = r.transcripts;
  const summaries = t?.summaries || [];
  return (
    <Section title="Earnings Calls" empty={!t || summaries.length === 0}>
      {t?.tone_trend && (
        <div className="text-sm mb-3">
          <span className="text-muted">Tone trend: </span>
          {t.tone_trend}
        </div>
      )}
      <div className="space-y-3">
        {summaries.map((q: any, i: number) => (
          <div key={i} className="border-t border-border/40 pt-2 first:border-0 first:pt-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-semibold">{q.quarter}</span>
              {q.earnings_date && (
                <span className="text-xs text-muted tnum">{q.earnings_date}</span>
              )}
              <Badge variant={toneVariant(q.tone) as any}>{q.tone}</Badge>
            </div>
            {q.key_topics?.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-1">
                {q.key_topics.map((k: string, j: number) => (
                  <Badge key={j} variant="muted">
                    {k}
                  </Badge>
                ))}
              </div>
            )}
            {q.management_guidance && (
              <div className="text-sm mb-1">
                <span className="text-muted">Guidance: </span>
                {q.management_guidance}
              </div>
            )}
            {q.analyst_concerns && (
              <div className="text-sm mb-1">
                <span className="text-muted">Analyst concerns: </span>
                {q.analyst_concerns}
              </div>
            )}
            {q.highlight_quote && (
              <blockquote className="border-l-2 border-border pl-2 text-sm italic text-muted">
                “{q.highlight_quote}”
              </blockquote>
            )}
            {q.source_url && (
              <a
                href={q.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-1 inline-block text-sm text-accent hover:underline"
              >
                Read full transcript ↗
              </a>
            )}
          </div>
        ))}
      </div>
      {t?.sources?.length > 0 && (
        <div className="mt-3 border-t border-border/40 pt-2">
          <div className="text-[10px] uppercase text-muted mb-1">Sources</div>
          <div className="space-y-0.5">
            {t.sources.map((s: any, i: number) => (
              <a
                key={i}
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block truncate text-xs text-muted hover:text-accent hover:underline"
                title={s.title || s.url}
              >
                {s.title || s.url}
              </a>
            ))}
          </div>
        </div>
      )}
    </Section>
  );
}

// ── SEC filings ───────────────────────────────────────────────────────────────

function filingVariant(form?: string) {
  const f = (form || "").toUpperCase();
  if (f === "10-K" || f === "10-Q") return "accent";
  if (f === "8-K") return "warn";
  if (f === "4") return "muted";
  return "muted";
}

export function FilingsPanel({ r }: { r: ResearchReport }) {
  const filings = (r.filings || [])
    .slice()
    .sort((a: any, b: any) => (a.filing_date < b.filing_date ? 1 : -1));
  return (
    <Section
      title="SEC Filings"
      empty={filings.length === 0}
      right={filings.length > 0 && <span className="text-xs text-muted">{filings.length}</span>}
    >
      <div className="max-h-80 overflow-y-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-panel">
            <tr>
              <Th>Form</Th>
              <Th>Filed</Th>
              <Th>Period</Th>
              <Th></Th>
            </tr>
          </thead>
          <tbody>
            {filings.map((f: any) => (
              <tr key={f.accession_number} className="border-t border-border/40">
                <Td>
                  <Badge variant={filingVariant(f.form) as any}>{f.form}</Badge>
                </Td>
                <Td>{f.filing_date}</Td>
                <Td className="text-muted">{f.period_of_report || "—"}</Td>
                <Td className="text-right">
                  {f.url ? (
                    <a
                      href={f.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-accent hover:underline"
                    >
                      open
                    </a>
                  ) : (
                    "—"
                  )}
                </Td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}

// ── Deep Research (white-paper, cited brief) ─────────────────────────────────

function srcVariant(t?: string) {
  return t === "sec_filing" ? "accent" : t === "website" ? "pos" : t === "news" ? "warn" : "muted";
}
function srcLabel(t?: string) {
  return t === "sec_filing" ? "SEC" : t === "website" ? "Website" : t === "news" ? "News" : "Source";
}

function Cite({ ids }: { ids?: number[] }) {
  if (!ids || ids.length === 0) return null;
  return (
    <sup className="ml-0.5 text-[10px] text-accent">
      [
      {ids.map((i, k) => (
        <span key={i}>
          {k > 0 ? "," : ""}
          <a href={`#dr-ref-${i}`} className="hover:underline">
            {i}
          </a>
        </span>
      ))}
      ]
    </sup>
  );
}

export function DeepResearchPanel({ r }: { r: ResearchReport }) {
  const d = r.deep_research || {};
  const customers = d.customers || [];
  const sc = d.supply_chain;
  const devs = d.recent_developments || [];
  const quotes = d.management_quotes || [];
  const so = d.second_order_thesis;
  const refs = d.references || [];
  const empty =
    !d.abstract &&
    !d.what_they_do &&
    customers.length === 0 &&
    !sc &&
    devs.length === 0 &&
    quotes.length === 0 &&
    !so;

  return (
    <Section title="Deep Research" empty={empty}>
      {/* Abstract */}
      {d.abstract && (
        <p className="mb-3 border-l-2 border-accent/50 pl-3 text-sm leading-relaxed">
          {d.abstract}
        </p>
      )}
      {d.what_they_do && (
        <div className="mb-4 text-sm">
          <span className="text-muted">What they do: </span>
          {d.what_they_do}
        </div>
      )}

      {/* Customers → use case */}
      {customers.length > 0 && (
        <div className="mb-4">
          <div className="mb-1 text-[10px] uppercase text-muted">Customers &amp; use cases</div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr>
                  <Th>Customer</Th>
                  <Th>Use case</Th>
                  <Th>Program</Th>
                </tr>
              </thead>
              <tbody>
                {customers.map((c: any, i: number) => (
                  <tr key={i} className="border-t border-border/40 align-top">
                    <Td className="font-medium">
                      {c.customer}
                      <Cite ids={c.citation_ids} />
                    </Td>
                    <Td>{c.use_case || "—"}</Td>
                    <Td className="text-muted">{c.program || "—"}</Td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Supply chain / chokepoint */}
      {sc && (
        <div className="mb-4">
          <div className="mb-1 text-[10px] uppercase text-muted">Supply chain &amp; chokepoint</div>
          <div className="flex flex-wrap items-center gap-2 mb-2">
            {sc.market_share_pct != null && (
              <Badge variant="accent">~{fmtPct(sc.market_share_pct)} share</Badge>
            )}
            {sc.sole_source === true && <Badge variant="pos">Sole-source</Badge>}
            <Cite ids={sc.citation_ids} />
          </div>
          {sc.share_basis && (
            <div className="text-sm mb-1">
              <span className="text-muted">Basis: </span>
              {sc.share_basis}
            </div>
          )}
          {sc.position_note && <div className="text-sm mb-1">{sc.position_note}</div>}
          {sc.geographic_note && (
            <div className="text-sm mb-1 text-muted">{sc.geographic_note}</div>
          )}
          {sc.global_players?.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-1">
              <span className="text-xs text-muted mr-1">Global players:</span>
              {sc.global_players.map((p: string, i: number) => (
                <Badge key={i} variant="muted">
                  {p}
                </Badge>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Recent developments */}
      {devs.length > 0 && (
        <div className="mb-4">
          <div className="mb-1 text-[10px] uppercase text-muted">Recent developments</div>
          <div className="space-y-1">
            {devs.map((dv: any, i: number) => (
              <div key={i} className="flex items-start gap-2 text-sm">
                {dv.date && <span className="text-muted tnum shrink-0">{dv.date}</span>}
                <span>
                  {dv.headline}
                  {dv.amount_usd != null && (
                    <span className="text-pos"> ({fmtUsd(dv.amount_usd)})</span>
                  )}
                  <Cite ids={dv.citation_ids} />
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Verbatim management quotes */}
      {quotes.length > 0 && (
        <div className="mb-4">
          <div className="mb-1 text-[10px] uppercase text-muted">From the filings</div>
          <div className="space-y-2">
            {quotes.map((q: any, i: number) => (
              <blockquote key={i} className="border-l-2 border-border pl-2 text-sm italic">
                “{q.quote}”
                <span className="mt-0.5 block text-[10px] not-italic text-muted">
                  {q.url ? (
                    <a
                      href={q.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="hover:underline"
                    >
                      {q.form} · {q.filing_date} ↗
                    </a>
                  ) : (
                    `${q.form} · ${q.filing_date}`
                  )}
                  <Cite ids={q.citation_id != null ? [q.citation_id] : []} />
                </span>
              </blockquote>
            ))}
          </div>
        </div>
      )}

      {/* Second-order thesis (speculative) */}
      {so?.thesis && (
        <div className="mb-4 rounded-md border border-warn/30 bg-warn/5 p-3">
          <div className="mb-1 flex items-center gap-2">
            <Badge variant="warn">Speculative</Badge>
            <span className="text-[10px] uppercase text-muted">Second-order thesis</span>
          </div>
          <p className="text-sm">
            {so.thesis}
            <Cite ids={so.citation_ids} />
          </p>
          {so.analogs?.length > 0 && (
            <div className="mt-2 text-sm">
              <span className="text-muted">Analogs: </span>
              {so.analogs.join(" · ")}
            </div>
          )}
        </div>
      )}

      {/* References / bibliography */}
      {refs.length > 0 && (
        <div className="mt-4 border-t border-border/40 pt-2">
          <div className="mb-1 text-[10px] uppercase text-muted">References</div>
          <ol className="space-y-1">
            {refs.map((ref: any) => (
              <li key={ref.id} id={`dr-ref-${ref.id}`} className="flex items-start gap-2 text-xs">
                <span className="tnum text-muted shrink-0">[{ref.id}]</span>
                <Badge variant={srcVariant(ref.source_type) as any}>
                  {srcLabel(ref.source_type)}
                </Badge>
                <span className="min-w-0">
                  {ref.url ? (
                    <a
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-muted hover:text-accent hover:underline"
                    >
                      {ref.title || ref.url}
                    </a>
                  ) : (
                    ref.title
                  )}
                  {ref.published_date && (
                    <span className="text-muted"> · {ref.published_date}</span>
                  )}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </Section>
  );
}

// ── Investment Memo ─────────────────────────────────────────────────────────

export function MemoPanel({ r }: { r: ResearchReport }) {
  const m = r.investment_memo;
  return (
    <Section title="Investment Memo" empty={!m}>
      {m?.executive_summary && <p className="text-sm mb-2">{m.executive_summary}</p>}
      {m?.key_insight && (
        <div className="text-sm mb-2">
          <span className="text-muted">Edge: </span>
          {m.key_insight}
        </div>
      )}
      {m?.max_loss_scenario && (
        <div className="text-sm mb-2 text-neg">Max loss: {m.max_loss_scenario}</div>
      )}
      {m?.conviction && (
        <KV
          k="Position sizing"
          v={`${m.conviction} · ${fmtNum(m.position_size_pct_low, 1)}–${fmtNum(
            m.position_size_pct_high,
            1
          )}%`}
        />
      )}
      {m?.key_catalysts?.length > 0 && (
        <div className="mt-2">
          <div className="text-xs uppercase text-muted mb-1">Catalysts</div>
          {m.key_catalysts.map((c: string, i: number) => (
            <div key={i} className="text-sm">
              • {c}
            </div>
          ))}
        </div>
      )}
      {m?.exit_triggers?.length > 0 && (
        <div className="mt-2">
          <div className="text-xs uppercase text-warn mb-1">Exit triggers</div>
          {m.exit_triggers.map((c: string, i: number) => (
            <div key={i} className="text-sm">
              ⚠ {c}
            </div>
          ))}
        </div>
      )}
    </Section>
  );
}

// ── Position recommendation (BUY/SELL/INCREASE/DECREASE/HOLD) ─────────────────

function actionBadgeVariant(action: string): "pos" | "neg" | "muted" {
  if (action === "BUY" || action === "INCREASE") return "pos";
  if (action === "SELL" || action === "DECREASE") return "neg";
  return "muted";
}

export function RecommendationPanel({ symbol }: { symbol: string }) {
  const recommend = useMutation({
    mutationFn: () => api.recommend(symbol),
  });

  const r = recommend.data;

  return (
    <Section
      title="Recommendation"
      right={
        <Button onClick={() => recommend.mutate()} disabled={recommend.isPending} size="sm">
          {recommend.isPending ? "Analyzing…" : r ? "Refresh" : "Get Recommendation"}
        </Button>
      }
    >
      {recommend.isError && (
        <div className="text-sm text-red-400">
          {(recommend.error as Error).message}
        </div>
      )}

      {!r && !recommend.isPending && !recommend.isError && (
        <div className="text-sm text-muted">
          Checks your current position, pulls cached research, and suggests an action.
        </div>
      )}

      {r && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={actionBadgeVariant(r.action)}>{r.action}</Badge>
            <Badge variant="muted">Conviction: {r.conviction}</Badge>
          </div>
          <p className="text-xs text-muted">
            Decision support only — you place any trade yourself.
          </p>

          {r.position.has_position ? (
            <div>
              {r.position.equity_quantity !== 0 && (
                <>
                  <KV
                    k="Position"
                    v={`${fmtNum(r.position.equity_quantity, 0)} sh @ avg ${fmtUsd(
                      r.position.average_open_price
                    )}`}
                  />
                  <KV k="Mark" v={fmtUsd(r.position.mark)} />
                  {r.position.unrealized_pnl_pct != null && (
                    <KV
                      k="Unrealized"
                      v={
                        <span className={pnlColor(r.position.unrealized_pnl_pct)}>
                          {fmtPct(r.position.unrealized_pnl_pct, { sign: true })}
                        </span>
                      }
                    />
                  )}
                </>
              )}
              {r.position.option_legs.map((leg, i) => (
                <KV key={i} k={`Option leg ${i + 1}`} v={leg} />
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted">No existing position.</div>
          )}

          {r.reasoning && <p className="text-sm">{r.reasoning}</p>}

          {r.key_factors.length > 0 && (
            <div>
              <div className="text-xs text-muted mb-1">Key factors</div>
              <ul className="text-sm list-disc list-inside space-y-0.5">
                {r.key_factors.map((f, i) => (
                  <li key={i}>{f}</li>
                ))}
              </ul>
            </div>
          )}

          {r.risks.length > 0 && (
            <div>
              <div className="text-xs text-muted mb-1">Risks</div>
              <ul className="text-sm list-disc list-inside space-y-0.5">
                {r.risks.map((risk, i) => (
                  <li key={i}>{risk}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </Section>
  );
}

// ── Price chart with fundamentals overlay ─────────────────────────────────────

const RANGES = { "1M": 30, "6M": 182, "1Y": 365, "5Y": 365 * 5 } as const;
type Range = keyof typeof RANGES;

/** Daily price line with the live quote appended, plus toggleable P/E (right
 *  axis) and a revenue/EPS sub-panel. Range tabs slice client-side. Renders
 *  nothing for tickers without cached research (404). */
export function PriceChartPanel({ symbol, quotes }: { symbol: string; quotes: QuotesState }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["price-history", symbol],
    queryFn: () => api.priceHistory(symbol),
    retry: false,
    staleTime: 5 * 60 * 1000,
  });

  const [range, setRange] = React.useState<Range>("1Y");
  const [showPe, setShowPe] = React.useState(false);
  const [showFund, setShowFund] = React.useState(false);

  const result = data?.result;
  const live = quotes.quotes[symbol]?.mid;

  // Merge bars + P/E into time-indexed points; append the live tick to today.
  const points = React.useMemo(() => {
    if (!result) return [] as { t: number; close: number; pe: number | null }[];
    const peByDate = new Map(result.pe_series.map((p) => [p.date, p.pe]));
    const rows = result.bars.map((b) => ({
      t: Date.parse(b.date),
      close: b.close,
      pe: peByDate.get(b.date) ?? null,
    }));
    if (live && rows.length) {
      const today = new Date().toISOString().slice(0, 10);
      const lastDate = result.bars[result.bars.length - 1].date;
      if (lastDate === today) {
        rows[rows.length - 1] = { ...rows[rows.length - 1], close: live };
      } else {
        rows.push({ t: Date.parse(today), close: live, pe: rows[rows.length - 1].pe });
      }
    }
    return rows;
  }, [result, live]);

  const view = React.useMemo(() => {
    const cutoff = Date.now() - RANGES[range] * 86400_000;
    return points.filter((p) => p.t >= cutoff);
  }, [points, range]);

  // Earnings dots in the visible window, y-positioned on the nearest price point.
  const dots = React.useMemo(() => {
    if (!result || view.length === 0) return [];
    const cutoff = Date.now() - RANGES[range] * 86400_000;
    return result.earnings
      .map((e) => {
        const t = Date.parse(e.date);
        if (t < cutoff || t > view[view.length - 1].t) return null;
        let best = view[0];
        for (const p of view) if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
        return { t, y: best.close, eps: e.eps, yoy: e.yoy_eps_growth };
      })
      .filter(Boolean) as { t: number; y: number; eps: number | null; yoy: number | null }[];
  }, [result, view, range]);

  if (isError) return null;
  if (!result) {
    return isLoading ? (
      <Section title="Price & fundamentals">
        <div className="text-sm text-muted">Loading price history…</div>
      </Section>
    ) : null;
  }

  const fmtDate = (t: number) =>
    new Date(t).toLocaleDateString(undefined, { month: "short", year: "2-digit" });
  const cutoff = Date.now() - RANGES[range] * 86400_000;
  const fundData = result.fundamentals.filter((f) => Date.parse(f.date) >= cutoff);

  return (
    <Section
      title="Price & fundamentals"
      right={
        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="flex items-center gap-1.5 text-muted">
            P/E <Toggle checked={showPe} onChange={setShowPe} />
          </label>
          <label className="flex items-center gap-1.5 text-muted">
            Rev/EPS <Toggle checked={showFund} onChange={setShowFund} />
          </label>
          <div className="flex overflow-hidden rounded-md border border-border">
            {(Object.keys(RANGES) as Range[]).map((r) => (
              <button
                key={r}
                onClick={() => setRange(r)}
                className={cn(
                  "px-2 py-1",
                  r === range ? "bg-accent text-white" : "text-muted hover:text-text"
                )}
              >
                {r}
              </button>
            ))}
          </div>
        </div>
      }
    >
      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart data={view} margin={{ top: 6, right: 8, left: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="pricefill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#5b8def" stopOpacity={0.35} />
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
            tickFormatter={fmtDate}
          />
          <YAxis
            yAxisId="price"
            {...chartAxis}
            tickLine={false}
            width={48}
            domain={["auto", "auto"]}
            tickFormatter={(v: number) => fmtNum(v, 0)}
          />
          {showPe && (
            <YAxis
              yAxisId="pe"
              orientation="right"
              {...chartAxis}
              tickLine={false}
              width={36}
              tickFormatter={(v: number) => fmtNum(v, 0)}
            />
          )}
          <Tooltip
            contentStyle={{ background: "#121722", border: `1px solid ${grid}` }}
            labelFormatter={(t: any) => new Date(t).toLocaleDateString()}
            formatter={(v: any, name: any) =>
              name === "pe" ? [fmtNum(v, 1), "P/E"] : [fmtNum(v), "Price"]
            }
          />
          <Area
            yAxisId="price"
            type="monotone"
            dataKey="close"
            stroke="#5b8def"
            fill="url(#pricefill)"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          {showPe && (
            <Line
              yAxisId="pe"
              type="monotone"
              dataKey="pe"
              stroke="#f5a524"
              strokeWidth={1.25}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          )}
          {dots.map((d) => (
            <ReferenceDot
              key={d.t}
              yAxisId="price"
              x={d.t}
              y={d.y}
              r={4}
              fill={d.yoy == null ? "#8b97ad" : d.yoy >= 0 ? "#19c37d" : "#ef4444"}
              stroke="#0b0e14"
              strokeWidth={1}
            />
          ))}
        </ComposedChart>
      </ResponsiveContainer>

      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
        {live ? <span className="text-pos">● live {fmtNum(live)}</span> : null}
        <span>● earnings — green = EPS up YoY, red = down</span>
      </div>

      {showFund &&
        (fundData.length > 0 ? (
          <ResponsiveContainer width="100%" height={130}>
            <ComposedChart data={fundData} margin={{ top: 12, right: 8, left: 8, bottom: 0 }}>
              <CartesianGrid stroke={grid} vertical={false} />
              <XAxis dataKey="fiscal_year" {...chartAxis} tickLine={false} />
              <YAxis
                yAxisId="rev"
                {...chartAxis}
                tickLine={false}
                width={48}
                tickFormatter={(v: number) => fmtUsd(v)}
              />
              <YAxis
                yAxisId="eps"
                orientation="right"
                {...chartAxis}
                tickLine={false}
                width={36}
                tickFormatter={(v: number) => fmtNum(v, 1)}
              />
              <Tooltip
                contentStyle={{ background: "#121722", border: `1px solid ${grid}` }}
                formatter={(v: any, name: any) =>
                  name === "eps" ? [fmtNum(v, 2), "EPS"] : [fmtUsd(v), "Revenue"]
                }
              />
              <Bar yAxisId="rev" dataKey="revenue" fill="#3b82f6" radius={[3, 3, 0, 0]} />
              <Line
                yAxisId="eps"
                type="monotone"
                dataKey="eps"
                stroke="#19c37d"
                strokeWidth={1.5}
                dot={{ r: 2 }}
                isAnimationActive={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        ) : (
          <div className="mt-2 text-xs text-muted">No fundamental periods in this range.</div>
        ))}
    </Section>
  );
}
