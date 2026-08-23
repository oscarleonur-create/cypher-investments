import * as React from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import type {
  BayesianOverrides,
  BayesianPriceResult,
  CatalystScenario,
  EcosystemFactor,
  EvidenceSignal,
  PriorDriver,
  ResearchReport,
} from "@/lib/types";
import type { QuotesState } from "@/lib/useQuotes";
import { cn, fmtNum, fmtPct, fmtPriceSafe, fmtUpside, fmtUsd, pnlColor } from "@/lib/utils";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { Slider, Toggle } from "./ui/slider";
import { KV, Section, Stat, Td, Th, ThesisBadge } from "./common";

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
  const scenarios = [t?.bear, t?.base, t?.bull].filter(Boolean);
  return (
    <Section title="Thesis & Edge" empty={!t && !vp && !c}>
      {t?.summary && <p className="text-sm mb-3">{t.summary}</p>}
      <div className="flex flex-wrap gap-2 mb-3">
        {t?.conviction && <Badge variant="accent">Conviction: {t.conviction}</Badge>}
        {vp?.mispricing_type && vp.mispricing_type !== "none" && (
          <Badge variant="muted">{String(vp.mispricing_type).replace(/_/g, " ")}</Badge>
        )}
      </div>

      {scenarios.length > 0 && (
        <div className="overflow-x-auto mb-3">
          <table className="w-full">
            <thead>
              <tr>
                <Th>Scenario</Th>
                <Th className="text-right">Prob</Th>
                <Th className="text-right">Target</Th>
                <Th className="text-right">Upside</Th>
              </tr>
            </thead>
            <tbody>
              {scenarios.map((s: any) => (
                <tr key={s.scenario} className="border-t border-border/40">
                  <Td className="capitalize">{s.scenario}</Td>
                  <Td className="text-right">{fmtPct(s.probability)}</Td>
                  <Td className="text-right">{fmtPriceSafe(s.target_price)}</Td>
                  <Td className={`text-right ${pnlColor(s.upside_pct)}`}>
                    {fmtUpside(s.upside_pct)}
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {vp?.our_key_insight && (
        <div className="text-sm mb-2">
          <span className="text-muted">Edge: </span>
          {vp.our_key_insight}
        </div>
      )}
      {c && (
        <div className="mt-2">
          <KV
            k={`Sell-side (${c.n_analysts} analysts)`}
            v={`${c.recommendation_key || "—"} · target ${
              c.target_price_mean ? fmtNum(c.target_price_mean) : "—"
            }`}
          />
          {c.consensus_upside_pct != null && (
            <KV k="Consensus upside" v={fmtPct(c.consensus_upside_pct, { sign: true })} />
          )}
        </div>
      )}
    </Section>
  );
}

// ── Valuation ───────────────────────────────────────────────────────────────

export function FairPricePanel({ r }: { r: ResearchReport }) {
  const fp = r.fair_price;
  const ok = fp && Number.isFinite(fp.fair_price) && fp.fair_price > 0;
  if (!ok) {
    return (
      <Section title="Fair Price" empty>
        {null}
      </Section>
    );
  }

  const conf = (fp.confidence || "LOW").toUpperCase();
  const confVariant = conf === "HIGH" ? "pos" : conf === "MEDIUM" ? "warn" : "muted";

  // Position markers on the low→high range bar (clamped to 0–100%).
  const span = Math.max(fp.high - fp.low, 1e-9);
  const pos = (v: number) => Math.min(100, Math.max(0, ((v - fp.low) / span) * 100));

  return (
    <Section
      title="Fair Price"
      right={<Badge variant={confVariant as any}>{conf} confidence</Badge>}
    >
      <div className="flex items-end justify-between gap-4">
        <Stat label="Fair value" value={fmtNum(fp.fair_price)} sub={`Current ${fmtNum(fp.current_price)}`} />
        <div className="text-right">
          <div className="text-xs uppercase tracking-wide text-muted">Upside</div>
          <div className={cn("text-lg font-semibold tnum", pnlColor(fp.upside_pct))}>
            {fmtUpside(fp.upside_pct)}
          </div>
        </div>
      </div>

      {/* Range bar: low ── current(▏) ── fair(●) ── high */}
      <div className="mt-4">
        <div className="relative h-2 rounded-full bg-panel-2">
          <div
            className="absolute top-1/2 h-3 w-0.5 -translate-y-1/2 bg-muted"
            style={{ left: `${pos(fp.current_price)}%` }}
            title={`Current ${fmtNum(fp.current_price)}`}
          />
          <div
            className="absolute top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent"
            style={{ left: `${pos(fp.fair_price)}%` }}
            title={`Fair ${fmtNum(fp.fair_price)}`}
          />
        </div>
        <div className="mt-1 flex justify-between text-xs text-muted tnum">
          <span>{fmtNum(fp.low)}</span>
          <span>{fmtNum(fp.high)}</span>
        </div>
      </div>

      {/* Per-method breakdown */}
      <div className="mt-4 space-y-1.5">
        {(fp.methods || []).map((m: any) => (
          <div key={m.name} className="flex items-center gap-2 text-sm">
            <span className="w-32 shrink-0 text-muted">{m.label}</span>
            <div className="h-1.5 flex-1 rounded-full bg-panel-2">
              <div
                className="h-1.5 rounded-full bg-accent/70"
                style={{ width: `${Math.round((m.weight || 0) * 100)}%` }}
              />
            </div>
            <span className="w-16 shrink-0 text-right tnum">{fmtNum(m.estimate)}</span>
            <span className="w-10 shrink-0 text-right tnum text-muted">
              {Math.round((m.weight || 0) * 100)}%
            </span>
          </div>
        ))}
      </div>
      {fp.note && <div className="mt-3 text-xs text-muted">{fp.note}</div>}
    </Section>
  );
}

export function ValuationPanel({ r }: { r: ResearchReport }) {
  const dcf = r.dcf;
  const m = r.multiples;
  const data = dcf
    ? [dcf.bear, dcf.base, dcf.bull]
        .filter(Boolean)
        .map((s: any) => ({
          name: s.assumptions?.scenario || "?",
          price: s.implied_price,
          upside: s.upside_pct,
        }))
    : [];
  const peers = m ? [m.subject, ...(m.peers || [])].filter(Boolean) : [];
  const dcfSane = data.every(
    (d) => Number.isFinite(d.price) && d.price > 0 && d.price < 1_000_000
  );

  return (
    <Section title="Valuation" empty={!dcf && !m}>
      {dcf && (
        <div className="mb-4">
          <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted mb-2">
            <span>Current {fmtNum(dcf.current_price)}</span>
            <span>WACC {fmtPct(dcf.wacc)}</span>
            {dcf.beta != null && <span>β {fmtNum(dcf.beta)}</span>}
            {dcf.implied_growth_rate != null && (
              <span>Implied growth {fmtPct(dcf.implied_growth_rate)}</span>
            )}
          </div>
          {data.length > 0 && dcfSane ? (
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={data} margin={{ top: 8, right: 8, left: 8, bottom: 0 }}>
                <CartesianGrid stroke={grid} vertical={false} />
                <XAxis dataKey="name" {...chartAxis} tickLine={false} />
                <YAxis {...chartAxis} tickLine={false} width={48} />
                <Tooltip
                  contentStyle={{ background: "#121722", border: `1px solid ${grid}` }}
                  formatter={(v: any, n: any) =>
                    n === "price" ? fmtNum(v) : fmtPct(v, { sign: true })
                  }
                />
                <Bar dataKey="price" radius={[4, 4, 0, 0]}>
                  {data.map((d, i) => (
                    <Cell key={i} fill={d.upside >= 0 ? "#19c37d" : "#ef4444"} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : data.length > 0 ? (
            <div className="text-sm text-muted">
              DCF output is out of plausible range — rebuild research to recompute.
              <div className="mt-1 flex gap-4">
                {data.map((d) => (
                  <span key={d.name} className="capitalize">
                    {d.name}: {fmtPriceSafe(d.price)} ({fmtUpside(d.upside)})
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      )}

      {peers.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <Th>Ticker</Th>
                <Th className="text-right">Mkt cap</Th>
                <Th className="text-right">Rev gr</Th>
                <Th className="text-right">Op mgn</Th>
                <Th className="text-right">EV/EBITDA</Th>
                <Th className="text-right">P/E</Th>
                <Th className="text-right">P/FCF</Th>
              </tr>
            </thead>
            <tbody>
              {peers.map((p: any, i: number) => (
                <tr
                  key={p.symbol + i}
                  className={`border-t border-border/40 ${
                    p.symbol === m.subject?.symbol ? "font-semibold" : ""
                  }`}
                >
                  <Td>{p.symbol}</Td>
                  <Td className="text-right">{fmtUsd(p.market_cap)}</Td>
                  <Td className="text-right">{fmtPct(p.revenue_growth_yoy, { sign: true })}</Td>
                  <Td className="text-right">{fmtPct(p.operating_margin)}</Td>
                  <Td className="text-right">{p.ev_to_ebitda ? fmtNum(p.ev_to_ebitda) : "—"}</Td>
                  <Td className="text-right">{p.pe_trailing ? fmtNum(p.pe_trailing) : "—"}</Td>
                  <Td className="text-right">{p.p_to_fcf ? fmtNum(p.p_to_fcf) : "—"}</Td>
                </tr>
              ))}
            </tbody>
          </table>
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
      <div className="grid gap-4 md:grid-cols-2">
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
        <div className="grid gap-4 md:grid-cols-2 mt-3">
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

// ── Bayesian fair value (interactive what-if) ─────────────────────────────────

function ConnGroup({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <details className="border-t border-border/40 py-2" open>
      <summary className="cursor-pointer list-none text-xs font-medium uppercase tracking-wide text-muted">
        {title}
      </summary>
      <div className="mt-2 space-y-1">{children}</div>
    </details>
  );
}

function driverDisplay(unit: string, v: number): string {
  if (unit === "x") return `${v.toFixed(1)}×`;
  if (unit === "ratio") return v.toFixed(2);
  return `${(v * 100).toFixed(1)}%`;
}

function driverStep(unit: string): number {
  return unit === "x" ? 0.5 : 0.005;
}

function DriverRow({
  d,
  mean,
  std,
  onMean,
  onStd,
}: {
  d: PriorDriver;
  mean: number;
  std: number;
  onMean: (v: number) => void;
  onStd: (v: number) => void;
}) {
  const stdMax = Math.max((d.max - d.min) / 3, driverStep(d.unit));
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="text-muted">{d.label}</span>
        <span className="tnum">
          {driverDisplay(d.unit, mean)}
          <span className="ml-1 text-xs text-muted">± {driverDisplay(d.unit, std)}</span>
        </span>
      </div>
      <Slider value={mean} min={d.min} max={d.max} step={driverStep(d.unit)} onChange={onMean} />
      <div className="mt-1 flex items-center gap-2">
        <span className="w-16 shrink-0 text-[10px] uppercase text-muted">Uncertainty</span>
        <Slider value={std} min={0} max={stdMax} step={stdMax / 100} onChange={onStd} />
      </div>
    </div>
  );
}

function EvidenceRow({
  e,
  weight,
  onWeight,
}: {
  e: EvidenceSignal;
  weight: number;
  onWeight: (v: number) => void;
}) {
  return (
    <div className="py-1.5">
      <div className="flex items-baseline justify-between gap-2 text-sm">
        <span className="truncate">{e.label}</span>
        <span className="tnum shrink-0 text-xs text-muted">weight {(weight * 100).toFixed(0)}%</span>
      </div>
      {e.note && <div className="text-xs text-muted">{e.note}</div>}
      <Slider value={weight} min={0} max={1} step={0.05} onChange={onWeight} />
    </div>
  );
}

function ScenarioEvTable({
  scenarios,
  currentPrice,
}: {
  scenarios: CatalystScenario[];
  currentPrice: number;
}) {
  const totalP = scenarios.reduce((s, sc) => s + sc.probability, 0);
  const ev = totalP > 0 ? scenarios.reduce((s, sc) => s + sc.probability * sc.target_price, 0) / totalP : 0;
  const evUpside = currentPrice > 0 ? ev / currentPrice - 1 : 0;

  return (
    <div className="mt-4">
      <div className="mb-1.5 text-xs font-medium uppercase tracking-wide text-muted">
        Scenario probability table
      </div>
      <table className="w-full text-xs tnum">
        <thead>
          <tr className="border-b border-border text-muted">
            <th className="pb-1 text-left font-normal">Scenario</th>
            <th className="pb-1 text-right font-normal">P</th>
            <th className="pb-1 text-right font-normal">Target</th>
            <th className="pb-1 text-right font-normal">Upside</th>
          </tr>
        </thead>
        <tbody>
          {scenarios.map((sc, i) => {
            const lbl = sc.label.toLowerCase();
            const isBull = lbl.includes("bull");
            const isBear = lbl.includes("bear");
            const rowColor = isBull ? "text-pos" : isBear ? "text-neg" : "";
            return (
              <tr key={i} className={`border-b border-border/40 ${rowColor}`}>
                <td className="py-1 pr-2 max-w-[180px] truncate">{sc.label}</td>
                <td className="py-1 text-right">{fmtPct(sc.probability)}</td>
                <td className="py-1 text-right">{fmtPriceSafe(sc.target_price)}</td>
                <td className={`py-1 text-right ${pnlColor(sc.upside_pct)}`}>
                  {fmtUpside(sc.upside_pct)}
                </td>
              </tr>
            );
          })}
        </tbody>
        <tfoot>
          <tr className="font-semibold">
            <td className="pt-2">Expected value (EV)</td>
            <td className="pt-2 text-right text-muted">{fmtPct(totalP)}</td>
            <td className="pt-2 text-right">{fmtPriceSafe(ev)}</td>
            <td className={`pt-2 text-right ${pnlColor(evUpside)}`}>{fmtUpside(evUpside)}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

export function BayesianPricingPanel({ symbol }: { symbol: string }) {
  const baseline = useQuery({
    queryKey: ["bayesian", symbol],
    queryFn: () => api.bayesian(symbol),
    retry: false,
  });

  const [overrides, setOverrides] = React.useState<BayesianOverrides>({});
  const [result, setResult] = React.useState<BayesianPriceResult | null>(null);

  // Seed the result + clear overrides whenever the baseline (re)loads.
  React.useEffect(() => {
    if (baseline.data) {
      setResult(baseline.data.result);
      setOverrides({});
    }
  }, [baseline.data]);

  const recompute = useMutation({
    mutationFn: (ov: BayesianOverrides) => api.recomputeBayesian(symbol, ov),
    onSuccess: (d) => setResult(d.result),
  });

  const hasOverrides = Object.values(overrides).some(
    (v) => v && typeof v === "object" && Object.keys(v).length > 0
  );

  // Debounce recompute so dragging a slider doesn't spam the backend.
  React.useEffect(() => {
    if (!hasOverrides) return;
    const t = setTimeout(() => recompute.mutate(overrides), 250);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overrides]);

  // Non-holding (404) → render nothing so the ticker page stays clean.
  if (baseline.isError) return null;
  if (!result || result.drivers.length === 0) {
    return baseline.isLoading ? (
      <Section title="Bayesian fair value">
        <div className="text-sm text-muted">Computing posterior…</div>
      </Section>
    ) : null;
  }

  const cur = result.current_price;

  const driverMean = (d: PriorDriver) => overrides.driver_mean?.[d.key] ?? d.mean;
  const driverStd = (d: PriorDriver) => overrides.driver_std?.[d.key] ?? d.std;
  const evWeight = (e: EvidenceSignal) => overrides.evidence_weight?.[e.key] ?? e.weight;
  const ecoActive = (f: EcosystemFactor) => overrides.ecosystem_active?.[f.key] ?? f.active;

  const setDriverMean = (key: string, v: number) =>
    setOverrides((o) => ({ ...o, driver_mean: { ...o.driver_mean, [key]: v } }));
  const setDriverStd = (key: string, v: number) =>
    setOverrides((o) => ({ ...o, driver_std: { ...o.driver_std, [key]: v } }));
  const setEvWeight = (key: string, v: number) =>
    setOverrides((o) => ({ ...o, evidence_weight: { ...o.evidence_weight, [key]: v } }));
  const setEcoActive = (key: string, v: boolean) =>
    setOverrides((o) => ({ ...o, ecosystem_active: { ...o.ecosystem_active, [key]: v } }));

  return (
    <Section
      title="Bayesian fair value"
      right={
        hasOverrides ? (
          <Button variant="ghost" size="sm" onClick={() => setOverrides({})}>
            Reset
          </Button>
        ) : null
      }
    >
      {result.note ? (
        <div
          className={cn(
            "mb-3 rounded-md border px-3 py-2 text-xs leading-relaxed",
            result.meaningful
              ? "border-amber-500/30 bg-amber-500/5 text-amber-300/90"
              : "border-amber-500/40 bg-amber-500/10 text-amber-200"
          )}
        >
          {!result.meaningful && <span className="font-medium">DCF not meaningful · </span>}
          {result.note}
        </div>
      ) : null}

      <div
        className={cn(
          "mb-3 grid grid-cols-2 gap-3 sm:grid-cols-4",
          !result.meaningful && "opacity-50"
        )}
      >
        <Stat
          label="Median fair value"
          value={fmtPriceSafe(result.median_price)}
          sub={`now ${fmtNum(cur)}`}
        />
        <Stat
          label="P(undervalued)"
          value={fmtPct(result.prob_undervalued)}
          className={pnlColor(result.prob_undervalued - 0.5)}
        />
        <Stat
          label="Exp. upside"
          value={fmtUpside(result.expected_upside_pct)}
          className={pnlColor(result.expected_upside_pct)}
        />
        <Stat
          label="90% interval"
          value={`${fmtPriceSafe(result.p5)} – ${fmtPriceSafe(result.p95)}`}
        />
      </div>

      <ResponsiveContainer width="100%" height={160}>
        <AreaChart data={result.histogram} margin={{ top: 6, right: 8, left: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="bayesfill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#5b8def" stopOpacity={0.5} />
              <stop offset="100%" stopColor="#5b8def" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis
            type="number"
            dataKey="x"
            domain={["dataMin", "dataMax"]}
            {...chartAxis}
            tickLine={false}
            tickFormatter={(v: number) => fmtNum(v, 0)}
          />
          <YAxis {...chartAxis} tickLine={false} width={32} />
          <Tooltip
            contentStyle={{ background: "#121722", border: `1px solid ${grid}` }}
            labelFormatter={(v: any) => `~${fmtNum(v)}`}
            formatter={(v: any) => [v, "draws"]}
          />
          <Area type="monotone" dataKey="count" stroke="#5b8def" fill="url(#bayesfill)" />
          <ReferenceLine
            x={cur}
            stroke="#8b97ad"
            strokeDasharray="4 3"
            label={{ value: "now", fill: "#8b97ad", fontSize: 10, position: "top" }}
          />
          <ReferenceLine
            x={result.median_price}
            stroke="#19c37d"
            label={{ value: "fair", fill: "#19c37d", fontSize: 10, position: "top" }}
          />
        </AreaChart>
      </ResponsiveContainer>

      <p className="mb-2 mt-1 text-xs text-muted">
        {result.n_draws.toLocaleString()} Monte-Carlo draws · adjust the connections below to see
        fair value shift{recompute.isPending ? " · recomputing…" : ""}.
      </p>

      <ConnGroup title="Valuation drivers (priors)">
        {result.drivers.map((d) => (
          <DriverRow
            key={d.key}
            d={d}
            mean={driverMean(d)}
            std={driverStd(d)}
            onMean={(v) => setDriverMean(d.key, v)}
            onStd={(v) => setDriverStd(d.key, v)}
          />
        ))}
      </ConnGroup>

      {result.evidence.length > 0 && (
        <ConnGroup title="Evidence signals (weights)">
          {result.evidence.map((e) => (
            <EvidenceRow
              key={e.key}
              e={e}
              weight={evWeight(e)}
              onWeight={(v) => setEvWeight(e.key, v)}
            />
          ))}
        </ConnGroup>
      )}

      {result.ecosystem.length > 0 && (
        <ConnGroup title="Ecosystem network (what-if)">
          {result.ecosystem.map((f) => (
            <div key={f.key} className="flex items-center justify-between gap-3 py-1.5">
              <div className="min-w-0">
                <div className="truncate text-sm">{f.label}</div>
                <div className="text-xs text-muted">
                  {f.kind} · {f.note}
                </div>
              </div>
              <Toggle checked={ecoActive(f)} onChange={(v) => setEcoActive(f.key, v)} />
            </div>
          ))}
        </ConnGroup>
      )}

      {result.catalyst_scenarios && result.catalyst_scenarios.length > 0 && (
        <ScenarioEvTable scenarios={result.catalyst_scenarios} currentPrice={result.current_price} />
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
        <div className="flex items-center gap-3 text-xs">
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
