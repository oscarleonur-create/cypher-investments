import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ResearchReport } from "@/lib/types";
import { fmtNum, fmtPct, fmtPriceSafe, fmtUpside, fmtUsd, pnlColor } from "@/lib/utils";
import { Badge } from "./ui/badge";
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
          {(insiders?.transactions || []).slice(0, 5).map((t: any, i: number) => (
            <KV
              key={i}
              k={`${t.insider_name} (${t.transaction_type})`}
              v={t.value_usd ? fmtUsd(t.value_usd) : `${fmtNum(t.shares, 0)} sh`}
            />
          ))}
        </div>
      </div>
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

export function CatalystsPanel({ r }: { r: ResearchReport }) {
  const cr = r.catalyst_risk;
  const catalysts = cr?.catalysts || [];
  const risks = cr?.risks || [];
  return (
    <Section title="Catalysts & Risks" empty={catalysts.length === 0 && risks.length === 0}>
      {catalysts.length > 0 && (
        <div className="mb-3 space-y-1">
          {catalysts.map((c: any, i: number) => (
            <div key={i} className="flex items-center gap-2 text-sm">
              <Badge variant={c.is_near_term ? "warn" : "muted"}>{c.catalyst_type}</Badge>
              <span className="text-muted tnum">{c.expected_date}</span>
              <span>{c.description}</span>
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
