import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { PriceCase, PriceRange } from "@/lib/types";
import { cn, fmtPct } from "@/lib/utils";
import { Section, Td, Th } from "./common";
import { Badge } from "./ui/badge";

const CASE_LABEL: Record<PriceCase["name"], string> = {
  bear: "Bear",
  base: "Base",
  bull: "Bull",
  market: "Market",
};

const VERDICT: Record<NonNullable<PriceRange["verdict"]>, { label: string; variant: string }> = {
  above_bull: { label: "Above bull", variant: "neg" },
  above_base: { label: "Base – bull", variant: "warn" },
  above_bear: { label: "Bear – base", variant: "pos" },
  below_bear: { label: "Below bear", variant: "pos" },
};

const money = (v: number) =>
  `$${v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

function growthText(c: PriceCase): string {
  if (c.growth == null) return "—";
  const held = c.held_years ? `${c.held_years}y` : "fades now";
  return `${fmtPct(c.growth, { sign: true })} · ${held}`;
}

/** Bear, base and bull on one line, with the price marked. Identity is carried
 *  by the labels, not by colour; the price is the only accent. */
function RangeBar({ cases, price }: { cases: PriceCase[]; price: number }) {
  const scenarios = cases.filter((c) => c.name !== "market");
  if (scenarios.length === 0) return null;
  const values = [...scenarios.map((c) => c.value_per_share), price];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = Math.max(hi - lo, 1e-9);
  const pos = (v: number) => 4 + ((v - lo) / span) * 92; // keep end labels inside
  const bear = scenarios.find((c) => c.name === "bear");
  const bull = scenarios.find((c) => c.name === "bull");
  return (
    <div className="mb-4 mt-6" role="img" aria-label="Value range against the price">
      <div className="relative h-10">
        {bear && bull && (
          <div
            className="absolute top-4 h-2 rounded-full bg-panel-2"
            style={{
              left: `${pos(Math.min(bear.value_per_share, bull.value_per_share))}%`,
              width: `${Math.abs(pos(bull.value_per_share) - pos(bear.value_per_share))}%`,
            }}
          />
        )}
        {scenarios.map((c) => (
          <div
            key={c.name}
            className="absolute top-3 -translate-x-1/2 text-center"
            style={{ left: `${pos(c.value_per_share)}%` }}
            title={`${CASE_LABEL[c.name]} ${money(c.value_per_share)}`}
          >
            <div className="mx-auto h-4 w-4 rounded-full border-2 border-panel bg-muted" />
            <div className="mt-1 whitespace-nowrap text-[10px] text-muted tnum">
              {CASE_LABEL[c.name]}
            </div>
          </div>
        ))}
        <div
          className="absolute -top-3 -translate-x-1/2 text-center"
          style={{ left: `${pos(price)}%` }}
          title={`Price ${money(price)}`}
        >
          <div className="whitespace-nowrap text-[10px] font-medium text-accent tnum">
            {money(price)}
          </div>
          <div className="mx-auto h-8 w-0.5 bg-accent" />
        </div>
      </div>
    </div>
  );
}

/** The one pricing card: the value range from the company's own filed
 *  margins, what the price itself assumes, and the reasoning in plain words.
 *  Every sentence is filled from the valuation snapshot — no model writes it. */
export function PriceRangeCard({ symbol }: { symbol: string }) {
  const client = useQueryClient();
  const key = ["price-range", symbol];
  const { data, isLoading, error } = useQuery({
    queryKey: key,
    queryFn: () => api.priceRange(symbol),
    enabled: Boolean(symbol),
    staleTime: 10 * 60_000,
    retry: false,
  });
  const [refreshing, setRefreshing] = useState(false);

  const refresh = async () => {
    setRefreshing(true);
    try {
      client.setQueryData<PriceRange>(key, await api.priceRange(symbol, true));
    } finally {
      setRefreshing(false);
    }
  };

  const right = (
    <div className="flex items-center gap-2">
      {data?.verdict && (
        <Badge variant={VERDICT[data.verdict].variant as any}>{VERDICT[data.verdict].label}</Badge>
      )}
      <button
        onClick={refresh}
        disabled={refreshing}
        title="Recompute now from the filings and the latest close"
        className="text-muted hover:text-text disabled:opacity-50"
      >
        <RefreshCw className={cn("h-4 w-4", refreshing && "animate-spin")} />
      </button>
    </div>
  );

  if (isLoading) {
    return (
      <Section title="Price range" right={right}>
        <div className="text-sm text-muted">Valuing from the filings…</div>
      </Section>
    );
  }
  if (error || !data) {
    return (
      <Section title="Price range" right={right}>
        <div className="text-sm text-muted">{(error as Error)?.message ?? "No valuation."}</div>
      </Section>
    );
  }

  const order: PriceCase["name"][] = ["bear", "base", "bull", "market"];
  const cases = [...data.cases].sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name));

  return (
    <Section title="Price range" right={right}>
      <div className="text-xs text-muted">
        {money(data.price)}
        {data.price_source && ` · ${data.price_source}`} ·{" "}
        {data.live ? "computed now" : `weekly valuation ${data.asof}`}
        {data.stale && <span className="text-warn"> · stale balance sheet</span>}
      </div>

      {data.refused ? (
        <div className="mt-3 text-sm text-warn">No value range: {data.refused}.</div>
      ) : (
        <RangeBar cases={cases} price={data.price} />
      )}

      {cases.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr>
                <Th>Case</Th>
                <Th className="text-right">Growth · held</Th>
                <Th className="text-right">FCF margin</Th>
                <Th className="text-right">Value</Th>
                <Th className="text-right">vs price</Th>
              </tr>
            </thead>
            <tbody>
              {cases.map((c) => (
                <tr
                  key={c.name}
                  className={cn("border-t border-border/40", c.name === "market" && "text-accent")}
                  title={c.margin_label}
                >
                  <Td>{CASE_LABEL[c.name]}</Td>
                  <Td className="text-right tnum">{growthText(c)}</Td>
                  <Td className="text-right tnum">{c.margin != null ? fmtPct(c.margin) : "—"}</Td>
                  <Td className="text-right tnum">{money(c.value_per_share)}</Td>
                  <Td className="text-right tnum">
                    {c.name === "market" ? "= price" : fmtPct(c.upside, { sign: true })}
                  </Td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* The rationale: why the range is where it is, sentence by sentence. */}
      <div className="mt-4 rounded-md border border-border/60 bg-panel-2 p-3">
        <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">Rationale</div>
        <div className="space-y-1.5 text-sm leading-relaxed">
          {data.rationale.map((line, i) => (
            <p key={i}>{line}</p>
          ))}
        </div>
      </div>

      <div className="mt-2 break-words text-xs text-muted">
        {data.assumptions}. Market = what today's price assumes; bear / base / bull = the
        company's own filed margins. An opinion, not advice.
      </div>
    </Section>
  );
}
