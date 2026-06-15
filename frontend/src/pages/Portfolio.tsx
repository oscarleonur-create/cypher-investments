import { useMemo } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight, RefreshCw } from "lucide-react";
import { api } from "@/lib/api";
import type { Holding, HoldingsResponse } from "@/lib/types";
import type { QuotesState } from "@/lib/useQuotes";
import { useJob } from "@/lib/useJob";
import { cn, fmtMoney, fmtNum, fmtPct, fmtUpside, fmtUsd, pnlColor } from "@/lib/utils";
import { AttentionDot, Stat, Th, ThesisBadge } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface Row extends Holding {
  price: number;
  isLive: boolean;
  dayChangePct: number | null;
  dayPnl: number;
  openPnl: number;
  mktValue: number;
}

function computeRow(h: Holding, q: QuotesState): Row {
  const live = q.quotes[h.symbol]?.mid || 0;
  const price = live > 0 ? live : h.mark_price || h.close_price || 0;
  const mktValue = price * h.quantity * h.multiplier;
  const openPnl = (price - h.average_open_price) * h.quantity * h.multiplier;
  const dayPnl = h.close_price > 0 ? (price - h.close_price) * h.quantity * h.multiplier : 0;
  const dayChangePct = h.close_price > 0 ? price / h.close_price - 1 : null;
  return {
    ...h,
    price,
    isLive: live > 0,
    dayChangePct,
    dayPnl,
    openPnl,
    mktValue,
  };
}

export default function Portfolio({ quotes }: { quotes: QuotesState }) {
  const { data, isLoading, error, refetch } = useQuery<HoldingsResponse>({
    queryKey: ["holdings"],
    queryFn: api.holdings,
    refetchInterval: 15000,
  });

  const reviewJob = useJob(() => refetch());

  const rows = useMemo(
    () => (data?.holdings || []).map((h) => computeRow(h, quotes)),
    [data, quotes]
  );

  const totals = useMemo(() => {
    let openPnl = 0;
    let dayPnl = 0;
    let mkt = 0;
    for (const r of rows) {
      openPnl += r.openPnl;
      dayPnl += r.dayPnl;
      mkt += r.mktValue;
    }
    return { openPnl, dayPnl, mkt };
  }, [rows]);

  const sorted = useMemo(
    () => [...rows].sort((a, b) => b.mktValue - a.mktValue),
    [rows]
  );

  const flags = useMemo(() => {
    let invalid = 0;
    let caution = 0;
    for (const r of rows) {
      const s = r.research?.thesis_status;
      if (s === "INVALIDATED") invalid++;
      else if (s === "CAUTION") caution++;
    }
    return { invalid, caution };
  }, [rows]);

  const startRefresh = async () => {
    const { job_id } = await api.refreshReview(false);
    reviewJob.start(job_id);
  };

  return (
    <div className="space-y-5">
      {/* Summary bar */}
      <Card className="p-4">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-4">
            <Stat label="Net Liq" value={fmtMoney(data?.balances.net_liq)} />
            <Stat
              label="Day P&L"
              value={
                <span className={pnlColor(totals.dayPnl)}>{fmtMoney(totals.dayPnl, true)}</span>
              }
            />
            <Stat
              label="Open P&L"
              value={
                <span className={pnlColor(totals.openPnl)}>{fmtMoney(totals.openPnl, true)}</span>
              }
            />
            <Stat label="Cash" value={fmtMoney(data?.balances.cash)} />
          </div>
          <div className="flex items-center gap-3">
            <div className="text-right text-xs text-muted">
              <div>
                <span className="text-neg font-medium">{flags.invalid}</span> invalidated ·{" "}
                <span className="text-warn font-medium">{flags.caution}</span> caution
              </div>
              <div>{rows.length} holdings</div>
            </div>
            <Button onClick={startRefresh} disabled={reviewJob.running} variant="outline" size="sm">
              <RefreshCw className={cn("h-4 w-4", reviewJob.running && "animate-spin")} />
              Refresh research
            </Button>
          </div>
        </div>
        {reviewJob.job && reviewJob.job.status !== "done" && (
          <div className="mt-2 text-xs text-muted">
            {reviewJob.job.status === "error"
              ? `Refresh failed: ${reviewJob.job.error}`
              : `Refreshing… ${reviewJob.job.message}`}
          </div>
        )}
      </Card>

      {/* Holdings table */}
      <Card className="overflow-hidden">
        {isLoading ? (
          <div className="p-6 text-sm text-muted">Loading holdings…</div>
        ) : error ? (
          <div className="p-6 text-sm text-neg">
            {(error as Error).message} — check TastyTrade credentials in .env.
          </div>
        ) : rows.length === 0 ? (
          <div className="p-6 text-sm text-muted">No equity holdings found.</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="border-b border-border bg-panel-2/40">
                <tr>
                  <Th>Symbol</Th>
                  <Th className="text-right">Price</Th>
                  <Th className="text-right">Day</Th>
                  <Th className="text-right">Qty</Th>
                  <Th className="text-right">Avg cost</Th>
                  <Th className="text-right">Mkt value</Th>
                  <Th className="text-right">Open P&L</Th>
                  <Th>Thesis</Th>
                  <Th className="text-right">Base ▲</Th>
                  <Th></Th>
                </tr>
              </thead>
              <tbody>
                {sorted.map((r) => (
                  <HoldingRow key={r.symbol} r={r} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function HoldingRow({ r }: { r: Row }) {
  return (
    <tr className="border-b border-border/40 hover:bg-panel-2/40">
      <td className="px-2 py-2">
        <Link to={`/ticker/${r.symbol}`} className="flex items-center gap-2">
          <AttentionDot level={r.research?.attention} />
          <span className="font-semibold">{r.symbol}</span>
          {!r.isLive && (
            <span className="text-[10px] uppercase text-muted" title="Last close (no live tick)">
              close
            </span>
          )}
        </Link>
      </td>
      <td className="px-2 py-2 text-right tnum">{fmtNum(r.price)}</td>
      <td className={cn("px-2 py-2 text-right tnum", pnlColor(r.dayChangePct))}>
        {r.dayChangePct == null ? "—" : fmtPct(r.dayChangePct, { sign: true })}
      </td>
      <td className="px-2 py-2 text-right tnum">{fmtNum(r.quantity, 0)}</td>
      <td className="px-2 py-2 text-right tnum">{fmtNum(r.average_open_price)}</td>
      <td className="px-2 py-2 text-right tnum">{fmtUsd(r.mktValue)}</td>
      <td className={cn("px-2 py-2 text-right tnum", pnlColor(r.openPnl))}>
        {fmtMoney(r.openPnl, true)}
      </td>
      <td className="px-2 py-2">
        <ThesisBadge status={r.research?.thesis_status} />
      </td>
      <td className={cn("px-2 py-2 text-right tnum", pnlColor(r.research?.base_upside))}>
        {fmtUpside(r.research?.base_upside)}
      </td>
      <td className="px-2 py-2 text-right">
        <Link to={`/ticker/${r.symbol}`} className="text-muted hover:text-text">
          <ChevronRight className="h-4 w-4 inline" />
        </Link>
      </td>
    </tr>
  );
}
