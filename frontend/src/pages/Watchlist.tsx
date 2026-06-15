import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Plus, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { WatchlistItem, WatchlistResponse } from "@/lib/types";
import type { QuotesState } from "@/lib/useQuotes";
import { useJob } from "@/lib/useJob";
import { cn, fmtNum, fmtUpside, pnlColor } from "@/lib/utils";
import { AttentionDot, Th, ThesisBadge } from "@/components/common";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

export default function Watchlist({ quotes }: { quotes: QuotesState }) {
  const { data, isLoading, error, refetch } = useQuery<WatchlistResponse>({
    queryKey: ["watchlist"],
    queryFn: api.watchlist,
    refetchInterval: 15000,
  });

  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const buildJob = useJob(() => refetch());

  const items = data?.watchlist || [];

  const add = async () => {
    const sym = input.trim().toUpperCase();
    if (!sym || busy) return;
    setBusy(true);
    try {
      const { job_id } = await api.addWatchlist(sym);
      setInput("");
      await refetch();
      buildJob.start(job_id);
    } catch (err) {
      alert((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async (sym: string) => {
    await api.removeWatchlist(sym);
    refetch();
  };

  return (
    <div className="space-y-5">
      <Card className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold">Watchlist</div>
            <div className="text-xs text-muted">
              Tickers you follow. Adding one builds the full research report.
            </div>
          </div>
          <form
            className="flex items-center gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              add();
            }}
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value.toUpperCase())}
              placeholder="Add ticker…"
              maxLength={8}
              className="h-8 w-32 rounded-lg border border-border bg-transparent px-3 text-sm uppercase outline-none focus:border-accent"
            />
            <Button type="submit" size="sm" disabled={busy || !input.trim()}>
              <Plus className="h-4 w-4" />
              Add
            </Button>
          </form>
        </div>
        {buildJob.job && buildJob.job.status !== "done" && (
          <div className="mt-2 text-xs text-muted">
            {buildJob.job.status === "error"
              ? `Build failed: ${buildJob.job.error}`
              : `Building ${buildJob.job.target}… ${buildJob.job.message}`}
          </div>
        )}
      </Card>

      <Card className="overflow-hidden">
        {isLoading ? (
          <div className="p-6 text-sm text-muted">Loading watchlist…</div>
        ) : error ? (
          <div className="p-6 text-sm text-neg">{(error as Error).message}</div>
        ) : items.length === 0 ? (
          <div className="p-6 text-sm text-muted">
            No tickers yet. Add one above to start following it.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="border-b border-border bg-panel-2/40">
                <tr>
                  <Th>Symbol</Th>
                  <Th className="text-right">Price</Th>
                  <Th className="text-right">Fair price</Th>
                  <Th className="text-right">Upside</Th>
                  <Th>Thesis</Th>
                  <Th></Th>
                </tr>
              </thead>
              <tbody>
                {items.map((it) => (
                  <WatchRow key={it.symbol} it={it} quotes={quotes} onRemove={remove} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function WatchRow({
  it,
  quotes,
  onRemove,
}: {
  it: WatchlistItem;
  quotes: QuotesState;
  onRemove: (sym: string) => void;
}) {
  const r = it.research;
  const live = quotes.quotes[it.symbol]?.mid || 0;
  const price = live > 0 ? live : r?.current_price || 0;
  const pending = !r?.has_report;

  return (
    <tr className="border-b border-border/40 hover:bg-panel-2/40">
      <td className="px-2 py-2">
        <Link to={`/ticker/${it.symbol}`} className="flex items-center gap-2">
          <AttentionDot level={r?.attention} />
          <span className="font-semibold">{it.symbol}</span>
          {pending && (
            <span className="text-[10px] uppercase text-muted" title="Research not built yet">
              building…
            </span>
          )}
        </Link>
      </td>
      <td className="px-2 py-2 text-right tnum">{price > 0 ? fmtNum(price) : "—"}</td>
      <td className="px-2 py-2 text-right tnum">
        {r?.fair_price ? fmtNum(r.fair_price) : "—"}
      </td>
      <td className={cn("px-2 py-2 text-right tnum", pnlColor(r?.fair_upside))}>
        {fmtUpside(r?.fair_upside)}
      </td>
      <td className="px-2 py-2">
        <ThesisBadge status={r?.thesis_status} />
      </td>
      <td className="px-2 py-2 text-right">
        <div className="flex items-center justify-end gap-1">
          <button
            onClick={() => onRemove(it.symbol)}
            title="Remove from watchlist"
            className="text-muted hover:text-neg"
          >
            <Trash2 className="h-4 w-4" />
          </button>
          <Link to={`/ticker/${it.symbol}`} className="text-muted hover:text-text">
            <ChevronRight className="h-4 w-4 inline" />
          </Link>
        </div>
      </td>
    </tr>
  );
}
