import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { OverviewBullet } from "@/lib/types";
import { cn, fmtEt } from "@/lib/utils";
import { Card } from "@/components/ui/card";

// The dot carries the tone; the text carries the meaning, so the card reads
// the same without colour.
const DOT: Record<OverviewBullet["tone"], string> = {
  pos: "bg-pos",
  neg: "bg-neg",
  warn: "bg-warn",
  neutral: "bg-muted/50",
};

/** Where the company stands today, in a few lines, each with its source.
 *  A summary of the page below it: nothing here is computed anywhere else. */
export function OverviewCard({ symbol }: { symbol: string }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ["overview", symbol],
    queryFn: () => api.overview(symbol),
    staleTime: 5 * 60_000,
  });

  if (isLoading) {
    return <Card className="p-4 text-sm text-muted">Loading overview…</Card>;
  }
  if (error || !data) {
    return (
      <Card className="p-4 text-sm text-muted">
        Overview unavailable{error ? `: ${(error as Error).message}` : ""}.
      </Card>
    );
  }

  return (
    <Card className="p-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-semibold uppercase tracking-wide text-muted">
          Overview
        </span>
        <span className="text-xs text-muted">as of {fmtEt(data.as_of)}</span>
      </div>
      <ul className="space-y-2">
        {data.bullets.map((b) => (
          <li
            key={b.topic}
            className="grid grid-cols-[10px_minmax(0,1fr)] gap-x-2 sm:grid-cols-[10px_6.5rem_minmax(0,1fr)]"
          >
            <span
              className={cn("mt-[7px] h-2 w-2 rounded-full", DOT[b.tone])}
              aria-hidden="true"
            />
            <span className="text-xs uppercase tracking-wide text-muted sm:pt-0.5">
              {b.topic}
            </span>
            <span className="col-start-2 min-w-0 text-sm sm:col-start-3">
              {b.text}
              <span className="ml-1 text-xs text-muted opacity-70">— {b.source}</span>
            </span>
          </li>
        ))}
      </ul>
      {data.gaps.length > 0 && (
        <div className="mt-3 text-xs text-muted">
          {data.gaps.map((g) => (
            <div key={g}>Missing: {g}</div>
          ))}
        </div>
      )}
    </Card>
  );
}
