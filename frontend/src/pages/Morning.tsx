import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { MorningCard, MorningItem } from "@/lib/types";
import { cn, fmtEt, fmtPct, pnlColor } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

const CALL: Record<string, "neg" | "warn" | "pos" | "muted"> = {
  EXIT: "neg",
  TRIM: "warn",
  REVIEW: "warn",
  WAIT: "warn",
  ADD: "pos",
  ENTER: "pos",
};

const DOT: Record<string, string> = {
  POSITIVE: "bg-pos",
  NEGATIVE: "bg-neg",
  MIXED: "bg-warn",
  NEUTRAL: "bg-muted",
};

/** One sourced line: what happened, then where it comes from and when. */
function Line({ i }: { i: MorningItem }) {
  return (
    <li className="flex gap-2 text-sm">
      <span
        className={cn(
          "mt-1.5 h-2 w-2 shrink-0 rounded-full",
          i.kind === "filing" ? "bg-accent" : DOT[i.direction ?? ""] ?? "bg-muted"
        )}
        title={i.kind === "filing" ? "filing" : `${i.direction?.toLowerCase()} · ${i.materiality?.toLowerCase()}`}
      />
      <span className="min-w-0">
        {i.text}{" "}
        <span className="whitespace-nowrap text-xs text-muted">
          —{" "}
          {i.url ? (
            <a href={i.url} target="_blank" rel="noreferrer" className="hover:text-accent hover:underline">
              {i.source}
            </a>
          ) : (
            i.source
          )}
          , {fmtEt(i.at)}
          {i.also.length > 0 && ` · also ${i.also.join(", ")}`}
        </span>
      </span>
    </li>
  );
}

function Position({ c, since }: { c: MorningCard; since: string }) {
  return (
    <Card>
      <CardContent className="space-y-2 pt-4">
        <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
          <Link to={`/ticker/${c.symbol}`} className="text-base font-bold hover:text-accent">
            {c.symbol}
          </Link>
          <span className="text-xs text-muted tnum">
            {c.shares} sh · {c.weight != null ? fmtPct(c.weight) : "—"} of the book
          </span>
          {c.from_cost != null && (
            <span className={cn("text-sm tnum", pnlColor(c.from_cost))}>
              {fmtPct(c.from_cost, { sign: true })} from cost
            </span>
          )}
          {c.day_move != null && (
            <span className={cn("text-sm tnum", pnlColor(c.day_move))}>
              {fmtPct(c.day_move, { sign: true })} {c.day_move_on ? `on ${c.day_move_on.slice(5)}` : "last session"}
            </span>
          )}
          {c.call && (
            <Badge variant={CALL[c.call] ?? "muted"} className="ml-auto">
              {c.call}
            </Badge>
          )}
        </div>
        {c.action && <div className="text-sm font-medium">→ {c.action}</div>}
        {c.answered && <div className="text-xs text-warn">Still in the book: {c.answered}</div>}
        {c.call_why && !c.action && <div className="text-xs text-muted">{c.call_why}</div>}

        {c.new.length > 0 ? (
          <ul className="space-y-1">
            {c.new.map((i, n) => (
              <Line key={n} i={i} />
            ))}
            {c.more > 0 && (
              <li className="pl-4 text-xs">
                <Link to={`/ticker/${c.symbol}`} className="text-muted hover:text-accent">
                  +{c.more} more on the ticker page
                </Link>
              </li>
            )}
          </ul>
        ) : (
          <div className="text-sm text-muted">Nothing material since {fmtEt(since)}.</div>
        )}

        {(c.week || c.results || c.watch) && (
          <div className="space-y-0.5 border-t border-border/50 pt-2 text-xs text-muted">
            {c.week && (
              <div>
                <span className="text-text">Week:</span> {c.week}{" "}
                <span>(news agent, {c.week_day})</span>
              </div>
            )}
            {(c.results || c.watch) && (
              <div>
                <span className="text-text">Next:</span>{" "}
                {c.results && (
                  <span className={cn(c.results_in != null && c.results_in <= 5 && "text-warn")}>
                    results {c.results}
                    {c.results_in != null ? ` (in ${c.results_in} sessions)` : ""}
                  </span>
                )}
                {c.results && c.watch ? " · " : ""}
                {c.watch}
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default function Morning() {
  const q = useQuery({ queryKey: ["morning"], queryFn: api.morning, refetchInterval: 300_000 });
  const m = q.data;
  const soon = (m?.cards ?? []).filter((c) => c.results_in != null && c.results_in <= 5);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-lg font-semibold">Morning read</h1>
        {m && (
          <span className="text-xs text-muted">
            new since {fmtEt(m.since)} · book as of {fmtEt(m.book_asof)}
            {m.net_liq ? ` · net liq $${Math.round(m.net_liq).toLocaleString("en-US")}` : ""}
          </span>
        )}
      </div>
      {q.isLoading && <div className="text-sm text-muted">Loading…</div>}
      {q.error && <div className="text-sm text-neg">{(q.error as Error).message}</div>}

      {m && (
        <>
          <Card>
            <CardContent className="space-y-3 pt-4 text-sm">
              <div>
                <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">To do</div>
                {m.actions.length ? (
                  <ul className="space-y-0.5">
                    {m.actions.map((a) => (
                      <li key={a}>
                        <Link to="/actions" className="hover:text-accent">
                          {a}
                        </Link>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div className="text-muted">Nothing to do.</div>
                )}
              </div>
              {(soon.length > 0 || m.macro.length > 0) && (
                <div>
                  <div className="mb-1 text-[10px] uppercase tracking-wide text-muted">Ahead</div>
                  <ul className="space-y-0.5">
                    {soon.map((c) => (
                      <li key={c.symbol}>
                        {c.symbol} results {c.results} (in {c.results_in} sessions)
                      </li>
                    ))}
                    {m.macro.map((x) => (
                      <li key={x}>{x}</li>
                    ))}
                  </ul>
                </div>
              )}
            </CardContent>
          </Card>

          <div className="space-y-3">
            {m.cards.map((c) => (
              <Position key={c.symbol} c={c} since={m.since} />
            ))}
          </div>

          {m.watched.length > 0 && (
            <Card>
              <CardContent className="space-y-2 pt-4">
                <div className="text-[10px] uppercase tracking-wide text-muted">
                  Watched names with material news
                </div>
                <ul className="space-y-1">
                  {m.watched.map((w) => (
                    <li key={w.symbol} className="text-sm">
                      <Link to={`/ticker/${w.symbol}`} className="font-semibold hover:text-accent">
                        {w.symbol}
                      </Link>
                      {w.more > 0 && <span className="text-xs text-muted"> (+{w.more} more)</span>}
                      <ul className="mt-0.5">
                        <Line i={w.item} />
                      </ul>
                    </li>
                  ))}
                </ul>
              </CardContent>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
