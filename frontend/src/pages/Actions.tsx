import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { Actionable } from "@/lib/types";
import { cn, fmtEt } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

/** Each verb's colour: selling red, deciding amber, buying green. */
const VERB: Record<Actionable["verb"], { tone: string; bar: string }> = {
  SELL: { tone: "text-neg", bar: "border-l-neg" },
  DECIDE: { tone: "text-warn", bar: "border-l-warn" },
  READ: { tone: "text-accent", bar: "border-l-accent" },
  BUY: { tone: "text-pos", bar: "border-l-pos" },
};

const ANSWER: Record<string, string> = { KEEP: "Keep", DONE: "Done", SKIP: "Skip" };

function Item({ a }: { a: Actionable }) {
  const qc = useQueryClient();
  const [keeping, setKeeping] = useState(false);
  const [note, setNote] = useState("");
  const m = useMutation({
    mutationFn: ({ answer, note }: { answer: string; note?: string }) =>
      api.answerActionable(a.id, answer, note),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["actionables"] }),
  });
  const style = VERB[a.verb];
  // The verb and the symbol are drawn on their own; the line goes on after them.
  const rest = a.title.replace(/^\S+\s+\S+\s+—\s+/, "");

  return (
    <Card className={cn("border-l-4", style.bar)}>
      <CardContent className="space-y-2 pt-4">
        <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <span className={cn("text-base font-bold tracking-wide", style.tone)}>{a.verb}</span>
          {a.symbol === "BOOK" ? (
            // A book-level decision (a correlated group): there is no ticker to open.
            <span className="text-base font-semibold">Book</span>
          ) : (
            <Link to={`/ticker/${a.symbol}`} className="text-base font-semibold hover:text-accent">
              {a.symbol}
            </Link>
          )}
          <span className="text-sm text-text">— {rest}</span>
        </div>
        <ul className="list-disc space-y-1 pl-5 text-sm text-text">
          {a.bullets.map((b, i) => (
            <li key={i}>{b}</li>
          ))}
        </ul>
        <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
          <span className="text-xs text-muted">{a.source}</span>
          <div className="flex flex-wrap gap-2">
            {a.answers.map((ans) =>
              ans === "KEEP" ? (
                <Button
                  key={ans}
                  size="sm"
                  variant="outline"
                  onClick={() => setKeeping((k) => !k)}
                  disabled={m.isPending}
                >
                  Keep…
                </Button>
              ) : (
                <Button
                  key={ans}
                  size="sm"
                  variant={ans === "DONE" ? "default" : "ghost"}
                  onClick={() => m.mutate({ answer: ans })}
                  disabled={m.isPending}
                >
                  {ANSWER[ans]}
                </Button>
              )
            )}
          </div>
        </div>
        {keeping && (
          <form
            className="flex flex-wrap gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              if (note.trim()) m.mutate({ answer: "KEEP", note });
            }}
          >
            <input
              autoFocus
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="Why you keep it — the rule stops asking"
              className="min-w-0 flex-1 rounded-md border border-border bg-panel-2 px-3 py-1.5 text-sm"
            />
            <Button size="sm" type="submit" disabled={!note.trim() || m.isPending}>
              Save
            </Button>
          </form>
        )}
        {m.error && <div className="text-xs text-neg">{(m.error as Error).message}</div>}
      </CardContent>
    </Card>
  );
}

function Group({ title, items }: { title: string; items: Actionable[] }) {
  if (items.length === 0) return null;
  return (
    <div className="space-y-2">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-muted">{title}</h2>
      {items.map((a) => (
        <Item key={a.id} a={a} />
      ))}
    </div>
  );
}

export default function Actions() {
  const q = useQuery({
    queryKey: ["actionables"],
    queryFn: api.actionables,
    refetchInterval: 60_000,
  });
  const [showAnswered, setShowAnswered] = useState(false);
  const data = q.data;
  const items = data?.items ?? [];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-baseline gap-3">
        <h1 className="text-lg font-semibold">Actions</h1>
        {data && (
          <span className="text-xs text-muted">
            {items.length} to do · book as of {fmtEt(data.book_asof)}
          </span>
        )}
      </div>
      {q.isLoading && <div className="text-sm text-muted">Loading…</div>}
      {q.error && <div className="text-sm text-neg">{(q.error as Error).message}</div>}
      {data && items.length === 0 && (
        <Card>
          <CardContent className="pt-4 text-sm text-muted">Nothing to do today.</CardContent>
        </Card>
      )}
      <Group title="Your positions" items={items.filter((a) => a.section === "book")} />
      <Group title="Tracking" items={items.filter((a) => a.section === "tracking")} />
      {data && data.answered.length > 0 && (
        <div className="space-y-1">
          <button
            className="text-xs text-muted hover:text-text"
            onClick={() => setShowAnswered((s) => !s)}
          >
            {showAnswered ? "▾" : "▸"} Answered ({data.answered.length})
          </button>
          {showAnswered && (
            <ul className="space-y-1 pl-4 text-xs text-muted">
              {data.answered.map((x) => (
                <li key={x.id}>
                  <span className="text-text">{x.title}</span> — {x.why}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
