import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { AlertCircle, ChevronDown, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type { ActionCard, ActionKind, ClaimVerdict, DecidedRef, Rationale } from "@/lib/types";
import { cn, fmtEt } from "@/lib/utils";
import { Section } from "@/components/common";

/** The system holds no view of its own. It reports that a condition the user
 *  defined in advance has or has not arrived — so the vocabulary is about
 *  attention, never about trades. */
const ACTION_STYLE: Record<ActionKind, { tone: string; label: string }> = {
  REVIEW_NOW: { tone: "text-neg", label: "review now" },
  REVIEW: { tone: "text-warn", label: "worth reading" },
  WRITE_THESIS: { tone: "text-accent", label: "write a rule" },
  HOLD: { tone: "text-pos", label: "nothing to do" },
  CANNOT_SAY: { tone: "text-muted", label: "cannot say" },
};

const CLAIM_MARK: Record<string, { mark: string; tone: string }> = {
  BROKEN: { mark: "✗", tone: "text-neg" },
  // Violated by the current state rather than by an event: true today, and
  // true yesterday. Marked apart so it is not read as fresh news.
  STANDING: { mark: "▲", tone: "text-warn" },
  INTACT: { mark: "✓", tone: "text-pos" },
  UNTESTED: { mark: "·", tone: "text-muted" },
  UNREACHABLE: { mark: "!", tone: "text-warn" },
};

const BEARING: Record<string, { mark: string; tone: string }> = {
  AGAINST: { mark: "✗", tone: "text-neg" },
  SUPPORTS: { mark: "✓", tone: "text-pos" },
  BLIND: { mark: "?", tone: "text-warn" },
  CONTEXT: { mark: "·", tone: "text-muted" },
};

/** What the system does next, which is all it can honestly act on: it cannot
 *  verify that a position was trimmed, only record what was said and behave
 *  accordingly. */
const VERDICTS: { value: string; label: string; hint: string }[] = [
  {
    value: "ACKNOWLEDGED",
    label: "seen",
    hint: "quiet until the number moves 10% past today's reading",
  },
  { value: "ACTED", label: "acted", hint: "quiet 3 days, then checked again" },
  { value: "DISMISSED", label: "not relevant", hint: "quiet until the rule itself changes" },
  { value: "THESIS_REVISED", label: "rewrote the rule", hint: "superseded" },
];

/** A rule the user can answer: one that is actually saying something. An
 *  intact or unreachable rule has nothing to decide about. */
function answerable(claim: ClaimVerdict): boolean {
  return (claim.status === "BROKEN" || claim.status === "STANDING") && !!claim.claim_id;
}

function Answer({ symbol, claim }: { symbol: string; claim: ClaimVerdict }) {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const qc = useQueryClient();
  const record = useMutation({
    mutationFn: (verdict: string) =>
      api.daemonDecide(symbol, claim.claim_id as string, verdict, note),
    onSuccess: () => {
      setOpen(false);
      setNote("");
      qc.invalidateQueries({ queryKey: ["daemon-actions"] });
    },
  });

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="shrink-0 text-[11px] text-muted underline hover:text-accent"
      >
        answer
      </button>
    );
  }

  return (
    <div className="mt-1 w-full space-y-1 rounded border border-border bg-panel-2 p-1.5">
      <input
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="why, in your own words — this is what you will read later"
        className="w-full bg-transparent text-xs outline-none placeholder:text-muted"
      />
      <div className="flex flex-wrap items-center gap-1">
        {VERDICTS.map((v) => (
          <button
            key={v.value}
            title={v.hint}
            disabled={record.isPending}
            onClick={() => record.mutate(v.value)}
            className="rounded border border-border px-1.5 py-0.5 text-[11px] hover:border-accent hover:text-accent disabled:opacity-50"
          >
            {v.label}
          </button>
        ))}
        <button
          onClick={() => setOpen(false)}
          className="px-1 text-[11px] text-muted hover:text-text"
        >
          cancel
        </button>
      </div>
      {record.isError && (
        <div className="text-[11px] text-neg">{(record.error as Error).message}</div>
      )}
    </div>
  );
}

function Decided({ refs, tone }: { refs: DecidedRef[]; tone: string }) {
  return (
    <>
      {refs.map((ref) => (
        <div key={ref.subject_id + ref.verdict} className="flex items-start gap-1.5 text-xs">
          <span className={cn("w-3 shrink-0", tone)}>{tone === "text-neg" ? "!" : "✓"}</span>
          <div className="min-w-0">
            <div>{ref.text}</div>
            <div className="text-muted">{ref.reason}</div>
            {ref.note && <div className="text-muted">“{ref.note}”</div>}
          </div>
        </div>
      ))}
    </>
  );
}

const SOURCE_LABEL: Record<string, string> = {
  YOUR_RULE: "your own rule, quoted back",
  ARITHMETIC: "arithmetic from the limit you set",
  NONE: "nothing to return to you",
};

function Card({ card }: { card: ActionCard }) {
  const [open, setOpen] = useState(card.action === "REVIEW_NOW");
  // A card built before the rationale existed, or one that refused, carries
  // an empty object rather than the shape — normalise once.
  const rationale: Rationale = {
    steps: (card.rationale as Rationale)?.steps ?? [],
    proposed: (card.rationale as Rationale)?.proposed ?? null,
  };
  const style = ACTION_STYLE[card.action];
  const Chevron = open ? ChevronDown : ChevronRight;

  return (
    <div className="border-b border-border/50 py-2 last:border-0">
      <button
        onClick={() => setOpen(!open)}
        className="flex w-full items-start gap-2 text-left"
      >
        <Chevron className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted" />
        <Link
          to={`/ticker/${card.symbol}`}
          onClick={(e) => e.stopPropagation()}
          className="w-14 shrink-0 font-semibold hover:text-accent hover:underline"
        >
          {card.symbol}
        </Link>
        <span className={cn("w-28 shrink-0 text-xs font-medium uppercase", style.tone)}>
          {style.label}
        </span>
        <span className="min-w-0 flex-1 text-sm">{card.headline}</span>
        {card.deadline && (
          <span className="shrink-0 text-xs text-neg">by {card.deadline}</span>
        )}
      </button>

      {open && (
        <div className="mt-2 space-y-2 pl-[22px]">
          {card.because.length > 0 && (
            <ul className="space-y-0.5">
              {card.because.map((reason, i) => (
                <li key={i} className="text-xs text-muted">
                  • {reason}
                </li>
              ))}
            </ul>
          )}

          {card.position_note && (
            <div className="text-xs tnum text-muted">{card.position_note}</div>
          )}

          {card.claims.length > 0 ? (
            <div className="space-y-0.5">
              <div className="text-[11px] uppercase tracking-wide text-muted">
                your rules
              </div>
              {card.claims.map((claim, i) => (
                <div key={i} className="flex flex-wrap items-start gap-1.5 text-xs">
                  <span className={cn("shrink-0", CLAIM_MARK[claim.status].tone)}>
                    {CLAIM_MARK[claim.status].mark}
                  </span>
                  <span className="min-w-0 flex-1">{claim.text}</span>
                  {answerable(claim) && <Answer symbol={card.symbol} claim={claim} />}
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-muted">No rules written for this name.</div>
          )}

          {/* The chain of stored facts. Each step says what it does to the
              picture, never whether it is good or bad. */}
          {rationale.steps.length > 0 && (
            <div className="space-y-0.5">
              <div className="text-[11px] uppercase tracking-wide text-muted">
                rationale
              </div>
              {rationale.steps.map((step, i) => (
                <div key={i} className="flex items-start gap-1.5 text-xs">
                  <span className={cn("w-3 shrink-0", BEARING[step.bearing].tone)}>
                    {BEARING[step.bearing].mark}
                  </span>
                  <span className="w-28 shrink-0 text-muted">{step.label}</span>
                  <span className="min-w-0">{step.fact}</span>
                </div>
              ))}
            </div>
          )}

          {/* A proposal always names where it came from. There are exactly two
              sources — a rule the user wrote, or arithmetic — and neither is
              the system forming a view. */}
          {rationale.proposed && (
            <div
              className={cn(
                "rounded border px-2 py-1.5",
                rationale.proposed.source === "YOUR_RULE"
                  ? "border-accent/40 bg-accent/10"
                  : rationale.proposed.source === "ARITHMETIC"
                    ? "border-border bg-panel-2"
                    : "border-border/60"
              )}
            >
              <div className="text-[11px] uppercase tracking-wide text-muted">
                proposed · {SOURCE_LABEL[rationale.proposed.source]}
              </div>
              <div className="mt-0.5 text-sm">{rationale.proposed.text}</div>
              {rationale.proposed.size && (
                <div className="tnum text-xs text-accent">= {rationale.proposed.size}</div>
              )}
              {rationale.proposed.quoted_from && (
                <div className="mt-0.5 text-xs text-muted">
                  from your rule: “{rationale.proposed.quoted_from}”
                </div>
              )}
            </div>
          )}

          {/* Answered once, and the number has moved past it. These are in
              `because` too; here they carry the history that makes them
              stronger than a first sighting rather than a repeat of one. */}
          {card.reopened.length > 0 && (
            <div className="space-y-0.5">
              <div className="text-[11px] uppercase tracking-wide text-neg">back in play</div>
              <Decided refs={card.reopened} tone="text-neg" />
            </div>
          )}

          {/* Shown, never dropped. Hiding what was decided would make the card
              quieter and less complete at the same time. */}
          {card.decided.length > 0 && (
            <div className="space-y-0.5">
              <div className="text-[11px] uppercase tracking-wide text-muted">you decided</div>
              <Decided refs={card.decided} tone="text-pos" />
            </div>
          )}

          {/* Always shown, never collapsed away: a card that proposed something
              on thin data must not be able to hide it. */}
          <div className="space-y-0.5">
            <div className="text-[11px] uppercase tracking-wide text-muted">evidence</div>
            {card.evidence.items.map((item) => (
              <div key={item.name} className="flex items-start gap-1.5 text-xs">
                <span
                  className={cn(
                    "w-3 shrink-0",
                    item.blocking ? "text-neg" : item.ok ? "text-pos" : "text-warn"
                  )}
                >
                  {item.blocking ? "!" : item.ok ? "✓" : "·"}
                </span>
                <span className="w-24 shrink-0 text-muted">{item.name}</span>
                <span className="min-w-0 text-muted">{item.detail}</span>
              </div>
            ))}
          </div>

          {card.what_would_sharpen_this.length > 0 && (
            <div className="space-y-0.5 border-t border-border/40 pt-1.5">
              {card.what_would_sharpen_this.map((hint, i) => (
                <div key={i} className="text-xs text-accent">
                  → {hint}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ActionCards({ symbol }: { symbol?: string } = {}) {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["daemon", "actions", symbol ?? "book"],
    queryFn: () => api.daemonActions(symbol),
  });

  if (isError) {
    return (
      <Section title="What to do">
        <div className="flex items-center gap-2 text-xs text-neg">
          <AlertCircle className="h-3.5 w-3.5" />
          {(error as Error).message}
        </div>
      </Section>
    );
  }

  const cards = data?.cards ?? [];
  const urgent = cards.filter((c) => c.action === "REVIEW_NOW").length;

  return (
    <Section
      title="What to do"
      empty={!isLoading && cards.length === 0}
      right={
        cards.length > 0 ? (
          <span className={cn("text-xs", urgent ? "text-neg" : "text-muted")}>
            {urgent > 0
              ? `${urgent} rule${urgent > 1 ? "s" : ""} broken`
              : `${cards.length} position${cards.length > 1 ? "s" : ""}`}
            {data?.cards[0] && ` · ${fmtEt(data.cards[0].assembled_at)}`}
          </span>
        ) : undefined
      }
    >
      <div>
        {cards.map((card) => (
          <Card key={card.symbol} card={card} />
        ))}
        {cards.length > 0 && (
          <p className="pt-2 text-xs text-muted">
            This reports whether conditions you defined in advance have arrived. It
            holds no view of its own and proposes no trade.
          </p>
        )}
      </div>
    </Section>
  );
}
