import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { AlertCircle, ChevronDown, ChevronRight } from "lucide-react";
import { api } from "@/lib/api";
import type { ActionCard, ActionKind } from "@/lib/types";
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
  INTACT: { mark: "✓", tone: "text-pos" },
  UNTESTED: { mark: "·", tone: "text-muted" },
  UNREACHABLE: { mark: "!", tone: "text-warn" },
};

function Card({ card }: { card: ActionCard }) {
  const [open, setOpen] = useState(card.action === "REVIEW_NOW");
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
                <div key={i} className="flex items-start gap-1.5 text-xs">
                  <span className={cn("shrink-0", CLAIM_MARK[claim.status].tone)}>
                    {CLAIM_MARK[claim.status].mark}
                  </span>
                  <span className="min-w-0">{claim.text}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-muted">No rules written for this name.</div>
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
