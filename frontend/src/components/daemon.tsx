import type {
  BookExposure,
  DaemonEvent,
  EventTier,
  ReconcileFinding,
  SourceItem,
  SourceTier,
} from "@/lib/types";
import { cn, fmtEt, fmtNum, fmtUsd } from "@/lib/utils";

/** Tier is the single most important thing on screen: it says whether this
 *  interrupts you, reaches a digest, or is only context. */
export function TierBadge({ tier }: { tier: EventTier }) {
  const style: Record<EventTier, string> = {
    A: "bg-neg/15 text-neg border-neg/30",
    B: "bg-warn/15 text-warn border-warn/30",
    C: "bg-panel-2 text-muted border-border",
  };
  const label: Record<EventTier, string> = {
    A: "ACT",
    B: "DIGEST",
    C: "CONTEXT",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold tracking-wide",
        style[tier]
      )}
      title={
        tier === "A"
          ? "Actionable with a deadline — this is allowed to interrupt you"
          : tier === "B"
            ? "Worth reading today, but not now"
            : "Logged context; can never trigger an alert"
      }
    >
      {label[tier]}
    </span>
  );
}

/** Where an item came from, which caps what it is permitted to do. */
export function SourceBadge({ tier, match }: { tier: SourceTier; match?: string }) {
  const style: Record<SourceTier, string> = {
    PRIMARY: "bg-pos/15 text-pos border-pos/30",
    BROKER: "bg-pos/15 text-pos border-pos/30",
    AGGREGATOR: "bg-accent/15 text-accent border-accent/30",
    UNTAGGED: "bg-panel-2 text-muted border-border",
  };
  const help: Record<SourceTier, string> = {
    PRIMARY: "The company or regulator, under legal liability. May interrupt.",
    BROKER: "Contractual data about your own account. May interrupt.",
    AGGREGATOR: "Dated, entity-tagged reporting. Reaches the digest.",
    UNTAGGED: "Untagged feed. Context only — can never trigger an alert.",
  };
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-medium",
        style[tier]
      )}
      title={`${help[tier]}${match ? ` Matched by ${match}.` : ""}`}
    >
      {tier}
    </span>
  );
}

function humanKind(kind: string) {
  return kind.replace(/^FILING_/, "").split("_").join(" ").toLowerCase();
}

/** The one line that says what an event actually means for the book. */
function eventDetail(event: DaemonEvent): string | null {
  const p = event.payload as Record<string, number | string | undefined>;
  if (p.dilution_pct != null && p.offering_usd != null) {
    return `${fmtUsd(Number(p.offering_usd))} = ${(Number(p.dilution_pct) * 100).toFixed(
      1
    )}% of market cap`;
  }
  if (p.offering_usd != null) return `${fmtUsd(Number(p.offering_usd))} offered (unsized vs cap)`;
  if (p.expected_book_move != null) {
    return `${p.factor} moved ${(Number(p.move) * 100).toFixed(2)}% (z ${fmtNum(
      Number(p.z),
      1
    )}) → book ${(Number(p.expected_book_move) * 100).toFixed(2)}%`;
  }
  if (p.residual_z != null) {
    return `${(Number(p.actual_return) * 100).toFixed(2)}% actual vs ${(
      Number(p.expected_return) * 100
    ).toFixed(2)}% expected (z ${fmtNum(Number(p.residual_z), 1)})`;
  }
  if (p.check != null) return `${p.check}: ${p.failed} symbol(s) failed`;
  if (p.label != null) return String(p.label);
  return null;
}

export function EventRow({ event }: { event: DaemonEvent }) {
  const detail = eventDetail(event);
  const payload = event.payload as { url?: string; accepted_at?: string };
  const url = payload.url;
  // The time that matters is when the thing *happened*, not when the daemon
  // noticed it. A 424B5 accepted on 21 Aug must not read as today because
  // today is when it was ingested.
  const occurred = payload.accepted_at ?? event.ts;
  const lagged = payload.accepted_at != null && payload.accepted_at.slice(0, 10) !== event.ts.slice(0, 10);
  return (
    <div className="flex items-start gap-3 border-b border-border/50 py-2 last:border-0">
      <TierBadge tier={event.tier} />
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-x-2">
          {event.symbol && <span className="font-semibold">{event.symbol}</span>}
          <span className="text-sm">{humanKind(event.kind)}</span>
          <span className="text-xs text-muted tnum" title={`ingested ${fmtEt(event.ts)}`}>
            {fmtEt(occurred)}
            {lagged && <span className="ml-1 text-warn">· backfilled</span>}
          </span>
        </div>
        {detail && <div className="text-xs text-muted tnum">{detail}</div>}
        {url && (
          <a
            href={url}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-accent hover:underline"
          >
            source
          </a>
        )}
      </div>
    </div>
  );
}

export function ExposureBars({ exposure }: { exposure: BookExposure }) {
  const max = Math.max(...exposure.factors.map((f) => Math.abs(f.net_loading)), 0.01);
  return (
    <div className="space-y-2">
      {exposure.factors.map((f) => {
        const pct = (Math.abs(f.net_loading) / max) * 50;
        const negative = f.net_loading < 0;
        return (
          <div key={f.factor} className="grid grid-cols-[110px_1fr_60px] items-center gap-2">
            <span className="text-xs text-muted">{f.factor}</span>
            <div className="relative h-4 rounded bg-panel-2">
              <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
              <div
                className={cn(
                  "absolute inset-y-0 rounded",
                  negative ? "bg-neg/70" : "bg-pos/70"
                )}
                style={
                  negative
                    ? { right: "50%", width: `${pct}%` }
                    : { left: "50%", width: `${pct}%` }
                }
                title={f.contributors
                  .map((c) => `${c.symbol} ${c.share >= 0 ? "+" : ""}${fmtNum(c.share, 2)}`)
                  .join("   ")}
              />
            </div>
            <span
              className={cn("text-xs tnum text-right", negative ? "text-neg" : "text-pos")}
            >
              {f.net_loading >= 0 ? "+" : ""}
              {fmtNum(f.net_loading, 2)}
            </span>
          </div>
        );
      })}
    </div>
  );
}

export function FindingRow({ finding }: { finding: ReconcileFinding }) {
  const colour = {
    OK: "text-pos",
    WARN: "text-warn",
    FAIL: "text-neg",
  }[finding.severity];
  return (
    <div className="flex items-start gap-3 border-b border-border/50 py-1.5 text-sm last:border-0">
      <span className={cn("w-12 shrink-0 text-xs font-semibold", colour)}>
        {finding.severity}
      </span>
      <span className="w-36 shrink-0 text-xs text-muted">{finding.check}</span>
      <span className="w-14 shrink-0 text-xs">{finding.symbol ?? "—"}</span>
      <span className="min-w-0 flex-1 text-xs text-muted">{finding.detail}</span>
    </div>
  );
}

export function SourceRow({ item }: { item: SourceItem }) {
  return (
    <div className="flex items-start gap-3 border-b border-border/50 py-2 last:border-0">
      <SourceBadge tier={item.tier} match={item.match} />
      <div className="min-w-0 flex-1">
        <a
          href={item.url}
          target="_blank"
          rel="noreferrer"
          className="text-sm hover:text-accent hover:underline"
        >
          {item.title}
        </a>
        <div className="text-xs text-muted tnum">
          {fmtEt(item.published_at)}
          {" · "}
          {item.doc_type ?? "news"}
          {item.item_codes.length > 0 && ` [${item.item_codes.join(", ")}]`}
          {" · "}
          {item.provider}
        </div>
      </div>
    </div>
  );
}
