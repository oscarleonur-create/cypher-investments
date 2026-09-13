import { useQuery } from "@tanstack/react-query";
import { AlertCircle, ExternalLink } from "lucide-react";
import { api } from "@/lib/api";
import type { Story, Verdict } from "@/lib/types";
import { cn, fmtEt, fmtMoney, fmtNum } from "@/lib/utils";
import { Card } from "@/components/ui/card";
import { SourceBadge, TierBadge } from "@/components/daemon";

/** Calibrated readings of a residual. The wording is deliberately weaker than
 *  instinct suggests: a first prototype called a -13.77% session
 *  "idiosyncratic" from a z of -1.12, when the alert threshold is 2.0. */
const VERDICT_TEXT: Record<Verdict, string> = {
  NOT_MARKET: "the market did not do this",
  UNEXPLAINED: "macro cannot explain this move",
  PARTLY_SPECIFIC: "partly company-specific, but inside this name's normal daily range",
  CONSISTENT: "consistent with what macro did that day",
  UNKNOWN: "no usable factor estimate for this name",
};

const VERDICT_TONE: Record<Verdict, string> = {
  NOT_MARKET: "text-neg",
  UNEXPLAINED: "text-warn",
  PARTLY_SPECIFIC: "text-muted",
  CONSISTENT: "text-muted",
  UNKNOWN: "text-muted",
};

function pct(v: number | null | undefined, sign = false) {
  if (v == null) return "—";
  return `${sign && v > 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[128px_1fr] gap-3 border-b border-border/40 py-2 last:border-0">
      <div className="text-[11px] uppercase tracking-wide text-muted">{label}</div>
      <div className="min-w-0 text-sm">{children}</div>
    </div>
  );
}

function StoryBody({ story }: { story: Story }) {
  const { anchor, position, reaction, attribution, corroboration, thesis } = story;
  const facts = anchor.facts as Record<string, number>;

  return (
    <div className="space-y-0">
      <Row label="What happened">
        <div className="flex flex-wrap items-baseline gap-2">
          <TierBadge tier={anchor.tier} />
          <span>{anchor.headline}</span>
          <span className="text-xs text-muted tnum">{fmtEt(anchor.occurred_at)}</span>
          {anchor.occurred_at.slice(0, 10) !== anchor.ingested_at.slice(0, 10) && (
            <span className="text-xs text-warn">learned later</span>
          )}
        </div>
        {facts.offering_usd != null && (
          <div className="mt-0.5 tnum">
            {fmtMoney(Number(facts.offering_usd))}
            {facts.dilution_pct != null && facts.market_cap != null && (
              <>
                {" = "}
                <span className="text-neg">
                  {(Number(facts.dilution_pct) * 100).toFixed(1)}% dilution
                </span>
                {" of a "}
                {fmtMoney(Number(facts.market_cap))} market cap
              </>
            )}
          </div>
        )}
        <div className="mt-0.5 text-xs text-muted">source: {anchor.source}</div>
      </Row>

      {anchor.quote && (
        <Row label="Evidence">
          <blockquote className="border-l-2 border-border pl-3 text-xs italic text-muted">
            “{anchor.quote.slice(0, 260)}…”
          </blockquote>
          {anchor.url && (
            <a
              href={anchor.url}
              target="_blank"
              rel="noreferrer"
              className="mt-1 inline-flex items-center gap-1 text-xs text-accent hover:underline"
            >
              read the filing <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </Row>
      )}

      <Row label="Your position">
        {position.confidence === "UNAVAILABLE" ? (
          <span className="text-muted">unknown — {position.note}</span>
        ) : !position.quantity ? (
          <span className="text-muted">not held{position.note && ` — ${position.note}`}</span>
        ) : (
          <>
            <span className="tnum">
              {position.quantity > 0 ? "+" : ""}
              {fmtNum(position.quantity, 0)} shares @ {fmtNum(position.avg_open_price, 2)} entry
              {position.weight_of_net_liq != null && ` · ${pct(position.weight_of_net_liq)} of net liq`}
            </span>
            {/* Never let a position from the wrong date pass unmarked. */}
            {!position.covers_event && (
              <div className="mt-0.5 flex items-start gap-1 text-xs text-warn">
                <AlertCircle className="mt-0.5 h-3 w-3 shrink-0" />
                <span>{position.note || "this snapshot postdates the event"}</span>
              </div>
            )}
          </>
        )}
      </Row>

      <Row label="What it cost">
        {reaction.confidence === "UNAVAILABLE" ? (
          <span className="text-muted">unknown — {reaction.note}</span>
        ) : (
          <>
            <span className="tnum">
              {fmtNum(reaction.before, 2)} → {fmtNum(reaction.after, 2)}{" "}
              <span className={cn((reaction.pct_move ?? 0) < 0 ? "text-neg" : "text-pos")}>
                ({pct(reaction.pct_move, true)})
              </span>
            </span>
            {reaction.dollars != null && (
              <div className="tnum">
                <span className={cn(reaction.dollars < 0 ? "text-neg" : "text-pos")}>
                  {fmtMoney(reaction.dollars, true)}
                </span>{" "}
                on the position
                {reaction.pct_of_book != null && ` = ${pct(reaction.pct_of_book, true)} of the book`}
              </div>
            )}
            <div className="text-xs text-muted">
              {reaction.session}
              {reaction.priced_next_session && " — the event landed after the close"}
            </div>
          </>
        )}
      </Row>

      <Row label="Was it macro?">
        {attribution.confidence === "UNAVAILABLE" ? (
          <span className="text-muted">unknown — {attribution.note}</span>
        ) : (
          <>
            <span className={VERDICT_TONE[attribution.verdict]}>
              {VERDICT_TEXT[attribution.verdict]}
            </span>
            <div className="tnum text-xs text-muted">
              macro expected {pct(attribution.expected_return, true)}, actual{" "}
              {pct(attribution.actual_return, true)} · residual z{" "}
              {fmtNum(attribution.residual_z, 2)} against {pct(attribution.resid_vol)}/day
            </div>
            {attribution.verdict === "PARTLY_SPECIFIC" && (
              <div className="text-xs text-muted">
                below the 2.0 alert threshold — a day like this is not unusual for this name
              </div>
            )}
          </>
        )}
      </Row>

      {corroboration.items.length > 0 && (
        <Row label="Corroboration">
          <div className="space-y-1">
            {corroboration.items.slice(0, 4).map((item) => (
              <div key={item.url} className="flex items-start gap-2">
                <SourceBadge tier={item.tier} />
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="min-w-0 text-xs hover:text-accent hover:underline"
                >
                  {item.title}
                </a>
              </div>
            ))}
          </div>
        </Row>
      )}

      <Row label="Thesis">
        {thesis.exists ? (
          <span>
            {thesis.title}{" "}
            <span className="text-xs text-muted">
              (conviction {thesis.conviction}, {thesis.status})
            </span>
          </span>
        ) : (
          <span className="text-muted">{thesis.note || "none"}</span>
        )}
      </Row>

      {story.unavailable.length > 0 && (
        <div className="pt-2 text-xs text-muted">
          Not established: {story.unavailable.join(", ")}
        </div>
      )}
    </div>
  );
}

/** The story behind a ticker's most recent material event. Assembled entirely
 *  from stored data — no model wrote any of this. */
export function StoryCard({ symbol }: { symbol: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["daemon", "story", symbol],
    queryFn: () => api.daemonStory(symbol, 1),
    enabled: Boolean(symbol),
  });

  if (isLoading || !data?.stories.length) return null;
  const story = data.stories[0];

  return (
    <Card className="p-4">
      <div className="mb-3 flex items-baseline justify-between gap-3">
        <h2 className="text-base font-semibold">{story.headline}</h2>
        <span className="text-[11px] uppercase tracking-wide text-muted">
          assembled, not generated
        </span>
      </div>
      <StoryBody story={story} />
    </Card>
  );
}
