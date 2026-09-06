import { useQuery } from "@tanstack/react-query";
import { api } from "@/lib/api";
import { cn, fmtEt, fmtNum } from "@/lib/utils";
import { Section } from "@/components/common";
import { EventRow, SourceBadge } from "@/components/daemon";

/** What the daemon holds on one name: filings and news on a clock, the factor
 *  fit, and this symbol's share of each book-level bet. */
export function TickerDaemonPanels({ symbol }: { symbol: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ["daemon", "symbol", symbol],
    queryFn: () => api.daemonSymbol(symbol),
    enabled: Boolean(symbol),
  });

  if (isLoading || !data) return null;

  return (
    <>
      <Section title="Filings & news" empty={!data.timeline.length}>
        <div>
          {data.timeline.map((row) => (
            <div
              key={row.url + row.published_at}
              className="flex items-start gap-3 border-b border-border/50 py-2 last:border-0"
            >
              <SourceBadge tier={row.tier} match={row.match} />
              <div className="min-w-0 flex-1">
                <a
                  href={row.url}
                  target="_blank"
                  rel="noreferrer"
                  className="text-sm hover:text-accent hover:underline"
                >
                  {row.title}
                </a>
                <div className="text-xs text-muted tnum">
                  {fmtEt(row.published_at)}
                  {row.doc_type && ` · ${row.doc_type}`}
                  {row.item_codes.length > 0 && ` [${row.item_codes.join(", ")}]`}
                </div>
              </div>
            </div>
          ))}
        </div>
      </Section>

      {data.events.length > 0 && (
        <Section title="Events">
          <div>
            {data.events.map((e) => (
              <EventRow key={e.id} event={e} />
            ))}
          </div>
        </Section>
      )}

      <Section
        title="Factor sensitivity"
        empty={!data.sensitivity}
        right={
          data.sensitivity && (
            <span className="text-xs text-muted tnum">
              R² {fmtNum(data.sensitivity.r2, 2)} · resid{" "}
              {(data.sensitivity.resid_vol * 100).toFixed(2)}%/day · n{" "}
              {data.sensitivity.n_obs}
            </span>
          )
        }
      >
        {data.sensitivity && (
          <div className="space-y-3">
            <table className="w-full text-xs">
              <thead className="text-muted">
                <tr>
                  <th className="text-left font-normal">factor</th>
                  <th className="text-right font-normal">loading</th>
                  <th className="text-right font-normal">t</th>
                  <th className="text-right font-normal">book share</th>
                </tr>
              </thead>
              <tbody>
                {data.sensitivity.loadings.map((l) => {
                  const contrib = data.contribution.find((c) => c.factor === l.factor);
                  return (
                    <tr key={l.factor} className={cn(!l.material && "text-muted")}>
                      <td className="py-0.5">
                        {l.factor}
                        {l.material && <span className="ml-1 text-accent">•</span>}
                      </td>
                      <td className={cn("tnum text-right", l.loading < 0 ? "text-neg" : "text-pos")}>
                        {l.loading >= 0 ? "+" : ""}
                        {fmtNum(l.loading, 2)}
                      </td>
                      <td className="tnum text-right text-muted">{fmtNum(l.tstat, 1)}</td>
                      <td className="tnum text-right">
                        {contrib
                          ? `${contrib.contribution >= 0 ? "+" : ""}${fmtNum(
                              contrib.contribution,
                              2
                            )}`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="text-xs text-muted">
              <span className="text-accent">•</span> material = |loading| ≥ 0.30 and |t| ≥ 2.
              Ridge estimates over correlated factors: read as relative bets, not causal
              betas. A high residual vol means macro explains little here, so divergence
              alerts are correspondingly rare.
            </p>
          </div>
        )}
      </Section>
    </>
  );
}
