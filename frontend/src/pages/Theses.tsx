import { useState } from "react";
import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Plus } from "lucide-react";
import { api } from "@/lib/api";
import type { Conviction, ThesisStatus, ThesisSummary } from "@/lib/types";
import { Th } from "@/components/common";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const CONVICTION_VARIANT: Record<Conviction, "pos" | "warn" | "muted"> = {
  HIGH: "pos",
  MEDIUM: "warn",
  LOW: "muted",
};

const STATUS_VARIANT: Record<ThesisStatus, "accent" | "pos" | "muted"> = {
  DRAFT: "muted",
  ACTIVE: "pos",
  ARCHIVED: "accent",
};

function fmtDate(iso: string): string {
  const d = new Date(iso);
  return Number.isNaN(d.getTime())
    ? "—"
    : d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
}

export default function Theses() {
  const [filter, setFilter] = useState("");
  const sym = filter.trim().toUpperCase();

  const { data, isLoading, error } = useQuery<{ theses: ThesisSummary[] }>({
    queryKey: ["theses", sym],
    queryFn: () => api.theses(sym || undefined),
  });

  const items = data?.theses || [];

  return (
    <div className="space-y-5">
      <Card className="p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-sm font-semibold">Investment Theses</div>
            <div className="text-xs text-muted">
              Your written research. Each thesis is a markdown write-up — for a ticker or a theme.
            </div>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={filter}
              onChange={(e) => setFilter(e.target.value.toUpperCase())}
              placeholder="Filter by ticker…"
              maxLength={8}
              className="h-8 w-36 rounded-lg border border-border bg-transparent px-3 text-sm uppercase outline-none focus:border-accent"
            />
            <Link to="/theses/new">
              <Button size="sm">
                <Plus className="h-4 w-4" />
                New thesis
              </Button>
            </Link>
          </div>
        </div>
      </Card>

      <Card className="overflow-hidden">
        {isLoading ? (
          <div className="p-6 text-sm text-muted">Loading theses…</div>
        ) : error ? (
          <div className="p-6 text-sm text-neg">{(error as Error).message}</div>
        ) : items.length === 0 ? (
          <div className="p-6 text-sm text-muted">
            {sym
              ? `No theses for ${sym} yet.`
              : "No theses yet. Click “New thesis” to start writing one."}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse">
              <thead className="border-b border-border bg-panel-2/40">
                <tr>
                  <Th>Title</Th>
                  <Th>Scope</Th>
                  <Th>Conviction</Th>
                  <Th>Status</Th>
                  <Th className="text-right">Updated</Th>
                  <Th></Th>
                </tr>
              </thead>
              <tbody>
                {items.map((t) => (
                  <ThesisRow key={t.id} t={t} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}

function ThesisRow({ t }: { t: ThesisSummary }) {
  return (
    <tr className="border-b border-border/40 hover:bg-panel-2/40">
      <td className="px-2 py-2">
        <Link to={`/theses/${t.id}`} className="font-medium hover:text-accent">
          {t.title || "Untitled thesis"}
        </Link>
        {t.tags.length > 0 && (
          <div className="mt-1 flex flex-wrap gap-1">
            {t.tags.map((tag) => (
              <span key={tag} className="text-[10px] text-muted">
                #{tag}
              </span>
            ))}
          </div>
        )}
      </td>
      <td className="px-2 py-2">
        {t.symbol ? (
          <Badge variant="default">{t.symbol}</Badge>
        ) : (
          <Badge variant="muted">Thematic</Badge>
        )}
      </td>
      <td className="px-2 py-2">
        <Badge variant={CONVICTION_VARIANT[t.conviction] || "muted"}>{t.conviction}</Badge>
      </td>
      <td className="px-2 py-2">
        <Badge variant={STATUS_VARIANT[t.status] || "muted"}>{t.status}</Badge>
      </td>
      <td className="px-2 py-2 text-right text-sm text-muted tnum">{fmtDate(t.updated_at)}</td>
      <td className="px-2 py-2 text-right">
        <Link to={`/theses/${t.id}`} className="text-muted hover:text-text">
          <ChevronRight className="h-4 w-4 inline" />
        </Link>
      </td>
    </tr>
  );
}
