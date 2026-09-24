import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Plus, X } from "lucide-react";
import { api } from "@/lib/api";
import type { AngleStatus, WatchAngle } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Card } from "@/components/ui/card";

/** Products and segments this holding is a bet on. Suggested from the
 *  holder's claims; only confirmed ones are searched, once a day, and what
 *  they find is context — it never interrupts. */
export function AnglesPanel({ symbol }: { symbol: string }) {
  const client = useQueryClient();
  const key = ["daemon", "angles", symbol];
  const { data } = useQuery({
    queryKey: key,
    queryFn: () => api.daemonAngles(symbol),
    enabled: Boolean(symbol),
  });
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);

  const decide = async (term: string, status: AngleStatus) => {
    setBusy(true);
    try {
      client.setQueryData(key, await api.decideAngle(symbol, term, status));
    } finally {
      setBusy(false);
    }
  };

  const angles = data?.angles ?? [];
  const confirmed = angles.filter((a) => a.status === "CONFIRMED");
  const suggested = angles.filter((a) => a.status === "SUGGESTED");

  return (
    <Card className="p-4">
      <div className="mb-2 flex items-baseline gap-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-muted">Angles</h3>
        <span className="text-xs text-muted">
          searched daily · news found through them is context, never an interrupt
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        {confirmed.map((a) => (
          <Chip key={a.term} angle={a}>
            <IconButton title={`Stop tracking ${a.term}`} onClick={() => decide(a.term, "REJECTED")} disabled={busy}>
              <X className="h-3 w-3" />
            </IconButton>
          </Chip>
        ))}
        {suggested.map((a) => (
          <Chip key={a.term} angle={a}>
            <IconButton title={`Track ${a.term} daily`} onClick={() => decide(a.term, "CONFIRMED")} disabled={busy}>
              <Check className="h-3 w-3" />
            </IconButton>
            <IconButton title={`${a.term} is not an angle`} onClick={() => decide(a.term, "REJECTED")} disabled={busy}>
              <X className="h-3 w-3" />
            </IconButton>
          </Chip>
        ))}
        <form
          className="flex items-center gap-1"
          onSubmit={(e) => {
            e.preventDefault();
            const term = draft.trim();
            if (term.length >= 2) {
              decide(term, "CONFIRMED");
              setDraft("");
            }
          }}
        >
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            placeholder="Add, e.g. Starship"
            maxLength={60}
            className="w-36 rounded border border-border bg-panel-2 px-2 py-0.5 text-xs outline-none focus:border-accent"
          />
          <IconButton title="Track daily" disabled={busy || draft.trim().length < 2}>
            <Plus className="h-3.5 w-3.5" />
          </IconButton>
        </form>
      </div>
      {confirmed.length === 0 && suggested.length > 0 && (
        <p className="mt-2 text-xs text-muted">
          Nothing is searched until you confirm an angle.
        </p>
      )}
    </Card>
  );
}

function Chip({ angle, children }: { angle: WatchAngle; children: React.ReactNode }) {
  const confirmed = angle.status === "CONFIRMED";
  return (
    <span
      title={confirmed ? "Tracked daily" : "Suggested from your claims"}
      className={cn(
        "inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs",
        confirmed ? "bg-accent/15 text-accent" : "border border-dashed border-border text-muted"
      )}
    >
      {angle.term}
      {children}
    </span>
  );
}

function IconButton({
  title,
  onClick,
  disabled,
  children,
}: {
  title: string;
  onClick?: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}) {
  return (
    <button
      type={onClick ? "button" : "submit"}
      title={title}
      aria-label={title}
      onClick={onClick}
      disabled={disabled}
      className="text-muted hover:text-text disabled:opacity-40"
    >
      {children}
    </button>
  );
}
