import * as React from "react";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { cn } from "@/lib/utils";

export function ThesisBadge({ status }: { status?: string | null }) {
  const s = (status || "UNKNOWN").toUpperCase();
  const variant =
    s === "ON_TRACK" ? "pos" : s === "CAUTION" ? "warn" : s === "INVALIDATED" ? "neg" : "muted";
  return <Badge variant={variant as any}>{s.replace(/_/g, " ")}</Badge>;
}

export function AttentionDot({ level }: { level?: string | null }) {
  const l = (level || "LOW").toUpperCase();
  const color = l === "HIGH" ? "bg-neg" : l === "MEDIUM" ? "bg-warn" : "bg-pos";
  return <span className={cn("inline-block h-2 w-2 rounded-full", color)} title={l} />;
}

export function Stat({
  label,
  value,
  sub,
  className,
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={className}>
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="text-lg font-semibold tnum">{value}</div>
      {sub != null && <div className="text-xs text-muted tnum">{sub}</div>}
    </div>
  );
}

/** A titled card section; renders a placeholder when empty. */
export function Section({
  title,
  empty,
  children,
  right,
}: {
  title: string;
  empty?: boolean;
  children: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between">
        <CardTitle className="uppercase text-muted">{title}</CardTitle>
        {right}
      </CardHeader>
      <CardContent>
        {empty ? <div className="text-sm text-muted">No data.</div> : children}
      </CardContent>
    </Card>
  );
}

export function KV({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1 border-b border-border/50 last:border-0">
      <span className="text-sm text-muted">{k}</span>
      <span className="text-sm tnum text-right">{v}</span>
    </div>
  );
}

export function Th({ children, className }: { children?: React.ReactNode; className?: string }) {
  return (
    <th className={cn("text-left font-medium text-muted px-2 py-1.5 text-xs", className)}>
      {children}
    </th>
  );
}

export function Td({ children, className }: { children?: React.ReactNode; className?: string }) {
  return <td className={cn("px-2 py-1.5 text-sm tnum", className)}>{children}</td>;
}
