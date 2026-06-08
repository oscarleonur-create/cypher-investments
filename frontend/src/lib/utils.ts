import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmtUsd(v: number | null | undefined, opts: { sign?: boolean } = {}): string {
  if (v == null || Number.isNaN(v)) return "—";
  const sign = opts.sign && v > 0 ? "+" : "";
  const abs = Math.abs(v);
  if (abs >= 1e12) return `${sign}$${(v / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(v / 1e9).toFixed(2)}B`;
  if (abs >= 1e6) return `${sign}$${(v / 1e6).toFixed(2)}M`;
  if (abs >= 1e3) return `${sign}$${(v / 1e3).toFixed(1)}K`;
  return `${sign}$${v.toFixed(2)}`;
}

export function fmtMoney(v: number | null | undefined, sign = false): string {
  if (v == null || Number.isNaN(v)) return "—";
  const s = sign && v > 0 ? "+" : "";
  return `${s}$${v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

export function fmtPct(v: number | null | undefined, opts: { sign?: boolean; scale?: boolean } = {}): string {
  if (v == null || Number.isNaN(v)) return "—";
  const x = opts.scale === false ? v : v * 100;
  const sign = opts.sign && x > 0 ? "+" : "";
  return `${sign}${x.toFixed(1)}%`;
}

export function fmtNum(v: number | null | undefined, digits = 2): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function pnlColor(v: number | null | undefined): string {
  if (v == null || v === 0) return "text-muted";
  return v > 0 ? "text-pos" : "text-neg";
}

// Guards against non-meaningful model output (a runaway DCF, or one that floors
// equity at 0 for a pre-profit cash-burner) so the UI shows "n/m" instead of
// +39,587,605,573,395% or a misleading -100%. Upside is a decimal fraction.
export function fmtUpside(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v) || Math.abs(v) > 10 || v <= -0.999) return "n/m";
  return fmtPct(v, { sign: true });
}

export function fmtPriceSafe(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return "—";
  if (!Number.isFinite(v) || v <= 0 || Math.abs(v) > 1_000_000) return "n/m";
  return fmtNum(v);
}
