import { cn } from "@/lib/utils";

/** A minimal range slider (native input) styled to match the dark theme. */
export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  disabled,
  className,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <input
      type="range"
      value={value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className={cn(
        "h-1.5 w-full cursor-pointer appearance-none rounded-full bg-panel-2 accent-accent disabled:opacity-50",
        className
      )}
    />
  );
}

/** A small on/off switch used for the ecosystem factor toggles. */
export function Toggle({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-5 w-9 shrink-0 items-center rounded-full transition-colors disabled:opacity-50",
        checked ? "bg-accent" : "border border-border bg-panel-2"
      )}
    >
      <span
        className={cn(
          "inline-block h-3.5 w-3.5 transform rounded-full bg-white transition",
          checked ? "translate-x-4" : "translate-x-1"
        )}
      />
    </button>
  );
}
