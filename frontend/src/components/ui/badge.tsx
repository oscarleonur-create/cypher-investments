import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-md px-2 py-0.5 text-xs font-medium border",
  {
    variants: {
      variant: {
        default: "border-border bg-panel-2 text-text",
        pos: "border-pos/30 bg-pos/15 text-pos",
        neg: "border-neg/30 bg-neg/15 text-neg",
        warn: "border-warn/30 bg-warn/15 text-warn",
        muted: "border-border bg-transparent text-muted",
        accent: "border-accent/30 bg-accent/15 text-accent",
      },
    },
    defaultVariants: { variant: "default" },
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badgeVariants> {}

export function Badge({ className, variant, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ variant }), className)} {...props} />;
}
