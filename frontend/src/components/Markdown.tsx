import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

// Shared markdown renderer for thesis bodies. `prose-invert` gives readable
// dark-theme typography; gfm enables tables and task-list checkboxes.
export function Markdown({ content, className }: { content: string; className?: string }) {
  return (
    <div
      className={cn(
        "prose prose-invert prose-sm max-w-none",
        "prose-headings:text-text prose-p:text-text/90 prose-li:text-text/90",
        "prose-strong:text-text prose-a:text-accent",
        className
      )}
    >
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content || "_Nothing written yet._"}</ReactMarkdown>
    </div>
  );
}
