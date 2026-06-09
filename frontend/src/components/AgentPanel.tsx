import * as React from "react";
import { Send, Wrench, History, Plus } from "lucide-react";
import { agentChat, api } from "@/lib/api";
import type { ChatMessage, ConversationSummary } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Section } from "./common";
import { Button } from "./ui/button";

type LiveMessage = ChatMessage & { toolEvents?: { name: string; ok?: boolean }[]; pending?: boolean };

const SUGGESTIONS = [
  "Summarize the bull and bear case in 3 bullets each.",
  "Is the most severe red flag a real concern? Check the latest filing.",
  "What are the near-term catalysts and how should I position?",
];

export function AgentPanel({ symbol }: { symbol: string }) {
  const [messages, setMessages] = React.useState<LiveMessage[]>([]);
  const [input, setInput] = React.useState("");
  const [streaming, setStreaming] = React.useState(false);
  const [conversationId, setConversationId] = React.useState<string | undefined>();
  const [history, setHistory] = React.useState<ConversationSummary[]>([]);
  const [showHistory, setShowHistory] = React.useState(false);
  const scrollRef = React.useRef<HTMLDivElement>(null);

  const loadHistory = React.useCallback(() => {
    api
      .conversations(symbol)
      .then((r) => setHistory(r.conversations))
      .catch(() => setHistory([]));
  }, [symbol]);

  React.useEffect(() => {
    // Reset when the ticker changes.
    setMessages([]);
    setConversationId(undefined);
    setShowHistory(false);
    loadHistory();
  }, [symbol, loadHistory]);

  React.useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [messages]);

  const openConversation = async (id: string) => {
    try {
      const conv = await api.conversation(id);
      setConversationId(conv.id);
      setMessages(
        conv.messages.map((m) => ({
          ...m,
          toolEvents: (m.tools ?? []).map((name) => ({ name })),
        })),
      );
      setShowHistory(false);
    } catch {
      /* ignore */
    }
  };

  const newConversation = () => {
    setConversationId(undefined);
    setMessages([]);
    setShowHistory(false);
  };

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || streaming) return;
    setInput("");
    setStreaming(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", content: q },
      { role: "assistant", content: "", toolEvents: [], pending: true },
    ]);

    const patchAssistant = (fn: (m: LiveMessage) => LiveMessage) =>
      setMessages((prev) => {
        const next = [...prev];
        for (let i = next.length - 1; i >= 0; i--) {
          if (next[i].role === "assistant") {
            next[i] = fn(next[i]);
            break;
          }
        }
        return next;
      });

    try {
      for await (const ev of agentChat(symbol, q, conversationId)) {
        if (ev.type === "meta") {
          setConversationId(ev.conversation_id);
        } else if (ev.type === "tool_call") {
          patchAssistant((m) => ({
            ...m,
            toolEvents: [...(m.toolEvents ?? []), { name: ev.name }],
          }));
        } else if (ev.type === "tool_result") {
          patchAssistant((m) => {
            const evs = [...(m.toolEvents ?? [])];
            for (let i = evs.length - 1; i >= 0; i--) {
              if (evs[i].name === ev.name && evs[i].ok === undefined) {
                evs[i] = { ...evs[i], ok: ev.ok };
                break;
              }
            }
            return { ...m, toolEvents: evs };
          });
        } else if (ev.type === "token") {
          patchAssistant((m) => ({ ...m, content: m.content + ev.text, pending: false }));
        } else if (ev.type === "done") {
          patchAssistant((m) => ({ ...m, content: ev.text || m.content, pending: false }));
        } else if (ev.type === "error") {
          patchAssistant((m) => ({
            ...m,
            content: (m.content ? m.content + "\n\n" : "") + `⚠ ${ev.message}`,
            pending: false,
          }));
        }
      }
    } catch (err) {
      patchAssistant((m) => ({
        ...m,
        content: `⚠ ${(err as Error).message}`,
        pending: false,
      }));
    } finally {
      setStreaming(false);
      loadHistory();
    }
  };

  return (
    <Section
      title="Research Agent"
      right={
        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" title="New chat" onClick={newConversation}>
            <Plus className="h-4 w-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            title="History"
            onClick={() => {
              loadHistory();
              setShowHistory((s) => !s);
            }}
          >
            <History className="h-4 w-4" />
          </Button>
        </div>
      }
    >
      {showHistory && (
        <div className="mb-3 rounded-lg border border-border bg-panel-2 p-2 text-sm">
          {history.length === 0 ? (
            <div className="text-muted">No past conversations.</div>
          ) : (
            history.map((c) => (
              <button
                key={c.id}
                onClick={() => openConversation(c.id)}
                className="flex w-full items-baseline justify-between gap-3 rounded px-2 py-1 text-left hover:bg-panel"
              >
                <span className="truncate">{c.title || "Untitled"}</span>
                <span className="shrink-0 text-xs text-muted">
                  {new Date(c.updated_at).toLocaleDateString()}
                </span>
              </button>
            ))
          )}
        </div>
      )}

      <div ref={scrollRef} className="max-h-[28rem] space-y-3 overflow-y-auto pr-1">
        {messages.length === 0 ? (
          <div className="space-y-2 text-sm text-muted">
            <p>Ask anything about {symbol}. The agent can read the report, search the web, and re-run analysis.</p>
            <div className="flex flex-col gap-1.5">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="rounded-lg border border-border px-3 py-1.5 text-left text-xs hover:bg-panel-2"
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => <MessageBubble key={i} m={m} />)
        )}
      </div>

      <form
        className="mt-3 flex items-end gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          send(input);
        }}
      >
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              send(input);
            }
          }}
          rows={1}
          placeholder={`Ask about ${symbol}…`}
          disabled={streaming}
          className="min-h-[2.25rem] flex-1 resize-none rounded-lg border border-border bg-transparent px-3 py-2 text-sm outline-none focus:border-accent disabled:opacity-50"
        />
        <Button type="submit" size="icon" disabled={streaming || !input.trim()} title="Send">
          <Send className="h-4 w-4" />
        </Button>
      </form>
    </Section>
  );
}

function MessageBubble({ m }: { m: LiveMessage }) {
  const isUser = m.role === "user";
  return (
    <div className={cn("flex flex-col", isUser ? "items-end" : "items-start")}>
      {!isUser && m.toolEvents && m.toolEvents.length > 0 && (
        <div className="mb-1 flex flex-wrap gap-1">
          {m.toolEvents.map((t, i) => (
            <span
              key={i}
              className={cn(
                "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]",
                t.ok === false
                  ? "border-neg/40 text-neg"
                  : t.ok === undefined
                    ? "border-border text-muted animate-pulse"
                    : "border-border text-muted",
              )}
            >
              <Wrench className="h-3 w-3" />
              {t.name}
            </span>
          ))}
        </div>
      )}
      <div
        className={cn(
          "max-w-[85%] whitespace-pre-wrap rounded-lg px-3 py-2 text-sm",
          isUser ? "bg-accent text-white" : "bg-panel-2 text-text",
        )}
      >
        {m.content || (m.pending ? <span className="text-muted">thinking…</span> : "")}
      </div>
    </div>
  );
}
