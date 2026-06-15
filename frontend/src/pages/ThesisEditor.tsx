import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Save, Trash2 } from "lucide-react";
import { api } from "@/lib/api";
import type { Conviction, Thesis, ThesisInput, ThesisStatus } from "@/lib/types";
import { thesisTemplate } from "@/lib/thesisTemplate";
import { Markdown } from "@/components/Markdown";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const CONVICTIONS: Conviction[] = ["HIGH", "MEDIUM", "LOW"];
const STATUSES: ThesisStatus[] = ["DRAFT", "ACTIVE", "ARCHIVED"];

const fieldClass =
  "h-8 rounded-lg border border-border bg-transparent px-3 text-sm outline-none focus:border-accent";

export default function ThesisEditor() {
  const { id } = useParams<{ id: string }>();
  const isNew = !id;
  const navigate = useNavigate();
  const qc = useQueryClient();

  const [title, setTitle] = useState("");
  const [symbol, setSymbol] = useState("");
  const [tags, setTags] = useState("");
  const [conviction, setConviction] = useState<Conviction>("MEDIUM");
  const [status, setStatus] = useState<ThesisStatus>("DRAFT");
  const [content, setContent] = useState(() => (isNew ? thesisTemplate() : ""));
  const [busy, setBusy] = useState(false);
  const loaded = useRef(false);

  const { data, isLoading, error } = useQuery<Thesis>({
    queryKey: ["thesis", id],
    queryFn: () => api.thesis(id!),
    enabled: !isNew,
  });

  // Populate the form once when an existing thesis loads.
  useEffect(() => {
    if (data && !loaded.current) {
      loaded.current = true;
      setTitle(data.title);
      setSymbol(data.symbol);
      setTags(data.tags.join(", "));
      setConviction(data.conviction);
      setStatus(data.status);
      setContent(data.content);
    }
  }, [data]);

  const save = async () => {
    if (busy) return;
    setBusy(true);
    const body: ThesisInput = {
      symbol: symbol.trim().toUpperCase(),
      title: title.trim(),
      content,
      tags: tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean),
      conviction,
      status,
    };
    try {
      const saved = isNew
        ? await api.createThesis(body)
        : await api.updateThesis(id!, body);
      qc.invalidateQueries({ queryKey: ["theses"] });
      qc.invalidateQueries({ queryKey: ["thesis", saved.id] });
      navigate(`/theses/${saved.id}`);
    } catch (err) {
      alert((err as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const remove = async () => {
    if (isNew || !confirm("Delete this thesis? This cannot be undone.")) return;
    setBusy(true);
    try {
      await api.deleteThesis(id!);
      qc.invalidateQueries({ queryKey: ["theses"] });
      navigate("/theses");
    } catch (err) {
      alert((err as Error).message);
      setBusy(false);
    }
  };

  if (!isNew && isLoading) {
    return <div className="p-6 text-sm text-muted">Loading thesis…</div>;
  }
  if (!isNew && error) {
    return <div className="p-6 text-sm text-neg">{(error as Error).message}</div>;
  }

  return (
    <div className="space-y-5">
      <Card className="p-4 space-y-3">
        <div className="flex items-center justify-between gap-3">
          <Link to="/theses" className="flex items-center gap-1 text-sm text-muted hover:text-text">
            <ArrowLeft className="h-4 w-4" />
            Theses
          </Link>
          <div className="flex items-center gap-2">
            {!isNew && (
              <Button variant="outline" size="sm" onClick={remove} disabled={busy}>
                <Trash2 className="h-4 w-4" />
                Delete
              </Button>
            )}
            <Button size="sm" onClick={save} disabled={busy}>
              <Save className="h-4 w-4" />
              {isNew ? "Create" : "Save"}
            </Button>
          </div>
        </div>

        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder="Thesis title…"
          className="w-full rounded-lg border border-border bg-transparent px-3 py-2 text-base font-semibold outline-none focus:border-accent"
        />

        <div className="flex flex-wrap items-center gap-3 text-sm">
          <label className="flex items-center gap-2">
            <span className="text-xs uppercase text-muted">Ticker</span>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              placeholder="(thematic)"
              maxLength={8}
              className={`${fieldClass} w-28 uppercase`}
            />
          </label>
          <label className="flex items-center gap-2">
            <span className="text-xs uppercase text-muted">Conviction</span>
            <select
              value={conviction}
              onChange={(e) => setConviction(e.target.value as Conviction)}
              className={fieldClass}
            >
              {CONVICTIONS.map((c) => (
                <option key={c} value={c} className="bg-panel">
                  {c}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <span className="text-xs uppercase text-muted">Status</span>
            <select
              value={status}
              onChange={(e) => setStatus(e.target.value as ThesisStatus)}
              className={fieldClass}
            >
              {STATUSES.map((s) => (
                <option key={s} value={s} className="bg-panel">
                  {s}
                </option>
              ))}
            </select>
          </label>
          <label className="flex flex-1 items-center gap-2">
            <span className="text-xs uppercase text-muted">Tags</span>
            <input
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="comma, separated, tags"
              className={`${fieldClass} min-w-40 flex-1`}
            />
          </label>
        </div>
      </Card>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card className="flex flex-col p-0">
          <div className="border-b border-border px-3 py-2 text-xs uppercase tracking-wide text-muted">
            Markdown
          </div>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            spellCheck
            className="min-h-[60vh] w-full flex-1 resize-y bg-transparent p-4 font-mono text-sm leading-relaxed outline-none"
          />
        </Card>
        <Card className="flex flex-col p-0">
          <div className="border-b border-border px-3 py-2 text-xs uppercase tracking-wide text-muted">
            Preview
          </div>
          <div className="min-h-[60vh] flex-1 overflow-auto p-4">
            <Markdown content={content} />
          </div>
        </Card>
      </div>
    </div>
  );
}
