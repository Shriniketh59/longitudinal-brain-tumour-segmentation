"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import PageHeader from "@/components/ui/PageHeader";
import { downloadReport } from "@/lib/api";
import { clearHistory, removeHistory, useHistory, type HistoryKind } from "@/lib/history";

const KIND_LABELS: Record<HistoryKind, string> = {
  classify: "Classification",
  segment: "Segmentation",
  volume: "3D volume",
  video: "Video",
  longitudinal: "Longitudinal",
};

const FILTERS: { value: HistoryKind | "all"; label: string }[] = [
  { value: "all", label: "All" },
  { value: "classify", label: "Classification" },
  { value: "segment", label: "Segmentation" },
  { value: "volume", label: "3D volume" },
  { value: "video", label: "Video" },
  { value: "longitudinal", label: "Longitudinal" },
];

const fmt = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });

export default function HistoryView() {
  const entries = useHistory();
  const [filter, setFilter] = useState<HistoryKind | "all">("all");
  const [query, setQuery] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    return entries.filter(
      (e) => (filter === "all" || e.kind === filter) && (!q || `${e.title} ${e.summary}`.toLowerCase().includes(q)),
    );
  }, [entries, filter, query]);

  async function pdf(id: string) {
    const entry = entries.find((e) => e.id === id);
    if (!entry) return;
    setBusyId(id);
    setError(null);
    try {
      await downloadReport(entry.report);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not generate the report.");
    } finally {
      setBusyId(null);
    }
  }

  return (
    <>
      <PageHeader
        eyebrow="History & reports"
        title="Study log"
        subtitle="Every analysis you run is saved in this browser. Re-open any study as a PDF report."
        actions={
          entries.length > 0 ? (
            <button
              type="button"
              className="btn"
              onClick={() => {
                if (window.confirm("Delete all saved studies from this browser?")) clearHistory();
              }}
            >
              Clear history
            </button>
          ) : undefined
        }
      />

      <div className="grid">
        {(["classify", "segment", "volume", "video", "longitudinal"] as HistoryKind[]).map((k) => (
          <div className="card" key={k}>
            <span className="stat-label">{KIND_LABELS[k]}</span>
            <p className="stat-value mono">{entries.filter((e) => e.kind === k).length}</p>
          </div>
        ))}
      </div>

      <section className="card">
        <div className="card-header" style={{ flexWrap: "wrap" }}>
          <div className="segmented" role="group" aria-label="Filter by type">
            {FILTERS.map((f) => (
              <button key={f.value} type="button" aria-pressed={filter === f.value} onClick={() => setFilter(f.value)}>
                {f.label}
              </button>
            ))}
          </div>
          <label className="field field-search">
            <span className="sr-only">Search studies</span>
            <input type="search" placeholder="Search studies" value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>
        </div>

        {error && (
          <div className="alert" role="alert" style={{ marginBottom: 14 }}>
            {error}
          </div>
        )}

        {shown.length === 0 ? (
          <div className="empty">
            {entries.length === 0 ? (
              <>
                No studies yet. Start with <Link href="/analyze">classification</Link>,{" "}
                <Link href="/segmentation">segmentation</Link> or the <Link href="/volume">3D volume lab</Link>.
              </>
            ) : (
              "No studies match this filter."
            )}
          </div>
        ) : (
          <ul className="history-list">
            {shown.map((e) => (
              <li key={e.id} className="history-item">
                <div className="history-thumb">
                  {e.thumb ? (
                    // Small stored data URL thumbnail.
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={e.thumb} alt="" />
                  ) : null}
                </div>
                <div className="history-body">
                  <div className="row" style={{ gap: 8 }}>
                    <span className="badge">{KIND_LABELS[e.kind]}</span>
                    <time className="card-kicker" dateTime={e.createdAt}>
                      {fmt.format(new Date(e.createdAt))}
                    </time>
                  </div>
                  <p className="history-title">{e.title}</p>
                  <p className="muted">{e.summary}</p>
                </div>
                <div className="history-actions">
                  <button type="button" className="btn btn-primary" onClick={() => pdf(e.id)} disabled={busyId !== null}>
                    {busyId === e.id ? "Generating…" : "PDF"}
                  </button>
                  <button type="button" className="btn" onClick={() => removeHistory(e.id)} aria-label={`Delete ${e.title}`}>
                    Delete
                  </button>
                </div>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
