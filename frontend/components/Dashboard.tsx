"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import HeroScanner from "@/components/HeroScanner";
import { IconCube, IconHistory, IconLayers, IconScan, IconTrend } from "@/components/shell/icons";
import { downloadReport } from "@/lib/api";
import { useHistory, type HistoryEntry, type HistoryKind } from "@/lib/history";
import { CLASS_COLORS, CLASS_LABELS, MALIGNANCY_COLORS, MALIGNANCY_LABELS } from "@/lib/labels";

const KIND_LABELS: Record<HistoryKind, string> = {
  classify: "Classification",
  segment: "Segmentation",
  volume: "3D volume",
  video: "Video",
  longitudinal: "Longitudinal",
};

const DAY = 86_400_000;
const rtf = new Intl.RelativeTimeFormat(undefined, { numeric: "auto" });

function ago(iso: string): string {
  const diff = (new Date(iso).getTime() - Date.now()) / 1000;
  const steps: [number, Intl.RelativeTimeFormatUnit][] = [[60, "second"], [3600, "minute"], [86400, "hour"], [604800, "day"], [Infinity, "week"]];
  const div = [1, 60, 3600, 86400, 604800];
  const i = steps.findIndex(([limit]) => Math.abs(diff) < limit);
  return rtf.format(Math.round(diff / div[i]), steps[i][1]);
}

/** Counts per day for the last 7 days (oldest first). */
function lastWeek(entries: HistoryEntry[]): number[] {
  const start = new Date();
  start.setHours(0, 0, 0, 0);
  const t0 = start.getTime() - 6 * DAY;
  const out = Array(7).fill(0);
  for (const e of entries) {
    const i = Math.floor((new Date(e.createdAt).getTime() - t0) / DAY);
    if (i >= 0 && i < 7) out[i] += 1;
  }
  return out;
}

function Spark({ values }: { values: number[] }) {
  const max = Math.max(1, ...values);
  return (
    <span className="spark" aria-hidden="true">
      {values.map((v, i) => (
        <span key={i} style={{ height: `${12 + (v / max) * 88}%`, opacity: v ? 1 : 0.35 }} />
      ))}
    </span>
  );
}

const FEATURES = [
  { href: "/analyze", icon: IconScan, title: "Tumor classification", text: "VGG16 · 4 tumor types · cancer-risk probability" },
  { href: "/analyze", icon: IconScan, title: "Grad-CAM explainability", text: "See where the model looked, with a focus check" },
  { href: "/segmentation", icon: IconLayers, title: "Slice segmentation", text: "U-Net masks for core, edema and enhancing tumor" },
  { href: "/volume", icon: IconCube, title: "3D volume & video", text: "Cine playback, expert vs AI masks, cm³ volumes" },
  { href: "/longitudinal", icon: IconTrend, title: "Longitudinal forecast", text: "Growth rate, response category, P(progression)" },
  { href: "/history", icon: IconHistory, title: "Reports & history", text: "Every study saved, A4 PDF in one click" },
];

export default function Dashboard() {
  const entries = useHistory();
  const [busyId, setBusyId] = useState<string | null>(null);

  const stats = useMemo(() => {
    const byKind = (k: HistoryKind[]) => entries.filter((e) => k.includes(e.kind));
    const classify = byKind(["classify"]);
    const labels: Record<string, number> = {};
    const risk: Record<string, number> = {};
    for (const e of classify) {
      if (e.meta?.label) labels[e.meta.label] = (labels[e.meta.label] ?? 0) + 1;
      if (e.meta?.malignancy) risk[e.meta.malignancy] = (risk[e.meta.malignancy] ?? 0) + 1;
    }
    return {
      tiles: [
        { label: "Total studies", value: entries.length, spark: lastWeek(entries) },
        { label: "Classifications", value: classify.length, spark: lastWeek(classify) },
        { label: "Segmentations", value: byKind(["segment"]).length, spark: lastWeek(byKind(["segment"])) },
        { label: "3D volumes & videos", value: byKind(["volume", "video"]).length, spark: lastWeek(byKind(["volume", "video"])) },
        { label: "Longitudinal tracks", value: byKind(["longitudinal"]).length, spark: lastWeek(byKind(["longitudinal"])) },
        { label: "Last 7 days", value: lastWeek(entries).reduce((a, b) => a + b, 0), spark: lastWeek(entries) },
      ],
      labels,
      labelTotal: Object.values(labels).reduce((a, b) => a + b, 0),
      risk,
      riskTotal: Object.values(risk).reduce((a, b) => a + b, 0),
    };
  }, [entries]);

  // Donut segments as conic-gradient stops: each class runs from the running total to total + share.
  const donut = useMemo(() => {
    const sorted = Object.entries(stats.labels).sort(([, a], [, b]) => b - a);
    const stops: string[] = [];
    sorted.reduce((from, [k, n]) => {
      const to = from + (n / stats.labelTotal) * 100;
      stops.push(`${CLASS_COLORS[k] ?? "var(--text-muted)"} ${from}% ${to}%`);
      return to;
    }, 0);
    return stops.join(", ");
  }, [stats]);


  async function pdf(e: HistoryEntry) {
    setBusyId(e.id);
    try {
      await downloadReport(e.report);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <>
      <header className="page-header">
        <div>
          <p className="eyebrow">
            Overview · <time suppressHydrationWarning>{new Date().toLocaleDateString(undefined, { weekday: "long", day: "numeric", month: "long" })}</time>
          </p>
          <h1 className="page-title">
            Neuro-oncology <span className="text-gradient">dashboard</span>
          </h1>
          <p className="page-subtitle">Your studies, results and tools in one place. Everything below updates as you analyse scans.</p>
        </div>
        <div className="row">
          <Link href="/analyze" className="btn btn-primary">New classification</Link>
          <Link href="/volume" className="btn">Open 3D lab</Link>
        </div>
      </header>

      <section className="dash-top" aria-label="Key figures">
        <div className="kpi-grid">
          {stats.tiles.map((t) => (
            <div className="card kpi" key={t.label}>
              <span className="stat-label">{t.label}</span>
              <div className="kpi-row">
                <span className="kpi-value mono">{t.value}</span>
                <Spark values={t.spark} />
              </div>
            </div>
          ))}
        </div>
        <div className="dash-scanner">
          <HeroScanner />
        </div>
      </section>

      <div className="grid-main">
        <section className="card" aria-labelledby="recent-title">
          <div className="card-header">
            <h2 className="card-title" id="recent-title">Recent studies</h2>
            <Link href="/history" className="card-kicker">View all →</Link>
          </div>
          {entries.length === 0 ? (
            <div className="empty">
              No studies yet. Start with a <Link href="/analyze">classification</Link> or load a case in the{" "}
              <Link href="/volume">3D lab</Link>.
            </div>
          ) : (
            <ul className="history-list">
              {entries.slice(0, 6).map((e) => (
                <li key={e.id} className="history-item">
                  <div className="history-thumb">
                    {/* Small stored data URL thumbnail. */}
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    {e.thumb ? <img src={e.thumb} alt="" /> : null}
                  </div>
                  <div className="history-body">
                    <div className="row" style={{ gap: 8 }}>
                      <span className="badge">{KIND_LABELS[e.kind]}</span>
                      <time className="card-kicker" dateTime={e.createdAt} suppressHydrationWarning>{ago(e.createdAt)}</time>
                    </div>
                    <p className="history-title">{e.title}</p>
                    <p className="muted">{e.summary}</p>
                  </div>
                  <div className="history-actions">
                    <button type="button" className="btn" onClick={() => pdf(e)} disabled={busyId !== null}>
                      {busyId === e.id ? "…" : "PDF"}
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </section>

        <aside className="stack" aria-label="Result summary">
          <section className="card">
            <div className="card-header">
              <h2 className="card-title">Prediction mix</h2>
              <span className="card-kicker">{stats.labelTotal} classified</span>
            </div>
            {stats.labelTotal ? (
              <div className="donut-wrap">
                <div className="donut-box">
                  <div className="donut" style={{ background: `conic-gradient(${donut})` }} role="img"
                    aria-label={Object.entries(stats.labels).map(([k, n]) => `${CLASS_LABELS[k] ?? k} ${n}`).join(", ")} />
                  <span className="donut-center mono" aria-hidden="true">{stats.labelTotal}</span>
                </div>
                <ul className="donut-legend">
                  {Object.entries(stats.labels).sort(([, a], [, b]) => b - a).map(([k, n]) => (
                    <li key={k}>
                      <span className="swatch" style={{ background: CLASS_COLORS[k] }} aria-hidden="true" />
                      {CLASS_LABELS[k] ?? k}
                      <span className="mono">{n}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="empty">Classify a scan to see the mix of tumor types.</div>
            )}
          </section>

          <section className="card">
            <div className="card-header">
              <h2 className="card-title">Cancer-risk estimates</h2>
              <span className="card-kicker">from tumor type</span>
            </div>
            {stats.riskTotal ? (
              <>
                <div className="stack-bar" role="img" aria-label="Share of studies by cancer-risk estimate">
                  {(["likely_cancerous", "likely_non_cancerous", "no_tumor_detected"] as const).map((k) => (
                    <span key={k} style={{
                      width: `${((stats.risk[k] ?? 0) / stats.riskTotal) * 100}%`,
                      background: MALIGNANCY_COLORS[k === "likely_cancerous" ? "malignant" : k === "likely_non_cancerous" ? "benign" : "no_tumor"],
                    }} />
                  ))}
                </div>
                <dl className="spec-list" style={{ marginTop: 14 }}>
                  {(["likely_cancerous", "likely_non_cancerous", "no_tumor_detected"] as const).map((k) => (
                    <div key={k} style={{ display: "contents" }}>
                      <dt>{MALIGNANCY_LABELS[k]}</dt>
                      <dd className="mono">{stats.risk[k] ?? 0}</dd>
                    </div>
                  ))}
                </dl>
              </>
            ) : (
              <div className="empty">No classifications yet.</div>
            )}
          </section>
        </aside>
      </div>

      <section aria-labelledby="tools-title">
        <div className="section-head">
          <h2 id="tools-title" className="section-title">Tools</h2>
          <span className="card-kicker">{FEATURES.length} workflows</span>
        </div>
        <div className="feature-grid">
          {FEATURES.map((f) => {
            const Icon = f.icon;
            return (
              <Link key={f.title} href={f.href} className="card feature">
                <span className="feature-icon"><Icon /></span>
                <span className="feature-title">{f.title}</span>
                <span className="feature-text">{f.text}</span>
                <span className="feature-arrow" aria-hidden="true">→</span>
              </Link>
            );
          })}
        </div>
      </section>

    </>
  );
}
