"use client";

import Link from "next/link";
import ConfusionMatrix from "@/components/metrics/ConfusionMatrix";
import { averages, useModelMetrics } from "@/components/metrics/useModelMetrics";
import PageHeader from "@/components/ui/PageHeader";
import { CLASS_COLORS, CLASS_LABELS } from "@/lib/labels";

const f3 = (v: number) => v.toFixed(3);

export default function ClassificationReportView() {
  const state = useModelMetrics();
  const c = state.status === "ok" ? state.data.classifier : null;
  const avg = c ? averages(c) : null;

  // Weakest class by recall, for the reading note.
  const weakest = c ? [...c.classes].sort((a, b) => c.per_class[a].recall - c.per_class[b].recall)[0] : null;

  return (
    <>
      <PageHeader
        eyebrow="VGG16 classifier"
        title="Classification report"
        subtitle="Accuracy, confusion matrix and per-class precision, recall and F1 on the held-out test split."
        actions={<Link href="/model" className="btn">← Model performance</Link>}
      />

      {state.status === "offline" && <div className="alert" role="alert">Metrics unavailable: the analysis API is offline.</div>}
      {state.status === "ok" && !c && <div className="card empty">The classifier has not been trained yet.</div>}

      {c && avg && (
        <>
          <section className="metric-grid" aria-label="Summary">
            {[
              { label: "Accuracy", value: `${(c.accuracy * 100).toFixed(2)}%` },
              { label: "Macro precision", value: f3(avg.macro.precision) },
              { label: "Macro recall", value: f3(avg.macro.recall) },
              { label: "Macro F1", value: f3(avg.macro.f1) },
              { label: "Test images", value: c.val_images.toLocaleString() },
            ].map((k) => (
              <div className="card kpi" key={k.label}>
                <span className="stat-label">{k.label}</span>
                <p className="kpi-value mono" style={{ marginTop: 12 }}>{k.value}</p>
              </div>
            ))}
          </section>

          <section className="card">
            <div className="card-header">
              <h2 className="card-title">Confusion matrix</h2>
              <span className="card-kicker">rows = true class · columns = predicted</span>
            </div>
            <ConfusionMatrix data={c} />
            <div className="legend" style={{ marginTop: 14 }}>
              <span><span className="swatch" style={{ background: "var(--accent-cyan)" }} aria-hidden="true" />Correct (diagonal)</span>
              <span><span className="swatch" style={{ background: "var(--status-danger)" }} aria-hidden="true" />Misclassified</span>
              <span>Shade = share of that true class</span>
            </div>
          </section>

          <section className="card">
            <div className="card-header">
              <h2 className="card-title">Per-class report</h2>
              <span className="card-kicker">{c.mode} · {c.epochs} epochs</span>
            </div>
            <div className="table-scroll">
              <table className="data-table report-table">
                <thead>
                  <tr>
                    <th scope="col">Class</th>
                    <th scope="col">Precision</th>
                    <th scope="col">Recall</th>
                    <th scope="col">F1-score</th>
                    <th scope="col">Support</th>
                  </tr>
                </thead>
                <tbody>
                  {c.classes.map((k) => {
                    const r = c.per_class[k];
                    return (
                      <tr key={k}>
                        <th scope="row">
                          <span className="swatch" style={{ background: CLASS_COLORS[k] }} aria-hidden="true" />
                          {CLASS_LABELS[k] ?? k}
                        </th>
                        <td className="mono">{f3(r.precision)}</td>
                        <td className="mono">{f3(r.recall)}</td>
                        <td className="mono">{f3(r.f1)}</td>
                        <td className="mono">{r.support}</td>
                      </tr>
                    );
                  })}
                </tbody>
                <tfoot>
                  <tr>
                    <th scope="row">Accuracy</th>
                    <td />
                    <td />
                    <td className="mono">{f3(c.accuracy)}</td>
                    <td className="mono">{avg.total}</td>
                  </tr>
                  <tr>
                    <th scope="row">Macro average</th>
                    <td className="mono">{f3(avg.macro.precision)}</td>
                    <td className="mono">{f3(avg.macro.recall)}</td>
                    <td className="mono">{f3(avg.macro.f1)}</td>
                    <td className="mono">{avg.total}</td>
                  </tr>
                  <tr>
                    <th scope="row">Weighted average</th>
                    <td className="mono">{f3(avg.weighted.precision)}</td>
                    <td className="mono">{f3(avg.weighted.recall)}</td>
                    <td className="mono">{f3(avg.weighted.f1)}</td>
                    <td className="mono">{avg.total}</td>
                  </tr>
                </tfoot>
              </table>
            </div>
          </section>

          {weakest && (
            <div className="alert alert-warn">
              <div>
                <strong>Reading note:</strong> the weakest class is {CLASS_LABELS[weakest] ?? weakest} (recall{" "}
                {(c.per_class[weakest].recall * 100).toFixed(1)}%). Grad-CAM showed the current model often bases glioma
                predictions on image borders rather than the tumor; the head-cropped retraining notebook addresses this.
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
