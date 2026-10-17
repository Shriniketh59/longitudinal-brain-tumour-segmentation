import type { Malignancy } from "@/lib/api";
import { MALIGNANCY_COLORS, MALIGNANCY_LABELS } from "@/lib/labels";

const ROWS = [
  { key: "malignant", label: "Cancerous (malignant)" },
  { key: "benign", label: "Non-cancerous (benign)" },
  { key: "no_tumor", label: "No tumor" },
] as const;

/** Cancerous vs non-cancerous probability, derived from tumor-type probabilities. */
export default function MalignancyCard({ data }: { data: Malignancy }) {
  const top = data.assessment === "likely_cancerous" ? "malignant" : data.assessment === "likely_non_cancerous" ? "benign" : "no_tumor";
  const total = data.malignant + data.benign + data.no_tumor || 1;

  return (
    <div className="card">
      <div className="card-header">
        <h2 className="card-title">Cancer risk estimate</h2>
        <span className="badge">
          <span className="dot" style={{ background: MALIGNANCY_COLORS[top] }} aria-hidden="true" />
          {data.confidence} confidence
        </span>
      </div>

      <p className="verdict-name" style={{ color: MALIGNANCY_COLORS[top], fontSize: 22 }}>
        {MALIGNANCY_LABELS[data.assessment]}
      </p>

      <div className="stack-bar" role="img" aria-label={ROWS.map((r) => `${r.label} ${(data[r.key] * 100).toFixed(1)}%`).join(", ")}>
        {ROWS.map((r) => (
          <span key={r.key} style={{ width: `${(data[r.key] / total) * 100}%`, background: MALIGNANCY_COLORS[r.key] }} />
        ))}
      </div>

      <dl className="spec-list" style={{ marginTop: 14 }}>
        {ROWS.map((r) => (
          <div key={r.key} style={{ display: "contents" }}>
            <dt>
              <span className="swatch" style={{ background: MALIGNANCY_COLORS[r.key] }} aria-hidden="true" />
              {r.label}
            </dt>
            <dd className="mono">{(data[r.key] * 100).toFixed(1)}%</dd>
          </div>
        ))}
        <dt>Model uncertainty</dt>
        <dd className="mono">{(data.uncertainty * 100).toFixed(0)}%</dd>
      </dl>

      <p className="field-hint" style={{ marginTop: 12 }}>
        Estimated from tumor type: glioma counts as cancerous; meningioma and pituitary adenoma as non-cancerous.
        Grade needs biopsy.
      </p>
    </div>
  );
}
