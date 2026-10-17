"use client";

import { useState } from "react";
import ScanPicker from "@/components/ScanPicker";
import PageHeader from "@/components/ui/PageHeader";
import MalignancyCard from "@/components/MalignancyCard";
import ReportPanel from "@/components/ReportPanel";
import ProbabilityBars from "@/components/ui/ProbabilityBars";
import { useObjectUrl } from "@/hooks/useObjectUrl";
import { classifyScan, type ClassifyResult, type ReportRequest } from "@/lib/api";
import { addHistory } from "@/lib/history";
import { composeImage } from "@/lib/image";
import { CLASS_COLORS, CLASS_LABELS, CLASS_NOTES, MALIGNANCY_LABELS } from "@/lib/labels";

export default function ClassifyWorkspace() {
  const [file, setFile] = useState<File | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [result, setResult] = useState<ClassifyResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [view, setView] = useState<"original" | "gradcam">("gradcam");
  const previewUrl = useObjectUrl(file);
  const camUrl = result?.gradcam ? `data:image/png;base64,${result.gradcam.overlay_png_base64}` : null;

  function pick(next: File, id: string | null) {
    setFile(next);
    setSampleId(id);
    setResult(null);
    setError(null);
  }

  function reset() {
    setFile(null);
    setSampleId(null);
    setResult(null);
    setError(null);
  }

  async function buildReport(r: ClassifyResult, side: number): Promise<ReportRequest> {
    const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
    const caution =
      r.gradcam && r.gradcam.focus_in_brain < 0.5
        ? " Grad-CAM shows most model attention outside the head; interpret with caution."
        : "";
    return {
      title: "Brain MRI slice classification",
      modality: "MRI brain · 2D axial slice",
      impression:
        `Appearance most consistent with ${(CLASS_LABELS[r.label] ?? r.label).toLowerCase()} ` +
        `(${pct(r.probabilities[r.label] ?? 0)} probability). Cancer-risk estimate: ` +
        `${MALIGNANCY_LABELS[r.malignancy.assessment].toLowerCase()} (${r.malignancy.confidence} confidence).${caution} ` +
        "Correlate with the full MRI study and histopathology.",
      sections: [
        {
          heading: "Input",
          rows: [
            ["File", file?.name ?? ""],
            ["Model", "VGG16 (ImageNet pretrained, fine-tuned, 4 classes)"],
          ],
        },
        {
          heading: "Prediction",
          rows: [
            ["Predicted class", CLASS_LABELS[r.label] ?? r.label],
            ["Confidence", pct(r.probabilities[r.label] ?? 0)],
          ],
        },
        {
          heading: "Cancer risk estimate (from tumor type)",
          rows: [
            ["Assessment", `${MALIGNANCY_LABELS[r.malignancy.assessment]} (${r.malignancy.confidence} confidence)`],
            ["Cancerous (malignant)", pct(r.malignancy.malignant)],
            ["Non-cancerous (benign)", pct(r.malignancy.benign)],
            ["No tumor", pct(r.malignancy.no_tumor)],
            ["Model uncertainty", pct(r.malignancy.uncertainty)],
            ["Basis", "Glioma counted as cancerous; meningioma and pituitary adenoma as non-cancerous. Grade needs histology."],
          ],
        },
        ...(r.gradcam
          ? [{
              heading: "Explainability (Grad-CAM)",
              rows: [
                ["Target class", CLASS_LABELS[r.gradcam.target_class] ?? r.gradcam.target_class],
                ["Layer", r.gradcam.layer],
                ["Strong activation on head tissue", pct(r.gradcam.focus_in_brain)],
                [
                  "Reading",
                  r.gradcam.focus_in_brain < 0.5
                    ? "Most of the model's focus is outside the head: treat this prediction with caution."
                    : "Model focus lies mainly on head tissue.",
                ],
              ] as [string, string][],
            }]
          : []),
        {
          heading: "Class probabilities",
          rows: Object.entries(r.probabilities)
            .sort(([, a], [, b]) => b - a)
            .map(([k, v]) => [CLASS_LABELS[k] ?? k, pct(v)]),
        },
      ],
      images: [
        ...(previewUrl ? [{ caption: "Input slice", data_url: await composeImage(previewUrl, [], side) }] : []),
        ...(r.gradcam
          ? [{
              caption: `Grad-CAM: regions driving "${CLASS_LABELS[r.label] ?? r.label}"`,
              data_url: await composeImage(`data:image/png;base64,${r.gradcam.overlay_png_base64}`, [], side),
            }]
          : []),
      ],
    };
  }

  async function run() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const r = await classifyScan(file);
      setResult(r);
      setView("gradcam");
      addHistory({
        kind: "classify",
        title: file.name,
        summary: `${CLASS_LABELS[r.label] ?? r.label} · ${((r.probabilities[r.label] ?? 0) * 100).toFixed(1)}% · ${MALIGNANCY_LABELS[r.malignancy.assessment]}`,
        thumb: previewUrl ? await composeImage(previewUrl, [], 160, 0.7).catch(() => null) : null,
        report: await buildReport(r, 320),
        meta: { label: r.label, confidence: r.probabilities[r.label], malignancy: r.malignancy.assessment },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  const confidence = result ? result.probabilities[result.label] ?? 0 : 0;

  return (
    <>
      <PageHeader
        eyebrow="Classification"
        title="Classify brain MRI slice"
        subtitle="Upload an axial slice or pick a sample. The fine-tuned VGG16 returns a probability for each of four classes."
      />

      <div className="grid-main">
        <section className="card" aria-label="Input scan">
          {file && previewUrl ? (
            <div className="viewer-scan">
              {camUrl && (
                <div className="viewer-actions">
                  <div className="segmented" role="group" aria-label="Viewer layer">
                    <button type="button" aria-pressed={view === "gradcam"} onClick={() => setView("gradcam")}>
                      Grad-CAM
                    </button>
                    <button type="button" aria-pressed={view === "original"} onClick={() => setView("original")}>
                      Original
                    </button>
                  </div>
                  {view === "gradcam" && (
                    <span className="legend">
                      <span className="heat-scale" aria-hidden="true" />
                      Low → high influence on the prediction
                    </span>
                  )}
                </div>
              )}
              <div className={`viewer${loading ? " is-scanning" : ""}`}>
                {/* Object URL or base64 PNG from the API; next/image cannot optimise either. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={camUrl && view === "gradcam" ? camUrl : previewUrl}
                  alt={camUrl && view === "gradcam" ? "Grad-CAM heatmap over the MRI slice" : "Selected MRI slice"}
                />
                <span className="viewer-tag mono">{camUrl && view === "gradcam" ? "Grad-CAM · VGG16 conv5" : file.name}</span>
              </div>
              <div className="viewer-actions">
                <button type="button" className="btn" onClick={reset} disabled={loading}>
                  Choose another scan
                </button>
                <button type="button" className="btn btn-primary" onClick={run} disabled={loading}>
                  {loading ? "Analyzing…" : "Run classifier"}
                </button>
              </div>
            </div>
          ) : (
            <ScanPicker onFile={pick} activeSampleId={sampleId} disabled={loading} />
          )}
        </section>

        <aside className="stack" aria-label="Results" aria-live="polite" aria-busy={loading}>
          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Prediction</h2>
              <span className="card-kicker">VGG16</span>
            </div>
            {result ? (
              <div className="verdict">
                <span className="verdict-name" style={{ color: CLASS_COLORS[result.label] }}>
                  {CLASS_LABELS[result.label] ?? result.label}
                </span>
                <span className="muted">
                  Confidence <span className="mono">{(confidence * 100).toFixed(1)}%</span>
                </span>
                <p className="class-note">{CLASS_NOTES[result.label]}</p>
                <div className="bar-track" style={{ marginTop: 6 }}>
                  <div className="verdict-bar" style={{ width: `${confidence * 100}%`, marginTop: 0, height: "100%" }} />
                </div>
              </div>
            ) : (
              <div className="empty">
                {file ? "Run the classifier to see a prediction." : "Select a scan to begin."}
              </div>
            )}
          </div>

          {result?.malignancy && <MalignancyCard data={result.malignancy} />}

          {result?.gradcam && (
            <div className={`alert${result.gradcam.focus_in_brain < 0.5 ? " alert-warn" : " alert-ok"}`} role="status">
              <div>
                <strong>Grad-CAM focus on head tissue: {(result.gradcam.focus_in_brain * 100).toFixed(0)}%</strong>
                <p className="field-hint" style={{ marginTop: 4 }}>
                  {result.gradcam.focus_in_brain < 0.5
                    ? "Most of the model's attention is on image borders or background, not anatomy. This prediction may rely on image artifacts; treat it with caution."
                    : "The model's attention lies mainly on head tissue."}
                </p>
              </div>
            </div>
          )}

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Class probabilities</h2>
            </div>
            {result ? (
              <ProbabilityBars
                items={Object.entries(result.probabilities).map(([key, value]) => ({
                  key,
                  label: CLASS_LABELS[key] ?? key,
                  value,
                  color: CLASS_COLORS[key] ?? "var(--text-muted)",
                }))}
              />
            ) : (
              <div className="empty">Probabilities appear after a run.</div>
            )}
          </div>

          {error && (
            <div className="alert" role="alert">
              {error}
            </div>
          )}

          <ReportPanel getReport={() => buildReport(result!, 512)} disabled={!result || loading} />

          <div className="alert alert-warn">
            Model output is for research and education. It is not a diagnosis.
          </div>
        </aside>
      </div>
    </>
  );
}
