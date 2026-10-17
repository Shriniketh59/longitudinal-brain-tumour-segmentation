"use client";

import { useState } from "react";
import ScanPicker from "@/components/ScanPicker";
import PageHeader from "@/components/ui/PageHeader";
import ReportPanel from "@/components/ReportPanel";
import ProbabilityBars from "@/components/ui/ProbabilityBars";
import { useObjectUrl } from "@/hooks/useObjectUrl";
import { segmentScan, type ReportRequest, type SegmentResult } from "@/lib/api";
import { addHistory } from "@/lib/history";
import { composeImage } from "@/lib/image";
import { SEG_COLORS, SEG_LABELS } from "@/lib/labels";

type View = "overlay" | "original" | "mask";

const VIEWS: { value: View; label: string }[] = [
  { value: "overlay", label: "Overlay" },
  { value: "original", label: "Original" },
  { value: "mask", label: "Mask" },
];

export default function SegmentWorkspace() {
  const [file, setFile] = useState<File | null>(null);
  const [sampleId, setSampleId] = useState<string | null>(null);
  const [result, setResult] = useState<SegmentResult | null>(null);
  const [view, setView] = useState<View>("overlay");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const previewUrl = useObjectUrl(file);

  function pick(next: File, id: string | null) {
    setFile(next);
    setSampleId(id);
    setResult(null);
    setError(null);
    setView("overlay");
  }

  function reset() {
    setFile(null);
    setSampleId(null);
    setResult(null);
    setError(null);
  }

  async function buildReport(r: SegmentResult, side: number): Promise<ReportRequest> {
    const total = r.image_size[0] * r.image_size[1];
    const tumor = Object.entries(r.pixel_counts)
      .filter(([k]) => k !== "background")
      .reduce((sum, [, n]) => sum + n, 0);
    const images: ReportRequest["images"] = [];
    if (previewUrl) images.push({ caption: "Input slice", data_url: await composeImage(previewUrl, [], side) });
    images.push({ caption: "AI segmentation overlay", data_url: await composeImage(`data:image/png;base64,${r.overlay_png_base64}`, [], side) });
    const area = (k: string) => (r.pixel_counts[k] ?? 0).toLocaleString();
    return {
      title: "Brain MRI slice segmentation",
      modality: "MRI brain · 2D axial slice",
      impression:
        tumor === 0
          ? "No tumor region segmented on this slice."
          : `Tumor occupies ${((tumor / total) * 100).toFixed(2)}% of the analysed slice ` +
            `(edema ${area("edema")} px, enhancing tumor ${area("enhancing")} px, necrotic core ${area("necrotic_core")} px). ` +
            "Areas are in pixels; physical size requires the scan's pixel spacing.",
      sections: [
        {
          heading: "Input",
          rows: [
            ["File", file?.name ?? ""],
            ["Image size", `${r.image_size[0]} × ${r.image_size[1]} px`],
            ["Model", "U-Net with VGG16 encoder (ImageNet pretrained)"],
          ],
        },
        {
          heading: "Tumor burden",
          rows: [
            ["Tumor pixels", tumor.toLocaleString()],
            ["Share of slice", `${((tumor / total) * 100).toFixed(2)}%`],
          ],
        },
        {
          heading: "Subregion area (pixels)",
          rows: Object.entries(r.pixel_counts)
            .filter(([k]) => k !== "background")
            .map(([k, n]) => [SEG_LABELS[k] ?? k, n.toLocaleString()]),
        },
      ],
      images,
    };
  }

  async function run() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const r = await segmentScan(file);
      setResult(r);
      setView("overlay");
      const tumor = Object.entries(r.pixel_counts)
        .filter(([k]) => k !== "background")
        .reduce((sum, [, n]) => sum + n, 0);
      addHistory({
        kind: "segment",
        title: file.name,
        summary: `Tumor ${((tumor / (r.image_size[0] * r.image_size[1])) * 100).toFixed(2)}% of slice`,
        thumb: await composeImage(`data:image/png;base64,${r.overlay_png_base64}`, [], 160, 0.7).catch(() => null),
        report: await buildReport(r, 320),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  const imageSrc =
    result && view === "overlay"
      ? `data:image/png;base64,${result.overlay_png_base64}`
      : result && view === "mask"
        ? `data:image/png;base64,${result.mask_png_base64}`
        : previewUrl;

  const totalPx = result ? result.image_size[0] * result.image_size[1] : 0;
  const tumorPx = result
    ? Object.entries(result.pixel_counts)
        .filter(([key]) => key !== "background")
        .reduce((sum, [, n]) => sum + n, 0)
    : 0;

  return (
    <>
      <PageHeader
        eyebrow="Segmentation"
        title="Tumor subregion segmentation"
        subtitle="A U-Net with a pretrained VGG16 encoder labels each pixel as necrotic core, edema, enhancing tumor or background."
      />

      <div className="grid-main">
        <section className="card" aria-label="Input scan">
          {file && imageSrc ? (
            <div className="viewer-scan">
              <div className="viewer-actions">
                <div className="segmented" role="group" aria-label="Viewer layer">
                  {VIEWS.map((v) => (
                    <button
                      key={v.value}
                      type="button"
                      aria-pressed={view === v.value}
                      disabled={!result && v.value !== "original"}
                      onClick={() => setView(v.value)}
                    >
                      {v.label}
                    </button>
                  ))}
                </div>
                <div className="legend" aria-label="Legend">
                  {Object.entries(SEG_COLORS).map(([key, color]) => (
                    <span key={key}>
                      <span className="swatch" style={{ background: color }} aria-hidden="true" />
                      {SEG_LABELS[key]}
                    </span>
                  ))}
                </div>
              </div>
              <div className={`viewer${loading ? " is-scanning" : ""}`}>
                {/* Object URL or base64 PNG from the API; next/image cannot optimise either. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={imageSrc} alt={result ? `Segmentation ${view} view` : "Selected MRI slice"} />
                <span className="viewer-tag mono">{result ? view : file.name}</span>
              </div>
              <div className="viewer-actions">
                <button type="button" className="btn" onClick={reset} disabled={loading}>
                  Choose another scan
                </button>
                <button type="button" className="btn btn-primary" onClick={run} disabled={loading}>
                  {loading ? "Segmenting…" : "Run segmentation"}
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
              <h2 className="card-title">Tumor burden</h2>
              <span className="card-kicker">This slice</span>
            </div>
            {result ? (
              <div className="verdict">
                <span className="verdict-name mono">{((tumorPx / totalPx) * 100).toFixed(2)}%</span>
                <span className="muted">
                  of slice area · <span className="mono">{tumorPx.toLocaleString()}</span> px
                </span>
              </div>
            ) : (
              <div className="empty">{file ? "Run segmentation to measure." : "Select a scan to begin."}</div>
            )}
          </div>

          <div className="card">
            <div className="card-header">
              <h2 className="card-title">Subregion area</h2>
            </div>
            {result ? (
              <ProbabilityBars
                unit="count"
                items={Object.entries(result.pixel_counts)
                  .filter(([key]) => key !== "background")
                  .map(([key, value]) => ({
                    key,
                    label: SEG_LABELS[key] ?? key,
                    value,
                    color: SEG_COLORS[key] ?? "var(--text-muted)",
                    detail: `${value.toLocaleString()} px`,
                  }))}
              />
            ) : (
              <div className="empty">Area per subregion appears after a run.</div>
            )}
          </div>

          {error && (
            <div className="alert" role="alert">
              {error}
            </div>
          )}

          <ReportPanel getReport={() => buildReport(result!, 512)} disabled={!result || loading} />

          <div className="alert alert-warn">
            Areas are in pixels; physical size needs the scan&apos;s voxel spacing.
          </div>
        </aside>
      </div>
    </>
  );
}
