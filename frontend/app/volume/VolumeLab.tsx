"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import AreaChart from "@/components/AreaChart";
import CinePlayer, { type CineBox, type CineLayer } from "@/components/CinePlayer";
import FastToggle from "@/components/FastToggle";
import ReportPanel from "@/components/ReportPanel";
import PageHeader from "@/components/ui/PageHeader";
import {
  analyzeVideoFrames,
  fetchHealth,
  fetchSampleCases,
  loadSampleVolume,
  uploadVolume,
  ApiError,
  type ReportRequest,
  type SampleCase,
  type SegModel,
  type VideoFrameResult,
  type VolumeResult,
} from "@/lib/api";
import { addHistory } from "@/lib/history";
import { composeImage, dataUrlToFile } from "@/lib/image";
import { CLASS_COLORS, CLASS_LABELS, MALIGNANCY_LABELS } from "@/lib/labels";
import { extractFrames, type ExtractedVideo } from "@/lib/video";

type Source = "brats" | "nifti" | "video";

const SOURCES: { value: Source; label: string }[] = [
  { value: "brats", label: "BraTS cases" },
  { value: "nifti", label: "Upload NIfTI" },
  { value: "video", label: "Upload video" },
];

const MODALITY_LABELS: Record<string, string> = {
  t1c: "T1 post-contrast",
  t1n: "T1 native",
  t2w: "T2 weighted",
  t2f: "T2 FLAIR",
};

const REGION_LABELS = {
  whole_tumor: "Whole tumor",
  tumor_core: "Tumor core",
  enhancing_tumor: "Enhancing tumor",
} as const;

type RegionKey = keyof typeof REGION_LABELS;

type VideoState = ExtractedVideo & {
  name: string;
  /** Per-frame detection/segmentation/classification; null until analysed. */
  results: (VideoFrameResult | null)[];
};

const VIDEO_BATCH = 12;

type VideoStats = {
  analysed: number;
  detected: number;
  votes: Record<string, number>;
  topLabel: string | null;
  topMalignancy: VideoFrameResult["malignancy"] extends infer M ? (M extends { assessment: infer A } ? A : never) | null : never;
  meanTumorProb: number | null;
  peakIndex: number | null;
  peakArea: number;
  areas: number[];
};

/** Summary over analysed frames: detection count, class votes, peak tumor frame. */
function computeVideoStats(v: VideoState): VideoStats | null {
  const done = v.results.filter((r): r is VideoFrameResult => r !== null);
  if (!done.length) return null;
  const votes: Record<string, number> = {};
  const malVotes: Record<string, number> = {};
  let peakIndex: number | null = null;
  let peakArea = 0;
  const probs: number[] = [];
  v.results.forEach((r, i) => {
    if (!r) return;
    if (r.label) votes[r.label] = (votes[r.label] ?? 0) + 1;
    if (r.detected && r.malignancy) malVotes[r.malignancy.assessment] = (malVotes[r.malignancy.assessment] ?? 0) + 1;
    if (r.detected && r.tumor_probability != null) probs.push(r.tumor_probability);
    if ((r.area_pct ?? 0) > peakArea) {
      peakArea = r.area_pct ?? 0;
      peakIndex = i;
    }
  });
  const top = (rec: Record<string, number>) => Object.entries(rec).sort(([, a], [, b]) => b - a)[0]?.[0] ?? null;
  const tumorVotes = Object.fromEntries(Object.entries(votes).filter(([k]) => k !== "no_tumor"));
  return {
    analysed: done.length,
    detected: done.filter((r) => r.detected).length,
    votes,
    topLabel: top(tumorVotes) ?? top(votes),
    topMalignancy: (top(malVotes) as VideoStats["topMalignancy"]) ?? null,
    meanTumorProb: probs.length ? probs.reduce((a, b) => a + b, 0) / probs.length : null,
    peakIndex,
    peakArea,
    areas: v.results.map((r) => r?.area_pct ?? 0),
  };
}

const png = (b64: string) => `data:image/png;base64,${b64}`;
const jpeg = (b64: string) => `data:image/jpeg;base64,${b64}`;

function argmax(values: number[]): number {
  let best = 0;
  values.forEach((v, i) => {
    if (v > values[best]) best = i;
  });
  return best;
}

export default function VolumeLab() {
  const [source, setSource] = useState<Source>("brats");
  const [cases, setCases] = useState<SampleCase[] | null>(null);
  const [caseId, setCaseId] = useState("");
  const [modality, setModality] = useState("t1c");
  const [niftiFile, setNiftiFile] = useState<File | null>(null);
  const [maskFile, setMaskFile] = useState<File | null>(null);
  const [flairFile, setFlairFile] = useState<File | null>(null);
  const [segModel, setSegModel] = useState<SegModel>("unet2d");
  const [fast, setFast] = useState(false);
  const [swinReady, setSwinReady] = useState(false);

  const [vol, setVol] = useState<VolumeResult | null>(null);
  const [video, setVideo] = useState<VideoState | null>(null);
  const [index, setIndex] = useState(0);
  const [visible, setVisible] = useState<Record<string, boolean>>({ gt: true, pred: true, ai: true });
  const [opacity, setOpacity] = useState(0.6);

  const [busy, setBusy] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchHealth()
      .then((h) => setSwinReady(!!h.swin3d_loaded))
      .catch(() => setSwinReady(false));
  }, []);

  useEffect(() => {
    fetchSampleCases()
      .then((list) => {
        setCases(list);
        if (list[0]) setCaseId(list[0].id);
      })
      .catch(() => setCases([]));
  }, []);

  const selectedCase = cases?.find((c) => c.id === caseId);

  // ---------------- volume loading ----------------

  async function finishVolume(result: VolumeResult) {
    setVideo(null);
    setVol(result);
    const peak = argmax(result.ground_truth?.slice_area_mm2 ?? result.prediction?.slice_area_mm2 ?? [0]);
    setIndex(peak);
    const mask = result.ground_truth?.masks[peak] ?? result.prediction?.masks[peak];
    const thumb = await composeImage(jpeg(result.frames[peak]), mask ? [{ src: png(mask), opacity: 0.6 }] : [], 160, 0.7).catch(() => null);
    const wt = result.ground_truth?.volumes_cm3.whole_tumor ?? result.prediction?.volumes_cm3.whole_tumor;
    addHistory({
      kind: "volume",
      title: result.id,
      summary: `${result.frames.length} slices · ${MODALITY_LABELS[result.modality] ?? result.modality}${wt !== undefined ? ` · WT ${wt.toFixed(1)} cm³` : ""}`,
      thumb,
      report: await buildVolumeReport(result, peak, 320),
      meta: wt !== undefined ? { wholeTumorCm3: wt } : undefined,
    });
  }

  async function loadCase() {
    if (!caseId) return;
    setBusy(
      segModel === "swin3d"
        ? `Running 3D transformer (${fast ? "about 25 s" : "about 1–1.5 min"} unless cached)…`
        : `Segmenting slices (${fast ? "about 25 s" : "about 45 s"} unless cached)…`,
    );
    setError(null);
    try {
      await finishVolume(await loadSampleVolume(caseId, modality, segModel, fast));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load the volume.");
    } finally {
      setBusy(null);
    }
  }

  async function analyseUpload() {
    if (!niftiFile) return;
    setBusy(segModel === "swin3d" ? `Uploading; 3D transformer ${fast ? "about 25 s" : "about 1–1.5 min"}…` : "Uploading and analysing volume…");
    setError(null);
    try {
      await finishVolume(await uploadVolume(niftiFile, maskFile, flairFile, segModel, fast));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not analyse the volume.");
    } finally {
      setBusy(null);
    }
  }

  // ---------------- video ----------------

  async function loadVideo(file: File) {
    setBusy("Extracting frames…");
    setError(null);
    setProgress({ done: 0, total: 1 });
    try {
      const extracted = await extractFrames(file, {
        maxFrames: 60,
        onProgress: (done, total) => setProgress({ done, total }),
      });
      setVol(null);
      const state: VideoState = { ...extracted, name: file.name, results: extracted.frames.map(() => null) };
      setVideo(state);
      setIndex(0);
      setBusy(null);
      setProgress(null);
      // Detection and segmentation start automatically once frames are ready.
      await analyseVideo(state);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not read the video.");
    } finally {
      setBusy(null);
      setProgress(null);
    }
  }

  async function analyseVideo(v: VideoState) {
    setBusy("Detecting and segmenting tumor in frames…");
    setError(null);
    const total = v.frames.length;
    const results: (VideoFrameResult | null)[] = [...v.results];
    setProgress({ done: 0, total });
    try {
      for (let start = 0; start < total; start += VIDEO_BATCH) {
        const idx = Array.from({ length: Math.min(VIDEO_BATCH, total - start) }, (_, k) => start + k);
        const files = await Promise.all(idx.map((i) => dataUrlToFile(v.frames[i], `frame-${i}.jpg`)));
        const res = await analyzeVideoFrames(files);
        idx.forEach((i, k) => (results[i] = res.frames[k]));
        setProgress({ done: start + idx.length, total });
        setVideo((cur) => (cur && cur.name === v.name ? { ...cur, results: [...results] } : cur));
      }
      const done: VideoState = { ...v, results };
      const stats = computeVideoStats(done);
      const peak = stats?.peakIndex ?? Math.floor(total / 2);
      setIndex(peak);
      const mask = results[peak]?.mask_png_base64;
      const thumb = await composeImage(v.frames[peak], mask ? [{ src: png(mask), opacity: 0.6 }] : [], 160, 0.7).catch(() => null);
      addHistory({
        kind: "video",
        title: v.name,
        summary: stats
          ? `Tumor in ${stats.detected}/${stats.analysed} frames${stats.topLabel ? ` · ${CLASS_LABELS[stats.topLabel] ?? stats.topLabel}` : ""}`
          : `${total} frames`,
        thumb,
        report: await buildVideoReport(done, peak, 320),
        meta: stats?.topLabel ? { label: stats.topLabel, malignancy: stats.topMalignancy ?? undefined } : undefined,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Analysis failed.";
      setError(err instanceof ApiError && err.status === 503 ? `${msg}. Load the model weights in backend/weights/.` : msg);
    } finally {
      setBusy(null);
      setProgress(null);
    }
  }

  async function buildVideoReport(v: VideoState, i: number, side: number): Promise<ReportRequest> {
    const stats = computeVideoStats(v);
    const images: ReportRequest["images"] = [];
    const add = async (k: number, caption: string) => {
      const m = v.results[k]?.mask_png_base64;
      images.push({ caption, data_url: await composeImage(v.frames[k], m ? [{ src: png(m), opacity: 0.6 }] : [], side) });
    };
    if (stats?.peakIndex != null) await add(stats.peakIndex, `Frame ${stats.peakIndex + 1} · largest tumor area (${stats.peakArea.toFixed(1)}%)`);
    if (i !== stats?.peakIndex) await add(i, `Frame ${i + 1} · viewer frame`);
    images.unshift({ caption: `Frame ${(stats?.peakIndex ?? i) + 1} · original`, data_url: await composeImage(v.frames[stats?.peakIndex ?? i], [], side) });

    const sections: ReportRequest["sections"] = [videoInfoSection(v, v.name)];
    if (stats) {
      sections.push({
        heading: "Detection & segmentation",
        rows: [
          ["Frames analysed", String(stats.analysed)],
          ["Frames with tumor detected", `${stats.detected} (${((stats.detected / stats.analysed) * 100).toFixed(0)}%)`],
          ["Largest tumor area", stats.peakIndex != null ? `${stats.peakArea.toFixed(2)}% of frame at frame ${stats.peakIndex + 1}` : "None"],
          ["Mean tumor probability (detected frames)", stats.meanTumorProb != null ? `${(stats.meanTumorProb * 100).toFixed(1)}%` : "—"],
          ["Cancer-risk estimate (majority)", stats.topMalignancy ? MALIGNANCY_LABELS[stats.topMalignancy] : "—"],
        ],
      });
      sections.push({
        heading: "Frame-wise classification",
        rows: Object.entries(stats.votes)
          .sort(([, a], [, b]) => b - a)
          .map(([k, n]) => [CLASS_LABELS[k] ?? k, `${n} frames (${((n / stats.analysed) * 100).toFixed(0)}%)`] as [string, string]),
      });
    }
    return {
      title: `Video analysis · ${v.name}`,
      modality: "MRI video · sampled frames",
      impression: stats
        ? stats.detected
          ? `Tumor detected in ${stats.detected} of ${stats.analysed} frames; predominant type ${(CLASS_LABELS[stats.topLabel ?? ""] ?? stats.topLabel ?? "unknown").toLowerCase()}` +
            `${stats.meanTumorProb != null ? ` (mean tumor probability ${(stats.meanTumorProb * 100).toFixed(0)}%)` : ""}. ` +
            `Largest tumor area ${stats.peakArea.toFixed(1)}% of the frame at frame ${(stats.peakIndex ?? 0) + 1}.` +
            `${stats.topMalignancy ? ` Cancer-risk estimate: ${MALIGNANCY_LABELS[stats.topMalignancy].toLowerCase()}.` : ""}`
          : `No tumor detected in ${stats.analysed} analysed frames.`
        : `${v.frames.length} frames reviewed; AI analysis not run.`,
      sections,
      images,
    };
  }

  // ---------------- player wiring ----------------

  const frames = useMemo(() => (vol ? vol.frames.map(jpeg) : video ? video.frames : []), [vol, video]);

  const layers: CineLayer[] = useMemo(() => {
    if (vol) {
      const out: CineLayer[] = [];
      if (vol.ground_truth) out.push({ key: "gt", label: "Ground truth", masks: vol.ground_truth.masks.map(png) });
      if (vol.prediction) out.push({ key: "pred", label: "AI", masks: vol.prediction.masks.map(png) });
      return out;
    }
    if (video && video.results.some((r) => r?.mask_png_base64)) {
      return [{ key: "ai", label: "AI mask + box", masks: video.results.map((r) => (r?.mask_png_base64 ? png(r.mask_png_base64) : null)) }];
    }
    return [];
  }, [vol, video]);

  const boxes: CineBox[] | undefined = useMemo(() => {
    if (!video) return undefined;
    return video.results.map((r) => {
      if (!r?.box) return null;
      const label = r.label ? `${CLASS_LABELS[r.label] ?? r.label} ${((r.probabilities?.[r.label] ?? 0) * 100).toFixed(0)}%` : "Tumor";
      const color = r.label && r.label !== "no_tumor" ? CLASS_COLORS[r.label] ?? "#e5484d" : "#e5484d";
      return { ...r.box, label, color };
    });
  }, [video]);

  const frameLabel = useCallback(
    (i: number) => {
      if (vol) return `Axial z=${vol.slice_indices[i]}`;
      const r = video?.results[i];
      if (!r) return `Frame ${i + 1}`;
      return `Frame ${i + 1} · ${r.detected ? `tumor ${r.area_pct != null ? `${r.area_pct.toFixed(1)}%` : ""}` : "no tumor detected"}`;
    },
    [vol, video],
  );

  const videoStats = useMemo(() => (video ? computeVideoStats(video) : null), [video]);

  // ---------------- reports ----------------

  async function snapshot(i: number, side: number) {
    if (!vol) return null;
    const overlays = [
      ...(vol.ground_truth && visible.gt ? [{ src: png(vol.ground_truth.masks[i]), opacity }] : []),
      ...(vol.prediction && visible.pred ? [{ src: png(vol.prediction.masks[i]), opacity }] : []),
    ];
    return composeImage(jpeg(vol.frames[i]), overlays, side);
  }

  async function buildVolumeReport(result: VolumeResult, i: number, side: number): Promise<ReportRequest> {
    const images: ReportRequest["images"] = [];
    const base = jpeg(result.frames[i]);
    images.push({ caption: `Axial z=${result.slice_indices[i]} · MRI`, data_url: await composeImage(base, [], side) });
    if (result.ground_truth) {
      images.push({ caption: `Axial z=${result.slice_indices[i]} · ground truth`, data_url: await composeImage(base, [{ src: png(result.ground_truth.masks[i]), opacity: 0.6 }], side) });
    }
    if (result.prediction) {
      images.push({ caption: `Axial z=${result.slice_indices[i]} · AI segmentation`, data_url: await composeImage(base, [{ src: png(result.prediction.masks[i]), opacity: 0.6 }], side) });
    }
    const sections: ReportRequest["sections"] = [
      {
        heading: "Study",
        rows: [
          ["Source", result.source],
          ["AI model", result.prediction?.model ?? "Not run"],
          ["Volume", result.id],
          ["Sequence", MODALITY_LABELS[result.modality] ?? result.modality],
          ["Dimensions (voxels)", result.shape.join(" × ")],
          ["Voxel spacing (mm)", result.spacing_mm.join(" × ")],
          ["Slices shown", String(result.frames.length)],
        ],
      },
    ];
    const vols: [string, string][] = [];
    (Object.keys(REGION_LABELS) as RegionKey[]).forEach((k) => {
      if (result.ground_truth) vols.push([`${REGION_LABELS[k]} · ground truth`, `${result.ground_truth.volumes_cm3[k].toFixed(2)} cm³`]);
      if (result.prediction) vols.push([`${REGION_LABELS[k]} · AI`, `${result.prediction.volumes_cm3[k].toFixed(2)} cm³`]);
    });
    if (vols.length) sections.push({ heading: "Volumetrics", rows: vols });
    if (result.prediction?.dice) {
      sections.push({
        heading: "Agreement with ground truth (Dice)",
        rows: (Object.keys(REGION_LABELS) as RegionKey[]).map((k) => [REGION_LABELS[k], result.prediction!.dice![k]?.toFixed(3) ?? "n/a"]),
      });
    }
    if (!result.prediction) {
      sections.push({ heading: "AI segmentation", rows: [["Status", "Not run: segmenter weights not loaded"]] });
    }
    const v = (x: number) => `${x.toFixed(1)} cm³`;
    const p = result.prediction;
    const g = result.ground_truth;
    const impression = p
      ? `${p.model ?? "AI segmentation"} estimates whole tumor ${v(p.volumes_cm3.whole_tumor)}, tumor core ${v(p.volumes_cm3.tumor_core)}, ` +
        `enhancing tumor ${v(p.volumes_cm3.enhancing_tumor)}.` +
        (g ? ` Expert-mask whole tumor ${v(g.volumes_cm3.whole_tumor)}${p.dice?.whole_tumor != null ? ` (Dice ${p.dice.whole_tumor.toFixed(2)})` : ""}.` : "")
      : g
        ? `Expert segmentation: whole tumor ${v(g.volumes_cm3.whole_tumor)}, tumor core ${v(g.volumes_cm3.tumor_core)}, enhancing tumor ${v(g.volumes_cm3.enhancing_tumor)}. AI segmentation not run.`
        : "Volume reviewed; no segmentation available.";
    return {
      title: `3D volume analysis · ${result.id}`,
      modality: `MRI brain · 3D ${MODALITY_LABELS[result.modality] ?? result.modality}`,
      impression,
      sections,
      images,
    };
  }

  function videoInfoSection(v: ExtractedVideo, name: string) {
    return {
      heading: "Video",
      rows: [
        ["File", name],
        ["Duration", `${v.duration.toFixed(2)} s`],
        ["Frames sampled", String(v.frames.length)],
        ["Frame size", `${v.width} × ${v.height}`],
      ] as [string, string][],
    };
  }

  async function getReport(): Promise<ReportRequest> {
    if (vol) {
      const report = await buildVolumeReport(vol, index, 512);
      const current = await snapshot(index, 512);
      if (current) report.images.push({ caption: `Viewer snapshot · axial z=${vol.slice_indices[index]}`, data_url: current });
      return report;
    }
    if (video) return buildVideoReport(video, index, 512);
    throw new Error("Nothing to report yet.");
  }

  // ---------------- render ----------------

  const hasMedia = frames.length > 0;
  const areaSeries = vol
    ? [
        ...(vol.ground_truth ? [{ label: "Ground truth area", values: vol.ground_truth.slice_area_mm2, color: "var(--accent-cyan)" }] : []),
        ...(vol.prediction ? [{ label: "AI area", values: vol.prediction.slice_area_mm2, color: "var(--accent-purple)", dashed: true }] : []),
      ]
    : [];

  return (
    <>
      <PageHeader
        eyebrow="3D volume & video lab"
        title="Volumetric MRI and video studies"
        subtitle="Play 3D MRI volumes slice by slice, compare AI segmentation with expert masks, measure tumor volume in cm³, and analyse brain MRI videos frame by frame."
      />

      <section className="card">
        <div className="card-header">
          <div className="segmented" role="group" aria-label="Data source">
            {SOURCES.map((s) => (
              <button key={s.value} type="button" aria-pressed={source === s.value} onClick={() => setSource(s.value)}>
                {s.label}
              </button>
            ))}
          </div>
          {busy && (
            <span className="badge" role="status">
              <span className="spinner" aria-hidden="true" />
              {busy}
              {progress && progress.total > 1 && <span className="mono"> {progress.done}/{progress.total}</span>}
            </span>
          )}
        </div>

        {source !== "video" && (
          <div className="row" style={{ marginBottom: 14 }}>
            <span className="field-hint">AI segmentation model</span>
            <div className="segmented" role="group" aria-label="AI segmentation model">
              <button type="button" aria-pressed={segModel === "unet2d"} onClick={() => setSegModel("unet2d")}>
                2D U-Net · slice by slice
              </button>
              <button
                type="button"
                aria-pressed={segModel === "swin3d"}
                onClick={() => setSegModel("swin3d")}
                disabled={!swinReady}
                title={swinReady ? "Whole-volume 3D Swin Transformer" : "Train the 3D model in Colab and place swin_unetr_brats.pt in backend/weights/"}
              >
                3D SwinUNETR · transformer
              </button>
            </div>
            {!swinReady && <span className="field-hint">3D model not trained yet</span>}
            <FastToggle fast={fast} onChange={setFast} disabled={!!busy} />
          </div>
        )}

        {source === "brats" && (
          <div className="source-row">
            {cases === null ? (
              <p className="muted">Loading case list…</p>
            ) : cases.length === 0 ? (
              <p className="muted">
                No BraTS cases on the server. Run <span className="mono">python -m scripts.download_brats</span> in{" "}
                <span className="mono">backend/</span>.
              </p>
            ) : (
              <>
                <label className="field">
                  <span>Case ({cases.length} available)</span>
                  <select value={caseId} onChange={(e) => setCaseId(e.target.value)}>
                    {cases.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.id}
                        {c.has_ground_truth ? " · expert mask" : ""}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span>Sequence</span>
                  <select value={modality} onChange={(e) => setModality(e.target.value)}>
                    {(selectedCase?.modalities ?? ["t1c"]).map((m) => (
                      <option key={m} value={m}>
                        {MODALITY_LABELS[m] ?? m}
                      </option>
                    ))}
                  </select>
                </label>
                <button type="button" className="btn btn-primary" onClick={loadCase} disabled={!!busy || !caseId}>
                  Load volume
                </button>
              </>
            )}
          </div>
        )}

        {source === "nifti" && (
          <div className="source-row">
            <label className="field">
              <span>{segModel === "swin3d" ? "T1 post-contrast volume (.nii, .nii.gz)" : "MRI volume (.nii, .nii.gz)"}</span>
              <input type="file" accept=".nii,.gz" onChange={(e) => setNiftiFile(e.target.files?.[0] ?? null)} />
            </label>
            <label className="field">
              <span>Expert mask (optional)</span>
              <input type="file" accept=".nii,.gz" onChange={(e) => setMaskFile(e.target.files?.[0] ?? null)} />
            </label>
            {segModel === "swin3d" && (
              <label className="field">
                <span>FLAIR volume (required for 3D model)</span>
                <input type="file" accept=".nii,.gz" onChange={(e) => setFlairFile(e.target.files?.[0] ?? null)} />
              </label>
            )}
            <button
              type="button"
              className="btn btn-primary"
              onClick={analyseUpload}
              disabled={!!busy || !niftiFile || (segModel === "swin3d" && !flairFile)}
            >
              Analyse volume
            </button>
          </div>
        )}

        {source === "video" && (
          <div className="source-row">
            <label className="field">
              <span>Brain MRI video (MP4, WebM, MOV)</span>
              <input
                type="file"
                accept="video/mp4,video/webm,video/quicktime,video/*"
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) loadVideo(f);
                }}
                disabled={!!busy}
              />
            </label>
            <button type="button" className="btn btn-primary" onClick={() => video && analyseVideo(video)} disabled={!!busy || !video}>
              {video?.results.some(Boolean) ? "Run detection again" : "Detect & segment"}
            </button>
            <p className="field-hint" style={{ flexBasis: "100%" }}>
              Up to 60 evenly spaced frames are extracted in your browser, then each frame is classified, segmented and
              checked for tumor automatically. Only the frames are sent for analysis, never the video file.
            </p>
          </div>
        )}

        {progress && progress.total > 1 && (
          <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total} aria-valuenow={progress.done}>
            <div style={{ width: `${(progress.done / progress.total) * 100}%` }} />
          </div>
        )}
        {error && (
          <div className="alert" role="alert" style={{ marginTop: 14 }}>
            {error}
          </div>
        )}
      </section>

      {!hasMedia ? (
        <div className="card empty empty-lg">
          <span className="dropzone-icon" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 3 3 8l9 5 9-5-9-5Z" /><path d="m3 13 9 5 9-5M3 17l9 5 9-5" /></svg>
          </span>
          <strong>No study loaded</strong>
          <span>Pick a BraTS case, upload a NIfTI volume, or upload a brain MRI video.</span>
        </div>
      ) : (
        <div className="grid-main">
          <section className="card stack" aria-label="Viewer">
            <CinePlayer
              frames={frames}
              layers={layers}
              boxes={video ? boxes : undefined}
              aspect={video ? video.width / video.height : 1}
              index={Math.min(index, frames.length - 1)}
              onIndexChange={setIndex}
              frameLabel={frameLabel}
              visibleLayers={visible}
              onToggleLayer={(k) => setVisible((v) => ({ ...v, [k]: !v[k] }))}
              opacity={opacity}
              onOpacityChange={setOpacity}
            />
            {video && videoStats && (
              <div>
                <div className="card-header" style={{ marginBottom: 8 }}>
                  <h2 className="card-title">Detection timeline</h2>
                  <span className="card-kicker">Click a frame to jump</span>
                </div>
                <div className="timeline" role="group" aria-label="Tumor detection per frame">
                  {video.results.map((r, i) => (
                    <button
                      key={i}
                      type="button"
                      aria-label={`Frame ${i + 1}: ${r ? (r.detected ? "tumor detected" : "no tumor") : "not analysed"}`}
                      aria-current={i === index}
                      onClick={() => setIndex(i)}
                      style={{
                        background: !r
                          ? undefined
                          : r.detected
                            ? (r.label && r.label !== "no_tumor" ? CLASS_COLORS[r.label] : "var(--status-danger)")
                            : "color-mix(in oklch, var(--status-success) 35%, var(--bg-elevated))",
                      }}
                    />
                  ))}
                </div>
                <div style={{ marginTop: 16 }}>
                  <AreaChart
                    series={[{ label: "Tumor area per frame", values: videoStats.areas, color: "var(--accent-cyan)" }]}
                    index={index}
                    onSeek={setIndex}
                    unit="% of frame"
                  />
                </div>
              </div>
            )}
            {vol && areaSeries.length > 0 && (
              <div>
                <div className="card-header" style={{ marginBottom: 8 }}>
                  <h2 className="card-title">Tumor area by slice</h2>
                  <span className="card-kicker">Click to jump</span>
                </div>
                <AreaChart series={areaSeries} index={index} onSeek={setIndex} />
              </div>
            )}
          </section>

          <aside className="stack" aria-label="Measurements" aria-live="polite">
            {vol && (
              <>
                <div className="card">
                  <div className="card-header">
                    <h2 className="card-title">Study</h2>
                    <span className="card-kicker">{vol.source}</span>
                  </div>
                  <dl className="spec-list">
                    <dt>Volume</dt>
                    <dd className="mono">{vol.id}</dd>
                    <dt>AI model</dt>
                    <dd>
                      {vol.prediction?.model ?? "Not run"}
                      {vol.cached && <span className="badge" style={{ marginLeft: 8 }}>instant · cached</span>}
                    </dd>
                    <dt>Sequence</dt>
                    <dd>{MODALITY_LABELS[vol.modality] ?? vol.modality}</dd>
                    <dt>Voxels</dt>
                    <dd className="mono">{vol.shape.join(" × ")}</dd>
                    <dt>Spacing</dt>
                    <dd className="mono">{vol.spacing_mm.join(" × ")} mm</dd>
                    <dt>Slices</dt>
                    <dd className="mono">{vol.frames.length}</dd>
                  </dl>
                </div>

                <div className="card">
                  <div className="card-header">
                    <h2 className="card-title">Tumor volume</h2>
                    <span className="card-kicker">cm³</span>
                  </div>
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th scope="col">Region</th>
                        {vol.ground_truth && <th scope="col">Expert</th>}
                        <th scope="col">AI</th>
                        {vol.prediction?.dice && <th scope="col">Dice</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {(Object.keys(REGION_LABELS) as RegionKey[]).map((k) => (
                        <tr key={k}>
                          <th scope="row">{REGION_LABELS[k]}</th>
                          {vol.ground_truth && <td className="mono">{vol.ground_truth.volumes_cm3[k].toFixed(2)}</td>}
                          <td className="mono">{vol.prediction ? vol.prediction.volumes_cm3[k].toFixed(2) : "—"}</td>
                          {vol.prediction?.dice && <td className="mono">{vol.prediction.dice[k]?.toFixed(3) ?? "n/a"}</td>}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  {!vol.prediction && <p className="field-hint">AI columns fill in once U-Net weights are loaded.</p>}
                </div>
              </>
            )}

            {video && (
              <>
                <div className="card">
                  <div className="card-header">
                    <h2 className="card-title">Detection summary</h2>
                    <span className="card-kicker">{videoStats ? `${videoStats.analysed}/${video.frames.length} analysed` : "pending"}</span>
                  </div>
                  {videoStats ? (
                    <div className="stack">
                      <div className="row" style={{ alignItems: "baseline", gap: 10 }}>
                        <span className="verdict-name mono" style={{ color: videoStats.detected ? "var(--status-danger)" : "var(--status-success)" }}>
                          {videoStats.detected}/{videoStats.analysed}
                        </span>
                        <span className="muted">frames with tumor detected</span>
                      </div>
                      <dl className="spec-list">
                        <dt>Predominant type</dt>
                        <dd>
                          {videoStats.topLabel ? (
                            <>
                              <span className="swatch" style={{ background: CLASS_COLORS[videoStats.topLabel] }} aria-hidden="true" />
                              {CLASS_LABELS[videoStats.topLabel] ?? videoStats.topLabel}
                            </>
                          ) : "—"}
                        </dd>
                        <dt>Tumor probability</dt>
                        <dd className="mono">{videoStats.meanTumorProb != null ? `${(videoStats.meanTumorProb * 100).toFixed(1)}% (mean)` : "—"}</dd>
                        <dt>Cancer risk</dt>
                        <dd>{videoStats.topMalignancy ? MALIGNANCY_LABELS[videoStats.topMalignancy] : "—"}</dd>
                        <dt>Largest tumor</dt>
                        <dd>
                          {videoStats.peakIndex != null ? (
                            <button type="button" className="btn btn-ghost" style={{ minHeight: 0, padding: "2px 8px" }} onClick={() => setIndex(videoStats.peakIndex!)}>
                              {videoStats.peakArea.toFixed(1)}% · frame {videoStats.peakIndex + 1} →
                            </button>
                          ) : "None"}
                        </dd>
                      </dl>
                      <div>
                        <p className="card-title" style={{ marginBottom: 8 }}>Frame-wise classification</p>
                        {Object.entries(videoStats.votes)
                          .sort(([, a], [, b]) => b - a)
                          .map(([k, n]) => (
                            <div className="bar-row" key={k}>
                              <div className="bar-head">
                                <span>
                                  <span className="swatch" style={{ background: CLASS_COLORS[k] }} aria-hidden="true" />
                                  {CLASS_LABELS[k] ?? k}
                                </span>
                                <span className="mono">{n} frames</span>
                              </div>
                              <div className="bar-track">
                                <div className="bar-fill" style={{ width: `${(n / videoStats.analysed) * 100}%`, background: CLASS_COLORS[k] ?? "var(--accent-cyan)" }} />
                              </div>
                            </div>
                          ))}
                      </div>
                    </div>
                  ) : (
                    <div className="empty">Analysis starts automatically after the frames are extracted.</div>
                  )}
                  <p className="field-hint" style={{ marginTop: 12 }}>
                    Tumor type comes from the VGG16 classifier. Masks and boxes come from the U-Net, which was trained on
                    skull-stripped glioma scans (BraTS): on other tumor types or scans that include the skull they can be
                    misplaced. A frame counts as a detection only when both models agree.
                  </p>
                </div>

                <div className="card">
                  <div className="card-header">
                    <h2 className="card-title">Video</h2>
                    <span className="card-kicker mono">{video.duration.toFixed(1)} s</span>
                  </div>
                  <dl className="spec-list">
                    <dt>File</dt>
                    <dd className="mono" style={{ overflowWrap: "anywhere" }}>{video.name}</dd>
                    <dt>Frames</dt>
                    <dd className="mono">{video.frames.length}</dd>
                    <dt>Size</dt>
                    <dd className="mono">{video.width} × {video.height}</dd>
                  </dl>
                </div>
              </>
            )}

            <ReportPanel getReport={getReport} disabled={!hasMedia || !!busy} />
          </aside>
        </div>
      )}
    </>
  );
}
