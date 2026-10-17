export type Malignancy = {
  malignant: number;
  benign: number;
  no_tumor: number;
  assessment: "likely_cancerous" | "likely_non_cancerous" | "no_tumor_detected";
  confidence: "high" | "moderate" | "low";
  /** Normalised entropy of the 4-class distribution: 0 certain, 1 uniform. */
  uncertainty: number;
  basis: string;
};

export type GradCam = {
  /** Heatmap blended over the input slice, original resolution. */
  overlay_png_base64: string;
  /** Strongest activation as fractions of width/height (0..1). */
  peak: { x: number; y: number };
  /** Share of strong activation on head tissue (0..1). Low = model looked at borders/background. */
  focus_in_brain: number;
  target_class: string;
  layer: string;
};

export type ClassifyResult = {
  label: string;
  probabilities: Record<string, number>;
  malignancy: Malignancy;
  gradcam: GradCam | null;
};

export type ClassifierMetrics = {
  accuracy: number;
  classes: string[];
  confusion_matrix: number[][];
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>;
  val_images: number;
  mode: string;
  epochs: number;
};

export type SegmenterMetrics = {
  dice: Record<"whole_tumor" | "tumor_core" | "enhancing_tumor", number | null>;
  best_val_loss: number;
  epochs: number;
  freeze_encoder: boolean;
  train_slices: number;
  val_slices: number;
};

export type SegmentResult = {
  overlay_png_base64: string;
  mask_png_base64: string;
  pixel_counts: Record<string, number>;
  image_size: [number, number];
};

export type HealthResult = {
  status: string;
  device: string;
  classifier_loaded: boolean;
  segmenter_loaded: boolean;
  classifier_preprocess?: string;
  swin3d_loaded?: boolean;
};

export type SegModel = "unet2d" | "swin3d";

export type Swin3dMetrics = {
  model: string;
  dice: Record<"tumor_core" | "whole_tumor" | "enhancing_tumor", number> | null;
  mean_dice: number;
  best_epoch: number;
  epochs: number;
  modalities: string[];
  roi: number[];
  pretrained_encoder: boolean;
  train_cases: number;
  val_cases: number;
};

export type RegionVolumes = {
  whole_tumor: number;
  tumor_core: number;
  enhancing_tumor: number;
};

export type VolumeLayer = {
  masks: string[];
  volumes_cm3: RegionVolumes;
  slice_area_mm2: number[];
};

export type VolumeResult = {
  /** True when the server returned a stored result instead of re-running the models. */
  cached?: boolean;
  id: string;
  modality: string;
  source: string;
  shape: [number, number, number];
  spacing_mm: [number, number, number];
  slice_indices: number[];
  frames: string[];
  ground_truth: VolumeLayer | null;
  prediction: (VolumeLayer & { model?: string; dice: Record<keyof RegionVolumes, number | null> | null }) | null;
};

export type SampleCase = {
  id: string;
  modalities: string[];
  has_ground_truth: boolean;
};

export type PatientInfo = {
  name?: string;
  patient_id?: string;
  age?: number | null;
  sex?: "" | "female" | "male" | "other";
  date_of_birth?: string | null;
  scan_date?: string | null;
  referring_physician?: string;
  institution?: string;
  clinical_indication?: string;
};

export type ReportRequest = {
  /** One or two sentence summary shown in the highlighted Impression box. */
  impression?: string;
  modality?: string;
  patient?: PatientInfo;
  title: string;
  study_id?: string;
  analyst?: string;
  notes?: string;
  sections: { heading: string; rows: [string, string][] }[];
  images: { caption: string; data_url: string }[];
};

/** Empty by default: requests go to this origin and Next.js proxies /api to the Python backend. */
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const payload = (await res.json().catch(() => null)) as { detail?: unknown } | null;
    const detail = typeof payload?.detail === "string" ? payload.detail : null;
    const fallback = res.status >= 500 && res.status !== 503 ? "The analysis server is unreachable." : `Request failed (${res.status})`;
    throw new ApiError(detail ?? fallback, res.status);
  }
  return (await res.json()) as T;
}

async function postFile<T>(path: string, file: File): Promise<T> {
  const body = new FormData();
  body.append("file", file);
  return parse<T>(await fetch(`${API_URL}${path}`, { method: "POST", body }));
}

export const classifyScan = (file: File) => postFile<ClassifyResult>("/api/classify", file);
export const segmentScan = (file: File) => postFile<SegmentResult>("/api/segment", file);

export async function fetchHealth(signal?: AbortSignal): Promise<HealthResult> {
  return parse<HealthResult>(await fetch(`${API_URL}/api/health`, { cache: "no-store", signal }));
}

export async function fetchModelMetrics(): Promise<{
  classifier: ClassifierMetrics | null;
  segmenter: SegmenterMetrics | null;
  swin3d?: Swin3dMetrics | null;
}> {
  return parse(await fetch(`${API_URL}/api/models/metrics`, { cache: "no-store" }));
}

export async function fetchSampleCases(): Promise<SampleCase[]> {
  const data = await parse<{ cases: SampleCase[] }>(await fetch(`${API_URL}/api/volume/samples`));
  return data.cases;
}

export async function loadSampleVolume(
  id: string,
  modality: string,
  model: SegModel = "unet2d",
  fast = false,
  signal?: AbortSignal,
): Promise<VolumeResult> {
  const qs = new URLSearchParams({ modality, model, fast: String(fast) });
  return parse<VolumeResult>(await fetch(`${API_URL}/api/volume/samples/${encodeURIComponent(id)}?${qs}`, { signal }));
}

export async function uploadVolume(
  file: File,
  mask: File | null,
  flair: File | null = null,
  model: SegModel = "unet2d",
  fast = false,
): Promise<VolumeResult> {
  const body = new FormData();
  body.append("fast", String(fast));
  body.append("file", file);
  if (mask) body.append("mask", mask);
  if (flair) body.append("flair", flair);
  body.append("model", model);
  return parse<VolumeResult>(await fetch(`${API_URL}/api/volume/upload`, { method: "POST", body }));
}

export type LongRegion = "whole_tumor" | "tumor_core" | "enhancing_tumor";

export type LongitudinalTimepoint = { date: string; label?: string; volumes_cm3: RegionVolumes };

export type LongitudinalResult = {
  region: LongRegion;
  timepoints: (LongitudinalTimepoint & { day: number })[];
  intervals: { from: string; to: string; days: number; change_cm3: number; change_pct: number | null }[];
  summary: {
    baseline_cm3: number;
    nadir_cm3: number;
    latest_cm3: number;
    total_change_pct: number | null;
    change_from_nadir_pct: number | null;
    growth_rate_pct_per_year: number;
    doubling_time_days: number | null;
    halving_time_days: number | null;
    response_category: "progressive_disease" | "stable_disease" | "partial_response" | "complete_response";
    fit_quality: string;
  };
  forecast: {
    days_ahead: number;
    date: string;
    volume_cm3: number;
    low_cm3: number | null;
    high_cm3: number | null;
    probability_progression: number | null;
  }[];
  curve: { day: number; date: string; volume_cm3: number; low_cm3: number | null; high_cm3: number | null }[];
  model: string;
  thresholds: { progression_pct: number; partial_response_pct: number };
};

export async function analyzeLongitudinal(timepoints: LongitudinalTimepoint[], region: LongRegion): Promise<LongitudinalResult> {
  return parse<LongitudinalResult>(
    await fetch(`${API_URL}/api/longitudinal/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ timepoints, region }),
    }),
  );
}

export type FrameBox = { x: number; y: number; w: number; h: number };

export type VideoFrameResult = {
  label?: string;
  probabilities?: Record<string, number>;
  tumor_probability?: number;
  malignancy?: Malignancy;
  /** Null when suppressed because the classifier found no tumor. */
  mask_png_base64?: string | null;
  suppressed?: string;
  /** Detection box around the largest tumor region, as fractions of frame width/height. */
  box?: FrameBox | null;
  /** Tumor area as % of the frame. */
  area_pct?: number;
  counts?: { necrotic_core: number; edema: number; enhancing: number };
  detected: boolean;
};

/** Classifies, segments and detects tumor in up to 16 frames per call. */
export async function analyzeVideoFrames(files: File[]): Promise<{ frames: VideoFrameResult[]; classifier: boolean; segmenter: boolean }> {
  const body = new FormData();
  files.forEach((f) => body.append("files", f));
  return parse(await fetch(`${API_URL}/api/video/analyze`, { method: "POST", body }));
}

/** Renders the report server-side and triggers a browser download of the PDF. */
export async function downloadReport(req: ReportRequest): Promise<void> {
  const res = await fetch(`${API_URL}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) await parse(res);
  const blob = await res.blob();
  const name = /filename="([^"]+)"/.exec(res.headers.get("Content-Disposition") ?? "")?.[1] ?? "neuroscan-report.pdf";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
