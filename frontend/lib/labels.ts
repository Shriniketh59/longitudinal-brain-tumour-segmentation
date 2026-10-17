export const CLASS_LABELS: Record<string, string> = {
  glioma: "Glioma",
  meningioma: "Meningioma",
  pituitary: "Pituitary adenoma",
  no_tumor: "No tumor",
};

/** Class accent colours, matched to the reference design system. */
export const CLASS_COLORS: Record<string, string> = {
  glioma: "var(--color-glioma)",
  meningioma: "var(--color-meningioma)",
  pituitary: "var(--color-pituitary)",
  no_tumor: "var(--color-notumor)",
};

export const SEG_LABELS: Record<string, string> = {
  background: "Background",
  necrotic_core: "Necrotic core",
  edema: "Peritumoral edema",
  enhancing: "Enhancing tumor",
};

/** Overlay colours used by the backend renderer (RGB) and the legend. */
export const SEG_COLORS: Record<string, string> = {
  necrotic_core: "#4C9BE8",
  edema: "#FFB547",
  enhancing: "#E56BB0",
};

export const MALIGNANCY_LABELS = {
  likely_cancerous: "Likely cancerous",
  likely_non_cancerous: "Likely non-cancerous",
  no_tumor_detected: "No tumor detected",
} as const;

export const MALIGNANCY_COLORS = {
  malignant: "var(--status-danger)",
  benign: "var(--status-warning)",
  no_tumor: "var(--status-success)",
} as const;

/** Short biology notes shown next to the predicted type. */
export const CLASS_NOTES: Record<string, string> = {
  glioma: "Primary glial tumor. Most gliomas are malignant; low-grade forms exist.",
  meningioma: "Arises from the meninges. About 80% are benign (WHO grade 1).",
  pituitary: "Pituitary adenomas are almost always non-cancerous.",
  no_tumor: "No tumor pattern recognised in this slice.",
};
