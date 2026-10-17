export type SampleScan = {
  id: string;
  file: string;
  /** Class folder name the classifier returns for this sample. */
  label: string;
  title: string;
  size: string;
};

export const SAMPLE_SCANS: SampleScan[] = [
  { id: "glioma-1", file: "glioma.jpg", label: "glioma", title: "Glioma · case A", size: "512 × 512" },
  { id: "glioma-2", file: "glioma2.jpg", label: "glioma", title: "Glioma · case B", size: "512 × 512" },
  { id: "meningioma", file: "meningioma.jpg", label: "meningioma", title: "Meningioma", size: "512 × 512" },
  { id: "pituitary", file: "pituitary.jpg", label: "pituitary", title: "Pituitary adenoma", size: "512 × 512" },
  { id: "no-tumor", file: "notumor.jpg", label: "no_tumor", title: "No tumor", size: "350 × 350" },
];

export async function loadSample(sample: SampleScan): Promise<File> {
  const res = await fetch(`/samples/${sample.file}`);
  if (!res.ok) throw new Error(`Could not load sample ${sample.file}`);
  const blob = await res.blob();
  return new File([blob], sample.file, { type: blob.type || "image/jpeg" });
}
