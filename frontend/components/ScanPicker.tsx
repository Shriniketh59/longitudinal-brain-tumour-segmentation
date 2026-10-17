"use client";

import Image from "next/image";
import { useState } from "react";
import { IconScan } from "@/components/shell/icons";
import { SAMPLE_SCANS, loadSample, type SampleScan } from "@/lib/samples";

type Props = {
  onFile: (file: File, sampleId: string | null) => void;
  activeSampleId: string | null;
  disabled?: boolean;
};

export default function ScanPicker({ onFile, activeSampleId, disabled }: Props) {
  const [dragOver, setDragOver] = useState(false);
  const [loadingId, setLoadingId] = useState<string | null>(null);
  const [sampleError, setSampleError] = useState<string | null>(null);

  function acceptFile(file: File | undefined) {
    if (!file) return;
    if (!/^image\/(png|jpeg)$/.test(file.type)) {
      setSampleError("Use a PNG or JPEG image.");
      return;
    }
    setSampleError(null);
    onFile(file, null);
  }

  async function pickSample(sample: SampleScan) {
    setLoadingId(sample.id);
    setSampleError(null);
    try {
      onFile(await loadSample(sample), sample.id);
    } catch (err) {
      setSampleError(err instanceof Error ? err.message : "Could not load sample.");
    } finally {
      setLoadingId(null);
    }
  }

  return (
    <div className="stack">
      <label
        className={`dropzone${dragOver ? " is-over" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          acceptFile(e.dataTransfer.files[0]);
        }}
      >
        <input
          type="file"
          accept="image/png,image/jpeg"
          disabled={disabled}
          onChange={(e) => acceptFile(e.target.files?.[0])}
        />
        <span className="dropzone-icon">
          <IconScan />
        </span>
        <span style={{ fontFamily: "var(--font-heading)", fontWeight: 600, fontSize: 16 }}>
          Drop an MRI slice here, or click to browse
        </span>
        <span className="muted">Axial 2D slice · PNG or JPEG · up to 10 MB</span>
      </label>

      <div className="stack" style={{ gap: 10 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3 className="card-title">Sample scans</h3>
          <span className="card-kicker">No upload needed</span>
        </div>
        <div className="sample-grid">
          {SAMPLE_SCANS.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className="sample"
              aria-pressed={activeSampleId === sample.id}
              aria-label={`Load sample: ${sample.title}`}
              disabled={disabled || loadingId !== null}
              onClick={() => pickSample(sample)}
            >
              <span className="sample-thumb">
                <Image
                  src={`/samples/${sample.file}`}
                  alt=""
                  fill
                  sizes="(max-width: 600px) 45vw, 160px"
                  style={{ objectFit: "cover" }}
                />
              </span>
              <span className="sample-name">{loadingId === sample.id ? "Loading…" : sample.title}</span>
              <span className="sample-meta mono">{sample.size}</span>
            </button>
          ))}
        </div>
        {sampleError && (
          <p className="muted" role="alert" style={{ color: "var(--status-danger)" }}>
            {sampleError}
          </p>
        )}
      </div>
    </div>
  );
}
