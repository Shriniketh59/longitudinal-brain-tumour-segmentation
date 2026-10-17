"use client";

import Link from "next/link";
import DiceTable from "@/components/metrics/DiceTable";
import { averages, useModelMetrics } from "@/components/metrics/useModelMetrics";
import PageHeader from "@/components/ui/PageHeader";

const mean = (d: Record<string, number | null> | null | undefined) => {
  const v = Object.values(d ?? {}).filter((x): x is number => x != null);
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
};

export default function PerformanceOverview() {
  const state = useModelMetrics();
  const data = state.status === "ok" ? state.data : null;
  const c = data?.classifier ?? null;
  const s = data?.segmenter ?? null;
  const w = data?.swin3d ?? null;
  const avg = c ? averages(c) : null;

  const kpis = [
    { label: "Classifier accuracy", value: c ? `${(c.accuracy * 100).toFixed(1)}%` : "—", note: c ? `${c.val_images} test images` : "Not trained" },
    { label: "Classifier macro F1", value: avg ? avg.macro.f1.toFixed(3) : "—", note: "Mean over 4 classes" },
    { label: "2D U-Net mean Dice", value: s ? mean(s.dice)!.toFixed(3) : "—", note: s ? `${s.val_slices} held-out slices` : "Not trained" },
    { label: "3D SwinUNETR mean Dice", value: w ? w.mean_dice.toFixed(3) : "—", note: w ? `${w.val_cases} held-out patients` : "Not trained" },
  ];

  return (
    <>
      <PageHeader
        eyebrow="Models"
        title="Model performance"
        subtitle="Scores on held-out data the models never saw during training."
        actions={<Link href="/model/classification" className="btn btn-primary">Classification report</Link>}
      />

      {state.status === "offline" && <div className="alert" role="alert">Metrics unavailable: the analysis API is offline.</div>}

      <section className="metric-grid" aria-label="Headline scores">
        {kpis.map((k) => (
          <div className="card kpi" key={k.label}>
            <span className="stat-label">{k.label}</span>
            <p className="kpi-value mono" style={{ marginTop: 12 }}>{k.value}</p>
            <p className="field-hint" style={{ marginTop: 6 }}>{k.note}</p>
          </div>
        ))}
      </section>

      <div className="two-col">
        <section className="card">
          <div className="card-header">
            <h2 className="card-title">2D U-Net · slice segmentation</h2>
            <span className="card-kicker">VGG16 encoder</span>
          </div>
          {s ? (
            <>
              <DiceTable dice={s.dice} />
              <p className="field-hint" style={{ marginTop: 12 }}>
                {s.epochs} epochs · {s.train_slices.toLocaleString()} training slices · {s.val_slices.toLocaleString()} held-out slices · BraTS 2023, split by patient.
              </p>
            </>
          ) : (
            <div className="empty">Not trained yet.</div>
          )}
        </section>

        <section className="card">
          <div className="card-header">
            <h2 className="card-title">3D SwinUNETR · volume segmentation</h2>
            <span className="card-kicker">Swin Transformer</span>
          </div>
          {w?.dice ? (
            <>
              <DiceTable dice={w.dice} />
              <p className="field-hint" style={{ marginTop: 12 }}>
                Whole 3D volumes · T1c + FLAIR · {w.epochs} epochs (best {w.best_epoch}) · {w.train_cases} training / {w.val_cases} held-out patients ·
                {w.pretrained_encoder ? " self-supervised pretrained encoder" : " encoder from scratch"}.
              </p>
            </>
          ) : (
            <div className="empty">Not trained yet.</div>
          )}
        </section>
      </div>

    </>
  );
}
