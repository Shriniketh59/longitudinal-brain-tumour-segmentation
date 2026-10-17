"use client";

import { useEffect, useState } from "react";
import FastToggle from "@/components/FastToggle";
import GrowthChart from "@/components/GrowthChart";
import ReportPanel from "@/components/ReportPanel";
import PageHeader from "@/components/ui/PageHeader";
import {
  analyzeLongitudinal,
  fetchHealth,
  fetchSampleCases,
  loadSampleVolume,
  uploadVolume,
  type LongRegion,
  type LongitudinalResult,
  type RegionVolumes,
  type ReportRequest,
  type SampleCase,
  type SegModel,
  type VolumeResult,
} from "@/lib/api";
import { addHistory } from "@/lib/history";
import { composeImage } from "@/lib/image";

type Source = "brats" | "upload";
type VolumeSource = "ai" | "expert";

type Row = {
  id: number;
  date: string;
  source: Source;
  caseId: string;
  file: File | null;
  flair: File | null;
  mask: File | null;
  status: "idle" | "running" | "done" | "error";
  message?: string;
  ai?: RegionVolumes;
  expert?: RegionVolumes;
  thumb?: string;
};

const REGIONS: { value: LongRegion; label: string }[] = [
  { value: "whole_tumor", label: "Whole tumor" },
  { value: "tumor_core", label: "Tumor core" },
  { value: "enhancing_tumor", label: "Enhancing" },
];

const CATEGORY = {
  progressive_disease: { label: "Progressive disease", color: "var(--status-danger)" },
  stable_disease: { label: "Stable disease", color: "var(--status-warning)" },
  partial_response: { label: "Partial response", color: "var(--status-success)" },
  complete_response: { label: "Complete response", color: "var(--status-success)" },
} as const;

const today = () => new Date().toISOString().slice(0, 10);
const addMonths = (iso: string, m: number) => {
  const d = new Date(iso + "T00:00:00");
  d.setMonth(d.getMonth() + m);
  return d.toISOString().slice(0, 10);
};
const pct = (v: number | null | undefined, digits = 1) => (v === null || v === undefined ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(digits)}%`);
const days = (v: number | null) => (v === null ? "—" : v > 3650 ? "> 10 years" : `${Math.round(v)} days`);

let nextId = 1;
const newRow = (date: string, caseId = ""): Row => ({
  id: nextId++, date, source: "brats", caseId, file: null, flair: null, mask: null, status: "idle",
});

export default function LongitudinalLab() {
  const [cases, setCases] = useState<SampleCase[]>([]);
  const [rows, setRows] = useState<Row[]>(() => [newRow(addMonths(today(), -6)), newRow(today())]);
  const [segModel, setSegModel] = useState<SegModel>("unet2d");
  const [fast, setFast] = useState(false);
  const [swinReady, setSwinReady] = useState(false);
  const [volSource, setVolSource] = useState<VolumeSource>("ai");
  const [region, setRegion] = useState<LongRegion>("whole_tumor");
  const [result, setResult] = useState<LongitudinalResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchSampleCases()
      .then((list) => {
        setCases(list);
        setRows((rs) => rs.map((r, i) => (r.caseId ? r : { ...r, caseId: list[i]?.id ?? list[0]?.id ?? "" })));
      })
      .catch(() => setCases([]));
    fetchHealth().then((h) => setSwinReady(!!h.swin3d_loaded)).catch(() => undefined);
  }, []);

  const update = (id: number, patch: Partial<Row>) => setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  // Changing a timepoint's input invalidates its measurement.
  const edit = (id: number, patch: Partial<Row>) => {
    update(id, { ...patch, status: "idle", ai: undefined, expert: undefined, thumb: undefined, message: undefined });
    setResult(null);
  };

  const done = rows.filter((r) => r.status === "done");
  const expertAvailable = done.length >= 2 && done.every((r) => r.expert);
  const aiAvailable = done.length >= 2 && done.every((r) => r.ai);
  const usingSource: VolumeSource = volSource === "expert" ? (expertAvailable ? "expert" : "ai") : aiAvailable ? "ai" : "expert";
  const bratsRows = rows.filter((r) => r.source === "brats");
  const mixedPatients = new Set(bratsRows.map((r) => r.caseId)).size > 1;

  async function measure(row: Row): Promise<Row> {
    let v: VolumeResult;
    if (row.source === "brats") {
      v = await loadSampleVolume(row.caseId, "t1c", segModel, fast);
    } else {
      if (!row.file) throw new Error("Choose a NIfTI volume");
      v = await uploadVolume(row.file, row.mask, row.flair, segModel, fast);
    }
    const peakArea = v.ground_truth?.slice_area_mm2 ?? v.prediction?.slice_area_mm2 ?? [0];
    const peak = peakArea.indexOf(Math.max(...peakArea));
    const mask = v.prediction?.masks[peak] ?? v.ground_truth?.masks[peak];
    const thumb = await composeImage(
      `data:image/jpeg;base64,${v.frames[peak]}`,
      mask ? [{ src: `data:image/png;base64,${mask}`, opacity: 0.6 }] : [],
      320,
    ).catch(() => undefined);
    return {
      ...row,
      status: "done",
      ai: v.prediction?.volumes_cm3,
      expert: v.ground_truth?.volumes_cm3,
      thumb,
      message: `${v.prediction ? v.prediction.model : "Expert mask only (AI model not loaded)"}${v.cached ? " · cached" : ""}`,
    };
  }

  async function runAll() {
    setBusy(true);
    setError(null);
    setResult(null);
    const out: Row[] = [];
    for (const row of rows) {
      if (row.status === "done") {
        out.push(row);
        continue;
      }
      update(row.id, { status: "running" });
      try {
        const measured = await measure(row);
        update(row.id, measured);
        out.push(measured);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Failed";
        update(row.id, { status: "error", message });
        out.push({ ...row, status: "error", message });
      }
    }
    setBusy(false);
    await compute(out, region, volSource);
  }

  async function compute(list: Row[], reg: LongRegion, src: VolumeSource) {
    const ok = list.filter((r) => r.status === "done");
    if (ok.length < 2) {
      setError("Measure at least two timepoints.");
      return;
    }
    const pick = (r: Row) => (src === "expert" ? r.expert ?? r.ai : r.ai ?? r.expert);
    if (ok.some((r) => !pick(r))) {
      setError("Some timepoints have no volumes for this source.");
      return;
    }
    try {
      setError(null);
      const res = await analyzeLongitudinal(
        ok.map((r, i) => ({ date: r.date, label: `T${i + 1}`, volumes_cm3: pick(r)! })),
        reg,
      );
      setResult(res);
      addHistory({
        kind: "longitudinal",
        title: `${ok.length} timepoints · ${REGIONS.find((x) => x.value === reg)?.label}`,
        summary: `${CATEGORY[res.summary.response_category].label} · ${pct(res.summary.growth_rate_pct_per_year)} per year`,
        thumb: ok[ok.length - 1].thumb ? await composeImage(ok[ok.length - 1].thumb!, [], 160, 0.7).catch(() => null) : null,
        report: buildReport(res, ok),
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    }
  }

  function buildReport(res: LongitudinalResult, ok: Row[]): ReportRequest {
    const regionLabel = REGIONS.find((x) => x.value === res.region)?.label ?? res.region;
    const s = res.summary;
    const f180 = res.forecast.find((f) => f.days_ahead === 180);
    return {
      title: `Longitudinal tumor tracking · ${regionLabel}`,
      modality: "MRI brain · serial studies",
      impression:
        `${CATEGORY[s.response_category].label} (volumetric, RANO-derived). ${regionLabel} volume ` +
        `${s.baseline_cm3.toFixed(1)} → ${s.latest_cm3.toFixed(1)} cm³ (${pct(s.total_change_pct)}) over ${res.timepoints.length} scans; ` +
        `growth ${pct(s.growth_rate_pct_per_year)} per year` +
        (s.doubling_time_days ? `, doubling time ${days(s.doubling_time_days)}` : "") +
        "." +
        (f180?.probability_progression != null
          ? ` Estimated probability of ≥${res.thresholds.progression_pct}% growth within 6 months: ${(f180.probability_progression * 100).toFixed(0)}%.`
          : ""),
      sections: [
        {
          heading: "Timepoints",
          rows: res.timepoints.map((t) => [
            `${t.label || ""} ${t.date}`.trim(),
            `WT ${t.volumes_cm3.whole_tumor.toFixed(2)} · TC ${t.volumes_cm3.tumor_core.toFixed(2)} · ET ${t.volumes_cm3.enhancing_tumor.toFixed(2)} cm³`,
          ]),
        },
        {
          heading: "Change and growth",
          rows: [
            ["Volume source", usingSource === "expert" ? "Expert masks" : `AI segmentation (${segModel === "swin3d" ? "3D SwinUNETR" : "2D U-Net"})`],
            ["Baseline → latest", `${s.baseline_cm3.toFixed(2)} → ${s.latest_cm3.toFixed(2)} cm³ (${pct(s.total_change_pct)})`],
            ["Change from nadir", pct(s.change_from_nadir_pct)],
            ["Growth rate", `${pct(s.growth_rate_pct_per_year)} per year`],
            ["Doubling time", days(s.doubling_time_days)],
            ["Halving time", days(s.halving_time_days)],
            ["Volumetric response (RANO-derived)", CATEGORY[s.response_category].label],
          ],
        },
        {
          heading: "Forecast (exponential growth model)",
          rows: res.forecast.map((f) => [
            `+${f.days_ahead} days (${f.date})`,
            `${f.volume_cm3.toFixed(2)} cm³${f.low_cm3 !== null ? ` [95% ${f.low_cm3.toFixed(2)}–${f.high_cm3!.toFixed(2)}]` : ""}${
              f.probability_progression !== null ? ` · P(progression) ${(f.probability_progression * 100).toFixed(0)}%` : ""
            }`,
          ]),
        },
        {
          heading: "Method",
          rows: [
            ["Growth model", res.model],
            ["Fit", s.fit_quality],
            ["Progression threshold", `+${res.thresholds.progression_pct}% volume vs nadir`],
            ["Partial response threshold", `${res.thresholds.partial_response_pct}% volume vs baseline`],
          ],
        },
      ],
      images: ok
        .filter((r) => r.thumb)
        .slice(-6)
        .map((r, i) => ({ caption: `T${i + 1} · ${r.date} · peak tumor slice`, data_url: r.thumb! })),
    };
  }

  const s = result?.summary;
  const f6 = result?.forecast.find((f) => f.days_ahead === 180);
  const getReport = async () => {
    if (!result) throw new Error("Run the analysis first.");
    return buildReport(result, done);
  };

  return (
    <>
      <PageHeader
        eyebrow="Longitudinal tracking"
        title="Tumor growth over time"
        subtitle="Segment the same patient's scans at several dates, then track volume change, growth rate, response category and a probabilistic growth forecast."
      />

      <section className="card stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <div className="row">
            <span className="field-hint">AI model</span>
            <div className="segmented" role="group" aria-label="AI segmentation model">
              <button type="button" aria-pressed={segModel === "unet2d"} onClick={() => setSegModel("unet2d")}>2D U-Net</button>
              <button type="button" aria-pressed={segModel === "swin3d"} onClick={() => setSegModel("swin3d")} disabled={!swinReady}
                title={swinReady ? "3D Swin Transformer" : "3D model not trained yet"}>
                3D SwinUNETR
              </button>
            </div>
            <FastToggle fast={fast} onChange={(v) => { setFast(v); }} disabled={busy} />
          </div>
          <button type="button" className="btn" onClick={() => setRows((rs) => [...rs, newRow(addMonths(rs[rs.length - 1]?.date ?? today(), 3), cases[rs.length]?.id ?? cases[0]?.id ?? "")])} disabled={busy || rows.length >= 12}>
            Add timepoint
          </button>
        </div>

        <ol className="tp-list">
          {rows.map((row, i) => (
            <li key={row.id} className="tp-row">
              <span className="tp-index mono">T{i + 1}</span>
              <label className="field tp-date">
                <span>Scan date</span>
                <input type="date" value={row.date} onChange={(e) => edit(row.id, { date: e.target.value })} disabled={busy} />
              </label>
              <label className="field tp-source">
                <span>Source</span>
                <select value={row.source} onChange={(e) => edit(row.id, { source: e.target.value as Source })} disabled={busy}>
                  <option value="brats">BraTS case</option>
                  <option value="upload">Upload NIfTI</option>
                </select>
              </label>
              {row.source === "brats" ? (
                <label className="field tp-input">
                  <span>Case</span>
                  <select value={row.caseId} onChange={(e) => edit(row.id, { caseId: e.target.value })} disabled={busy}>
                    {cases.map((c) => <option key={c.id} value={c.id}>{c.id}</option>)}
                  </select>
                </label>
              ) : (
                <div className="tp-input tp-files">
                  <label className="field"><span>{segModel === "swin3d" ? "T1c" : "Volume"}</span>
                    <input type="file" accept=".nii,.gz" onChange={(e) => edit(row.id, { file: e.target.files?.[0] ?? null })} disabled={busy} /></label>
                  {segModel === "swin3d" && (
                    <label className="field"><span>FLAIR</span>
                      <input type="file" accept=".nii,.gz" onChange={(e) => edit(row.id, { flair: e.target.files?.[0] ?? null })} disabled={busy} /></label>
                  )}
                  <label className="field"><span>Expert mask (optional)</span>
                    <input type="file" accept=".nii,.gz" onChange={(e) => edit(row.id, { mask: e.target.files?.[0] ?? null })} disabled={busy} /></label>
                </div>
              )}
              <div className="tp-status">
                {row.status === "running" && <span className="badge"><span className="spinner" aria-hidden="true" />Segmenting…</span>}
                {row.status === "done" && (
                  <span className="badge" title={row.message}>
                    <span className="dot ok" aria-hidden="true" />
                    WT {(row.ai ?? row.expert)!.whole_tumor.toFixed(1)} cm³
                  </span>
                )}
                {row.status === "error" && <span className="badge" style={{ color: "var(--status-danger)" }}>{row.message}</span>}
                <button type="button" className="icon-btn" aria-label={`Remove timepoint ${i + 1}`} disabled={busy || rows.length <= 2}
                  onClick={() => { setRows((rs) => rs.filter((r) => r.id !== row.id)); setResult(null); }}>
                  <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg>
                </button>
              </div>
            </li>
          ))}
        </ol>

        {mixedPatients && (
          <div className="alert alert-warn">
            The BraTS sample cases are different patients scanned once each. Use them only to try the feature; real tracking
            needs the same patient&apos;s scans over time (for example the LUMIERE dataset).
          </div>
        )}

        <div className="row" style={{ justifyContent: "space-between" }}>
          <p className="field-hint">Each timepoint is segmented once; change a row to re-measure it. Runs one scan at a time.</p>
          <button type="button" className="btn btn-primary" onClick={runAll} disabled={busy}>
            {busy ? "Measuring…" : "Measure & analyse"}
          </button>
        </div>
        {error && <div className="alert" role="alert">{error}</div>}
      </section>

      {result && s && (
        <>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div className="segmented" role="group" aria-label="Tumor region">
              {REGIONS.map((r) => (
                <button key={r.value} type="button" aria-pressed={region === r.value}
                  onClick={() => { setRegion(r.value); compute(done, r.value, volSource); }}>
                  {r.label}
                </button>
              ))}
            </div>
            <div className="segmented" role="group" aria-label="Volume source">
              <button type="button" aria-pressed={usingSource === "ai"} disabled={!aiAvailable}
                onClick={() => { setVolSource("ai"); compute(done, region, "ai"); }}>AI volumes</button>
              <button type="button" aria-pressed={usingSource === "expert"} disabled={!expertAvailable}
                onClick={() => { setVolSource("expert"); compute(done, region, "expert"); }}>Expert masks</button>
            </div>
          </div>

          <div className="grid">
            <div className="card">
              <span className="stat-label">Response (volumetric, RANO-derived)</span>
              <p className="stat-value" style={{ color: CATEGORY[s.response_category].color, fontSize: 22 }}>{CATEGORY[s.response_category].label}</p>
              <p className="field-hint">{pct(s.change_from_nadir_pct)} from nadir · {pct(s.total_change_pct)} from baseline</p>
            </div>
            <div className="card">
              <span className="stat-label">Growth rate</span>
              <p className="stat-value mono">{pct(s.growth_rate_pct_per_year, 0)}<span className="muted" style={{ fontSize: 14 }}> / year</span></p>
              <p className="field-hint">{s.doubling_time_days ? `Doubling time ${days(s.doubling_time_days)}` : s.halving_time_days ? `Halving time ${days(s.halving_time_days)}` : "No measurable change"}</p>
            </div>
            <div className="card">
              <span className="stat-label">Forecast in 6 months</span>
              <p className="stat-value mono">{f6 ? `${f6.volume_cm3.toFixed(1)} cm³` : "—"}</p>
              <p className="field-hint">
                {f6?.low_cm3 != null ? `95% range ${f6.low_cm3.toFixed(1)}–${f6.high_cm3!.toFixed(1)} cm³` : "Needs 3+ scans for a range"}
              </p>
            </div>
            <div className="card">
              <span className="stat-label">P(progression) in 6 months</span>
              <p className="stat-value mono">{f6?.probability_progression != null ? `${(f6.probability_progression * 100).toFixed(0)}%` : "—"}</p>
              <p className="field-hint">Chance volume grows ≥ {result.thresholds.progression_pct}% {f6?.probability_progression == null && "· needs 3+ scans"}</p>
            </div>
          </div>

          <div className="grid-main">
            <section className="card">
              <div className="card-header">
                <h2 className="card-title">Volume over time</h2>
                <span className="card-kicker">{REGIONS.find((r) => r.value === region)?.label} · {usingSource === "expert" ? "expert masks" : "AI"}</span>
              </div>
              <GrowthChart data={result} />
            </section>

            <aside className="stack">
              <div className="card">
                <div className="card-header"><h2 className="card-title">Intervals</h2></div>
                <table className="data-table">
                  <thead><tr><th scope="col">Period</th><th scope="col">Days</th><th scope="col">Change</th></tr></thead>
                  <tbody>
                    {result.intervals.map((it) => (
                      <tr key={it.from}>
                        <th scope="row" className="mono" style={{ fontSize: 12 }}>{it.from} → {it.to}</th>
                        <td className="mono">{it.days}</td>
                        <td className="mono">{it.change_cm3 > 0 ? "+" : ""}{it.change_cm3.toFixed(2)} cm³ ({pct(it.change_pct, 0)})</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="card">
                <div className="card-header"><h2 className="card-title">Forecast</h2><span className="card-kicker">exponential model</span></div>
                <table className="data-table">
                  <thead><tr><th scope="col">Horizon</th><th scope="col">Volume</th><th scope="col">P(prog.)</th></tr></thead>
                  <tbody>
                    {result.forecast.map((fc) => (
                      <tr key={fc.days_ahead}>
                        <th scope="row">+{fc.days_ahead} d</th>
                        <td className="mono">{fc.volume_cm3.toFixed(1)}{fc.low_cm3 !== null && <span className="muted"> ({fc.low_cm3.toFixed(1)}–{fc.high_cm3!.toFixed(1)})</span>}</td>
                        <td className="mono">{fc.probability_progression !== null ? `${(fc.probability_progression * 100).toFixed(0)}%` : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <p className="field-hint" style={{ marginTop: 10 }}>
                  Assumes growth continues at the fitted rate. Treatment, surgery or biology can change it at any time.
                </p>
              </div>
              <ReportPanel getReport={getReport} disabled={!result || busy} />
            </aside>
          </div>
        </>
      )}
    </>
  );
}
