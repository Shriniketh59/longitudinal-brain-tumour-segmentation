"use client";

import { useEffect, useId, useState } from "react";
import { downloadReport, type PatientInfo, type ReportRequest } from "@/lib/api";

type Props = {
  /** Builds the analysis-specific part of the report (title, impression, sections, images). */
  getReport: () => Promise<ReportRequest>;
  disabled?: boolean;
};

// Only clinic-level details are remembered on this device; patient details never are.
const PREF_KEY = "ns-report-prefs";

type Prefs = { institution: string; referring_physician: string; analyst: string };

function readPrefs(): Prefs {
  try {
    return { institution: "", referring_physician: "", analyst: "", ...JSON.parse(localStorage.getItem(PREF_KEY) ?? "{}") };
  } catch {
    return { institution: "", referring_physician: "", analyst: "" };
  }
}

export default function ReportPanel({ getReport, disabled }: Props) {
  const id = useId();
  const [patient, setPatient] = useState<PatientInfo>({ sex: "" });
  const [studyId, setStudyId] = useState("");
  const [analyst, setAnalyst] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Restore clinic preferences after mount (localStorage is unavailable during server render).
    const p = readPrefs();
    queueMicrotask(() => {
      setPatient((cur) => ({ ...cur, institution: p.institution, referring_physician: p.referring_physician }));
      setAnalyst(p.analyst);
    });
  }, []);

  const set = (patch: Partial<PatientInfo>) => setPatient((p) => ({ ...p, ...patch }));

  async function generate() {
    setBusy(true);
    setError(null);
    try {
      try {
        localStorage.setItem(PREF_KEY, JSON.stringify({
          institution: patient.institution ?? "", referring_physician: patient.referring_physician ?? "", analyst,
        }));
      } catch {
        // storage blocked: preferences just aren't remembered
      }
      const base = await getReport();
      const age = patient.age === null || patient.age === undefined || Number.isNaN(patient.age) ? null : patient.age;
      await downloadReport({
        ...base,
        study_id: studyId.trim(),
        analyst: analyst.trim(),
        notes: notes.trim(),
        patient: {
          ...patient,
          age,
          name: patient.name?.trim() ?? "",
          patient_id: patient.patient_id?.trim() ?? "",
          date_of_birth: patient.date_of_birth || null,
          scan_date: patient.scan_date || null,
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not generate the report.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="card" aria-labelledby={`${id}-title`}>
      <div className="card-header">
        <h2 className="card-title" id={`${id}-title`}>PDF report</h2>
        <span className="card-kicker">A4 · clinical format</span>
      </div>

      <div className="form">
        <fieldset className="fieldset">
          <legend>Patient</legend>
          <label className="field">
            <span>Full name</span>
            <input value={patient.name ?? ""} onChange={(e) => set({ name: e.target.value })} maxLength={120} autoComplete="off" />
          </label>
          <div className="field-row">
            <label className="field">
              <span>Patient ID / MRN</span>
              <input value={patient.patient_id ?? ""} onChange={(e) => set({ patient_id: e.target.value })} maxLength={60} autoComplete="off" />
            </label>
            <label className="field">
              <span>Age</span>
              <input type="number" min={0} max={130} inputMode="numeric" value={patient.age ?? ""}
                onChange={(e) => set({ age: e.target.value === "" ? null : Math.max(0, Math.min(130, Number(e.target.value))) })} />
            </label>
          </div>
          <div className="field-row">
            <label className="field">
              <span>Sex</span>
              <select value={patient.sex ?? ""} onChange={(e) => set({ sex: e.target.value as PatientInfo["sex"] })}>
                <option value="">Not stated</option>
                <option value="female">Female</option>
                <option value="male">Male</option>
                <option value="other">Other</option>
              </select>
            </label>
            <label className="field">
              <span>Date of birth</span>
              <input type="date" value={patient.date_of_birth ?? ""} onChange={(e) => set({ date_of_birth: e.target.value })} />
            </label>
          </div>
        </fieldset>

        <fieldset className="fieldset">
          <legend>Study</legend>
          <div className="field-row">
            <label className="field">
              <span>Scan date</span>
              <input type="date" value={patient.scan_date ?? ""} onChange={(e) => set({ scan_date: e.target.value })} />
            </label>
            <label className="field">
              <span>Accession / study ID</span>
              <input value={studyId} onChange={(e) => setStudyId(e.target.value)} maxLength={80} />
            </label>
          </div>
          <div className="field-row">
            <label className="field">
              <span>Referring physician</span>
              <input value={patient.referring_physician ?? ""} onChange={(e) => set({ referring_physician: e.target.value })} maxLength={120} />
            </label>
            <label className="field">
              <span>Institution</span>
              <input value={patient.institution ?? ""} onChange={(e) => set({ institution: e.target.value })} maxLength={120} placeholder="Shown in the header" />
            </label>
          </div>
          <label className="field">
            <span>Clinical indication</span>
            <textarea value={patient.clinical_indication ?? ""} onChange={(e) => set({ clinical_indication: e.target.value })} maxLength={1000} rows={2}
              placeholder="Reason for the scan, symptoms, history" />
          </label>
        </fieldset>

        <fieldset className="fieldset">
          <legend>Report</legend>
          <label className="field">
            <span>Analyst</span>
            <input value={analyst} onChange={(e) => setAnalyst(e.target.value)} maxLength={80} placeholder="Defaults to NeuroScan AI" />
          </label>
          <label className="field">
            <span>Notes</span>
            <textarea value={notes} onChange={(e) => setNotes(e.target.value)} maxLength={2000} rows={3} placeholder="Additional observations" />
          </label>
        </fieldset>

        <button type="button" className="btn btn-primary btn-block" onClick={generate} disabled={disabled || busy}>
          {busy ? "Generating PDF…" : "Download PDF report"}
        </button>
        <p className="field-hint">
          All fields are optional. Patient details go into this PDF only; they are not saved in the app.
        </p>
        {disabled && <p className="field-hint">Run an analysis first.</p>}
        {error && <p className="field-hint" role="alert" style={{ color: "var(--status-danger)" }}>{error}</p>}
      </div>
    </section>
  );
}
