import type { LongitudinalResult } from "@/lib/api";

const W = 720;
const H = 300;
const PAD = { l: 52, r: 16, t: 16, b: 36 };

const fmtDate = (iso: string) => new Date(iso + "T00:00:00").toLocaleDateString(undefined, { month: "short", year: "2-digit" });

/** Measured volumes, fitted growth curve, 95% prediction band and forecast beyond the last scan. */
export default function GrowthChart({ data }: { data: LongitudinalResult }) {
  const last = data.timepoints[data.timepoints.length - 1].day;
  const maxDay = data.curve[data.curve.length - 1].day || 1;
  // Scale to measured and forecast volumes; a very wide prediction band is clipped, not allowed to flatten the chart.
  const values = [...data.timepoints.map((t) => t.volumes_cm3[data.region]), ...data.curve.map((c) => c.volume_cm3)];
  const maxV = Math.max(1e-3, ...values) * 1.35;
  const clampV = (v: number) => Math.min(v, maxV);
  const x = (d: number) => PAD.l + (d / maxDay) * (W - PAD.l - PAD.r);
  const y = (v: number) => H - PAD.b - (v / maxV) * (H - PAD.t - PAD.b);

  const past = data.curve.filter((c) => c.day <= last + 1e-6);
  const future = data.curve.filter((c) => c.day >= last - 1e-6);
  const line = (pts: typeof data.curve) => pts.map((c) => `${x(c.day)},${y(c.volume_cm3)}`).join(" ");
  const hasBand = data.curve.every((c) => c.low_cm3 !== null && c.high_cm3 !== null);
  const band = hasBand
    ? [...data.curve.map((c) => `${x(c.day)},${y(clampV(c.high_cm3!))}`), ...[...data.curve].reverse().map((c) => `${x(c.day)},${y(clampV(c.low_cm3!))}`)].join(" ")
    : "";
  const bandClipped = hasBand && data.curve.some((c) => c.high_cm3! > maxV);
  const yTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * maxV);
  const xTicks = [...data.timepoints.map((t) => ({ day: t.day, date: t.date })), { day: maxDay, date: data.curve[data.curve.length - 1].date }];

  return (
    <figure style={{ margin: 0 }}>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Tumor volume over time with forecast. Latest ${data.summary.latest_cm3.toFixed(1)} cubic centimetres.`}
        style={{ width: "100%", height: "auto", display: "block" }}
      >
        {yTicks.map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="chart-grid" />
            <text x={PAD.l - 8} y={y(v) + 4} textAnchor="end" className="chart-label">
              {v.toFixed(v < 10 ? 1 : 0)}
            </text>
          </g>
        ))}
        {xTicks.map((t, i) => (
          <text key={i} x={x(t.day)} y={H - 12} textAnchor="middle" className="chart-label">
            {fmtDate(t.date)}
          </text>
        ))}
        <line x1={x(last)} x2={x(last)} y1={PAD.t} y2={H - PAD.b} className="chart-today" />
        <text x={x(last) + 6} y={PAD.t + 12} className="chart-label">
          forecast →
        </text>
        {hasBand && <polygon points={band} className="chart-band" />}
        <polyline points={line(past)} className="chart-fit" />
        <polyline points={line(future)} className="chart-fit chart-forecast" />
        {data.timepoints.map((t) => (
          <circle key={t.date} cx={x(t.day)} cy={y(t.volumes_cm3[data.region])} r={5} className="chart-point">
            <title>{`${t.date}: ${t.volumes_cm3[data.region].toFixed(2)} cm³`}</title>
          </circle>
        ))}
        <text x={14} y={PAD.t + 4} className="chart-label" transform={`rotate(-90 14 ${PAD.t + 4})`} textAnchor="end">
          cm³
        </text>
      </svg>
      <figcaption className="legend" style={{ marginTop: 8 }}>
        <span><span className="swatch" style={{ background: "var(--accent-cyan)", borderRadius: "50%" }} aria-hidden="true" />Measured</span>
        <span><span className="swatch" style={{ background: "var(--accent-purple)" }} aria-hidden="true" />Exponential fit and forecast</span>
        {hasBand ? (
          <span>
            <span className="swatch" style={{ background: "color-mix(in srgb, var(--accent-purple) 25%, transparent)" }} aria-hidden="true" />
            95% prediction band{bandClipped && " (extends beyond the chart: few scans, high uncertainty)"}
          </span>
        ) : (
          <span>Add a third scan for an uncertainty band</span>
        )}
      </figcaption>
    </figure>
  );
}
