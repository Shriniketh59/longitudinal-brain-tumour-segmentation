type Series = { label: string; values: number[]; color: string; dashed?: boolean };

/** Tumor area per slice. Marker follows the current slice; click/keys are handled by the cine player. */
export default function AreaChart({
  series,
  index,
  onSeek,
  height = 120,
  unit = "mm²",
}: {
  series: Series[];
  index: number;
  onSeek?: (i: number) => void;
  height?: number;
  unit?: string;
}) {
  const n = Math.max(0, ...series.map((s) => s.values.length));
  if (n < 2) return null;
  const max = Math.max(1, ...series.flatMap((s) => s.values));
  const w = 600;
  const pad = 4;
  const x = (i: number) => pad + (i / (n - 1)) * (w - pad * 2);
  const y = (v: number) => height - pad - (v / max) * (height - pad * 2);
  const peak = series[0] ? Math.max(...series[0].values) : 0;

  return (
    <figure className="chart" style={{ margin: 0 }}>
      <svg
        viewBox={`0 0 ${w} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Tumor area by frame. Peak ${peak.toFixed(1)} ${unit}.`}
        style={{ width: "100%", height, display: "block", cursor: onSeek ? "pointer" : undefined }}
        onClick={(e) => {
          if (!onSeek) return;
          const rect = e.currentTarget.getBoundingClientRect();
          const ratio = (e.clientX - rect.left) / rect.width;
          onSeek(Math.round(Math.min(1, Math.max(0, ratio)) * (n - 1)));
        }}
      >
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1={0} x2={w} y1={height * f} y2={height * f} className="chart-grid" />
        ))}
        {series.map((s) => (
          <polyline
            key={s.label}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? "6 5" : undefined}
            vectorEffect="non-scaling-stroke"
            points={s.values.map((v, i) => `${x(i)},${y(v)}`).join(" ")}
          />
        ))}
        <line x1={x(index)} x2={x(index)} y1={0} y2={height} className="chart-cursor" vectorEffect="non-scaling-stroke" />
      </svg>
      <figcaption className="legend" style={{ marginTop: 8 }}>
        {series.map((s) => (
          <span key={s.label}>
            <span className="swatch" style={{ background: s.color }} aria-hidden="true" />
            {s.label}
          </span>
        ))}
        <span className="mono">max {max.toFixed(max < 10 ? 1 : 0)} {unit}</span>
      </figcaption>
    </figure>
  );
}
