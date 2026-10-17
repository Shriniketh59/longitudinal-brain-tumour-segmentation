export type BarItem = {
  key: string;
  label: string;
  value: number;
  color: string;
  detail?: string;
};

/** Horizontal bars for a set of 0–1 values or percentages. Sorted by value, largest first. */
export default function ProbabilityBars({ items, unit = "percent" }: { items: BarItem[]; unit?: "percent" | "count" }) {
  const sorted = [...items].sort((a, b) => b.value - a.value);
  const max = Math.max(...sorted.map((i) => i.value), 1e-9);

  return (
    <div>
      {sorted.map((item) => {
        const width = unit === "percent" ? item.value * 100 : (item.value / max) * 100;
        const shown = unit === "percent" ? `${(item.value * 100).toFixed(1)}%` : item.detail ?? item.value.toLocaleString();
        return (
          <div className="bar-row" key={item.key}>
            <div className="bar-head">
              <span>
                <span className="swatch" style={{ background: item.color }} aria-hidden="true" />
                {item.label}
              </span>
              <span className="mono">{shown}</span>
            </div>
            <div className="bar-track" role="presentation">
              <div
                className="bar-fill"
                style={{ width: `${width.toFixed(2)}%`, background: item.color }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
