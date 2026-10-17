import type { ClassifierMetrics } from "@/lib/api";
import { CLASS_LABELS } from "@/lib/labels";

/** Rows are true classes, columns predictions. Cell shade = share of the true class (row %). */
export default function ConfusionMatrix({ data }: { data: ClassifierMetrics }) {
  const n = data.classes.length;
  return (
    <div className="table-scroll">
      <table className="cm" style={{ minWidth: 120 + n * 120 }}>
        <caption className="sr-only">Confusion matrix: rows are true classes, columns are predicted classes</caption>
        <thead>
          <tr>
            <th scope="col" className="cm-corner">
              <span>True ↓</span>
              <span>Predicted →</span>
            </th>
            {data.classes.map((k) => (
              <th key={k} scope="col">{CLASS_LABELS[k] ?? k}</th>
            ))}
            <th scope="col">Recall</th>
          </tr>
        </thead>
        <tbody>
          {data.confusion_matrix.map((row, i) => {
            const total = row.reduce((a, b) => a + b, 0) || 1;
            return (
              <tr key={data.classes[i]}>
                <th scope="row">{CLASS_LABELS[data.classes[i]] ?? data.classes[i]}</th>
                {row.map((count, j) => {
                  const share = count / total;
                  const diag = i === j;
                  return (
                    <td
                      key={j}
                      className={diag ? "cm-cell cm-diag" : "cm-cell"}
                      style={{
                        background: `color-mix(in oklch, ${diag ? "var(--accent-cyan)" : "var(--status-danger)"} ${Math.round(share * (diag ? 70 : 90))}%, transparent)`,
                      }}
                    >
                      <span className="cm-count mono">{count}</span>
                      <span className="cm-pct mono">{(share * 100).toFixed(1)}%</span>
                    </td>
                  );
                })}
                <td className="cm-cell cm-recall mono">{((row[i] / total) * 100).toFixed(1)}%</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
