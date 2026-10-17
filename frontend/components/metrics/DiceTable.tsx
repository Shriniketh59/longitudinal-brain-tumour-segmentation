const REGIONS = { whole_tumor: "Whole tumor", tumor_core: "Tumor core", enhancing_tumor: "Enhancing tumor" } as const;

/** Dice per BraTS region with a bar for quick comparison. */
export default function DiceTable({ dice }: { dice: Partial<Record<keyof typeof REGIONS, number | null>> }) {
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th scope="col">Region</th>
          <th scope="col" style={{ width: "45%" }}>Dice</th>
        </tr>
      </thead>
      <tbody>
        {(Object.keys(REGIONS) as (keyof typeof REGIONS)[]).map((k) => {
          const v = dice[k];
          return (
            <tr key={k}>
              <th scope="row">{REGIONS[k]}</th>
              <td>
                <span className="dice-cell">
                  <span className="bar-track"><span className="bar-fill" style={{ width: `${(v ?? 0) * 100}%`, background: "var(--accent-cyan)", display: "block" }} /></span>
                  <span className="mono">{v != null ? v.toFixed(3) : "n/a"}</span>
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
