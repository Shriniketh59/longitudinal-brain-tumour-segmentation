"use client";

/** Switch between full-quality and fast inference. Fast: 2D at 192 px, 3D at 75% scale (~2-3x quicker). */
export default function FastToggle({ fast, onChange, disabled }: { fast: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  return (
    <label className="switch" title="Fast mode: 2D U-Net at 192 px, 3D SwinUNETR at 75% scale. About 2-3x quicker, slightly less precise.">
      <input type="checkbox" role="switch" checked={fast} onChange={(e) => onChange(e.target.checked)} disabled={disabled} />
      <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
      <span className="switch-label">Fast mode</span>
    </label>
  );
}
