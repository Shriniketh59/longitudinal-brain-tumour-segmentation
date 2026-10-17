// Animated medical background: millimetre grid, sweeping heartbeat traces, rotating scanner
// rings and drifting cells. Pure SVG + CSS, transform/opacity animations only.

const BEAT = "h60 l6 -6 l6 6 h14 l5 -22 l7 40 l5 -18 h16 l8 -8 l8 8 h65";

/** One heartbeat path repeated across 2400px; the strip scrolls left forever. */
function trace(repeats: number): string {
  return `M0 50 ${Array.from({ length: repeats }, () => BEAT).join(" ")}`;
}

// Fixed positions so server and client render the same markup.
const CELLS = [
  { x: 8, size: 10, dur: 26, delay: -3 },
  { x: 18, size: 6, dur: 31, delay: -14 },
  { x: 27, size: 14, dur: 38, delay: -22 },
  { x: 39, size: 7, dur: 29, delay: -8 },
  { x: 52, size: 11, dur: 35, delay: -30 },
  { x: 61, size: 5, dur: 24, delay: -11 },
  { x: 70, size: 16, dur: 42, delay: -5 },
  { x: 79, size: 8, dur: 33, delay: -19 },
  { x: 88, size: 12, dur: 37, delay: -27 },
  { x: 95, size: 6, dur: 28, delay: -16 },
];

export default function MedicalBackdrop() {
  const d = trace(12);
  return (
    <div className="backdrop" aria-hidden="true">
      <div className="backdrop-grid" />
      <div className="backdrop-glow glow-a" />
      <div className="backdrop-glow glow-b" />

      <svg className="bg-rings" viewBox="0 0 400 400" focusable="false">
        <circle cx="200" cy="200" r="190" className="ring r1" />
        <circle cx="200" cy="200" r="150" className="ring r2" />
        <ellipse cx="200" cy="200" rx="110" ry="130" className="ring r3" />
        <circle cx="200" cy="200" r="70" className="ring r4" />
        <path d="M200 0 V400 M0 200 H400" className="ring-cross" />
      </svg>

      {[18, 52, 84].map((top, i) => (
        <div key={top} className={`ecg-strip ecg-${i + 1}`} style={{ top: `${top}%` }}>
          <svg viewBox="0 0 2400 100" preserveAspectRatio="none" focusable="false">
            <path d={d} className="ecg-line" />
          </svg>
        </div>
      ))}

      {CELLS.map((c, i) => (
        <span
          key={i}
          className="cell"
          style={{
            left: `${c.x}%`,
            width: c.size,
            height: c.size,
            animationDuration: `${c.dur}s`,
            animationDelay: `${c.delay}s`,
          }}
        />
      ))}
    </div>
  );
}
