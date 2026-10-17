// Heartbeat monitor trace for the top bar. Two beats in one path; a CSS stroke-dash animation moves the bright segment.
const TRACE =
  "M0 12 H28 L32 9 L36 12 H44 L47 4 L51 20 L54 12 H62 L67 8 L72 12 " +
  "H128 L132 9 L136 12 H144 L147 4 L151 20 L154 12 H162 L167 8 L172 12 H200";

export default function Pulse() {
  return (
    <svg className="pulse" viewBox="0 0 200 24" aria-hidden="true" focusable="false" preserveAspectRatio="none">
      <path className="pulse-base" d={TRACE} />
      <path d={TRACE} pathLength={300} />
    </svg>
  );
}
