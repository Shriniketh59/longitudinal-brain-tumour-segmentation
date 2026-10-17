import Image from "next/image";

/** Animated MRI "scanner" panel for the dashboard hero, built on a real sample slice. */
export default function HeroScanner() {
  return (
    <figure className="scanner" aria-label="Sample axial MRI slice under analysis">
      <div className="scanner-frame">
        <Image
          src="/samples/glioma.jpg"
          alt="Axial brain MRI slice showing a glioma"
          fill
          priority
          sizes="(max-width: 1080px) 90vw, 440px"
          className="scanner-img"
        />
        <svg className="scanner-ring" viewBox="0 0 200 200" aria-hidden="true" focusable="false">
          <circle cx="100" cy="100" r="92" pathLength={100} />
          <circle cx="100" cy="100" r="80" pathLength={100} className="inner" />
        </svg>
        <div className="scanner-line" aria-hidden="true" />
        <div className="scanner-target" aria-hidden="true" />
        <span className="reticle tl" aria-hidden="true" />
        <span className="reticle tr" aria-hidden="true" />
        <span className="reticle bl" aria-hidden="true" />
        <span className="reticle br" aria-hidden="true" />
        <span className="hud hud-tl mono" aria-hidden="true">AXIAL</span>
        <span className="hud hud-tr mono" aria-hidden="true">512 × 512</span>
        <span className="hud hud-bl mono" aria-hidden="true">
          <span className="dot ok" /> SCANNING
        </span>
        <span className="hud hud-br mono" aria-hidden="true">VGG16 · U-NET</span>
      </div>
      <figcaption className="scanner-caption">Sample slice · public brain tumor MRI dataset</figcaption>
    </figure>
  );
}
