"use client";

import { useEffect, useRef, useState } from "react";

export type CineLayer = {
  key: string;
  label: string;
  /** One transparent PNG/JPEG data URL per frame (or null for none). */
  masks: (string | null)[];
};

export type CineBox = { x: number; y: number; w: number; h: number; label: string; color: string } | null;

type Props = {
  frames: string[];
  /** Optional detection boxes per frame, as fractions of the frame (shown with the first layer). */
  boxes?: CineBox[];
  /** Frame width / height; keeps masks and boxes aligned on non-square video. */
  aspect?: number;
  layers?: CineLayer[];
  index: number;
  onIndexChange: (i: number) => void;
  frameLabel?: (i: number) => string;
  visibleLayers: Record<string, boolean>;
  onToggleLayer: (key: string) => void;
  opacity: number;
  onOpacityChange: (v: number) => void;
};

const SPEEDS = [4, 8, 12, 24];

export default function CinePlayer({
  frames,
  layers = [],
  boxes,
  aspect = 1,
  index,
  onIndexChange,
  frameLabel,
  visibleLayers,
  onToggleLayer,
  opacity,
  onOpacityChange,
}: Props) {
  const [playing, setPlaying] = useState(false);
  const [fps, setFps] = useState(12);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const indexRef = useRef(index);
  useEffect(() => {
    indexRef.current = index;
  }, [index]);
  const count = frames.length;

  useEffect(() => {
    if (!playing || count < 2) return;
    const timer = setInterval(() => onIndexChange((indexRef.current + 1) % count), 1000 / fps);
    return () => clearInterval(timer);
  }, [playing, fps, count, onIndexChange]);

  function step(delta: number) {
    setPlaying(false);
    onIndexChange((index + delta + count) % count);
  }

  function onKeyDown(e: React.KeyboardEvent) {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.key === " ") {
      e.preventDefault();
      setPlaying((p) => !p);
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      step(1);
    } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      step(-1);
    }
  }

  if (count === 0) return null;
  const filter = `brightness(${brightness}%) contrast(${contrast}%)`;

  return (
    <div className="cine" tabIndex={0} onKeyDown={onKeyDown} aria-label="Cine player. Space plays or pauses, arrow keys step frames.">
      <div className="viewer cine-viewer">
        <div className="cine-stage" style={aspect >= 1 ? { width: "100%", aspectRatio: aspect } : { height: "100%", aspectRatio: aspect }}>
          {/* Data URLs from the API or the local video; next/image cannot optimise them. */}
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={frames[index]} alt={frameLabel ? frameLabel(index) : `Frame ${index + 1}`} style={{ filter }} />
          {layers.map((layer) =>
            visibleLayers[layer.key] && layer.masks[index] ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img key={layer.key} src={layer.masks[index]!} alt="" className="cine-mask" style={{ opacity }} />
            ) : null,
          )}
          {boxes?.[index] && (layers.length === 0 || visibleLayers[layers[0].key]) && (
            <div
              className="cine-box"
              style={{
                left: `${boxes[index]!.x * 100}%`,
                top: `${boxes[index]!.y * 100}%`,
                width: `${boxes[index]!.w * 100}%`,
                height: `${boxes[index]!.h * 100}%`,
                borderColor: boxes[index]!.color,
              }}
            >
              <span className="cine-box-label mono" style={{ background: boxes[index]!.color }}>{boxes[index]!.label}</span>
            </div>
          )}
        </div>
        <span className="viewer-tag mono">{frameLabel ? frameLabel(index) : `Frame ${index + 1}`}</span>
        <span className="viewer-tag viewer-tag-right mono">
          {index + 1} / {count}
        </span>
      </div>

      <div className="cine-controls">
        <div className="row" style={{ gap: 6 }}>
          <button type="button" className="icon-btn" onClick={() => step(-1)} aria-label="Previous frame">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M15 6l-6 6 6 6" /></svg>
          </button>
          <button type="button" className="icon-btn icon-btn-primary" onClick={() => setPlaying((p) => !p)} aria-label={playing ? "Pause" : "Play"}>
            {playing ? (
              <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><rect x="6" y="5" width="4" height="14" rx="1" /><rect x="14" y="5" width="4" height="14" rx="1" /></svg>
            ) : (
              <svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M7 5v14l12-7z" /></svg>
            )}
          </button>
          <button type="button" className="icon-btn" onClick={() => step(1)} aria-label="Next frame">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg>
          </button>
        </div>
        <label className="cine-scrub">
          <span className="sr-only">Frame</span>
          <input
            type="range"
            min={0}
            max={count - 1}
            value={index}
            onChange={(e) => {
              setPlaying(false);
              onIndexChange(Number(e.target.value));
            }}
          />
        </label>
        <label className="field-inline">
          <span className="sr-only">Playback speed</span>
          <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>
                {s} fps
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="cine-adjust">
        <label className="slider">
          <span>Brightness</span>
          <input type="range" min={50} max={180} value={brightness} onChange={(e) => setBrightness(Number(e.target.value))} />
        </label>
        <label className="slider">
          <span>Contrast</span>
          <input type="range" min={50} max={200} value={contrast} onChange={(e) => setContrast(Number(e.target.value))} />
        </label>
        {layers.length > 0 && (
          <label className="slider">
            <span>Overlay</span>
            <input type="range" min={0} max={100} value={Math.round(opacity * 100)} onChange={(e) => onOpacityChange(Number(e.target.value) / 100)} />
          </label>
        )}
        {layers.length > 0 && (
          <div className="segmented" role="group" aria-label="Overlay layers">
            {layers.map((layer) => (
              <button key={layer.key} type="button" aria-pressed={!!visibleLayers[layer.key]} onClick={() => onToggleLayer(layer.key)}>
                {layer.label}
              </button>
            ))}
          </div>
        )}
        <button
          type="button"
          className="btn btn-ghost"
          onClick={() => {
            setBrightness(100);
            setContrast(100);
          }}
        >
          Reset
        </button>
      </div>
    </div>
  );
}
