"use client";

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("Could not load image"));
    img.src = src;
  });
}

/**
 * Draws a base image with optional transparent overlays on top and returns a
 * JPEG data URL no larger than `maxSide` on its longest edge.
 */
export async function composeImage(
  base: string,
  overlays: { src: string; opacity: number }[] = [],
  maxSide = 512,
  quality = 0.88,
): Promise<string> {
  const img = await loadImage(base);
  const scale = Math.min(1, maxSide / Math.max(img.naturalWidth, img.naturalHeight));
  const w = Math.max(1, Math.round(img.naturalWidth * scale));
  const h = Math.max(1, Math.round(img.naturalHeight * scale));
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas not supported");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);
  ctx.drawImage(img, 0, 0, w, h);
  for (const layer of overlays) {
    const over = await loadImage(layer.src);
    ctx.globalAlpha = layer.opacity;
    ctx.imageSmoothingEnabled = false; // keep label edges crisp
    ctx.drawImage(over, 0, 0, w, h);
    ctx.globalAlpha = 1;
  }
  return canvas.toDataURL("image/jpeg", quality);
}

export async function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

export async function dataUrlToFile(dataUrl: string, name: string): Promise<File> {
  const blob = await (await fetch(dataUrl)).blob();
  return new File([blob], name, { type: blob.type });
}
