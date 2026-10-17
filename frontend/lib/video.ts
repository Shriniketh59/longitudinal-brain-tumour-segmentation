"use client";

export type ExtractedVideo = {
  frames: string[];
  width: number;
  height: number;
  duration: number;
};

function once(target: HTMLMediaElement, event: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const ok = () => {
      cleanup();
      resolve();
    };
    const fail = () => {
      cleanup();
      reject(new Error("This video format cannot be decoded by your browser."));
    };
    const cleanup = () => {
      target.removeEventListener(event, ok);
      target.removeEventListener("error", fail);
    };
    target.addEventListener(event, ok);
    target.addEventListener("error", fail);
  });
}

/**
 * Samples evenly spaced frames from a video file in the browser (no upload).
 * Frames come back as JPEG data URLs, longest side capped at `maxSide`.
 */
export async function extractFrames(
  file: File,
  { maxFrames = 60, maxSide = 512, onProgress }: { maxFrames?: number; maxSide?: number; onProgress?: (done: number, total: number) => void } = {},
): Promise<ExtractedVideo> {
  const url = URL.createObjectURL(file);
  const video = document.createElement("video");
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.src = url;

  try {
    await once(video, "loadedmetadata");
    const duration = video.duration;
    if (!Number.isFinite(duration) || duration <= 0) throw new Error("Video has no readable duration.");

    const scale = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
    const width = Math.round(video.videoWidth * scale);
    const height = Math.round(video.videoHeight * scale);
    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas not supported");

    const total = Math.max(1, Math.min(maxFrames, Math.floor(duration * 30)));
    const frames: string[] = [];
    for (let i = 0; i < total; i++) {
      // Sample mid-interval so the first and last frames are not black fades.
      video.currentTime = Math.min(duration - 0.001, ((i + 0.5) / total) * duration);
      await once(video, "seeked");
      ctx.drawImage(video, 0, 0, width, height);
      frames.push(canvas.toDataURL("image/jpeg", 0.85));
      onProgress?.(i + 1, total);
    }
    return { frames, width, height, duration };
  } finally {
    video.removeAttribute("src");
    video.load();
    URL.revokeObjectURL(url);
  }
}
