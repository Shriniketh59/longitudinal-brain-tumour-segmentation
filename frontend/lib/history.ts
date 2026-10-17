"use client";

import { useSyncExternalStore } from "react";
import type { ReportRequest } from "./api";

export type HistoryKind = "classify" | "segment" | "volume" | "video" | "longitudinal";

export type HistoryEntry = {
  id: string;
  kind: HistoryKind;
  createdAt: string;
  title: string;
  summary: string;
  /** Small JPEG data URL. */
  thumb: string | null;
  /** Everything needed to regenerate the PDF later; images already downscaled. */
  report: ReportRequest;
  /** Structured results for dashboard charts (optional; older entries lack it). */
  meta?: {
    label?: string;
    confidence?: number;
    malignancy?: "likely_cancerous" | "likely_non_cancerous" | "no_tumor_detected";
    wholeTumorCm3?: number;
  };
};

const KEY = "ns-history-v1";
const MAX_ENTRIES = 40;
const EMPTY: HistoryEntry[] = [];
const listeners = new Set<() => void>();
let cache: HistoryEntry[] | null = null;

function read(): HistoryEntry[] {
  if (cache) return cache;
  try {
    const raw = localStorage.getItem(KEY);
    cache = raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    cache = [];
  }
  return cache;
}

function write(entries: HistoryEntry[]) {
  cache = entries;
  // Drop oldest entries until it fits the storage quota.
  for (let list = entries; list.length >= 0; list = list.slice(0, -1)) {
    try {
      localStorage.setItem(KEY, JSON.stringify(list));
      cache = list;
      break;
    } catch {
      if (list.length === 0) break;
    }
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  const onStorage = (e: StorageEvent) => {
    if (e.key === KEY) {
      cache = null;
      listener();
    }
  };
  window.addEventListener("storage", onStorage);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", onStorage);
  };
}

export function useHistory(): HistoryEntry[] {
  return useSyncExternalStore(subscribe, read, () => EMPTY);
}

export function addHistory(entry: Omit<HistoryEntry, "id" | "createdAt">): HistoryEntry {
  const full: HistoryEntry = {
    ...entry,
    id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : String(Date.now()),
    createdAt: new Date().toISOString(),
  };
  write([full, ...read()].slice(0, MAX_ENTRIES));
  return full;
}

export function removeHistory(id: string) {
  write(read().filter((e) => e.id !== id));
}

export function clearHistory() {
  write([]);
}
