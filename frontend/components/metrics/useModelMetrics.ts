"use client";

import { useEffect, useState } from "react";
import { fetchModelMetrics, type ClassifierMetrics, type SegmenterMetrics, type Swin3dMetrics } from "@/lib/api";

export type AllMetrics = {
  classifier: ClassifierMetrics | null;
  segmenter: SegmenterMetrics | null;
  swin3d?: Swin3dMetrics | null;
};

export type MetricsState = { status: "loading" } | { status: "offline" } | { status: "ok"; data: AllMetrics };

/** Evaluation results written by the training notebooks, fetched once per page. */
export function useModelMetrics(): MetricsState {
  const [state, setState] = useState<MetricsState>({ status: "loading" });
  useEffect(() => {
    fetchModelMetrics()
      .then((data) => setState({ status: "ok", data }))
      .catch(() => setState({ status: "offline" }));
  }, []);
  return state;
}

/** Macro and support-weighted averages of per-class precision, recall and F1. */
export function averages(c: ClassifierMetrics) {
  const rows = c.classes.map((k) => c.per_class[k]);
  const total = rows.reduce((s, r) => s + r.support, 0) || 1;
  const macro = (key: "precision" | "recall" | "f1") => rows.reduce((s, r) => s + r[key], 0) / rows.length;
  const weighted = (key: "precision" | "recall" | "f1") => rows.reduce((s, r) => s + r[key] * r.support, 0) / total;
  return {
    macro: { precision: macro("precision"), recall: macro("recall"), f1: macro("f1") },
    weighted: { precision: weighted("precision"), recall: weighted("recall"), f1: weighted("f1") },
    total,
  };
}
