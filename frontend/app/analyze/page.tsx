import type { Metadata } from "next";
import ClassifyWorkspace from "./ClassifyWorkspace";

export const metadata: Metadata = { title: "Classify slice" };

export default function AnalyzePage() {
  return <ClassifyWorkspace />;
}
