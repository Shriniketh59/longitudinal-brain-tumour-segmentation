import type { Metadata } from "next";
import PerformanceOverview from "./PerformanceOverview";

export const metadata: Metadata = { title: "Model performance" };

export default function ModelPage() {
  return <PerformanceOverview />;
}
