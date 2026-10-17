import type { Metadata } from "next";
import ClassificationReportView from "./ClassificationReportView";

export const metadata: Metadata = { title: "Classification report" };

export default function ClassificationReportPage() {
  return <ClassificationReportView />;
}
