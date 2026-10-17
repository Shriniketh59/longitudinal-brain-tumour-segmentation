import type { Metadata } from "next";
import SegmentWorkspace from "./SegmentWorkspace";

export const metadata: Metadata = { title: "Segment slice" };

export default function SegmentationPage() {
  return <SegmentWorkspace />;
}
