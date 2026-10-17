import type { Metadata } from "next";
import LongitudinalLab from "./LongitudinalLab";

export const metadata: Metadata = { title: "Longitudinal tracking" };

export default function LongitudinalPage() {
  return <LongitudinalLab />;
}
