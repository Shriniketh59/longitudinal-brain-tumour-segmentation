import type { Metadata } from "next";
import VolumeLab from "./VolumeLab";

export const metadata: Metadata = { title: "3D volume & video" };

export default function VolumePage() {
  return <VolumeLab />;
}
