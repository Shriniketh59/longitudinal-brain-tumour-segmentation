"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { IconCpu, IconCube, IconDashboard, IconHistory, IconLayers, IconReport, IconScan, IconTrend } from "./icons";

const NAV = [
  { href: "/", label: "Dashboard", icon: IconDashboard, match: (p: string) => p === "/" },
  { href: "/analyze", label: "Classify slice", icon: IconScan, match: (p: string) => p.startsWith("/analyze") },
  { href: "/segmentation", label: "Segment slice", icon: IconLayers, match: (p: string) => p.startsWith("/segmentation") },
  { href: "/volume", label: "3D volume & video", icon: IconCube, match: (p: string) => p.startsWith("/volume") },
  { href: "/longitudinal", label: "Longitudinal", icon: IconTrend, match: (p: string) => p.startsWith("/longitudinal") },
  { href: "/history", label: "History & reports", icon: IconHistory, match: (p: string) => p.startsWith("/history") },
  { href: "/model", label: "Performance", icon: IconCpu, match: (p: string) => p === "/model" },
  { href: "/model/classification", label: "Classification report", icon: IconReport, match: (p: string) => p.startsWith("/model/classification") },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="sidebar" aria-label="Primary">
      <Link href="/" className="brand" aria-label="NeuroScan AI home">
        <span className="brand-mark" aria-hidden="true" />
        NeuroScan AI
      </Link>

      <nav>
        <div className="nav-label">Workspace</div>
        <div className="nav">
          {NAV.map((item) => {
            const Icon = item.icon;
            const current = item.match(pathname);
            return (
              <Link
                key={item.href}
                href={item.href}
                className="nav-link"
                aria-current={current ? "page" : undefined}
              >
                <Icon className="nav-icon" />
                {item.label}
              </Link>
            );
          })}
        </div>
      </nav>

      <div className="sidebar-foot">
        Research and education only. Not a diagnostic device.
      </div>
    </aside>
  );
}
