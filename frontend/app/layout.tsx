import type { Metadata, Viewport } from "next";
import { IBM_Plex_Mono, IBM_Plex_Sans, Source_Serif_4 } from "next/font/google";
import AppShell from "@/components/shell/AppShell";
import "./globals.css";

// Journal-style serif headings, clinical sans body, monospaced labels like scan metadata.
const serif = Source_Serif_4({ subsets: ["latin"], weight: ["500", "600"], style: ["normal", "italic"], variable: "--font-serif", display: "swap" });
const sans = IBM_Plex_Sans({ subsets: ["latin"], weight: ["400", "500", "600"], variable: "--font-sans", display: "swap" });
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-plexmono", display: "swap" });

export const metadata: Metadata = {
  title: { default: "NeuroScan AI", template: "%s · NeuroScan AI" },
  description: "Brain MRI classification and segmentation workstation. Research use only, not cleared for clinical use.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#14171c" },
    { media: "(prefers-color-scheme: light)", color: "#f4f6f9" },
  ],
};

// Applied before paint so the saved theme never flashes.
const themeScript = `try{var t=localStorage.getItem('ns-theme');if(t==='light'||t==='dark'){document.documentElement.dataset.theme=t;}}catch(e){}`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${serif.variable} ${sans.variable} ${plexMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
