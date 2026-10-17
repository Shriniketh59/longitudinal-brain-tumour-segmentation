import type { NextConfig } from "next";

// The Python API runs privately on BACKEND_URL; the browser only talks to
// this Next.js server on :3000, which proxies /api/* to it.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  devIndicators: false,
  compress: true,
  images: {
    formats: ["image/avif", "image/webp"],
  },
  experimental: {
    // NIfTI volumes are large and 3D analysis on CPU can take a while.
    proxyClientMaxBodySize: "200mb",
    proxyTimeout: 180_000,
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
  },
};

export default nextConfig;
