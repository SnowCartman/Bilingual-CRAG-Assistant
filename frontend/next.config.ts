import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // standalone produces .next/standalone/server.js + a self-contained
  // node_modules/, which the Docker runner stage copies in. Cuts the image
  // from ~1.2GB (full node_modules) to ~180MB.
  output: "standalone",
};

export default nextConfig;
