/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  images: {
    unoptimized: true,
  },
  experimental: {
    outputFileTracingIncludes: {
      "/api/**/*": ["./src/lib/verified_samples_cache.json"],
    },
  },
};

module.exports = nextConfig;
