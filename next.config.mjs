/** @type {import('next').NextConfig} */
const nextConfig = {
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  images: {
    remotePatterns: [
      {
        protocol: 'https',
        hostname: 'ufc-fighters-img.s3.amazonaws.com',
        pathname: '/images/**',        // ajustez si dossier différent
      },
    ],
  },
}

export default nextConfig
