import { defineNitroConfig } from 'nitropack/config'

const preset =
  process.env.NITRO_PRESET ||
  (process.env.VERCEL || process.env.VERCEL_ENV ? 'vercel' : 'node-server')

export default defineNitroConfig({
  compatibilityDate: '2026-07-18',
  preset,
  srcDir: 'server',
  // One serverless function so /register and /servers share in-memory state
  inlineDynamicImports: true,
  publicAssets: [
    {
      dir: 'public',
      baseURL: '/',
      maxAge: 60 * 60,
    },
  ],
  runtimeConfig: {
    /** Shared 7-char join token (A–Z / 0–9). Set BROKER_TOKEN on Vercel. */
    brokerToken: process.env.BROKER_TOKEN || process.env.BROKER_SECRET || '',
    brokerSecret: process.env.BROKER_SECRET || process.env.BROKER_TOKEN || '',
  },
  routeRules: {
    '/health': { cors: true },
    '/handshake': { cors: true },
    '/heartbeat': { cors: true },
    '/register': { cors: true },
    '/servers': { cors: true },
    '/pick': { cors: true },
  },
  vercel: {
    functions: {
      maxDuration: 30,
    },
  },
})
