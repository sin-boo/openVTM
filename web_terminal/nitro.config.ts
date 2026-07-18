import { defineNitroConfig } from 'nitropack/config'

export default defineNitroConfig({
  // node-server by default; set NITRO_PRESET=vercel | cloudflare_module for deploy
  compatibilityDate: '2026-07-18',
  srcDir: 'server',
  runtimeConfig: {
    brokerSecret: process.env.BROKER_SECRET || 'change-me',
  },
  routeRules: {
    '/': { cors: false },
    '/health': { cors: true },
    '/register': { cors: true },
    '/servers': { cors: true },
    '/pick': { cors: true },
  },
})
