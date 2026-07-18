import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'node:path'

export default defineConfig({
  root: resolve(__dirname),
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/health': 'http://127.0.0.1:3000',
      '/servers': 'http://127.0.0.1:3000',
      '/pick': 'http://127.0.0.1:3000',
      '/register': 'http://127.0.0.1:3000',
    },
  },
  build: {
    outDir: resolve(__dirname, '../public'),
    emptyOutDir: true,
  },
})
