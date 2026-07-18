import { cpSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const src = join(root, 'public')
const vercelOut = join(root, '.vercel', 'output')
const dest = join(vercelOut, 'static')

if (!existsSync(vercelOut)) {
  console.log('skip copy-static (no .vercel/output — not a Vercel preset build)')
  process.exit(0)
}

if (!existsSync(src)) {
  console.error('public/ missing — run vite build first')
  process.exit(1)
}

mkdirSync(dest, { recursive: true })
cpSync(src, dest, { recursive: true })
const files = readdirSync(dest, { recursive: true })
console.log(`copied public/ -> .vercel/output/static (${files.length} entries)`)
