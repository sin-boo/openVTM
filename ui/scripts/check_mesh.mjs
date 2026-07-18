/** FaceMesh is dots-only now; keep a smoke check for the component module. */
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const here = path.dirname(fileURLToPath(import.meta.url))
const meshPath = path.resolve(here, '../src/components/FaceMesh.tsx')
const require = createRequire(import.meta.url)

// TSX isn't directly require-able; just verify the file exists and mentions dots API.
import fs from 'node:fs'
const src = fs.readFileSync(meshPath, 'utf8')
if (!src.includes('DEFAULT_MESH_SETTINGS') || !src.includes('Waiting for landmarks')) {
  console.error('FaceMesh smoke check FAILED')
  process.exit(1)
}
console.log('FaceMesh dots module OK', pathToFileURL(meshPath).href)
void require
