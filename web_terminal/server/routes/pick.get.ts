import { pickBest, storageMode } from '../utils/servers'

export default defineEventHandler(async () => {
  const best = await pickBest()
  // Always 200 so browsers / dashboards don't spam "Failed to load resource".
  if (!best) {
    return { ok: false, error: 'no servers available', storage: storageMode() }
  }
  return { ok: true, server: best, storage: storageMode() }
})
