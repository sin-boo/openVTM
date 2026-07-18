import { pickBest } from '../utils/servers'

export default defineEventHandler((event) => {
  const best = pickBest()
  if (!best) {
    setResponseStatus(event, 503)
    return { ok: false, error: 'no servers available' }
  }
  return { ok: true, server: best }
})
