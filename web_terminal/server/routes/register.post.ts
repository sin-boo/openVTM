import { ttlSeconds, upsertServer } from '../utils/servers'

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig(event)
  const secret = String(config.brokerSecret || '').trim()
  const header = getHeader(event, 'authorization') || ''
  const token = header.startsWith('Bearer ') ? header.slice(7).trim() : ''
  if (!secret || token !== secret) {
    setResponseStatus(event, 401)
    return { ok: false, error: 'unauthorized' }
  }

  let body: Record<string, unknown>
  try {
    body = (await readBody(event)) as Record<string, unknown>
  } catch {
    setResponseStatus(event, 400)
    return { ok: false, error: 'invalid JSON' }
  }

  const id = String(body.id || '').trim()
  const public_url = String(body.public_url || '')
    .trim()
    .replace(/\/$/, '')
  if (!id || !public_url) {
    setResponseStatus(event, 400)
    return { ok: false, error: 'id and public_url required' }
  }
  if (!/^https?:\/\//i.test(public_url)) {
    setResponseStatus(event, 400)
    return { ok: false, error: 'public_url must be http(s)' }
  }

  const load = Number(body.load ?? 0)
  const ready = Boolean(body.ready)
  const vramRaw = body.vram_free
  const vram_free =
    vramRaw === undefined || vramRaw === null ? null : Number(vramRaw)

  const server = upsertServer({
    id,
    public_url,
    load: Number.isFinite(load) ? load : 0,
    ready,
    vram_free:
      vram_free !== null && Number.isFinite(vram_free) ? vram_free : null,
  })

  return { ok: true, server, ttl_seconds: ttlSeconds() }
})
