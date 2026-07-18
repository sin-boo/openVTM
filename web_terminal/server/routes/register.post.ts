import { upsertServer, ttlSeconds, storageMode } from '../utils/servers'
import { assertJoinToken } from '../utils/join'
import { randomServerName } from '../utils/names'

/**
 * Legacy register — prefer /handshake + /heartbeat.
 * Still accepts Bearer join token and assigns a random name if id omitted.
 */
export default defineEventHandler(async (event) => {
  const header = getHeader(event, 'authorization') || ''
  const bearer = header.startsWith('Bearer ') ? header.slice(7).trim() : ''

  let body: Record<string, unknown>
  try {
    body = (await readBody(event)) as Record<string, unknown>
  } catch {
    setResponseStatus(event, 400)
    return { ok: false, error: 'invalid JSON' }
  }

  const token = String(body.token || '').trim() || bearer
  if (!assertJoinToken(event, token)) {
    setResponseStatus(event, 401)
    return { ok: false, error: 'unauthorized' }
  }

  const public_url = String(body.public_url || '')
    .trim()
    .replace(/\/$/, '')
  if (!public_url || !/^https?:\/\//i.test(public_url)) {
    setResponseStatus(event, 400)
    return { ok: false, error: 'public_url must be http(s)' }
  }

  const id = String(body.id || '').trim() || randomServerName()
  const load = Number(body.load ?? 0)
  const ready = Boolean(body.ready)
  const vramRaw = body.vram_free
  const vram_free =
    vramRaw === undefined || vramRaw === null ? null : Number(vramRaw)

  const server = await upsertServer({
    id,
    public_url,
    load: Number.isFinite(load) ? load : 0,
    ready,
    vram_free:
      vram_free !== null && Number.isFinite(vram_free) ? vram_free : null,
  })

  return {
    ok: true,
    server,
    ttl_seconds: ttlSeconds(),
    storage: storageMode(),
  }
})
