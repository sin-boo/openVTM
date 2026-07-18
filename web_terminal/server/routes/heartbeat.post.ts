import { heartbeatSession, ttlSeconds } from '../utils/servers'

/**
 * Authenticated keep-alive (every ~2s). Session from /handshake — no join token.
 * Body: { session, ready?, load?, vram_free?, public_url? }
 */
export default defineEventHandler(async (event) => {
  let body: Record<string, unknown>
  try {
    body = (await readBody(event)) as Record<string, unknown>
  } catch {
    setResponseStatus(event, 400)
    return { ok: false, error: 'invalid JSON' }
  }

  const session = String(body.session || '').trim()
  if (!session) {
    setResponseStatus(event, 400)
    return { ok: false, error: 'session required' }
  }

  const load =
    body.load === undefined || body.load === null ? undefined : Number(body.load)
  const vramRaw = body.vram_free
  const vram_free =
    vramRaw === undefined || vramRaw === null ? undefined : Number(vramRaw)

  const server = heartbeatSession({
    session,
    ready: body.ready === undefined ? undefined : Boolean(body.ready),
    load,
    vram_free:
      vram_free !== undefined && Number.isFinite(vram_free) ? vram_free : undefined,
    public_url:
      body.public_url === undefined
        ? undefined
        : String(body.public_url).trim(),
  })

  if (!server) {
    setResponseStatus(event, 401)
    return {
      ok: false,
      error: 'session expired or unknown — call /handshake again',
    }
  }

  return { ok: true, server, ttl_seconds: ttlSeconds() }
})
