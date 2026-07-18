import { handshakeServer, ttlSeconds, storageMode } from '../utils/servers'
import { assertJoinToken } from '../utils/join'

/**
 * First contact from a GPU host.
 * Body: { token, public_url, ready?, load? }
 * Returns session for subsequent /heartbeat calls (no token needed).
 */
export default defineEventHandler(async (event) => {
  let body: Record<string, unknown>
  try {
    body = (await readBody(event)) as Record<string, unknown>
  } catch {
    setResponseStatus(event, 400)
    return { ok: false, error: 'invalid JSON' }
  }

  const token = String(body.token || body.join_token || '').trim()
  // Also accept Authorization: Bearer <token>
  const header = getHeader(event, 'authorization') || ''
  const bearer = header.startsWith('Bearer ') ? header.slice(7).trim() : ''
  const provided = token || bearer

  if (!assertJoinToken(event, provided)) {
    setResponseStatus(event, 401)
    return { ok: false, error: 'invalid join token' }
  }

  const public_url = String(body.public_url || '')
    .trim()
    .replace(/\/$/, '')
  if (!public_url || !/^https?:\/\//i.test(public_url)) {
    setResponseStatus(event, 400)
    return { ok: false, error: 'public_url must be http(s)' }
  }

  const load = Number(body.load ?? 0)
  const ready = Boolean(body.ready)
  const { server, session } = await handshakeServer({
    public_url,
    ready,
    load: Number.isFinite(load) ? load : 0,
  })

  return {
    ok: true,
    session,
    server,
    ttl_seconds: ttlSeconds(),
    storage: storageMode(),
    message: 'handshake accepted — use session for /heartbeat',
  }
})
