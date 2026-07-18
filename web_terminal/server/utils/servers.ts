import { randomServerName } from './names'

export type ServerRecord = {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
  updated_at: number
}

type Stored = ServerRecord & { expires_at: number; session: string }

/** Miss ~3 heartbeats at 2s interval before dropping. */
const TTL_MS = 8_000

const store = new Map<string, Stored>()
/** session -> server id */
const sessions = new Map<string, string>()

export function ttlSeconds(): number {
  return TTL_MS / 1000
}

function prune(): void {
  const now = Date.now()
  for (const [id, rec] of store) {
    if (rec.expires_at <= now) {
      store.delete(id)
      sessions.delete(rec.session)
    }
  }
}

function publicRecord(rec: Stored): ServerRecord {
  const { expires_at: _e, session: _s, ...rest } = rec
  return rest
}

function newSessionId(): string {
  const bytes = new Uint8Array(18)
  crypto.getRandomValues(bytes)
  return Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('')
}

export function listServers(): ServerRecord[] {
  prune()
  return [...store.values()]
    .map(publicRecord)
    .sort((a, b) => b.updated_at - a.updated_at)
}

export function pickBest(): ServerRecord | null {
  const servers = listServers()
  if (servers.length === 0) return null
  const sorted = [...servers].sort((a, b) => {
    if (a.ready !== b.ready) return a.ready ? -1 : 1
    if (a.load !== b.load) return a.load - b.load
    return b.updated_at - a.updated_at
  })
  return sorted[0] ?? null
}

/** First contact: validate join token outside this helper, then create session. */
export function handshakeServer(input: {
  public_url: string
  ready?: boolean
  load?: number
}): { server: ServerRecord; session: string } {
  prune()
  const now = Date.now()
  const id = randomServerName()
  const session = newSessionId()
  const record: Stored = {
    id,
    public_url: input.public_url,
    load: Number.isFinite(input.load) ? Number(input.load) : 0,
    ready: Boolean(input.ready),
    vram_free: null,
    updated_at: now,
    expires_at: now + TTL_MS,
    session,
  }
  store.set(id, record)
  sessions.set(session, id)
  return { server: publicRecord(record), session }
}

export function heartbeatSession(input: {
  session: string
  ready?: boolean
  load?: number
  vram_free?: number | null
  public_url?: string
}): ServerRecord | null {
  prune()
  const id = sessions.get(input.session)
  if (!id) return null
  const prev = store.get(id)
  if (!prev || prev.session !== input.session) return null

  const now = Date.now()
  const load = input.load !== undefined ? Number(input.load) : prev.load
  const record: Stored = {
    ...prev,
    public_url: input.public_url?.replace(/\/$/, '') || prev.public_url,
    ready: input.ready !== undefined ? Boolean(input.ready) : prev.ready,
    load: Number.isFinite(load) ? load : prev.load,
    vram_free:
      input.vram_free !== undefined ? input.vram_free : prev.vram_free,
    updated_at: now,
    expires_at: now + TTL_MS,
  }
  store.set(id, record)
  return publicRecord(record)
}

/** Legacy upsert kept for older clients still hitting /register with Bearer. */
export function upsertServer(input: {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
  session?: string
}): ServerRecord {
  prune()
  const now = Date.now()
  const session = input.session || newSessionId()
  const record: Stored = {
    id: input.id,
    public_url: input.public_url,
    load: input.load,
    ready: input.ready,
    vram_free: input.vram_free ?? null,
    updated_at: now,
    expires_at: now + TTL_MS,
    session,
  }
  store.set(record.id, record)
  sessions.set(session, record.id)
  return publicRecord(record)
}
