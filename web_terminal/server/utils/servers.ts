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

/** Miss a few heartbeats at 2s interval before dropping. */
const TTL_MS = 15_000
const TTL_S = Math.ceil(TTL_MS / 1000)

const KEY_IDS = 'broker:ids'
const recKey = (id: string) => `broker:rec:${id}`
const sessKey = (session: string) => `broker:sess:${session}`

/** In-memory fallback — OK for local `nitro dev`, NOT for multi-instance Vercel. */
const memStore = new Map<string, Stored>()
const memSessions = new Map<string, string>()

export function ttlSeconds(): number {
  return TTL_S
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

function upstashConfigured(): boolean {
  return Boolean(
    process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN,
  )
}

async function redisCmd<T = unknown>(args: Array<string | number>): Promise<T> {
  const url = process.env.UPSTASH_REDIS_REST_URL
  const token = process.env.UPSTASH_REDIS_REST_TOKEN
  if (!url || !token) {
    throw new Error('Upstash not configured')
  }
  const res = await fetch(`${url.replace(/\/$/, '')}`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(args),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Upstash ${res.status}: ${text.slice(0, 200)}`)
  }
  const payload = (await res.json()) as { result?: T; error?: string }
  if (payload.error) throw new Error(payload.error)
  return payload.result as T
}

async function redisPipeline(
  commands: Array<Array<string | number>>,
): Promise<unknown[]> {
  const url = process.env.UPSTASH_REDIS_REST_URL
  const token = process.env.UPSTASH_REDIS_REST_TOKEN
  if (!url || !token) {
    throw new Error('Upstash not configured')
  }
  const res = await fetch(`${url.replace(/\/$/, '')}/pipeline`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(commands),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`Upstash pipeline ${res.status}: ${text.slice(0, 200)}`)
  }
  const payload = (await res.json()) as Array<{ result?: unknown; error?: string }>
  return payload.map((row) => {
    if (row.error) throw new Error(row.error)
    return row.result
  })
}

function memPrune(): void {
  const now = Date.now()
  for (const [id, rec] of memStore) {
    if (rec.expires_at <= now) {
      memStore.delete(id)
      memSessions.delete(rec.session)
    }
  }
}

async function saveRecord(rec: Stored): Promise<void> {
  if (!upstashConfigured()) {
    memStore.set(rec.id, rec)
    memSessions.set(rec.session, rec.id)
    return
  }
  const payload = JSON.stringify(rec)
  await redisPipeline([
    ['SET', recKey(rec.id), payload, 'EX', TTL_S],
    ['SET', sessKey(rec.session), rec.id, 'EX', TTL_S],
    ['SADD', KEY_IDS, rec.id],
  ])
}

async function loadBySession(session: string): Promise<Stored | null> {
  if (!upstashConfigured()) {
    memPrune()
    const id = memSessions.get(session)
    if (!id) return null
    const prev = memStore.get(id)
    if (!prev || prev.session !== session) return null
    return prev
  }
  const id = await redisCmd<string | null>(['GET', sessKey(session)])
  if (!id) return null
  const raw = await redisCmd<string | null>(['GET', recKey(id)])
  if (!raw) return null
  try {
    return JSON.parse(raw) as Stored
  } catch {
    return null
  }
}

export async function listServers(): Promise<ServerRecord[]> {
  if (!upstashConfigured()) {
    memPrune()
    return [...memStore.values()]
      .map(publicRecord)
      .sort((a, b) => b.updated_at - a.updated_at)
  }

  const ids = (await redisCmd<string[]>(['SMEMBERS', KEY_IDS])) || []
  if (ids.length === 0) return []

  const results = await redisPipeline(ids.map((id) => ['GET', recKey(id)]))
  const alive: ServerRecord[] = []
  const stale: string[] = []

  for (let i = 0; i < ids.length; i++) {
    const raw = results[i]
    if (typeof raw !== 'string' || !raw) {
      stale.push(ids[i]!)
      continue
    }
    try {
      const rec = JSON.parse(raw) as Stored
      if (rec.expires_at <= Date.now()) {
        stale.push(ids[i]!)
        continue
      }
      alive.push(publicRecord(rec))
    } catch {
      stale.push(ids[i]!)
    }
  }

  if (stale.length) {
    await redisPipeline([
      ['SREM', KEY_IDS, ...stale],
      ...stale.map((id) => ['DEL', recKey(id)] as Array<string | number>),
    ]).catch(() => undefined)
  }

  return alive.sort((a, b) => b.updated_at - a.updated_at)
}

export async function pickBest(): Promise<ServerRecord | null> {
  const servers = await listServers()
  if (servers.length === 0) return null
  const sorted = [...servers].sort((a, b) => {
    if (a.ready !== b.ready) return a.ready ? -1 : 1
    if (a.load !== b.load) return a.load - b.load
    return b.updated_at - a.updated_at
  })
  return sorted[0] ?? null
}

/** First contact: validate join token outside this helper, then create session. */
export async function handshakeServer(input: {
  public_url: string
  ready?: boolean
  load?: number
}): Promise<{ server: ServerRecord; session: string }> {
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
  await saveRecord(record)
  return { server: publicRecord(record), session }
}

export async function heartbeatSession(input: {
  session: string
  ready?: boolean
  load?: number
  vram_free?: number | null
  public_url?: string
}): Promise<ServerRecord | null> {
  const prev = await loadBySession(input.session)
  if (!prev) return null

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
  await saveRecord(record)
  return publicRecord(record)
}

/** Legacy upsert kept for older clients still hitting /register with Bearer. */
export async function upsertServer(input: {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
  session?: string
}): Promise<ServerRecord> {
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
  await saveRecord(record)
  return publicRecord(record)
}

export function storageMode(): 'upstash' | 'memory' {
  return upstashConfigured() ? 'upstash' : 'memory'
}
