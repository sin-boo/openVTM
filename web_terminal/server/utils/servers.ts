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

type SessionClaims = {
  id: string
  public_url: string
  ready: boolean
  load: number
  vram_free?: number | null
  exp: number
}

/** Miss a few heartbeats at 2s interval before dropping. */
const TTL_MS = 15_000
const TTL_S = Math.ceil(TTL_MS / 1000)

const KEY_IDS = 'broker:ids'
const recKey = (id: string) => `broker:rec:${id}`
const sessKey = (session: string) => `broker:sess:${session}`

/** In-memory fallback — OK for local `nitro dev`; flaky for /servers on multi-instance Vercel. */
const memStore = new Map<string, Stored>()

export function ttlSeconds(): number {
  return TTL_S
}

function publicRecord(rec: Stored): ServerRecord {
  const { expires_at: _e, session: _s, ...rest } = rec
  return rest
}

function sessionSecret(): string {
  return (
    process.env.BROKER_TOKEN ||
    process.env.BROKER_SECRET ||
    'dev-only-insecure-broker-secret'
  )
}

function b64urlFromBytes(bytes: ArrayBuffer | Uint8Array): string {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes)
  let bin = ''
  for (let i = 0; i < u8.length; i++) bin += String.fromCharCode(u8[i]!)
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/g, '')
}

function bytesFromB64url(s: string): Uint8Array {
  const pad = s.length % 4 === 0 ? '' : '='.repeat(4 - (s.length % 4))
  const b64 = s.replace(/-/g, '+').replace(/_/g, '/') + pad
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

async function hmacSign(message: string, secret: string): Promise<string> {
  const enc = new TextEncoder()
  const key = await crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'HMAC', hash: 'SHA-256' },
    false,
    ['sign'],
  )
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(message))
  return b64urlFromBytes(sig)
}

async function mintSession(claims: SessionClaims): Promise<string> {
  const body = b64urlFromBytes(new TextEncoder().encode(JSON.stringify(claims)))
  const sig = await hmacSign(body, sessionSecret())
  return `${body}.${sig}`
}

async function verifySession(session: string): Promise<SessionClaims | null> {
  const parts = session.split('.')
  if (parts.length !== 2) return null
  const [body, sig] = parts
  if (!body || !sig) return null
  const expected = await hmacSign(body, sessionSecret())
  if (expected !== sig) return null
  try {
    const json = new TextDecoder().decode(bytesFromB64url(body))
    const claims = JSON.parse(json) as SessionClaims
    if (!claims?.id || !claims?.public_url || !claims?.exp) return null
    if (claims.exp * 1000 < Date.now()) return null
    return claims
  } catch {
    return null
  }
}

/** Stable display name per public_url so re-handshake does not spam new names. */
export function stableServerName(public_url: string): string {
  const enc = new TextEncoder().encode(public_url)
  let h = 2166136261
  for (let i = 0; i < enc.length; i++) {
    h ^= enc[i]!
    h = Math.imul(h, 16777619)
  }
  const adj = [
    'amber',
    'brisk',
    'coral',
    'delta',
    'ember',
    'frost',
    'gleam',
    'harbor',
    'ivory',
    'jade',
    'keen',
    'lunar',
    'mist',
    'nova',
    'onyx',
    'plume',
    'quartz',
    'rapid',
    'solar',
    'tide',
    'ultra',
    'vivid',
    'wave',
    'zenith',
  ]
  const nouns = [
    'otter',
    'falcon',
    'cedar',
    'pixel',
    'ridge',
    'comet',
    'orchid',
    'anvil',
    'beacon',
    'cinder',
    'drizzle',
    'echo',
    'flint',
    'grove',
    'horizon',
    'iris',
    'jasper',
    'kite',
    'lagoon',
    'maple',
    'nectar',
    'orbit',
    'pebble',
    'quill',
  ]
  const a = adj[Math.abs(h) % adj.length]!
  const n = nouns[Math.abs(h >> 8) % nouns.length]!
  const suffix = (Math.abs(h >> 16) % 90) + 10
  return `${a}-${n}-${suffix}`
}

function upstashConfigured(): boolean {
  return Boolean(
    process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN,
  )
}

async function redisCmd<T = unknown>(args: Array<string | number>): Promise<T> {
  const url = process.env.UPSTASH_REDIS_REST_URL
  const token = process.env.UPSTASH_REDIS_REST_TOKEN
  if (!url || !token) throw new Error('Upstash not configured')
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
  if (!url || !token) throw new Error('Upstash not configured')
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
    if (rec.expires_at <= now) memStore.delete(id)
  }
}

async function saveRecord(rec: Stored): Promise<void> {
  if (!upstashConfigured()) {
    memStore.set(rec.id, rec)
    return
  }
  const payload = JSON.stringify(rec)
  await redisPipeline([
    ['SET', recKey(rec.id), payload, 'EX', TTL_S],
    ['SET', sessKey(rec.session), rec.id, 'EX', TTL_S],
    ['SADD', KEY_IDS, rec.id],
  ])
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

export async function handshakeServer(input: {
  public_url: string
  ready?: boolean
  load?: number
}): Promise<{ server: ServerRecord; session: string }> {
  const now = Date.now()
  const id = stableServerName(input.public_url)
  const claims: SessionClaims = {
    id,
    public_url: input.public_url,
    load: Number.isFinite(input.load) ? Number(input.load) : 0,
    ready: Boolean(input.ready),
    vram_free: null,
    exp: Math.floor((now + TTL_MS) / 1000),
  }
  const session = await mintSession(claims)
  const record: Stored = {
    id,
    public_url: claims.public_url,
    load: claims.load,
    ready: claims.ready,
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
}): Promise<{ server: ServerRecord; session: string } | null> {
  const claims = await verifySession(input.session)
  if (!claims) return null

  const now = Date.now()
  const load = input.load !== undefined ? Number(input.load) : claims.load
  const nextClaims: SessionClaims = {
    id: claims.id,
    public_url: input.public_url?.replace(/\/$/, '') || claims.public_url,
    ready: input.ready !== undefined ? Boolean(input.ready) : claims.ready,
    load: Number.isFinite(load) ? load : claims.load,
    vram_free:
      input.vram_free !== undefined ? input.vram_free : claims.vram_free ?? null,
    exp: Math.floor((now + TTL_MS) / 1000),
  }
  const session = await mintSession(nextClaims)
  const record: Stored = {
    id: nextClaims.id,
    public_url: nextClaims.public_url,
    load: nextClaims.load,
    ready: nextClaims.ready,
    vram_free: nextClaims.vram_free ?? null,
    updated_at: now,
    expires_at: now + TTL_MS,
    session,
  }
  await saveRecord(record)
  return { server: publicRecord(record), session }
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
  const id = input.id || stableServerName(input.public_url) || randomServerName()
  const claims: SessionClaims = {
    id,
    public_url: input.public_url,
    load: input.load,
    ready: input.ready,
    vram_free: input.vram_free ?? null,
    exp: Math.floor((now + TTL_MS) / 1000),
  }
  const session = input.session || (await mintSession(claims))
  const record: Stored = {
    id,
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
