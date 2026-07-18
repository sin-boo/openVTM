export type ServerRecord = {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
  updated_at: number
}

type Stored = ServerRecord & { expires_at: number }

const TTL_MS = 45_000
const store = new Map<string, Stored>()

export function ttlSeconds(): number {
  return TTL_MS / 1000
}

function prune(): void {
  const now = Date.now()
  for (const [id, rec] of store) {
    if (rec.expires_at <= now) store.delete(id)
  }
}

export function upsertServer(input: {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
}): ServerRecord {
  prune()
  const now = Date.now()
  const record: Stored = {
    id: input.id,
    public_url: input.public_url,
    load: input.load,
    ready: input.ready,
    vram_free: input.vram_free ?? null,
    updated_at: now,
    expires_at: now + TTL_MS,
  }
  store.set(record.id, record)
  const { expires_at: _, ...publicRec } = record
  return publicRec
}

export function listServers(): ServerRecord[] {
  prune()
  return [...store.values()]
    .map(({ expires_at: _, ...rec }) => rec)
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
