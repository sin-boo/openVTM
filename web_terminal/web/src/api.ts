export type ServerRecord = {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
  updated_at: number
}

export type HealthResponse = {
  ok: boolean
  service?: string
  join_token?: string
  auth?: {
    token_chars: number
    handshake: string
    heartbeat: string
    interval_s: number
  }
}

export async function fetchHealth(): Promise<HealthResponse | null> {
  try {
    const r = await fetch('/health')
    const j = (await r.json()) as HealthResponse
    return r.ok && j.ok ? j : null
  } catch {
    return null
  }
}

export async function fetchServers(): Promise<ServerRecord[]> {
  const r = await fetch('/servers')
  const j = (await r.json()) as { servers?: ServerRecord[] }
  return Array.isArray(j.servers) ? j.servers : []
}

export async function fetchPick(): Promise<{
  ok: boolean
  server?: ServerRecord
  error?: string
}> {
  const r = await fetch('/pick')
  return (await r.json()) as {
    ok: boolean
    server?: ServerRecord
    error?: string
  }
}
