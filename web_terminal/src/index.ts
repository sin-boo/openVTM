/**
 * Discovery-only broker: GPUs heartbeat here; clients pick a public_url.
 * Images never pass through this Worker.
 */

export interface Env {
  SERVERS: KVNamespace
  BROKER_SECRET: string
}

export type ServerRecord = {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
  updated_at: number
}

const INDEX_KEY = "index"
const TTL_SECONDS = 45
const KEY_PREFIX = "server:"

function corsHeaders(): HeadersInit {
  return {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
  }
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: {
      "Content-Type": "application/json",
      ...corsHeaders(),
    },
  })
}

function unauthorized(): Response {
  return json({ ok: false, error: "unauthorized" }, 401)
}

function serverKey(id: string): string {
  return `${KEY_PREFIX}${id}`
}

async function readIndex(env: Env): Promise<string[]> {
  const raw = await env.SERVERS.get(INDEX_KEY)
  if (!raw) return []
  try {
    const parsed = JSON.parse(raw) as unknown
    if (!Array.isArray(parsed)) return []
    return parsed.filter((x): x is string => typeof x === "string")
  } catch {
    return []
  }
}

async function writeIndex(env: Env, ids: string[]): Promise<void> {
  const unique = [...new Set(ids)]
  await env.SERVERS.put(INDEX_KEY, JSON.stringify(unique), {
    expirationTtl: Math.max(TTL_SECONDS * 4, 120),
  })
}

async function listLiveServers(env: Env): Promise<ServerRecord[]> {
  const ids = await readIndex(env)
  const out: ServerRecord[] = []
  const still: string[] = []
  for (const id of ids) {
    const raw = await env.SERVERS.get(serverKey(id))
    if (!raw) continue
    try {
      const rec = JSON.parse(raw) as ServerRecord
      if (!rec?.id || !rec?.public_url) continue
      out.push(rec)
      still.push(id)
    } catch {
      /* skip bad */
    }
  }
  if (still.length !== ids.length) {
    await writeIndex(env, still)
  }
  return out
}

function pickBest(servers: ServerRecord[]): ServerRecord | null {
  if (servers.length === 0) return null
  const sorted = [...servers].sort((a, b) => {
    if (a.ready !== b.ready) return a.ready ? -1 : 1
    if (a.load !== b.load) return a.load - b.load
    return b.updated_at - a.updated_at
  })
  return sorted[0] ?? null
}

function checkAuth(request: Request, env: Env): boolean {
  const secret = (env.BROKER_SECRET || "").trim()
  if (!secret || secret === "change-me") {
    // Allow register only if a real secret is configured in production;
    // for local/dev with change-me, still require the header match.
  }
  const header = request.headers.get("Authorization") || ""
  const token = header.startsWith("Bearer ") ? header.slice(7).trim() : ""
  return Boolean(secret) && token === secret
}

async function handleRegister(request: Request, env: Env): Promise<Response> {
  if (!checkAuth(request, env)) return unauthorized()
  let body: Record<string, unknown>
  try {
    body = (await request.json()) as Record<string, unknown>
  } catch {
    return json({ ok: false, error: "invalid JSON" }, 400)
  }

  const id = String(body.id || "").trim()
  const public_url = String(body.public_url || "").trim().replace(/\/$/, "")
  if (!id || !public_url) {
    return json({ ok: false, error: "id and public_url required" }, 400)
  }
  if (!/^https?:\/\//i.test(public_url)) {
    return json({ ok: false, error: "public_url must be http(s)" }, 400)
  }

  const load = Number(body.load ?? 0)
  const ready = Boolean(body.ready)
  const vram_free =
    body.vram_free === undefined || body.vram_free === null
      ? null
      : Number(body.vram_free)

  const record: ServerRecord = {
    id,
    public_url,
    load: Number.isFinite(load) ? load : 0,
    ready,
    vram_free: vram_free !== null && Number.isFinite(vram_free) ? vram_free : null,
    updated_at: Date.now(),
  }

  await env.SERVERS.put(serverKey(id), JSON.stringify(record), {
    expirationTtl: TTL_SECONDS,
  })
  const ids = await readIndex(env)
  if (!ids.includes(id)) ids.push(id)
  await writeIndex(env, ids)

  return json({ ok: true, server: record, ttl_seconds: TTL_SECONDS })
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: corsHeaders() })
    }

    const url = new URL(request.url)
    const path = url.pathname.replace(/\/$/, "") || "/"

    if (request.method === "GET" && (path === "/" || path === "/health")) {
      return json({ ok: true, service: "openvtm-web-terminal" })
    }

    if (request.method === "POST" && path === "/register") {
      return handleRegister(request, env)
    }

    if (request.method === "GET" && path === "/servers") {
      const servers = await listLiveServers(env)
      return json({ ok: true, servers, count: servers.length })
    }

    if (request.method === "GET" && path === "/pick") {
      const servers = await listLiveServers(env)
      const best = pickBest(servers)
      if (!best) {
        return json({ ok: false, error: "no servers available" }, 503)
      }
      return json({ ok: true, server: best })
    }

    return json({ ok: false, error: "not found" }, 404)
  },
}
