import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  fetchHealth,
  fetchPick,
  fetchServers,
  type HealthResponse,
  type ServerRecord,
} from './api'

function ageLabel(updatedAt: number): string {
  const sec = Math.max(0, Math.round((Date.now() - updatedAt) / 1000))
  if (sec < 5) return 'just now'
  if (sec < 60) return `${sec}s ago`
  return `${Math.floor(sec / 60)}m ${sec % 60}s ago`
}

function statusKind(s: ServerRecord): 'ready' | 'busy' | 'down' {
  if (!s.ready) return 'down'
  if (Number(s.load) > 0) return 'busy'
  return 'ready'
}

function statusLabel(kind: 'ready' | 'busy' | 'down'): string {
  if (kind === 'ready') return 'Ready'
  if (kind === 'busy') return 'Busy'
  return 'Not ready'
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [online, setOnline] = useState<boolean | null>(null)
  const [servers, setServers] = useState<ServerRecord[]>([])
  const [pick, setPick] = useState<{
    ok: boolean
    server?: ServerRecord
    error?: string
  } | null>(null)
  const [refreshedAt, setRefreshedAt] = useState<Date | null>(null)
  const [loading, setLoading] = useState(false)
  const [copied, setCopied] = useState(false)

  const readyCount = useMemo(
    () => servers.filter((s) => s.ready).length,
    [servers],
  )

  const refresh = useCallback(async () => {
    setLoading(true)
    const h = await fetchHealth()
    setHealth(h)
    setOnline(Boolean(h))
    if (!h) {
      setServers([])
      setPick({ ok: false, error: 'broker unreachable' })
      setRefreshedAt(new Date())
      setLoading(false)
      return
    }
    const [list, chosen] = await Promise.all([fetchServers(), fetchPick()])
    setServers(list)
    setPick(chosen)
    setRefreshedAt(new Date())
    setLoading(false)
  }, [])

  useEffect(() => {
    void refresh()
    const id = window.setInterval(() => {
      void refresh()
    }, 2000)
    return () => window.clearInterval(id)
  }, [refresh])

  const copyToken = async () => {
    if (!health?.join_token) return
    try {
      await navigator.clipboard.writeText(health.join_token)
      setCopied(true)
      window.setTimeout(() => setCopied(false), 1500)
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="shell">
      <div className="atmosphere" aria-hidden="true" />

      <header className="hero">
        <div>
          <p className="brand">openVTM</p>
          <h1>Web Terminal</h1>
          <p className="lede">
            GPU hosts handshake with the shared join token, get a random server
            name + session, then ping every 2s. Images never pass through here.
          </p>
        </div>
        <div
          className={`pill ${online === true ? 'ok' : online === false ? 'bad' : ''}`}
        >
          <span className="dot" />
          {online === null
            ? 'Checking broker…'
            : online
              ? 'Broker online'
              : 'Broker offline'}
        </div>
      </header>

      <section className="token-panel">
        <div>
          <p className="label">Fleet join token</p>
          <p className="token-value mono">
            {health?.token_configured && health.join_token
              ? health.join_token
              : 'NOT SET'}
          </p>
          <p className="sub">
            Set <code>BROKER_TOKEN</code> in Vercel env (7 chars). Never commit
            the token to git — copy it from here only after env is configured.
          </p>
        </div>
        <button
          type="button"
          className="btn"
          onClick={() => void copyToken()}
          disabled={!health?.token_configured || !health.join_token}
        >
          {copied ? 'Copied' : 'Copy token'}
        </button>
      </section>

      <section className="metrics">
        <article>
          <p className="label">Live servers</p>
          <p className="value">{servers.length}</p>
        </article>
        <article>
          <p className="label">Ready</p>
          <p className="value">{readyCount}</p>
        </article>
        <article>
          <p className="label">Best pick</p>
          <p className="value mono">
            {pick?.ok && pick.server ? pick.server.id : 'none'}
          </p>
        </article>
        <article>
          <p className="label">Last refresh</p>
          <p className="value mono">
            {refreshedAt ? refreshedAt.toLocaleTimeString() : '—'}
          </p>
        </article>
      </section>

      <section className="panel">
        <div className="panel-head">
          <div>
            <h2>Registered servers</h2>
            <p className="sub">
              Random names from handshake · heartbeat every 2s · drop after ~8s
              silence
            </p>
          </div>
          <button
            type="button"
            className="btn"
            onClick={() => void refresh()}
            disabled={loading}
          >
            {loading ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Status</th>
                <th>Server name</th>
                <th>Public URL</th>
                <th>Load</th>
                <th>Heartbeat</th>
              </tr>
            </thead>
            <tbody>
              {servers.length === 0 ? (
                <tr>
                  <td colSpan={5} className="empty">
                    No live servers. On the GPU host set{' '}
                    <code>BROKER_URL</code>, <code>BROKER_TOKEN</code>, and{' '}
                    <code>PUBLIC_URL</code>, then run{' '}
                    <code>./start-server-linux.sh</code>.
                  </td>
                </tr>
              ) : (
                servers.map((s) => {
                  const kind = statusKind(s)
                  return (
                    <tr key={s.id}>
                      <td>
                        <span className={`badge ${kind}`}>
                          {statusLabel(kind)}
                        </span>
                      </td>
                      <td className="mono">{s.id}</td>
                      <td>
                        <a
                          className="mono link"
                          href={s.public_url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {s.public_url}
                        </a>
                      </td>
                      <td className="mono">{Number(s.load).toFixed(2)}</td>
                      <td className="mono">{ageLabel(s.updated_at)}</td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Current pick</h2>
          <span className="tag">
            {pick?.ok ? 'selected' : pick ? 'empty' : 'idle'}
          </span>
        </div>
        <pre className="code">
          {JSON.stringify(
            pick ?? { ok: false, error: 'no servers available' },
            null,
            2,
          )}
        </pre>
      </section>

      <footer>
        <span>Nitro API · React dashboard</span>
        <span>{new Date().toLocaleString()}</span>
      </footer>
    </div>
  )
}
