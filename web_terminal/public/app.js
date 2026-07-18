const els = {
  brokerPill: document.getElementById('brokerPill'),
  brokerLabel: document.getElementById('brokerLabel'),
  metricCount: document.getElementById('metricCount'),
  metricReady: document.getElementById('metricReady'),
  metricPick: document.getElementById('metricPick'),
  metricRefresh: document.getElementById('metricRefresh'),
  serverRows: document.getElementById('serverRows'),
  pickBody: document.getElementById('pickBody'),
  pickTag: document.getElementById('pickTag'),
  refreshBtn: document.getElementById('refreshBtn'),
  clock: document.getElementById('clock'),
}

function ageLabel(updatedAt) {
  const sec = Math.max(0, Math.round((Date.now() - updatedAt) / 1000))
  if (sec < 5) return 'just now'
  if (sec < 60) return `${sec}s ago`
  return `${Math.floor(sec / 60)}m ${sec % 60}s ago`
}

function setBroker(state, label) {
  els.brokerPill.dataset.state = state
  els.brokerLabel.textContent = label
}

function renderServers(servers) {
  if (!servers.length) {
    els.serverRows.innerHTML =
      '<tr class="empty"><td colspan="5">No live servers. Waiting for GPU heartbeats…</td></tr>'
    return
  }

  els.serverRows.innerHTML = servers
    .map((s) => {
      const busy = Number(s.load) > 0
      const cls = s.ready ? (busy ? 'busy' : 'ready') : 'down'
      const label = s.ready ? (busy ? 'Busy' : 'Ready') : 'Not ready'
      const url = String(s.public_url || '')
      return `<tr>
        <td><span class="status ${cls}">${label}</span></td>
        <td class="mono">${escapeHtml(s.id)}</td>
        <td><a href="${escapeAttr(url)}" target="_blank" rel="noreferrer">${escapeHtml(url)}</a></td>
        <td class="mono">${Number(s.load).toFixed(2)}</td>
        <td class="mono">${ageLabel(s.updated_at)}</td>
      </tr>`
    })
    .join('')
}

function escapeHtml(s) {
  return String(s)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
}

function escapeAttr(s) {
  return escapeHtml(s).replaceAll("'", '&#39;')
}

async function refresh() {
  const now = new Date()
  els.clock.textContent = now.toLocaleString()
  els.metricRefresh.textContent = now.toLocaleTimeString()

  try {
    const health = await fetch('/health').then((r) => r.json())
    if (!health.ok) throw new Error('health not ok')
    setBroker('ok', 'Broker online')
  } catch {
    setBroker('bad', 'Broker unreachable')
    els.metricCount.textContent = '—'
    els.metricReady.textContent = '—'
    els.metricPick.textContent = '—'
    els.pickTag.textContent = 'offline'
    els.pickBody.textContent = JSON.stringify(
      { ok: false, error: 'broker unreachable' },
      null,
      2,
    )
    els.serverRows.innerHTML =
      '<tr class="empty"><td colspan="5">Cannot reach /health</td></tr>'
    return
  }

  let servers = []
  try {
    const list = await fetch('/servers').then((r) => r.json())
    servers = Array.isArray(list.servers) ? list.servers : []
  } catch {
    servers = []
  }

  els.metricCount.textContent = String(servers.length)
  els.metricReady.textContent = String(servers.filter((s) => s.ready).length)
  renderServers(servers)

  try {
    const pickRes = await fetch('/pick')
    const pick = await pickRes.json()
    els.pickBody.textContent = JSON.stringify(pick, null, 2)
    if (pick.ok && pick.server) {
      els.metricPick.textContent = pick.server.id
      els.pickTag.textContent = 'selected'
    } else {
      els.metricPick.textContent = 'none'
      els.pickTag.textContent = 'empty'
    }
  } catch (err) {
    els.metricPick.textContent = 'error'
    els.pickTag.textContent = 'error'
    els.pickBody.textContent = JSON.stringify(
      { ok: false, error: String(err) },
      null,
      2,
    )
  }
}

els.refreshBtn.addEventListener('click', () => {
  void refresh()
})

void refresh()
setInterval(() => {
  void refresh()
}, 5000)
