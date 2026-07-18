export type StatusResponse = {
  state: string
  message: string
  checkpoint: string
  device: string
  error: string
  mode?: 'local' | 'server'
  server_mode?: boolean
  streaming?: boolean
  param_names: string[]
  ip_adapters: string[]
  pose_drives: string[]
  defaults: {
    prompt: string
    negative: string
    steps: number
    cfg: number
    checkpoint: string
  }
  has_reference: boolean
  reference_name: string
  logs: string[]
  tracking?: {
    mode: string
    active: boolean
    message: string
    calibration_ready: boolean
    calibration_progress: number
  }
}

export type GenerateResponse = {
  ok: boolean
  elapsed: number
  fps: number
  seed: number
  pose_tags: string
  prompt: string
  filename: string
  image: string
  url: string
}

export type TrackingStatus = {
  active: boolean
  mode: 'local' | 'server' | string
  tracking: boolean
  has_face: boolean
  calibrating: boolean
  calibration_progress: number
  calibration_ready: boolean
  camera_index: number | null
  mirror: boolean
  packets_received: number
  age_seconds: number
  face_id: number | null
  got_3d: boolean
  fit_error: number
  pose: Record<string, number>
  landmarks: number[][]
  confidences: number[]
  message: string
  sequence: number
  timestamp: number
}

export type CameraInfo = { index: number; name: string }

export type BrokerServer = {
  id: string
  public_url: string
  load: number
  ready: boolean
  vram_free?: number | null
  updated_at: number
}

/** Absolute API base. Empty string = same-origin / Vite proxy (localhost). */
let _apiBase = ''

export function getApiBase(): string {
  return _apiBase
}

export function setApiBase(url: string): void {
  _apiBase = (url || '').replace(/\/$/, '')
}

function apiUrl(path: string): string {
  if (!_apiBase) return path
  return `${_apiBase}${path.startsWith('/') ? path : `/${path}`}`
}

/**
 * If VITE_BROKER_URL is set, ask the discovery broker for the best GPU
 * and point subsequent API calls at its public_url. Otherwise keep localhost.
 */
export async function resolveApiBaseFromBroker(): Promise<string> {
  const broker = (import.meta.env.VITE_BROKER_URL as string | undefined)?.trim()
  if (!broker) {
    setApiBase('')
    return ''
  }
  const root = broker.replace(/\/$/, '')
  const res = await fetch(`${root}/pick`)
  const body = (await res.json()) as {
    ok?: boolean
    server?: BrokerServer
    error?: string
  }
  if (!res.ok || !body.ok || !body.server?.public_url) {
    throw new Error(body.error || `broker pick failed (${res.status})`)
  }
  setApiBase(body.server.public_url)
  return body.server.public_url
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail || JSON.stringify(body)
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () =>
    fetch(apiUrl('/api/health')).then((r) => json<{ ok: string; mode?: string }>(r)),
  status: () => fetch(apiUrl('/api/status')).then((r) => json<StatusResponse>(r)),
  checkpoints: () =>
    fetch(apiUrl('/api/checkpoints')).then((r) =>
      json<{ checkpoints: { label: string; name: string }[]; default: string }>(r),
    ),
  load: (checkpoint?: string) =>
    fetch(apiUrl('/api/load'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint: checkpoint || null }),
    }).then((r) => json<{ started: boolean }>(r)),
  switchCheckpoint: (label: string) =>
    fetch(apiUrl('/api/checkpoint'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    }).then((r) => json<{ ok: boolean; step: number; label: string }>(r)),
  uploadReference: async (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch(apiUrl('/api/reference'), { method: 'POST', body: fd }).then((r) =>
      json<{ ok: boolean; name: string; preview: string }>(r),
    )
  },
  defaultReference: () =>
    fetch(apiUrl('/api/reference/default'), { method: 'POST' }).then((r) =>
      json<{ ok: boolean; name: string; preview: string }>(r),
    ),
  generate: (body: Record<string, unknown>) =>
    fetch(apiUrl('/api/generate'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<GenerateResponse>(r)),

  trackingCameras: () =>
    fetch(apiUrl('/api/tracking/cameras')).then((r) =>
      json<{ cameras: CameraInfo[]; default_index?: number; mode: string }>(r),
    ),
  trackingStatus: () =>
    fetch(apiUrl('/api/tracking/status')).then((r) => json<TrackingStatus>(r)),
  trackingStart: (camera_index?: number | null, mirror = false) =>
    fetch(apiUrl('/api/tracking/start'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_index: camera_index ?? null, mirror }),
    }).then((r) => json<{ ok: boolean; camera_index?: number }>(r)),
  trackingStop: () =>
    fetch(apiUrl('/api/tracking/stop'), { method: 'POST' }).then((r) =>
      json<{ ok: boolean }>(r),
    ),
  trackingCalibrate: () =>
    fetch(apiUrl('/api/tracking/calibrate'), { method: 'POST' }).then((r) =>
      json<{ ok: boolean; progress: number }>(r),
    ),
  trackingMirror: (mirror: boolean) =>
    fetch(apiUrl('/api/tracking/mirror'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mirror }),
    }).then((r) => json<{ ok: boolean; mirror: boolean }>(r)),
  trackingFrame: (body: Record<string, unknown>) =>
    fetch(apiUrl('/api/tracking/frame'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<{ ok: boolean; sequence: number }>(r)),
}

function wsUrl(path: string): string {
  if (_apiBase) {
    const u = new URL(_apiBase)
    const proto = u.protocol === 'https:' ? 'wss' : 'ws'
    return `${proto}://${u.host}${path}`
  }
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}${path}`
}

/** Open a websocket to privacy-safe tracking snapshots (pose + landmarks only). */
export function openTrackingSocket(
  onMessage: (data: TrackingStatus) => void,
  onError?: (err: Event) => void,
): WebSocket {
  const ws = new WebSocket(wsUrl('/api/tracking/ws'))
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data) as TrackingStatus)
    } catch {
      /* ignore */
    }
  }
  if (onError) ws.onerror = onError
  return ws
}

export type StreamFrameMessage = {
  type: 'frame'
  sequence: number
  elapsed: number
  fps: number
  fps_ema: number
  seed: number
  pose_tags: string
  image: string
}

export type StreamStatusMessage = {
  type: 'status'
  streaming: boolean
  message: string
}

export type StreamErrorMessage = {
  type: 'error'
  message: string
}

export type StreamMessage = StreamFrameMessage | StreamStatusMessage | StreamErrorMessage

export type StreamStartPayload = {
  prompt: string
  negative: string
  steps: number
  cfg: number
  seed: number
  ip_adapter: string
  ip_scale: number
  pose_drive: string
  pose_target_rms: number
  norm_mode?: string
  pose?: Record<string, number>
  jpeg_quality?: number
}

/** Live pose→image stream (GPU-paced, JPEG frames). */
export function openGenerateStreamSocket(
  onMessage: (data: StreamMessage) => void,
  onError?: (err: Event) => void,
): WebSocket {
  const ws = new WebSocket(wsUrl('/api/generate/ws'))
  ws.onmessage = (ev) => {
    try {
      onMessage(JSON.parse(ev.data) as StreamMessage)
    } catch {
      /* ignore */
    }
  }
  if (onError) ws.onerror = onError
  return ws
}

export function streamStart(ws: WebSocket, body: StreamStartPayload) {
  ws.send(JSON.stringify({ type: 'start', ...body }))
}

export function streamPose(ws: WebSocket, pose: Record<string, number>) {
  ws.send(JSON.stringify({ type: 'pose', pose }))
}

export function streamStop(ws: WebSocket) {
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'stop' }))
  }
}
