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
  health: () => fetch('/api/health').then((r) => json<{ ok: string; mode?: string }>(r)),
  status: () => fetch('/api/status').then((r) => json<StatusResponse>(r)),
  checkpoints: () =>
    fetch('/api/checkpoints').then((r) =>
      json<{ checkpoints: { label: string; name: string }[]; default: string }>(r),
    ),
  load: (checkpoint?: string) =>
    fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint: checkpoint || null }),
    }).then((r) => json<{ started: boolean }>(r)),
  switchCheckpoint: (label: string) =>
    fetch('/api/checkpoint', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label }),
    }).then((r) => json<{ ok: boolean; step: number; label: string }>(r)),
  uploadReference: async (file: File) => {
    const fd = new FormData()
    fd.append('file', file)
    return fetch('/api/reference', { method: 'POST', body: fd }).then((r) =>
      json<{ ok: boolean; name: string; preview: string }>(r),
    )
  },
  defaultReference: () =>
    fetch('/api/reference/default', { method: 'POST' }).then((r) =>
      json<{ ok: boolean; name: string; preview: string }>(r),
    ),
  generate: (body: Record<string, unknown>) =>
    fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<GenerateResponse>(r)),

  trackingCameras: () =>
    fetch('/api/tracking/cameras').then((r) =>
      json<{ cameras: CameraInfo[]; default_index?: number; mode: string }>(r),
    ),
  trackingStatus: () => fetch('/api/tracking/status').then((r) => json<TrackingStatus>(r)),
  trackingStart: (camera_index?: number | null, mirror = false) =>
    fetch('/api/tracking/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera_index: camera_index ?? null, mirror }),
    }).then((r) => json<{ ok: boolean; camera_index?: number }>(r)),
  trackingStop: () =>
    fetch('/api/tracking/stop', { method: 'POST' }).then((r) => json<{ ok: boolean }>(r)),
  trackingCalibrate: () =>
    fetch('/api/tracking/calibrate', { method: 'POST' }).then((r) =>
      json<{ ok: boolean; progress: number }>(r),
    ),
  trackingMirror: (mirror: boolean) =>
    fetch('/api/tracking/mirror', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mirror }),
    }).then((r) => json<{ ok: boolean; mirror: boolean }>(r)),
  trackingFrame: (body: Record<string, unknown>) =>
    fetch('/api/tracking/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }).then((r) => json<{ ok: boolean; sequence: number }>(r)),
}

/** Open a websocket to privacy-safe tracking snapshots (pose + landmarks only). */
export function openTrackingSocket(
  onMessage: (data: TrackingStatus) => void,
  onError?: (err: Event) => void,
): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${window.location.host}/api/tracking/ws`)
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
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${window.location.host}/api/generate/ws`)
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
