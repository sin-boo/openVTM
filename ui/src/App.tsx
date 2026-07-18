import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  api,
  openGenerateStreamSocket,
  openTrackingSocket,
  streamPose,
  streamStart,
  streamStop,
  type CameraInfo,
  type StatusResponse,
  type StreamMessage,
  type TrackingStatus,
} from './api'
import FaceMesh, { DEFAULT_MESH_SETTINGS, type Landmark, type MeshSettings } from './components/FaceMesh'
import './App.css'

const SLIDERS: { key: string; label: string; min: number; max: number; step: number; default: number }[] = [
  { key: 'ParamAngleX', label: 'AngleX', min: -30, max: 30, step: 0.5, default: 0 },
  { key: 'ParamAngleY', label: 'AngleY', min: -30, max: 30, step: 0.5, default: 0 },
  { key: 'ParamAngleZ', label: 'AngleZ', min: -30, max: 30, step: 0.5, default: 0 },
  { key: 'ParamMouthOpenY', label: 'MouthOpen', min: 0, max: 1, step: 0.01, default: 0 },
  { key: 'ParamEyeLOpen', label: 'EyeLOpen', min: 0, max: 1, step: 0.01, default: 1 },
  { key: 'ParamEyeROpen', label: 'EyeROpen', min: 0, max: 1, step: 0.01, default: 1 },
  { key: 'ParamEyeBallX', label: 'EyeBallX', min: -1, max: 1, step: 0.01, default: 0 },
  { key: 'ParamEyeBallY', label: 'EyeBallY', min: -1, max: 1, step: 0.01, default: 0 },
  { key: 'MouthSmile', label: 'Smile', min: -1, max: 1, step: 0.01, default: 0 },
]

function defaultPose(): Record<string, number> {
  const o: Record<string, number> = {}
  for (const s of SLIDERS) o[s.key] = s.default
  return o
}

function applyTrackedPose(pose: Record<string, number>): Record<string, number> {
  const next = defaultPose()
  for (const s of SLIDERS) {
    if (pose[s.key] !== undefined && Number.isFinite(pose[s.key])) {
      next[s.key] = Number(pose[s.key])
    }
  }
  return next
}

export default function App() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [checkpoints, setCheckpoints] = useState<string[]>([])
  const [ckpt, setCkpt] = useState('')
  const [pose, setPose] = useState(defaultPose)
  // Prompt fields stay empty by default and are never overwritten by server defaults or tracking.
  const [prompt, setPrompt] = useState('')
  const [negative, setNegative] = useState('')
  const [steps, setSteps] = useState(8)
  const [cfg, setCfg] = useState(4.5)
  const [seed, setSeed] = useState(42)
  const [ipAdapter, setIpAdapter] = useState('plus (closer to image)')
  const [ipPct, setIpPct] = useState(85)
  const [poseDrive, setPoseDrive] = useState('prompt tags (visible motion)')
  const [poseRms, setPoseRms] = useState(12)
  const [refPreview, setRefPreview] = useState<string | null>(null)
  const [refName, setRefName] = useState('')
  const [resultImage, setResultImage] = useState<string | null>(null)
  const [streaming, setStreaming] = useState(false)
  const [streamFps, setStreamFps] = useState(0)
  const [error, setError] = useState('')

  const [cameras, setCameras] = useState<CameraInfo[]>([])
  const [cameraIndex, setCameraIndex] = useState(0)
  const [mirror, setMirror] = useState(false)
  const [track, setTrack] = useState<TrackingStatus | null>(null)
  const [trackingBusy, setTrackingBusy] = useState(false)
  const [mesh, setMesh] = useState<MeshSettings>(DEFAULT_MESH_SETTINGS)
  const [meshRecenterKey, setMeshRecenterKey] = useState(0)
  const wsRef = useRef<WebSocket | null>(null)
  const streamWsRef = useRef<WebSocket | null>(null)
  const poseRef = useRef(pose)
  const drivePoseRef = useRef(true)
  /** First camera-list fetch should apply server preferred index (skip virtual cams). */
  const cameraPickReady = useRef(false)
  const ckptAppliedRef = useRef('')

  useEffect(() => {
    poseRef.current = pose
    const sw = streamWsRef.current
    if (sw && sw.readyState === WebSocket.OPEN && streaming) {
      streamPose(sw, pose)
    }
  }, [pose, streaming])

  const serverMode = Boolean(status?.server_mode || status?.mode === 'server')

  const refresh = useCallback(async () => {
    try {
      const s = await api.status()
      setStatus(s)
    } catch (e) {
      setError(String(e))
    }
  }, [])

  const refreshCameras = useCallback(async () => {
    try {
      const cams = await api.trackingCameras()
      setCameras(cams.cameras)
      if (cams.cameras.length) {
        const preferred =
          typeof cams.default_index === 'number'
            ? cams.default_index
            : cams.cameras[0].index
        if (!cameraPickReady.current) {
          cameraPickReady.current = true
          setCameraIndex(preferred)
        } else {
          setCameraIndex((prev) =>
            cams.cameras.some((c) => c.index === prev) ? prev : preferred,
          )
        }
      }
    } catch (e) {
      setCameras([])
      setError(`Camera list failed: ${String(e)}`)
    }
  }, [])

  const onTrackingMessage = useCallback((data: TrackingStatus) => {
    setTrack(data)
    // Drive pose from tracking, but never touch prompt/negative.
    if (drivePoseRef.current && data.has_face && data.pose) {
      const next = applyTrackedPose(data.pose)
      setPose(next)
      const sw = streamWsRef.current
      if (sw && sw.readyState === WebSocket.OPEN) {
        streamPose(sw, next)
      }
    }
  }, [])

  useEffect(() => {
    void (async () => {
      try {
        const c = await api.checkpoints()
        setCheckpoints(c.checkpoints.map((x) => x.label))
        setCkpt(c.default || c.checkpoints.at(-1)?.label || '')
        const s = await api.status()
        setStatus(s)
        // Prompt stays empty; negative gets the quality default (user can edit).
        setSteps(s.defaults.steps)
        setCfg(s.defaults.cfg)
        if (s.defaults.negative) setNegative(s.defaults.negative)
        if (s.ip_adapters?.length) setIpAdapter(s.ip_adapters[0])
        if (s.pose_drives?.length) setPoseDrive(s.pose_drives[0])

        // Cameras must not depend on model load — load 409/slow init was skipping this.
        if (!(s.server_mode || s.mode === 'server')) {
          void refreshCameras()
        }

        // Models auto-load on backend startup; just pull the default reference preview.
        try {
          const ref = await api.defaultReference()
          setRefPreview(ref.preview)
          setRefName(ref.name)
        } catch {
          /* optional — backend may already have loaded it */
        }
      } catch (e) {
        setError(String(e))
      }
    })()
    const id = window.setInterval(() => void refresh(), 1000)
    return () => window.clearInterval(id)
  }, [refresh, refreshCameras])

  useEffect(() => {
    let closed = false
    const ws = openTrackingSocket(onTrackingMessage)
    wsRef.current = ws
    return () => {
      closed = true
      wsRef.current = null
      // Avoid "closed before connection established" noise from Strict Mode remounts.
      if (ws.readyState === WebSocket.CONNECTING) {
        ws.onopen = () => {
          if (closed) ws.close()
        }
        ws.onerror = null
        return
      }
      if (ws.readyState === WebSocket.OPEN) ws.close()
    }
  }, [onTrackingMessage])
  const ready = status?.state === 'ready' || status?.state === 'streaming'
  const loading = status?.state === 'loading'
  const canStream = Boolean(refPreview) && ready && !loading
  const trackingActive = Boolean(track?.tracking || track?.active)

  // Apply selected checkpoint as soon as models are ready (and whenever the user changes it).
  useEffect(() => {
    if (!ready || !ckpt || streaming || loading) return
    if (ckptAppliedRef.current === ckpt && status?.checkpoint === ckpt) return
    if (status?.checkpoint === ckpt) {
      ckptAppliedRef.current = ckpt
      return
    }
    let cancelled = false
    void (async () => {
      setError('')
      try {
        await api.switchCheckpoint(ckpt)
        ckptAppliedRef.current = ckpt
        if (!cancelled) await refresh()
      } catch (e) {
        if (!cancelled) setError(String(e))
      }
    })()
    return () => {
      cancelled = true
    }
  }, [ckpt, ready, streaming, loading, status?.checkpoint, refresh])

  const logs = useMemo(() => status?.logs?.slice().reverse() ?? [], [status])
  const landmarks = useMemo<Landmark[]>(() => {
    const pts = track?.landmarks || []
    return pts.map((p) => [Number(p[0]), Number(p[1])] as Landmark)
  }, [track])

  async function onPickRef(file: File | null) {
    if (!file) return
    setError('')
    try {
      const r = await api.uploadReference(file)
      setRefPreview(r.preview)
      setRefName(r.name)
    } catch (e) {
      setError(String(e))
    }
  }

  function onStreamMessage(msg: StreamMessage) {
    if (msg.type === 'frame') {
      setResultImage(msg.image)
      setStreamFps(msg.fps_ema || msg.fps || 0)
      return
    }
    if (msg.type === 'status') {
      setStreaming(Boolean(msg.streaming))
      if (!msg.streaming) setStreamFps(0)
      void refresh()
      return
    }
    if (msg.type === 'error') {
      setError(msg.message)
      setStreaming(false)
      setStreamFps(0)
      void refresh()
    }
  }

  function closeStreamSocket() {
    const ws = streamWsRef.current
    streamWsRef.current = null
    if (!ws) return
    try {
      streamStop(ws)
    } catch {
      /* ignore */
    }
    try {
      ws.close()
    } catch {
      /* ignore */
    }
  }

  function onStartStream() {
    if (!canStream || streaming) return
    setError('')
    closeStreamSocket()
    const ws = openGenerateStreamSocket(onStreamMessage, () => {
      setError('Stream socket error')
      setStreaming(false)
    })
    streamWsRef.current = ws
    ws.onopen = () => {
      streamStart(ws, {
        pose: poseRef.current,
        prompt,
        negative,
        steps,
        cfg,
        seed,
        ip_adapter: ipAdapter,
        ip_scale: ipPct / 100,
        pose_drive: poseDrive,
        pose_target_rms: poseRms / 100,
        norm_mode: 'slider_to_unit',
      })
      setStreaming(true)
    }
    ws.onclose = () => {
      if (streamWsRef.current === ws) {
        streamWsRef.current = null
        setStreaming(false)
        setStreamFps(0)
      }
    }
  }

  function onStopStream() {
    closeStreamSocket()
    setStreaming(false)
    setStreamFps(0)
    void refresh()
  }

  useEffect(() => {
    return () => closeStreamSocket()
  }, [])

  async function onToggleTracking() {
    setError('')
    setTrackingBusy(true)
    try {
      if (trackingActive) {
        await api.trackingStop()
      } else {
        await api.trackingStart(cameraIndex, mirror)
        setMeshRecenterKey((k) => k + 1)
      }
      const st = await api.trackingStatus()
      setTrack(st)
    } catch (e) {
      setError(String(e))
    } finally {
      setTrackingBusy(false)
    }
  }

  async function onCalibrate() {
    setError('')
    try {
      await api.trackingCalibrate()
      // Re-lock mesh framing on the current face (look straight ahead).
      setMeshRecenterKey((k) => k + 1)
    } catch (e) {
      setError(String(e))
    }
  }

  async function onMirrorChange(next: boolean) {
    setMirror(next)
    try {
      await api.trackingMirror(next)
      // Landmark topology flips with mirror — recapture neutral mesh.
      setMeshRecenterKey((k) => k + 1)
    } catch {
      /* ignore if not tracking */
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>SDAnime Pose</h1>
          <p className="sub">
            {status?.message || 'Connecting…'}
            {status?.device ? ` · ${status.device}` : ''}
            {` · mode=${serverMode ? 'server' : 'local'}`}
            {loading ? ' · loading…' : ''}
          </p>
        </div>
        <div className="header-actions">
          {streaming ? (
            <button type="button" onClick={onStopStream}>
              Stop stream{streamFps > 0 ? ` (${streamFps.toFixed(1)} FPS)` : ''}
            </button>
          ) : (
            <button type="button" className="primary" onClick={onStartStream} disabled={!canStream}>
              {loading ? 'Loading models…' : 'Start stream'}
            </button>
          )}
        </div>
      </header>

      {error ? <div className="error">{error}</div> : null}

      <div className="layout">
        <aside className="sidebar">
          <section>
            <h2>Reference</h2>
            <div className="row">
              <label className="file-btn">
                Upload…
                <input
                  type="file"
                  accept="image/*"
                  hidden
                  onChange={(e) => void onPickRef(e.target.files?.[0] ?? null)}
                />
              </label>
              <button
                type="button"
                onClick={() =>
                  void api.defaultReference().then((r) => {
                    setRefPreview(r.preview)
                    setRefName(r.name)
                  })
                }
              >
                Default
              </button>
            </div>
            <div className="ref-box">{refPreview ? <img src={refPreview} alt="ref" /> : <span>No reference</span>}</div>
            <div className="muted">{refName || '—'}</div>
          </section>

          <section>
            <h2>Checkpoint</h2>
            <select
              value={ckpt}
              disabled={streaming || loading}
              onChange={(e) => setCkpt(e.target.value)}
            >
              {checkpoints.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
            <div className="muted mt">
              {loading
                ? 'Models loading… checkpoint applies when ready'
                : status?.checkpoint
                  ? `Active: ${status.checkpoint}`
                  : 'Select a checkpoint — applied automatically'}
            </div>
          </section>

          <section>
            <h2>Face tracking</h2>
            {serverMode ? (
              <p className="muted">
                Server mode: camera disabled. Pose arrives as JSON at <code>/api/tracking/frame</code>.
              </p>
            ) : (
              <>
                <label>
                  Camera
                  <select
                    value={cameraIndex}
                    onChange={(e) => setCameraIndex(Number(e.target.value))}
                    disabled={trackingActive}
                  >
                    {cameras.length === 0 ? (
                      <option value={0}>No cameras found — click Refresh</option>
                    ) : (
                      cameras.map((c) => (
                        <option key={c.index} value={c.index}>
                          {c.index}: {c.name}
                        </option>
                      ))
                    )}
                  </select>
                </label>
                <div className="row mt">
                  <button
                    type="button"
                    onClick={() => void refreshCameras()}
                    disabled={trackingBusy}
                  >
                    Refresh cams
                  </button>
                  <button type="button" onClick={() => void onToggleTracking()} disabled={trackingBusy}>
                    {trackingActive ? 'Stop track' : 'Start face track'}
                  </button>
                  <button type="button" onClick={() => void onCalibrate()} disabled={!trackingActive}>
                    Calibrate
                  </button>
                </div>
                <label className="check mt">
                  <input
                    type="checkbox"
                    checked={mirror}
                    onChange={(e) => void onMirrorChange(e.target.checked)}
                  />
                  Mirror
                </label>
              </>
            )}
            <div className="muted mt">{track?.message || 'Tracking: off'}</div>
            {track && !track.calibration_ready && track.tracking ? (
              <div className="muted">
                Calibrating… {Math.round((track.calibration_progress || 0) * 100)}%
              </div>
            ) : null}
            <FaceMesh
              landmarks={landmarks}
              confidences={track?.confidences}
              settings={mesh}
              recenterKey={meshRecenterKey}
            />
            <div className="mesh-settings mt">
              <div className="mesh-settings-title">Mesh settings</div>
              {(
                [
                  { key: 'smoothing', label: 'Smoothing', min: 0, max: 0.9, step: 0.05 },
                  { key: 'size', label: 'Dot size', min: 1.5, max: 7, step: 0.1 },
                  { key: 'glow', label: 'Glow', min: 0, max: 1, step: 0.05 },
                  { key: 'opacity', label: 'Opacity', min: 0.25, max: 1, step: 0.05 },
                  { key: 'confThreshold', label: 'Confidence', min: 0, max: 0.6, step: 0.02 },
                  { key: 'zoom', label: 'Zoom fit', min: 0.04, max: 0.28, step: 0.01 },
                  { key: 'widthScale', label: 'Width', min: 0.9, max: 1.8, step: 0.02 },
                ] as const
              ).map((s) => (
                <label key={s.key} className="slider mesh-slider">
                  <span>{s.label}</span>
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={s.step}
                    value={mesh[s.key]}
                    onChange={(e) =>
                      setMesh((m) => ({ ...m, [s.key]: Number(e.target.value) }))
                    }
                  />
                  <em>{mesh[s.key].toFixed(2)}</em>
                </label>
              ))}
              <button
                type="button"
                className="mt"
                onClick={() => setMesh(DEFAULT_MESH_SETTINGS)}
              >
                Reset mesh
              </button>
            </div>
          </section>
          <section>
            <h2>Pose</h2>
            {SLIDERS.map((s) => (
              <label key={s.key} className="slider">
                <span>{s.label}</span>
                <input
                  type="range"
                  min={s.min}
                  max={s.max}
                  step={s.step}
                  value={pose[s.key] ?? s.default}
                  onChange={(e) => {
                    drivePoseRef.current = false
                    setPose((p) => ({ ...p, [s.key]: Number(e.target.value) }))
                  }}
                />
                <em>{(pose[s.key] ?? s.default).toFixed(2)}</em>
              </label>
            ))}
            <div className="row mt">
              <button
                type="button"
                onClick={() => {
                  drivePoseRef.current = true
                  setPose(defaultPose())
                }}
              >
                Reset pose
              </button>
              {trackingActive ? (
                <button type="button" onClick={() => { drivePoseRef.current = true }}>
                  Follow track
                </button>
              ) : null}
            </div>
          </section>
        </aside>

        <main className="main">
          <section className="controls">
            <label>
              Prompt
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                placeholder="Optional — e.g. wearing a hat, no animal ears…"
              />
            </label>
            <label>
              Negative
              <textarea
                value={negative}
                onChange={(e) => setNegative(e.target.value)}
                rows={3}
                placeholder="lowres, bad anatomy, bad hands…"
              />
            </label>

            <div className="grid3">
              <label>
                Steps
                <input type="number" min={4} max={40} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
              </label>
              <label>
                CFG
                <input
                  type="number"
                  min={1}
                  max={12}
                  step={0.5}
                  value={cfg}
                  onChange={(e) => setCfg(Number(e.target.value))}
                />
              </label>
              <label>
                Seed
                <input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
              </label>
            </div>

            <div className="grid2">
              <label>
                IP-Adapter
                <select value={ipAdapter} onChange={(e) => setIpAdapter(e.target.value)}>
                  {(status?.ip_adapters || [ipAdapter]).map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                IP scale {ipPct}%
                <input type="range" min={0} max={200} value={ipPct} onChange={(e) => setIpPct(Number(e.target.value))} />
              </label>
            </div>

            <div className="grid2">
              <label>
                Pose drive
                <select value={poseDrive} onChange={(e) => setPoseDrive(e.target.value)}>
                  {(status?.pose_drives || [poseDrive]).map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Adapter RMS {poseRms}
                <input type="range" min={0} max={50} value={poseRms} onChange={(e) => setPoseRms(Number(e.target.value))} />
              </label>
            </div>
          </section>

          <section className="preview">
            {resultImage ? (
              <img src={resultImage} alt="result" />
            ) : (
              <div className="placeholder">
                {loading ? 'Loading models…' : streaming ? 'Streaming…' : 'Press Start stream'}
              </div>
            )}
            {streaming ? (
              <div className="sub" style={{ marginTop: 8 }}>
                Live stream{streamFps > 0 ? ` · ${streamFps.toFixed(2)} FPS` : ''} · fixed seed {seed}
              </div>
            ) : null}
          </section>

          <section className="log">
            <h2>Log</h2>
            <pre>{logs.join('\n') || 'Waiting for backend logs…'}</pre>
          </section>
        </main>
      </div>
    </div>
  )
}
