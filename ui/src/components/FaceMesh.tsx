/** Privacy-safe face mesh: landmark dots only. Never renders video/camera.
 *
 * Calibration (recenterKey) captures a neutral face, then each frame is
 * rigidly aligned back to that neutral. Expression (mouth, eyes, brows)
 * stays as real tracker geometry — no template residual remapping.
 */

import { useEffect, useMemo, useRef, useState } from 'react'

export type Landmark = [number, number]

export type MeshSettings = {
  /** 0 = raw, 1 = heavy temporal smoothing */
  smoothing: number
  /** Dot radius in SVG units */
  size: number
  /** Overall opacity 0..1 */
  opacity: number
  /** Soft glow halo around each dot */
  glow: number
  /** Hide points below this confidence */
  confThreshold: number
  /** Inset when fitting face into the view (0..0.4) */
  zoom: number
  /**
   * Horizontal stretch when fitting. Landmarks arrive in 0..1 image space
   * (x/w, y/h), so faces look too narrow on landscape cameras without this.
   */
  widthScale: number
}

export const DEFAULT_MESH_SETTINGS: MeshSettings = {
  smoothing: 0.45,
  size: 2.35,
  opacity: 0.95,
  glow: 0.4,
  confThreshold: 0.12,
  zoom: 0.14,
  widthScale: 1.32,
}

type Props = {
  landmarks: Landmark[]
  confidences?: number[]
  width?: number
  height?: number
  settings?: Partial<MeshSettings>
  /** Bump on Calibrate (and track start) to capture a new neutral face. */
  recenterKey?: number
  label?: string
}

type Pt = { x: number; y: number; ok: boolean; i: number }

type ViewLock = { cx: number; cy: number; bw: number; bh: number }

type Similarity = { tx: number; ty: number; scale: number; cos: number; sin: number }

/**
 * Rigid-align anchors only. Mouth (48–67) is intentionally excluded so
 * lip motion cannot warp the similarity fit or get template-remapped.
 */
const STABLE_IDX = [
  0, 8, 16, // jaw L / chin / jaw R
  19, 24, // brow mids
  27, 30, 33, // nose bridge / tip / base mid
  36, 39, 42, 45, // eye outer/inner corners
]

/** Inner lip indices — denser cluster; draw slightly smaller. */
const INNER_MOUTH = new Set([60, 61, 62, 63, 64, 65, 66, 67])

function pointOk(
  pt: Landmark | undefined,
  confidences: number[] | undefined,
  i: number,
  confThreshold: number,
): boolean {
  if (!pt) return false
  const c = confidences?.[i] ?? 1
  return c >= confThreshold && Number.isFinite(pt[0]) && Number.isFinite(pt[1])
}

function estimateSimilarity(src: Landmark[], dst: Landmark[], idxs: number[]): Similarity | null {
  const pairs: { sx: number; sy: number; dx: number; dy: number }[] = []
  for (const i of idxs) {
    const s = src[i]
    const d = dst[i]
    if (!s || !d) continue
    if (![s[0], s[1], d[0], d[1]].every(Number.isFinite)) continue
    pairs.push({ sx: s[0], sy: s[1], dx: d[0], dy: d[1] })
  }
  if (pairs.length < 3) return null

  let msx = 0
  let msy = 0
  let mdx = 0
  let mdy = 0
  for (const p of pairs) {
    msx += p.sx
    msy += p.sy
    mdx += p.dx
    mdy += p.dy
  }
  const n = pairs.length
  msx /= n
  msy /= n
  mdx /= n
  mdy /= n

  let sxx = 0
  let sxy = 0
  let syx = 0
  let syy = 0
  let varSrc = 0
  for (const p of pairs) {
    const sx = p.sx - msx
    const sy = p.sy - msy
    const dx = p.dx - mdx
    const dy = p.dy - mdy
    sxx += sx * dx
    sxy += sx * dy
    syx += sy * dx
    syy += sy * dy
    varSrc += sx * sx + sy * sy
  }
  if (varSrc < 1e-12) return null

  const a = sxx + syy
  const b = sxy - syx
  const norm = Math.hypot(a, b) || 1
  const cos = a / norm
  const sin = b / norm
  const scale = (a * cos + b * sin) / varSrc
  if (!Number.isFinite(scale) || scale <= 1e-6) return null

  const tx = mdx - scale * (cos * msx - sin * msy)
  const ty = mdy - scale * (sin * msx + cos * msy)
  return { tx, ty, scale, cos, sin }
}

function applySimilarity(pt: Landmark, s: Similarity): Landmark {
  const x = s.scale * (s.cos * pt[0] - s.sin * pt[1]) + s.tx
  const y = s.scale * (s.sin * pt[0] + s.cos * pt[1]) + s.ty
  return [x, y]
}

function boundsOf(
  points: Landmark[],
  confidences: number[] | undefined,
  confThreshold: number,
): ViewLock | null {
  let minX = Infinity
  let maxX = -Infinity
  let minY = Infinity
  let maxY = -Infinity
  let n = 0
  for (let i = 0; i < points.length; i++) {
    if (!pointOk(points[i], confidences, i, confThreshold)) continue
    const [x, y] = points[i]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
    n++
  }
  if (n < 8) return null
  return {
    cx: (minX + maxX) / 2,
    cy: (minY + maxY) / 2,
    bw: Math.max(1e-6, maxX - minX),
    bh: Math.max(1e-6, maxY - minY),
  }
}

function applyLock(
  points: Pt[],
  lock: ViewLock,
  width: number,
  height: number,
  padding: number,
  widthScale = 1,
): Pt[] {
  const pad = Math.max(0, Math.min(0.45, padding))
  const stretch = Math.max(0.5, Math.min(2.5, widthScale))
  const availW = width * (1 - 2 * pad)
  const availH = height * (1 - 2 * pad)
  // Fit the stretched bbox so wider faces don't clip the sides.
  const scale = Math.min(availW / (lock.bw * stretch), availH / lock.bh)
  const sx = scale * stretch
  const sy = scale
  const ox = width / 2
  const oy = height / 2
  return points.map((p) =>
    p.ok ? { ...p, x: ox + (p.x - lock.cx) * sx, y: oy + (p.y - lock.cy) * sy } : p,
  )
}

/**
 * Rigidly warp live → calibrated neutral pose.
 * Keeps real mouth / eye / brow shapes; only undoes head translate/rotate/scale.
 */
function stabilizeToNeutral(
  live: Landmark[],
  neutral: Landmark[],
  confidences: number[] | undefined,
  confThreshold: number,
): Pt[] {
  const sim = estimateSimilarity(live, neutral, STABLE_IDX)
  return live.map((pt, i) => {
    const ok = pointOk(pt, confidences, i, confThreshold)
    if (!ok) return { x: pt[0], y: pt[1], ok: false, i }
    if (!sim) return { x: pt[0], y: pt[1], ok: true, i }
    const [x, y] = applySimilarity(pt, sim)
    return { x, y, ok: true, i }
  })
}

export default function FaceMesh({
  landmarks,
  confidences,
  width = 280,
  height = 210,
  settings,
  recenterKey = 0,
  label = 'Tracking mesh (no camera)',
}: Props) {
  const cfg: MeshSettings = { ...DEFAULT_MESH_SETTINGS, ...settings }
  const prevRef = useRef<(Landmark | null)[]>(Array.from({ length: 68 }, () => null))
  const [smoothed, setSmoothed] = useState<Landmark[] | null>(null)
  const [neutral, setNeutral] = useState<Landmark[] | null>(null)
  const [viewLock, setViewLock] = useState<ViewLock | null>(null)
  const pendingNeutral = useRef(true)

  useEffect(() => {
    pendingNeutral.current = true
    setNeutral(null)
    setViewLock(null)
    prevRef.current = Array.from({ length: 68 }, () => null)
    setSmoothed(null)
  }, [recenterKey])

  useEffect(() => {
    if (!landmarks || landmarks.length < 68) {
      prevRef.current = Array.from({ length: 68 }, () => null)
      setSmoothed(null)
      return
    }

    const keep = Math.max(0, Math.min(0.95, cfg.smoothing))
    const take = 1 - keep
    const snap = pendingNeutral.current
    const next: Landmark[] = landmarks.map((pt, i) => {
      const ok = pointOk(pt, confidences, i, cfg.confThreshold)
      if (!ok) {
        prevRef.current[i] = null
        return pt
      }
      const prev = prevRef.current[i]
      if (!prev || keep <= 0.001 || snap) {
        const fresh: Landmark = [pt[0], pt[1]]
        prevRef.current[i] = fresh
        return fresh
      }
      const blended: Landmark = [prev[0] * keep + pt[0] * take, prev[1] * keep + pt[1] * take]
      prevRef.current[i] = blended
      return blended
    })
    setSmoothed(next)
  }, [landmarks, confidences, cfg.smoothing, cfg.confThreshold])

  useEffect(() => {
    if (!smoothed || smoothed.length < 68) return
    if (!pendingNeutral.current && neutral && viewLock) return
    let okCount = 0
    for (let i = 0; i < 68; i++) {
      if (pointOk(smoothed[i], confidences, i, cfg.confThreshold)) okCount++
    }
    if (okCount < 50) return
    const lock = boundsOf(smoothed, confidences, cfg.confThreshold)
    if (!lock) return
    pendingNeutral.current = false
    setNeutral(smoothed.map((p) => [p[0], p[1]] as Landmark))
    setViewLock(lock)
  }, [smoothed, confidences, cfg.confThreshold, neutral, viewLock])

  const fitted = useMemo(() => {
    if (!smoothed || smoothed.length < 68 || !neutral || !viewLock) return []
    const stabilized = stabilizeToNeutral(smoothed, neutral, confidences, cfg.confThreshold)
    return applyLock(stabilized, viewLock, width, height, cfg.zoom, cfg.widthScale)
  }, [
    smoothed,
    neutral,
    viewLock,
    confidences,
    cfg.confThreshold,
    cfg.zoom,
    cfg.widthScale,
    width,
    height,
  ])

  const hasFace = fitted.some((p) => p.ok)
  const baseR = Math.max(1.1, cfg.size)
  const glowOp = Math.max(0, Math.min(1, cfg.glow)) * 0.32 * cfg.opacity

  return (
    <div className="face-mesh">
      <div className="face-mesh-label">{label}</div>
      <svg
        width={width}
        height={height}
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Face landmark dots"
      >
        <defs>
          <radialGradient id="meshDotGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#dce6ff" stopOpacity={0.85} />
            <stop offset="100%" stopColor="#6ea8fe" stopOpacity={0} />
          </radialGradient>
        </defs>
        <rect x={0} y={0} width={width} height={height} fill="#0b0d12" rx={8} />
        {!hasFace ? (
          <text x={width / 2} y={height / 2} textAnchor="middle" fill="#5c6578" fontSize={12}>
            {landmarks.length >= 68 ? 'Calibrating mesh… look forward' : 'Waiting for landmarks…'}
          </text>
        ) : (
          fitted.map((p) => {
            if (!p.ok) return null
            const r = INNER_MOUTH.has(p.i) ? baseR * 0.72 : baseR
            const glowR = r * (1.45 + cfg.glow * 1.5)
            return (
              <g key={p.i} opacity={cfg.opacity}>
                {cfg.glow > 0.02 ? (
                  <circle cx={p.x} cy={p.y} r={glowR} fill="url(#meshDotGlow)" opacity={glowOp} />
                ) : null}
                <circle cx={p.x} cy={p.y} r={r} fill="#e8eeff" />
                <circle cx={p.x} cy={p.y} r={r * 0.42} fill="#ffffff" opacity={0.5} />
              </g>
            )
          })
        )}
      </svg>
    </div>
  )
}
