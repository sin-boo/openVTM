const ALPHANUM = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789' // no 0/O/1/I

export function normalizeJoinToken(raw: string): string {
  return raw
    .toUpperCase()
    .replace(/[^A-Z0-9]/g, '')
    .slice(0, 7)
}

export function isValidJoinToken(token: string): boolean {
  return /^[A-Z0-9]{7}$/.test(token)
}

export function generateJoinToken(): string {
  let out = ''
  for (let i = 0; i < 7; i++) {
    out += ALPHANUM[Math.floor(Math.random() * ALPHANUM.length)]!
  }
  return out
}

/** Resolve the shared fleet token from runtime config / env. */
export function resolveJoinToken(configured: string): string {
  const normalized = normalizeJoinToken(configured || '')
  if (isValidJoinToken(normalized)) return normalized
  // Fixed fallback when BROKER_TOKEN unset (override in production).
  return 'VTMPREV'
}
