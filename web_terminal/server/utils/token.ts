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

/**
 * Join token comes only from env (BROKER_TOKEN / BROKER_SECRET).
 * Never hardcode a default secret in source.
 */
export function resolveJoinToken(configured: string): string | null {
  const normalized = normalizeJoinToken(configured || '')
  if (isValidJoinToken(normalized)) return normalized
  return null
}
