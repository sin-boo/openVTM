import {
  isValidJoinToken,
  normalizeJoinToken,
  resolveJoinToken,
} from '../utils/token'

let cachedToken: string | null | undefined

export function getJoinToken(
  event: Parameters<typeof useRuntimeConfig>[0],
): string | null {
  if (cachedToken !== undefined) return cachedToken
  const config = useRuntimeConfig(event)
  const fromEnv = String(config.brokerToken || config.brokerSecret || '')
  cachedToken = resolveJoinToken(fromEnv)
  return cachedToken
}

export function assertJoinToken(
  event: Parameters<typeof useRuntimeConfig>[0],
  provided: string,
): boolean {
  const expected = getJoinToken(event)
  if (!expected) return false
  const got = normalizeJoinToken(provided)
  return isValidJoinToken(got) && got === expected
}
