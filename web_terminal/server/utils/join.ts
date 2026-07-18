import {
  isValidJoinToken,
  normalizeJoinToken,
  resolveJoinToken,
} from '../utils/token'

let cachedToken: string | null = null

export function getJoinToken(event: Parameters<typeof useRuntimeConfig>[0]): string {
  if (cachedToken && isValidJoinToken(cachedToken)) return cachedToken
  const config = useRuntimeConfig(event)
  const fromEnv = normalizeJoinToken(
    String(config.brokerToken || config.brokerSecret || ''),
  )
  cachedToken = resolveJoinToken(fromEnv)
  return cachedToken
}

export function assertJoinToken(
  event: Parameters<typeof useRuntimeConfig>[0],
  provided: string,
): boolean {
  const expected = getJoinToken(event)
  const got = normalizeJoinToken(provided)
  return isValidJoinToken(got) && got === expected
}
