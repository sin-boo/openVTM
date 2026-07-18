import { getJoinToken } from '../utils/join'

export default defineEventHandler((event) => {
  const join_token = getJoinToken(event)
  return {
    ok: true,
    service: 'openvtm-web-terminal',
    runtime: 'nitro',
    time: new Date().toISOString(),
    /** Present only when BROKER_TOKEN is set in the environment — never from source. */
    join_token: join_token || null,
    token_configured: Boolean(join_token),
    auth: {
      token_chars: 7,
      handshake: 'POST /handshake',
      heartbeat: 'POST /heartbeat',
      interval_s: 2,
    },
  }
})
