import { getJoinToken } from '../utils/join'

export default defineEventHandler((event) => {
  const join_token = getJoinToken(event)
  return {
    ok: true,
    service: 'openvtm-web-terminal',
    runtime: 'nitro',
    time: new Date().toISOString(),
    join_token,
    /** GPU hosts: POST /handshake with this token, then POST /heartbeat every 2s with session. */
    auth: {
      token_chars: 7,
      handshake: 'POST /handshake',
      heartbeat: 'POST /heartbeat',
      interval_s: 2,
    },
  }
})
