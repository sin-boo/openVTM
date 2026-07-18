import { listServers } from '../utils/servers'

export default defineEventHandler(() => {
  const servers = listServers()
  return {
    ok: true,
    servers,
    count: servers.length,
  }
})
