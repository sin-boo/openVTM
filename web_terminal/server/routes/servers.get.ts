import { listServers, storageMode } from '../utils/servers'

export default defineEventHandler(async () => {
  const servers = await listServers()
  return {
    ok: true,
    servers,
    count: servers.length,
    storage: storageMode(),
  }
})
