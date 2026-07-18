export default defineEventHandler(() => {
  return {
    ok: true,
    service: 'openvtm-web-terminal',
    runtime: 'nitro',
    time: new Date().toISOString(),
  }
})
