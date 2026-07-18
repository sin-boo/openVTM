import { INDEX_HTML } from '../utils/ui-html'

/** Always return the React shell HTML (assets load from /assets/*). */
export default defineEventHandler((event) => {
  setHeader(event, 'Content-Type', 'text/html; charset=utf-8')
  setHeader(event, 'Cache-Control', 'public, max-age=0, must-revalidate')
  return INDEX_HTML
})
