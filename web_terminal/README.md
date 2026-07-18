# openVTM web terminal (discovery broker)

Cloudflare Worker that tracks which GPU API servers are online.
**Does not proxy images** — clients call the returned `public_url` directly.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | no | Broker alive |
| POST | `/register` | `Bearer BROKER_SECRET` | GPU heartbeat |
| GET | `/servers` | no | List live servers |
| GET | `/pick` | no | Best server (`ready`, then lowest `load`) |

Heartbeat TTL: **45s**. Miss heartbeats → server disappears from `/pick`.

### Register body

```json
{
  "id": "vast-45250875",
  "public_url": "http://137.175.76.24:45323",
  "ready": true,
  "load": 0,
  "vram_free": null
}
```

### Pick response

```json
{
  "ok": true,
  "server": {
    "id": "vast-45250875",
    "public_url": "http://137.175.76.24:45323",
    "ready": true,
    "load": 0,
    "updated_at": 1710000000000
  }
}
```

## Deploy

```bash
cd web_terminal
npm install
npx wrangler login
npx wrangler kv namespace create SERVERS
# paste id into wrangler.toml (id + preview_id)
npx wrangler secret put BROKER_SECRET
npm run deploy
```

## GPU env (on Vast)

```bash
export BROKER_URL=https://openvtm-web-terminal.<account>.workers.dev
export BROKER_SECRET=your-secret
export PUBLIC_URL=http://137.175.76.24:45323   # Vast mapped port for 8765
export SERVER_ID=vast-45250875                 # optional
./start-server-linux.sh
```
