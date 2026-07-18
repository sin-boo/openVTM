# openVTM Web Terminal (Nitro)

Discovery broker **with a status UI**. GPUs heartbeat in; clients call `/pick` for a `public_url`.  
**Images never pass through this app.**

## Run locally

```bash
cd web_terminal
npm install
export BROKER_SECRET=your-secret   # Windows: set BROKER_SECRET=...
npm run dev
```

Open **http://localhost:3000** — dashboard shows broker health and live servers.

Production:

```bash
npm run build
BROKER_SECRET=your-secret npm start
```

Deploy elsewhere with Nitro presets, e.g. `NITRO_PRESET=vercel npm run build`.

## UI

Browser dashboard at `/`:

- Broker online / offline
- Live server count + ready count
- Current best pick
- Table of registered GPUs (heartbeat age, load, public URL)
- Auto-refresh every 5s

## API (same as before)

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | no | Broker alive |
| POST | `/register` | `Bearer BROKER_SECRET` | GPU heartbeat |
| GET | `/servers` | no | List live servers |
| GET | `/pick` | no | Best server |

Heartbeat TTL: **45s**.

### Register body

```json
{
  "id": "vast-45250875",
  "public_url": "http://137.175.76.24:45323",
  "ready": true,
  "load": 0
}
```

## GPU env (Vast)

Point `BROKER_URL` at this Nitro app (local, Vercel, or your host):

```bash
export BROKER_URL=https://your-web-terminal.example.com
export BROKER_SECRET=your-secret
export PUBLIC_URL=http://137.175.76.24:45323
export SERVER_ID=vast-45250875
./start-server-linux.sh
```

Desktop UI: `VITE_BROKER_URL` = same base URL (calls `/pick` on startup).
