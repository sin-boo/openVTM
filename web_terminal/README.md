# openVTM Web Terminal

**Nitro** discovery API + **React** status dashboard.

## Auth flow

1. Set the same **7-character** join token on Vercel and every GPU (`BROKER_TOKEN`).
2. GPU finishes loading → `POST /handshake` with `{ token, public_url }`.
3. Broker assigns a **random server name** + **session**.
4. GPU `POST /heartbeat` every **2 seconds** with `{ session }` (no token again).
5. Miss ~8s of heartbeats → server drops offline.

Dashboard shows the join token (copy button) and live servers.

## Local

```bash
npm install
export BROKER_TOKEN=<YOUR_7_CHAR_TOKEN>   # set in env only — never commit
npm run dev
```

- UI: http://localhost:5173  
- API: http://localhost:3000  

## Vercel

Set env **`BROKER_TOKEN`** (7 chars). Build: `npm run build`.

## GPU (Vast / Linux)

```bash
export BROKER_URL=https://webtermial.vercel.app
export BROKER_TOKEN=<YOUR_7_CHAR_TOKEN>  # same as Vercel env — never commit
export PUBLIC_URL=http://IP:MAPPED   # Vast open-port mapping for 8765
./start-server-linux.sh
```
