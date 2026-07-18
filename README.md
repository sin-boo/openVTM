# openVTM / SDAnime Pose (server)

Linux GPU API for pose-driven anime generation (Anything V5 + IP-Adapter + PoseAdapter).

Models are **not** in this repo. They download from Hugging Face on first start:
[`sinBoo1/models-VT-prototype`](https://huggingface.co/sinBoo1/models-VT-prototype).

## Quick start (Vast / Linux GPU)

```bash
chmod +x start-server-linux.sh
./start-server-linux.sh
```

That script is **zero-touch** (no prompts):

1. Create `.venv` and install **CUDA torch** (`cu128` on RTX 5090 / Blackwell, else `cu124`+)
2. Install headless server deps (`requirements.server.txt`)
3. Download models from Hugging Face into `data/models/`
4. CUDA smoke-test, then start the API in **server mode** on `0.0.0.0:8765`

Health check (on the machine):

```bash
curl http://127.0.0.1:8765/api/health
```

On Vast, open **port 8765** in the instance template. The script auto-builds `PUBLIC_URL` from `PUBLIC_IPADDR` + `VAST_TCP_PORT_8765` when those env vars exist.

Optional: if the HF repo is private, export a token first:

```bash
export HF_TOKEN=hf_...
./start-server-linux.sh
```

## Discovery broker (web terminal)

Nitro + React dashboard. GPUs **handshake** with a shared 7-char `BROKER_TOKEN`, get a random name + session, then **heartbeat every 2s**.

Broker is **optional**. Pass env vars (or a saved `.broker.env`) — the start script will not prompt:

```bash
export BROKER_URL=https://webtermial.vercel.app
export BROKER_TOKEN=<YOUR_7_CHAR_TOKEN>   # same 7 chars — never commit
# PUBLIC_URL auto-fills on Vast when port 8765 is open; or set manually:
# export PUBLIC_URL=http://137.175.76.24:45323
./start-server-linux.sh
```

Interactive broker setup only if you ask for it: `BROKER_PROMPT=1 ./start-server-linux.sh`

Clients: `GET $BROKER_URL/pick` → use `server.public_url` for generate.  
Desktop: set `VITE_BROKER_URL` to the web terminal URL.

See [`web_terminal/README.md`](web_terminal/README.md).

## API (one port)

- `GET /api/health`
- `GET /api/status`
- `POST /api/load`
- `POST /api/reference` — character image
- `POST /api/generate` — pose → image
- `POST /api/tracking/frame` — JSON pose only (no webcam in server mode)

## Windows (local)

- `start_normal.bat` — API + Vite UI
- `start_server.bat` — API-only server mode (no model download)
