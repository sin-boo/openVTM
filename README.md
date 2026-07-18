# openVTM / SDAnime Pose (server)

Linux GPU API for pose-driven anime generation (Anything V5 + IP-Adapter + PoseAdapter).

Models are **not** in this repo. They download from Hugging Face on first start:
[`sinBoo1/models-VT-prototype`](https://huggingface.co/sinBoo1/models-VT-prototype).

## Quick start (Vast / Linux GPU)

```bash
chmod +x start-server-linux.sh
./start-server-linux.sh
```

That script will:

1. Create `.venv` and install deps (CUDA torch when `nvidia-smi` works)
2. Download models into `data/models/`
3. Start the API in **server mode** on `0.0.0.0:8765`

Health check (on the machine):

```bash
curl http://127.0.0.1:8765/api/health
```

On Vast, use the **mapped** public port for 8765 (see instance **Open Ports**, e.g. `PUBLIC_IP:45323 -> 8765`).

Optional: if the HF repo is private, export a token first:

```bash
export HF_TOKEN=hf_...
./start-server-linux.sh
```

## Discovery broker (web terminal)

Images do **not** go through the broker. It only answers “which GPU is online?”

See [`web_terminal/README.md`](web_terminal/README.md). On the GPU machine:

```bash
export BROKER_URL=https://openvtm-web-terminal.<account>.workers.dev
export BROKER_SECRET=your-secret
export PUBLIC_URL=http://137.175.76.24:45323   # Vast mapped URL for port 8765
export SERVER_ID=vast-45250875                 # optional
./start-server-linux.sh
```

Clients:

```bash
curl "$BROKER_URL/pick"
# → { "ok": true, "server": { "public_url": "http://...", ... } }
# then call that public_url for /api/reference and /api/generate
```

Desktop UI: set `VITE_BROKER_URL` to the Worker URL so the app calls `/pick` on startup.

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
