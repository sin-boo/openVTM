"""Heartbeat this GPU API into the discovery broker (Cloudflare Worker).

Env:
  BROKER_URL   — e.g. https://openvtm-web-terminal.example.workers.dev
  BROKER_SECRET — Bearer token matching Worker secret
  PUBLIC_URL   — client-facing base, e.g. http://137.175.76.24:45323
  SERVER_ID    — optional stable id (default: hostname)
  BROKER_INTERVAL_S — seconds between heartbeats (default 15)
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Callable

LogFn = Callable[[str], None]


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def broker_config() -> dict[str, str] | None:
    broker = _env("BROKER_URL").rstrip("/")
    public = _env("PUBLIC_URL").rstrip("/")
    secret = _env("BROKER_SECRET")
    if not broker or not public:
        return None
    if not secret:
        return None
    server_id = _env("SERVER_ID") or socket.gethostname() or "gpu"
    return {
        "broker_url": broker,
        "public_url": public,
        "secret": secret,
        "server_id": server_id,
    }


def _local_status(port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{int(port)}/api/status"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def _post_register(cfg: dict[str, str], payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{cfg['broker_url']}/register",
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {cfg['secret']}",
        },
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        resp.read()


def heartbeat_once(cfg: dict[str, str], *, local_port: int) -> dict[str, Any]:
    status = _local_status(local_port)
    state = str(status.get("state") or "")
    streaming = bool(status.get("streaming"))
    ready = state in {"ready", "streaming", "generating"}
    load = 1.0 if streaming or state == "generating" else 0.0
    payload = {
        "id": cfg["server_id"],
        "public_url": cfg["public_url"],
        "ready": ready,
        "load": load,
        "vram_free": None,
    }
    _post_register(cfg, payload)
    return payload


def start_broker_heartbeat(
    *,
    local_port: int,
    log: LogFn | None = None,
) -> threading.Thread | None:
    """Start daemon thread. Returns None if broker env is not configured."""
    cfg = broker_config()
    if cfg is None:
        return None

    interval = float(_env("BROKER_INTERVAL_S") or "15")
    interval = max(5.0, interval)
    _log = log or (lambda _m: None)

    def _loop() -> None:
        _log(
            f"broker heartbeat → {cfg['broker_url']} as {cfg['server_id']} "
            f"public_url={cfg['public_url']} every {interval:.0f}s"
        )
        while True:
            try:
                payload = heartbeat_once(cfg, local_port=local_port)
                _log(
                    f"broker register ok ready={payload['ready']} load={payload['load']}"
                )
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                _log(f"broker register HTTP {exc.code}: {body}")
            except Exception:
                _log("broker register failed:\n" + traceback.format_exc())
            time.sleep(interval)

    t = threading.Thread(target=_loop, name="broker-heartbeat", daemon=True)
    t.start()
    return t
