"""Register this GPU with the web-terminal broker, then heartbeat every 2s.

Env:
  BROKER_URL    — e.g. https://webtermial.vercel.app
  BROKER_TOKEN  — shared 7-char join token (or BROKER_SECRET)
  PUBLIC_URL    — client-facing base, e.g. http://137.175.76.24:45323
  BROKER_INTERVAL_S — heartbeat seconds (default 2)
"""

from __future__ import annotations

import json
import os
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
    token = _env("BROKER_TOKEN") or _env("BROKER_SECRET")
    token = "".join(ch for ch in token.upper() if ch.isalnum())[:7]
    if not broker or not public or len(token) != 7:
        return None
    return {
        "broker_url": broker,
        "public_url": public,
        "token": token,
    }


def _local_status(port: int) -> dict[str, Any]:
    url = f"http://127.0.0.1:{int(port)}/api/status"
    try:
        with urllib.request.urlopen(url, timeout=3.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def _post_json(url: str, payload: dict[str, Any], *, headers: dict[str, str] | None = None) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _snapshot(local_port: int) -> tuple[bool, float]:
    status = _local_status(local_port)
    state = str(status.get("state") or "")
    streaming = bool(status.get("streaming"))
    ready = state in {"ready", "streaming", "generating"}
    load = 1.0 if streaming or state == "generating" else 0.0
    return ready, load


def handshake(cfg: dict[str, str], *, local_port: int) -> dict[str, Any]:
    ready, load = _snapshot(local_port)
    return _post_json(
        f"{cfg['broker_url']}/handshake",
        {
            "token": cfg["token"],
            "public_url": cfg["public_url"],
            "ready": ready,
            "load": load,
        },
    )


def heartbeat(cfg: dict[str, str], session: str, *, local_port: int) -> dict[str, Any]:
    ready, load = _snapshot(local_port)
    return _post_json(
        f"{cfg['broker_url']}/heartbeat",
        {
            "session": session,
            "ready": ready,
            "load": load,
            "public_url": cfg["public_url"],
        },
    )


def start_broker_heartbeat(
    *,
    local_port: int,
    log: LogFn | None = None,
) -> threading.Thread | None:
    """Handshake once with join token, then heartbeat every 2s with session."""
    cfg = broker_config()
    if cfg is None:
        return None

    interval = float(_env("BROKER_INTERVAL_S") or "2")
    interval = max(1.0, min(interval, 10.0))
    _log = log or (lambda _m: None)

    def _loop() -> None:
        session: str | None = None
        server_name = "?"
        _log(
            f"broker → {cfg['broker_url']} public_url={cfg['public_url']} "
            f"token=******* heartbeat={interval:.0f}s"
        )

        while True:
            try:
                if not session:
                    # Wait until models look ready (or 3 min), then handshake.
                    deadline = time.time() + 180.0
                    while time.time() < deadline:
                        ready, _load = _snapshot(local_port)
                        if ready:
                            break
                        _log("broker waiting for models ready before handshake…")
                        time.sleep(3.0)

                    res = handshake(cfg, local_port=local_port)
                    if not res.get("ok") or not res.get("session"):
                        raise RuntimeError(f"handshake rejected: {res}")
                    session = str(res["session"])
                    server = res.get("server") or {}
                    server_name = str(server.get("id") or "?")
                    _log(
                        f"broker handshake ok name={server_name} "
                        f"ready={server.get('ready')}"
                    )
                else:
                    res = heartbeat(cfg, session, local_port=local_port)
                    if not res.get("ok"):
                        _log(f"broker session lost ({res.get('error')}) — re-handshake")
                        session = None
                        continue
                    server = res.get("server") or {}
                    _log(
                        f"broker heartbeat ok name={server.get('id', server_name)} "
                        f"ready={server.get('ready')} load={server.get('load')}"
                    )
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode("utf-8", errors="replace")[:200]
                except Exception:
                    pass
                _log(f"broker HTTP {exc.code}: {body}")
                if exc.code in {401, 403}:
                    session = None
            except Exception:
                _log("broker failed:\n" + traceback.format_exc())
                # Keep trying; clear session on hard failures after handshake
            time.sleep(interval)

    t = threading.Thread(target=_loop, name="broker-heartbeat", daemon=True)
    t.start()
    return t
