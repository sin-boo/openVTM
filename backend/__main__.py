"""App entry: local FastAPI + native pywebview window (or API-only / Vite-dev)."""

from __future__ import annotations

import codecs


def _idna_encode(s: str, errors: str = "strict") -> tuple[bytes, int]:
    return s.encode("ascii", errors), len(s)


def _idna_decode(b: bytes, errors: str = "strict") -> tuple[str, int]:
    return b.decode("ascii", errors), len(b)


# PyInstaller often omits encodings.idna from base_library.zip; register a stub.
codecs.register(
    lambda name: codecs.CodecInfo(
        name="idna", encode=_idna_encode, decode=_idna_decode
    )
    if name == "idna"
    else None
)

import argparse
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

# Ensure project root is on sys.path when run as `python -m backend`.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _log_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "sdanime_pose.log"
    return _ROOT / "outputs" / "sdanime_pose.log"


def _file_log(msg: str) -> None:
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _port_open(host: str, port: int) -> bool:
    """Check localhost port without relying on exotic encodings when possible."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.settimeout(0.35)
        target = "127.0.0.1" if host in {"localhost", "127.0.0.1", "::1"} else host
        s.connect((target, int(port)))
        return True
    except OSError:
        return False
    finally:
        try:
            s.close()
        except OSError:
            pass


def _wait_for_server(host: str, port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if _port_open(host, port):
                return True
        except Exception as exc:
            _file_log(f"port check error: {exc}")
        time.sleep(0.25)
    return False


def _run_uvicorn(host: str, port: int, *, mount_ui: bool) -> None:
    try:
        _file_log("importing uvicorn / backend.api…")
        import uvicorn

        from backend.api import app, configure_runtime, mount_frontend

        configure_runtime()
        if mount_ui:
            mount_frontend()
        _file_log(f"uvicorn binding {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="warning", reload=False)
    except Exception:
        _file_log("uvicorn crashed:\n" + traceback.format_exc())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SDAnime Pose desktop app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--ui",
        choices=("webview", "none", "dev"),
        default="webview",
        help="webview=native window, none=API only, dev=open Vite URL in webview",
    )
    parser.add_argument(
        "--vite-url",
        default="http://127.0.0.1:5173",
        help="Vite dev server URL when --ui=dev",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument(
        "--server-mode",
        action="store_true",
        help="Docker/server mode: no webcam; accept JSON tracking frames only",
    )
    args = parser.parse_args(argv)

    if args.server_mode:
        import os

        os.environ["SDANIME_SERVER_MODE"] = "1"

    # Server mode should bind more openly so remote clients can push frames.
    if args.server_mode and args.host in {"127.0.0.1", "localhost"}:
        args.host = "0.0.0.0"

    _file_log(
        f"starting ui={args.ui} frozen={getattr(sys, 'frozen', False)} "
        f"server_mode={bool(args.server_mode)}"
    )

    mount_ui = args.ui == "webview"
    if args.ui == "dev":
        mount_ui = False

    server = threading.Thread(
        target=_run_uvicorn,
        args=(args.host, args.port),
        kwargs={"mount_ui": mount_ui},
        daemon=True,
        name="uvicorn",
    )
    server.start()

    if not _wait_for_server(args.host if args.host != "0.0.0.0" else "127.0.0.1", args.port, timeout=180.0):
        _file_log("ERROR: FastAPI failed to start within 180s — see log above")
        if getattr(sys, "frozen", False):
            input("Press Enter to close…")
        return 1

    bind_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    base = f"http://{bind_host}:{args.port}"
    _file_log(f"API ready at {base}/api/health")

    try:
        from backend.broker_heartbeat import start_broker_heartbeat

        hb = start_broker_heartbeat(local_port=args.port, log=_file_log)
        if hb is None:
            _file_log(
                "broker heartbeat off "
                "(set BROKER_URL, BROKER_SECRET, PUBLIC_URL to register)"
            )
    except Exception:
        _file_log("broker heartbeat failed to start:\n" + traceback.format_exc())

    if args.ui == "none":
        _file_log("API-only mode. Press Ctrl+C to stop.")
        try:
            while server.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        _shutdown_all(exit_code=0)
        return 0

    url = args.vite_url if args.ui == "dev" else base
    try:
        import webview
    except ImportError:
        _file_log(
            "pywebview is not installed. Run: pip install pywebview\n"
            f"Meanwhile open {url} in a browser, or use --ui=none"
        )
        _shutdown_all(exit_code=1)
        return 1

    try:
        _file_log(f"opening webview -> {url}")
        webview.create_window(
            "SDAnime Pose",
            url=url,
            width=args.width,
            height=args.height,
            min_size=(960, 700),
        )
        webview.start()
    except Exception:
        _file_log("webview crashed:\n" + traceback.format_exc())
        _shutdown_all(exit_code=1)
        return 1
    _shutdown_all(exit_code=0)
    return 0


def _shutdown_all(*, exit_code: int = 0) -> None:
    """Unload models + kill face tracker after the UI/API session ends."""
    try:
        from backend.api import shutdown_runtime

        shutdown_runtime()
    except Exception:
        _file_log("shutdown_runtime failed:\n" + traceback.format_exc())
    # Force-exit so daemon uvicorn / CUDA threads cannot keep VRAM pinned.
    try:
        import os

        os._exit(int(exit_code))
    except Exception:
        pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        _file_log("fatal:\n" + traceback.format_exc())
        try:
            from backend.api import shutdown_runtime

            shutdown_runtime()
        except Exception:
            pass
        if getattr(sys, "frozen", False):
            try:
                input("Press Enter to close…")
            except Exception:
                time.sleep(10)
        raise
