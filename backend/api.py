"""FastAPI surface for the SDAnime Pose desktop app."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from backend.engine import (
    DEFAULT_CFG,
    DEFAULT_NEGATIVE,
    DEFAULT_STEPS,
    IP_ADAPTER_WEIGHTS,
    POSE_DRIVE_MODES,
    PoseEngine,
    best_pose_checkpoint_label,
    list_pose_checkpoints,
)
from backend.paths import default_ref_path, outputs_dir, ui_dist_dir
from backend.stream import StreamController, StreamSettings
from backend.tracking import get_tracking_service, set_server_mode
from backend.tracking import openseeface as osf
from backend.utils.params import PARAM_NAMES

HOST = "127.0.0.1"
PORT = 8765

_SERVER_MODE = os.environ.get("SDANIME_SERVER_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

app = FastAPI(title="SDAnime Pose", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_engine: PoseEngine | None = None
_stream: StreamController | None = None
_ref_image: Image.Image | None = None
_ref_name: str = ""
_status: dict[str, Any] = {
    "state": "idle",
    "message": "Not loaded",
    "checkpoint": "",
    "device": "",
    "error": "",
}
_lock = threading.Lock()
_logs: list[str] = []


def _log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    _logs.append(line)
    if len(_logs) > 800:
        del _logs[:200]
    # Surface live load stages in the header status line.
    if _status.get("state") == "loading":
        _set_status(message=msg)


def configure_runtime(*, server_mode: bool | None = None) -> None:
    """Apply local/server mode before (or while) serving."""
    global _SERVER_MODE
    if server_mode is None:
        server_mode = os.environ.get("SDANIME_SERVER_MODE", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    _SERVER_MODE = bool(server_mode)
    os.environ["SDANIME_SERVER_MODE"] = "1" if _SERVER_MODE else "0"
    set_server_mode(_SERVER_MODE)
    _log(f"runtime mode={'server' if _SERVER_MODE else 'local'}")


def get_engine() -> PoseEngine:
    global _engine
    if _engine is None:
        _engine = PoseEngine(log=_log)
    return _engine


def _track_pose() -> dict[str, float] | None:
    snap = get_tracking_service().snapshot_dict()
    if not snap.get("has_face"):
        return None
    pose = snap.get("pose")
    return pose if isinstance(pose, dict) and pose else None


def get_stream() -> StreamController:
    global _stream
    if _stream is None:
        _stream = StreamController(
            engine=get_engine(),
            get_ref=lambda: _ref_image,
            get_track_pose=_track_pose,
            log=_log,
        )
    return _stream


class LoadRequest(BaseModel):
    checkpoint: str | None = None


class CheckpointRequest(BaseModel):
    label: str


class GenerateRequest(BaseModel):
    pose: dict[str, float] = Field(default_factory=dict)
    prompt: str = ""
    negative: str = DEFAULT_NEGATIVE
    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
    seed: int = 42
    ip_adapter: str = "plus (closer to image)"
    ip_scale: float = 0.85
    pose_drive: str = POSE_DRIVE_MODES[0]
    pose_target_rms: float = 0.12
    norm_mode: str = "slider_to_unit"


class TrackingStartRequest(BaseModel):
    camera_index: int | None = None
    mirror: bool = False


class TrackingMirrorRequest(BaseModel):
    mirror: bool = False


class TrackingFrameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pose: dict[str, float] = Field(default_factory=dict)
    landmarks: list[list[float]] | None = None
    confidences: list[float] | None = None
    sequence: int | None = None
    face_id: int | None = None
    got_3d: bool = True
    fit_error: float = 0.0
    timestamp: float | None = None


_FORBIDDEN_FRAME_KEYS = frozenset(
    {
        "image",
        "frame",
        "pixels",
        "bitmap",
        "jpeg",
        "png",
        "base64",
        "b64",
        "webcam",
        "video",
        "rgb",
        "rgba",
        "data_url",
        "dataurl",
    }
)


def _pose_from_dict(data: dict[str, float]) -> list[float]:
    values = {name: 0.0 for name in PARAM_NAMES}
    for name in ("ParamEyeLOpen", "ParamEyeROpen", "EyeOpenLeft", "EyeOpenRight"):
        values[name] = 1.0
    for k, v in data.items():
        if k in values:
            values[k] = float(v)
    values["FaceAngleX"] = values["ParamAngleX"]
    values["FaceAngleY"] = values["ParamAngleY"]
    values["FaceAngleZ"] = values["ParamAngleZ"]
    values["MouthOpen"] = values["ParamMouthOpenY"]
    values["EyeOpenLeft"] = values["ParamEyeLOpen"]
    values["EyeOpenRight"] = values["ParamEyeROpen"]
    return [values[n] for n in PARAM_NAMES]


def _set_status(**kwargs: Any) -> None:
    _status.update(kwargs)


def shutdown_runtime() -> None:
    """Stop stream/tracking and free GPU models. Safe to call more than once."""
    global _engine, _stream, _ref_image, _ref_name
    _log("Shutting down runtime…")
    try:
        if _stream is not None:
            _stream.stop()
    except Exception as exc:
        _log(f"stream stop: {exc}")
    try:
        get_tracking_service().stop()
    except Exception as exc:
        _log(f"tracking stop: {exc}")
    try:
        osf._kill_orphaned_facetrackers()
    except Exception as exc:
        _log(f"facetracker cleanup: {exc}")
    try:
        if _engine is not None:
            _engine.unload()
    except Exception as exc:
        _log(f"engine unload: {exc}")
    _engine = None
    _stream = None
    _ref_image = None
    _ref_name = ""
    _set_status(state="idle", message="Unloaded", device="", error="")
    _log("Runtime shutdown complete")


@app.on_event("startup")
def _on_startup() -> None:
    configure_runtime(server_mode=_SERVER_MODE)
    _log("Startup: auto-loading models in background…")
    # Fire-and-forget — UI only needs Start/Stop stream once ready.
    load_models(LoadRequest())


@app.on_event("shutdown")
def _on_shutdown() -> None:
    shutdown_runtime()


@app.post("/api/unload")
def unload_models() -> dict[str, Any]:
    with _lock:
        shutdown_runtime()
    return {"ok": True}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"ok": "true", "mode": "server" if _SERVER_MODE else "local"}


@app.get("/api/status")
def status() -> dict[str, Any]:
    track = get_tracking_service().snapshot_dict()
    return {
        **_status,
        "mode": "server" if _SERVER_MODE else "local",
        "server_mode": _SERVER_MODE,
        "param_names": list(PARAM_NAMES),
        "ip_adapters": list(IP_ADAPTER_WEIGHTS.keys()),
        "pose_drives": list(POSE_DRIVE_MODES),
        "defaults": {
            "prompt": "",
            "negative": DEFAULT_NEGATIVE,
            "steps": DEFAULT_STEPS,
            "cfg": DEFAULT_CFG,
            "checkpoint": best_pose_checkpoint_label(),
        },
        "has_reference": _ref_image is not None,
        "reference_name": _ref_name,
        "streaming": get_stream().streaming,
        "logs": _logs[-200:],
        "tracking": {
            "mode": track["mode"],
            "active": track["active"],
            "message": track["message"],
            "calibration_ready": track["calibration_ready"],
            "calibration_progress": track["calibration_progress"],
        },
    }


@app.get("/api/checkpoints")
def checkpoints() -> dict[str, Any]:
    items = [{"label": label, "name": path.name} for label, path in list_pose_checkpoints()]
    return {"checkpoints": items, "default": best_pose_checkpoint_label()}


@app.post("/api/load")
def load_models(body: LoadRequest = LoadRequest()) -> dict[str, Any]:
    with _lock:
        # Idempotent while in-flight: React Strict Mode double-mount must not 409.
        if _status["state"] == "loading":
            _log("Load already in progress — ignoring duplicate request")
            return {"ok": True, "already": "loading"}
        if _status["state"] in {"ready", "streaming"} and get_engine().pipe is not None:
            _log("Models already loaded — skipping full reload")
            return {"ok": True, "already": "ready"}
        _set_status(state="loading", message="Loading models…", error="")
        _log("Load queued: preparing Stable Diffusion + IP-Adapter + PoseAdapter…")

    def work() -> None:
        t0 = time.perf_counter()
        try:
            eng = get_engine()
            ckpt_map = {label: path for label, path in list_pose_checkpoints()}
            path = None
            label = body.checkpoint or best_pose_checkpoint_label()
            if body.checkpoint and body.checkpoint in ckpt_map:
                path = ckpt_map[body.checkpoint]
            _log(f"Load start — checkpoint={label or '(best)'} device={eng.device}")
            step = eng.load(path)
            global _ref_image, _ref_name
            if _ref_image is None:
                ref = default_ref_path()
                if ref.is_file():
                    _log(f"Loading default reference: {ref.name}")
                    _ref_image = Image.open(ref).convert("RGB")
                    _ref_name = ref.name
                else:
                    _log("No default reference found — upload one before streaming")
            elapsed = time.perf_counter() - t0
            _log(f"Load finished in {elapsed:.1f}s — ready to stream")
            _set_status(
                state="ready",
                message=f"Ready — {label} (step {step})",
                checkpoint=label,
                device=eng.device,
                error="",
            )
        except Exception as exc:
            _log(f"Load FAILED: {exc}")
            _set_status(
                state="error",
                message=f"Load failed: {exc}",
                error=traceback.format_exc(),
            )

    threading.Thread(target=work, daemon=True, name="model-load").start()
    return {"started": True}


@app.post("/api/checkpoint")
def switch_checkpoint(body: CheckpointRequest) -> dict[str, Any]:
    eng = get_engine()
    if eng.pipe is None:
        raise HTTPException(400, "Models still loading — try again when ready")
    ckpt_map = {label: path for label, path in list_pose_checkpoints()}
    path = ckpt_map.get(body.label)
    if path is None:
        raise HTTPException(404, f"Unknown checkpoint: {body.label}")
    with _lock:
        _log(f"Switching pose checkpoint → {body.label} ({path.name})")
        t0 = time.perf_counter()
        step = eng.load_pose_adapter(path)
        _log(f"Checkpoint loaded in {time.perf_counter() - t0:.2f}s (step {step})")
        _set_status(
            state="ready",
            message=f"Ready — {body.label} (step {step})",
            checkpoint=body.label,
        )
    return {"ok": True, "step": step, "label": body.label}


@app.post("/api/reference")
async def upload_reference(file: UploadFile = File(...)) -> dict[str, Any]:
    global _ref_image, _ref_name
    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(400, f"Bad image: {exc}") from exc
    _ref_image = image
    _ref_name = file.filename or "reference.png"
    thumb = image.copy()
    thumb.thumbnail((160, 160))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"ok": True, "name": _ref_name, "preview": f"data:image/png;base64,{b64}"}


@app.post("/api/reference/default")
def load_default_reference() -> dict[str, Any]:
    global _ref_image, _ref_name
    path = default_ref_path()
    if not path.is_file():
        raise HTTPException(404, f"Missing default ref: {path}")
    _ref_image = Image.open(path).convert("RGB")
    _ref_name = path.name
    thumb = _ref_image.copy()
    thumb.thumbnail((160, 160))
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"ok": True, "name": _ref_name, "preview": f"data:image/png;base64,{b64}"}


@app.post("/api/generate")
def generate(body: GenerateRequest) -> dict[str, Any]:
    eng = get_engine()
    if eng.pipe is None or eng.pose_adapter is None:
        raise HTTPException(400, "Models not loaded")
    if _ref_image is None:
        raise HTTPException(400, "Upload a reference image first")
    if get_stream().streaming:
        raise HTTPException(409, "Streaming — stop stream before one-shot generate")

    weight = IP_ADAPTER_WEIGHTS.get(body.ip_adapter)
    if weight is None:
        raise HTTPException(400, f"Unknown IP adapter: {body.ip_adapter}")

    with _lock:
        if _status["state"] == "generating":
            raise HTTPException(409, "Already generating")
        _set_status(state="generating", message="Generating…")

    try:
        eng.set_ip_adapter(weight, scale=float(body.ip_scale))
        eng.steps = int(body.steps)
        eng.cfg = float(body.cfg)
        eng.pose_drive = body.pose_drive
        eng.pose_target_rms = float(body.pose_target_rms)
        pose = _pose_from_dict(body.pose)
        result = eng.generate(
            pose=pose,
            ref=_ref_image.copy(),
            prompt=body.prompt,
            negative=body.negative or DEFAULT_NEGATIVE,
            seed=int(body.seed),
            steps=int(body.steps),
            cfg=float(body.cfg),
            pose_drive=body.pose_drive,
            pose_target_rms=float(body.pose_target_rms),
            norm_mode=body.norm_mode,
        )
        out_dir = outputs_dir()
        name = f"pose_{int(time.time())}_{result.seed}.png"
        out_path = out_dir / name
        result.image.save(out_path)
        buf = io.BytesIO()
        result.image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        _set_status(
            state="ready",
            message=f"Done {result.elapsed:.2f}s ({result.timings.get('fps', 0):.2f} FPS)",
        )
        return {
            "ok": True,
            "elapsed": result.elapsed,
            "fps": result.timings.get("fps", 0),
            "seed": result.seed,
            "pose_tags": result.pose_tags,
            "prompt": result.prompt,
            "filename": name,
            "image": f"data:image/png;base64,{b64}",
            "url": f"/api/outputs/{name}",
        }
    except Exception as exc:
        _set_status(state="error", message=f"Generate failed: {exc}", error=traceback.format_exc())
        raise HTTPException(500, str(exc)) from exc


@app.websocket("/api/generate/ws")
async def generate_ws(ws: WebSocket) -> None:
    """Live pose→image stream. Client: start | pose | stop. Server: frame | status | error."""
    await ws.accept()
    ctrl = get_stream()

    async def pump_frames() -> None:
        while True:
            frame = ctrl.pop_frame()
            if frame is None:
                await asyncio.sleep(0.01)
                if not ctrl.streaming:
                    # Drain once more then exit pump when stopped and empty.
                    frame = ctrl.pop_frame()
                    if frame is None:
                        return
                    await ws.send_json(frame)
                continue
            await ws.send_json(frame)

    pump_task: asyncio.Task[None] | None = None
    try:
        await ws.send_json({"type": "status", "streaming": False, "message": "connected"})
        while True:
            raw = await ws.receive_json()
            if not isinstance(raw, dict):
                await ws.send_json({"type": "error", "message": "JSON object required"})
                continue
            action = str(raw.get("type") or raw.get("action") or "").lower()

            if action == "pose":
                pose = raw.get("pose")
                if isinstance(pose, dict):
                    ctrl.update_pose(pose)
                continue

            if action == "stop":
                ctrl.stop()
                if pump_task is not None:
                    pump_task.cancel()
                    pump_task = None
                _set_status(state="ready", message="Stream stopped")
                await ws.send_json({"type": "status", "streaming": False, "message": "stopped"})
                continue

            if action == "start":
                if ctrl.streaming:
                    await ws.send_json({"type": "error", "message": "Already streaming"})
                    continue
                with _lock:
                    if _status["state"] == "generating":
                        await ws.send_json({"type": "error", "message": "One-shot generate in progress"})
                        continue
                    _set_status(state="streaming", message="Streaming…")
                try:
                    settings = StreamSettings(
                        prompt=str(raw.get("prompt") or ""),
                        negative=str(raw.get("negative") or DEFAULT_NEGATIVE),
                        steps=int(raw.get("steps") or DEFAULT_STEPS),
                        cfg=float(raw.get("cfg") or DEFAULT_CFG),
                        seed=int(raw.get("seed") if raw.get("seed") is not None else 42),
                        ip_adapter=str(raw.get("ip_adapter") or "plus (closer to image)"),
                        ip_scale=float(raw.get("ip_scale") if raw.get("ip_scale") is not None else 0.85),
                        pose_drive=str(raw.get("pose_drive") or POSE_DRIVE_MODES[0]),
                        pose_target_rms=float(
                            raw.get("pose_target_rms")
                            if raw.get("pose_target_rms") is not None
                            else 0.12
                        ),
                        norm_mode=str(raw.get("norm_mode") or "slider_to_unit"),
                        jpeg_quality=int(raw.get("jpeg_quality") or 85),
                    )
                    pose = raw.get("pose") if isinstance(raw.get("pose"), dict) else None
                    ctrl.start(settings, pose=pose)
                except Exception as exc:
                    _set_status(state="ready", message=f"Stream failed: {exc}")
                    await ws.send_json({"type": "error", "message": str(exc)})
                    continue
                await ws.send_json({"type": "status", "streaming": True, "message": "streaming"})
                if pump_task is None or pump_task.done():
                    pump_task = asyncio.create_task(pump_frames())
                continue

            await ws.send_json({"type": "error", "message": f"Unknown action: {action}"})
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        if ctrl.streaming:
            ctrl.stop()
            _set_status(state="ready", message="Stream stopped")
        if pump_task is not None:
            pump_task.cancel()


@app.get("/api/outputs/{name}")
def get_output(name: str) -> FileResponse:
    safe = Path(name).name
    path = outputs_dir() / safe
    if not path.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/tracking/cameras")
def tracking_cameras() -> dict[str, Any]:
    svc = get_tracking_service()
    if svc.server_mode:
        raise HTTPException(400, "Camera list unavailable in server mode")
    cams = svc.list_cameras()
    # Prefer a non-virtual device when present; virtual cams stay in the list.
    pairs = [(int(c["index"]), str(c["name"])) for c in cams]
    default_index = osf.pick_default_camera(pairs) if pairs else 0
    return {"cameras": cams, "default_index": default_index, "mode": "local"}


@app.post("/api/tracking/start")
def tracking_start(body: TrackingStartRequest = TrackingStartRequest()) -> dict[str, Any]:
    svc = get_tracking_service()
    if svc.server_mode:
        raise HTTPException(400, "Camera tracking disabled in server mode")
    try:
        return svc.start(body.camera_index, mirror=body.mirror)
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.post("/api/tracking/stop")
def tracking_stop() -> dict[str, Any]:
    return get_tracking_service().stop()


@app.post("/api/tracking/calibrate")
def tracking_calibrate() -> dict[str, Any]:
    svc = get_tracking_service()
    if svc.server_mode:
        raise HTTPException(400, "Calibration is local-mode only")
    return svc.calibrate()


@app.post("/api/tracking/mirror")
def tracking_mirror(body: TrackingMirrorRequest) -> dict[str, Any]:
    return get_tracking_service().set_mirror(body.mirror)


@app.get("/api/tracking/status")
def tracking_status() -> dict[str, Any]:
    return get_tracking_service().snapshot_dict()


@app.post("/api/tracking/frame")
async def tracking_frame(request: Request) -> dict[str, Any]:
    svc = get_tracking_service()
    if not svc.server_mode:
        raise HTTPException(400, "External frames only accepted in server mode")
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(400, f"Invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, "JSON object required")
    bad = {str(k).lower() for k in payload.keys()} & _FORBIDDEN_FRAME_KEYS
    if bad:
        raise HTTPException(400, f"Image/frame payloads are forbidden: {sorted(bad)}")
    try:
        body = TrackingFrameRequest.model_validate(payload)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from exc
    try:
        return svc.ingest_frame(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(500, str(exc)) from exc


@app.websocket("/api/tracking/ws")
async def tracking_ws(ws: WebSocket) -> None:
    await ws.accept()
    svc = get_tracking_service()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=8)
    loop = asyncio.get_running_loop()

    def on_snap(_snap: Any) -> None:
        data = svc.snapshot_dict()
        try:
            loop.call_soon_threadsafe(queue.put_nowait, data)
        except Exception:
            pass

    svc.add_listener(on_snap)
    try:
        await ws.send_json(svc.snapshot_dict())
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=1.0)
                await ws.send_json(data)
            except asyncio.TimeoutError:
                await ws.send_json(svc.snapshot_dict())
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
    finally:
        svc.remove_listener(on_snap)


def mount_frontend() -> None:
    dist = ui_dist_dir()
    if dist.is_dir() and (dist / "index.html").is_file():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="ui")
        _log(f"Serving UI from {dist}")
    else:
        _log(f"No UI dist at {dist} (dev mode — use Vite)")


def create_app(*, mount_ui: bool = True) -> FastAPI:
    if mount_ui:
        mount_frontend()
    return app
