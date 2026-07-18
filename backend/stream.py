"""Continuous pose→image stream (real_stream-style as-fast-as-GPU loop).

Each frame is a fresh denoise from a fixed seed + cached IP embeds.
No latent carry-over — temporal coherence comes from fixed seed/ref + live pose.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from PIL import Image

from backend.engine import DEFAULT_NEGATIVE, IP_ADAPTER_WEIGHTS, PoseEngine
from backend.utils.params import PARAM_NAMES

LogFn = Callable[[str], None]


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


@dataclass
class StreamSettings:
    prompt: str = ""
    negative: str = DEFAULT_NEGATIVE
    steps: int = 8
    cfg: float = 4.5
    seed: int = 42
    ip_adapter: str = "plus (closer to image)"
    ip_scale: float = 0.85
    pose_drive: str = "prompt tags (visible motion)"
    pose_target_rms: float = 0.12
    norm_mode: str = "slider_to_unit"
    jpeg_quality: int = 85


@dataclass
class StreamController:
    """One GPU worker; coalesce latest pose; push JPEG frames."""

    engine: PoseEngine
    get_ref: Callable[[], Image.Image | None]
    get_track_pose: Callable[[], dict[str, float] | None]
    log: LogFn = field(default=lambda _m: None)

    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _streaming: bool = field(default=False, init=False)
    _wake: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _worker: threading.Thread | None = field(default=None, init=False, repr=False)
    _settings: StreamSettings = field(default_factory=StreamSettings, init=False)
    _pose: list[float] | None = field(default=None, init=False)
    _frame_q: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    _frame_q_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _sequence: int = field(default=0, init=False)
    _fps_ema: float = field(default=0.0, init=False)

    @property
    def streaming(self) -> bool:
        return self._streaming

    def start(self, settings: StreamSettings, pose: dict[str, float] | None = None) -> None:
        with self._lock:
            if self._streaming:
                raise RuntimeError("Already streaming")
            if self.engine.pipe is None or self.engine.pose_adapter is None:
                raise RuntimeError("Models not loaded")
            if self.get_ref() is None:
                raise RuntimeError("Upload a reference image first")
            weight = IP_ADAPTER_WEIGHTS.get(settings.ip_adapter)
            if weight is None:
                raise RuntimeError(f"Unknown IP adapter: {settings.ip_adapter}")

            self._settings = settings
            if pose:
                self._pose = _pose_from_dict(pose)
            self._sequence = 0
            self._fps_ema = 0.0
            self._streaming = True
            self._wake.set()

            # Apply adapter/settings once up front.
            self.log(
                f"Stream starting — steps={settings.steps} cfg={settings.cfg} "
                f"seed={settings.seed} drive={settings.pose_drive}"
            )
            self.engine.set_ip_adapter(weight, scale=float(settings.ip_scale))
            self.engine.steps = int(settings.steps)
            self.engine.cfg = float(settings.cfg)
            self.engine.pose_drive = settings.pose_drive
            self.engine.pose_target_rms = float(settings.pose_target_rms)

            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._loop, name="sdanime-stream", daemon=True
                )
                self._worker.start()
        self.log("Stream started")

    def stop(self) -> None:
        with self._lock:
            if not self._streaming:
                return
            self._streaming = False
            self._wake.set()
        self.log("Stream stopped")

    def update_pose(self, pose: dict[str, float]) -> None:
        self._pose = _pose_from_dict(pose)

    def pop_frame(self) -> dict[str, Any] | None:
        with self._frame_q_lock:
            if not self._frame_q:
                return None
            # Keep only the newest frame if UI lags.
            frame = self._frame_q[-1]
            self._frame_q.clear()
            return frame

    def _resolve_pose(self) -> list[float] | None:
        tracked = self.get_track_pose()
        if tracked:
            return _pose_from_dict(tracked)
        return self._pose

    def _push_frame(self, payload: dict[str, Any]) -> None:
        with self._frame_q_lock:
            self._frame_q.append(payload)
            # Cap backlog — stream is latest-wins.
            if len(self._frame_q) > 2:
                del self._frame_q[:-1]

    def _loop(self) -> None:
        while True:
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            if not self._streaming:
                continue

            while self._streaming:
                pose = self._resolve_pose()
                ref = self.get_ref()
                if pose is None or ref is None:
                    time.sleep(0.05)
                    continue

                settings = self._settings
                try:
                    result = self.engine.generate(
                        pose=pose,
                        ref=ref,  # same PIL → IP cache hits via content hash
                        prompt=settings.prompt,
                        negative=settings.negative or DEFAULT_NEGATIVE,
                        seed=int(settings.seed),
                        steps=int(settings.steps),
                        cfg=float(settings.cfg),
                        pose_drive=settings.pose_drive,
                        pose_target_rms=float(settings.pose_target_rms),
                        norm_mode=settings.norm_mode,
                    )
                except Exception as exc:
                    self._push_frame({"type": "error", "message": str(exc)})
                    self._streaming = False
                    break

                self._sequence += 1
                fps = 1.0 / result.elapsed if result.elapsed > 0 else 0.0
                if self._fps_ema <= 0:
                    self._fps_ema = fps
                else:
                    self._fps_ema = 0.85 * self._fps_ema + 0.15 * fps

                buf = io.BytesIO()
                result.image.save(buf, format="JPEG", quality=int(settings.jpeg_quality))
                b64 = base64.b64encode(buf.getvalue()).decode("ascii")
                self._push_frame(
                    {
                        "type": "frame",
                        "sequence": self._sequence,
                        "elapsed": result.elapsed,
                        "fps": fps,
                        "fps_ema": self._fps_ema,
                        "seed": result.seed,
                        "pose_tags": result.pose_tags,
                        "image": f"data:image/jpeg;base64,{b64}",
                    }
                )
                # Immediately schedule next frame while still streaming (GPU-paced).
