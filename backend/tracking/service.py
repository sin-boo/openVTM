"""Mode-aware tracking service: local OpenSeeFace or server JSON frames."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from backend.tracking import openseeface as osf
from backend.utils.params import NUM_PARAMS, PARAM_NAMES, params_from_mapping

FACE_MAX_AGE_S = 0.75


@dataclass
class TrackingSnapshot:
    active: bool
    mode: str
    tracking: bool
    has_face: bool
    calibrating: bool
    calibration_progress: float
    calibration_ready: bool
    camera_index: int | None
    mirror: bool
    packets_received: int
    age_seconds: float
    face_id: int | None
    got_3d: bool
    fit_error: float
    pose: dict[str, float]
    landmarks: list[list[float]]
    confidences: list[float]
    message: str = ""
    sequence: int = 0
    timestamp: float = 0.0


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


class TrackingService:
    def __init__(self, *, server_mode: bool = False) -> None:
        self.server_mode = bool(server_mode)
        self._lock = threading.RLock()
        self._mirror = False
        self._tracking = False
        self._camera_index: int | None = None
        self._sequence = 0
        self._pose: dict[str, float] = {n: 0.0 for n in PARAM_NAMES}
        for n in ("ParamEyeLOpen", "ParamEyeROpen", "EyeOpenLeft", "EyeOpenRight"):
            self._pose[n] = 1.0
        self._landmarks: list[list[float]] = []
        self._confidences: list[float] = []
        self._has_face = False
        self._face_id: int | None = None
        self._got_3d = False
        self._fit_error = 0.0
        self._last_update = 0.0
        self._message = "Tracking: off"
        self._calibrator = osf.HeadAngleCalibrator()
        self._smoother = osf.SmoothedParams(alpha=0.35, angle_alpha=0.75)
        self._receiver: osf.OpenSeeFaceReceiver | None = None
        self._tracker: osf.OpenSeeFaceTrackerProcess | None = None
        self._poll_thread: threading.Thread | None = None
        self._stop_poll = threading.Event()
        self._listeners: list[Callable[[TrackingSnapshot], None]] = []

    def set_server_mode(self, enabled: bool) -> None:
        with self._lock:
            if bool(enabled) == self.server_mode:
                return
            if self._tracking:
                self.stop()
            self.server_mode = bool(enabled)
            self._message = "Tracking: server mode" if self.server_mode else "Tracking: off"

    def add_listener(self, cb: Callable[[TrackingSnapshot], None]) -> None:
        with self._lock:
            self._listeners.append(cb)

    def remove_listener(self, cb: Callable[[TrackingSnapshot], None]) -> None:
        with self._lock:
            if cb in self._listeners:
                self._listeners.remove(cb)

    def _emit(self) -> None:
        snap = self.snapshot()
        listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(snap)
            except Exception:
                pass

    def list_cameras(self) -> list[dict[str, Any]]:
        if self.server_mode:
            return []
        cams = osf.list_cameras()
        return [{"index": i, "name": n} for i, n in cams]

    def start(self, camera_index: int | None = None, *, mirror: bool = False) -> dict[str, Any]:
        if self.server_mode:
            raise RuntimeError("Camera tracking is disabled in server mode")
        with self._lock:
            self._mirror = bool(mirror)
            cams = osf.list_cameras()
            if camera_index is None:
                camera_index = osf.pick_default_camera(cams)
            self.stop_unlocked()
            self._receiver = osf.OpenSeeFaceReceiver()
            self._tracker = osf.OpenSeeFaceTrackerProcess()
            self._receiver.start()
            try:
                self._tracker.start(int(camera_index))
            except Exception:
                self._receiver.stop()
                self._receiver = None
                self._tracker = None
                raise
            self._calibrator.reset()
            self._smoother.reset()
            self._tracking = True
            self._camera_index = int(camera_index)
            self._message = f"Tracking: starting camera {camera_index}"
            self._stop_poll.clear()
            self._poll_thread = threading.Thread(
                target=self._poll_loop, name="track-poll", daemon=True
            )
            self._poll_thread.start()
            return {"ok": True, "camera_index": self._camera_index}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self.stop_unlocked()
            self._message = "Tracking: off"
            self._emit()
            return {"ok": True}

    def stop_unlocked(self) -> None:
        self._stop_poll.set()
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        self._poll_thread = None
        if self._tracker is not None:
            self._tracker.stop()
            self._tracker = None
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None
        self._tracking = False
        self._camera_index = None
        self._has_face = False
        self._landmarks = []
        self._confidences = []

    def calibrate(self) -> dict[str, Any]:
        with self._lock:
            self._calibrator.reset()
            self._smoother.reset()
            self._message = "Tracking: calibrating… look straight ahead"
            self._emit()
            return {"ok": True, "progress": self._calibrator.progress}

    def set_mirror(self, mirror: bool) -> dict[str, Any]:
        with self._lock:
            self._mirror = bool(mirror)
            return {"ok": True, "mirror": self._mirror}

    def ingest_frame(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.server_mode:
            raise RuntimeError("External frame ingestion is only available in server mode")
        # Privacy: reject any image-like fields.
        lowered = {str(k).lower() for k in payload.keys()}
        bad = lowered & _FORBIDDEN_FRAME_KEYS
        if bad:
            raise ValueError(f"Image/frame payloads are forbidden: {sorted(bad)}")
        for k, v in payload.items():
            if isinstance(v, str) and (
                v.startswith("data:image") or len(v) > 50_000
            ):
                raise ValueError("Image/base64 payloads are forbidden")

        pose_in = payload.get("pose") or {}
        if not isinstance(pose_in, dict):
            raise ValueError("pose must be an object of param->float")
        values = params_from_mapping({str(k): v for k, v in pose_in.items()})
        for i, v in enumerate(values):
            if not math.isfinite(float(v)):
                values[i] = 0.0

        landmarks_raw = payload.get("landmarks") or []
        conf_raw = payload.get("confidences") or []
        landmarks: list[list[float]] = []
        confidences: list[float] = []
        if landmarks_raw:
            if not isinstance(landmarks_raw, list) or len(landmarks_raw) != 68:
                raise ValueError("landmarks must be a list of 68 [x,y] pairs in 0..1")
            for pt in landmarks_raw:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    raise ValueError("each landmark must be [x,y]")
                x = float(pt[0])
                y = float(pt[1])
                if not math.isfinite(x) or not math.isfinite(y):
                    x, y = 0.0, 0.0
                landmarks.append([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])
        if conf_raw:
            if not isinstance(conf_raw, list) or len(conf_raw) != 68:
                raise ValueError("confidences must be a list of 68 floats")
            confidences = [float(c) if math.isfinite(float(c)) else 0.0 for c in conf_raw]
        elif landmarks:
            confidences = [1.0] * 68

        with self._lock:
            self._tracking = True
            self._has_face = True
            self._pose = osf.values_to_pose_dict(values)
            self._landmarks = landmarks
            self._confidences = confidences
            self._sequence = int(payload.get("sequence") or (self._sequence + 1))
            self._last_update = time.time()
            self._face_id = int(payload.get("face_id") or 0)
            self._got_3d = bool(payload.get("got_3d", True))
            self._fit_error = float(payload.get("fit_error") or 0.0)
            self._message = "Tracking: receiving remote frames"
            self._emit()
            return {"ok": True, "sequence": self._sequence}

    def _poll_loop(self) -> None:
        while not self._stop_poll.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                with self._lock:
                    self._message = f"Tracking poll error: {exc}"
            time.sleep(0.05)

    def _poll_once(self) -> None:
        with self._lock:
            if not self._tracking or self.server_mode:
                return
            tracker = self._tracker
            receiver = self._receiver
            if tracker is None or receiver is None:
                return
            if not tracker.running:
                code = tracker.exit_code
                self._tracking = False
                self._has_face = False
                self._message = f"Tracking: process died (exit {code})"
                self._emit()
                return
            face = receiver.latest
            age = receiver.age_seconds()
            pkts = receiver.packets_received
            if face is None or age > FACE_MAX_AGE_S:
                self._has_face = False
                self._landmarks = []
                self._confidences = []
                self._message = (
                    f"Tracking: waiting for face… (cam {self._camera_index}, udp={pkts})"
                )
                self._emit()
                return

            values = osf.face_to_param_values(
                face,
                mirror=self._mirror,
                calibrator=self._calibrator,
            )
            values = self._smoother.update(values)
            landmarks = list(face.landmarks2d)
            conf = list(face.confidences)
            if self._mirror:
                landmarks = osf.mirror_landmarks(landmarks)
                conf = osf.mirror_confidences(conf)

            self._pose = osf.values_to_pose_dict(values)
            self._landmarks = [[float(x), float(y)] for x, y in landmarks]
            self._confidences = [float(c) for c in conf]
            self._has_face = True
            self._face_id = int(face.face_id)
            self._got_3d = bool(face.got_3d)
            self._fit_error = float(face.fit_error)
            self._last_update = time.time()
            self._sequence += 1
            cal = (
                "ready"
                if self._calibrator.ready
                else f"calibrating {self._calibrator.progress * 100:.0f}%"
            )
            self._message = (
                f"Tracking: face #{face.face_id} "
                f"({'3D' if face.got_3d else '2D'})  {cal}  pkt_age={age:.2f}s"
            )
            self._emit()

    def snapshot(self) -> TrackingSnapshot:
        with self._lock:
            age = (
                (time.time() - self._last_update)
                if self._last_update > 0
                else math.inf
            )
            pkts = self._receiver.packets_received if self._receiver else 0
            return TrackingSnapshot(
                active=self._tracking,
                mode="server" if self.server_mode else "local",
                tracking=self._tracking,
                has_face=self._has_face and age <= FACE_MAX_AGE_S,
                calibrating=self._tracking and not self._calibrator.ready and not self.server_mode,
                calibration_progress=float(self._calibrator.progress),
                calibration_ready=bool(self._calibrator.ready) or self.server_mode,
                camera_index=self._camera_index,
                mirror=self._mirror,
                packets_received=int(pkts),
                age_seconds=float(age if math.isfinite(age) else 999.0),
                face_id=self._face_id,
                got_3d=self._got_3d,
                fit_error=self._fit_error,
                pose=dict(self._pose),
                landmarks=[list(p) for p in self._landmarks],
                confidences=list(self._confidences),
                message=self._message,
                sequence=self._sequence,
                timestamp=self._last_update,
            )

    def snapshot_dict(self) -> dict[str, Any]:
        s = self.snapshot()
        return {
            "active": s.active,
            "mode": s.mode,
            "tracking": s.tracking,
            "has_face": s.has_face,
            "calibrating": s.calibrating,
            "calibration_progress": s.calibration_progress,
            "calibration_ready": s.calibration_ready,
            "camera_index": s.camera_index,
            "mirror": s.mirror,
            "packets_received": s.packets_received,
            "age_seconds": s.age_seconds,
            "face_id": s.face_id,
            "got_3d": s.got_3d,
            "fit_error": s.fit_error,
            "pose": s.pose,
            "landmarks": s.landmarks,
            "confidences": s.confidences,
            "message": s.message,
            "sequence": s.sequence,
            "timestamp": s.timestamp,
            "param_names": list(PARAM_NAMES),
            "num_params": NUM_PARAMS,
        }


_SERVICE: TrackingService | None = None
_SERVICE_LOCK = threading.Lock()


def get_tracking_service() -> TrackingService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = TrackingService(server_mode=False)
        return _SERVICE


def set_server_mode(enabled: bool) -> TrackingService:
    svc = get_tracking_service()
    svc.set_server_mode(enabled)
    return svc
