"""OpenSeeFace UDP client + Live2D/VTS param mapping (privacy-safe landmarks).

Adapted from ``real_stream/openseeface_client.py``. Always starts the tracker
with ``-v 0`` so OpenCV never shows the user's face. 2D landmarks are kept as
normalized coordinates for the wire-mesh UI only — never camera pixels.
"""

from __future__ import annotations

import math
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from backend.paths import app_root, outputs_dir
from backend.utils.params import (
    PARAM_NAMES,
    flip_params_horizontal,
    params_from_mapping,
)

# Must match OpenSeeFace Unity/OpenSee.cs packetFrameSize.
PACKET_FRAME_SIZE = (
    8  # timestamp
    + 4  # id
    + 2 * 4  # resolution (float width, float height)
    + 2 * 4  # eye open
    + 1  # got3D
    + 4  # fit error
    + 4 * 4  # quaternion
    + 3 * 4  # euler
    + 3 * 4  # translation
    + 68 * 4  # landmark conf
    + 68 * 2 * 4  # 2D landmarks
    + 70 * 3 * 4  # 3D points
    + 14 * 4  # features
)

DEFAULT_UDP_IP = "127.0.0.1"
DEFAULT_UDP_PORT = 11573
ANGLE_CLAMP = 30.0
# Mild overall gain; yaw gets an extra bump so left/right reaches prompt tags sooner.
HEAD_ANGLE_GAIN = 1.25
YAW_ANGLE_GAIN = 1.55
_CAM_LINE_RE = re.compile(r"^(\d+)\s*:\s*(.+)$")


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_deg(angle: float) -> float:
    a = (float(angle) + 180.0) % 360.0 - 180.0
    return a


def resolve_openseeface_dir() -> Path:
    """Locate the vendored OpenSeeFace tree (dev monorepo or packaged copy)."""
    env = os.environ.get("SDANIME_OPENSEEFACE_DIR", "").strip()
    if env:
        p = Path(env)
        if (p / "facetracker.py").is_file() or (p / "facetracker.exe").is_file():
            return p

    root = app_root()
    candidates = [
        root / "tools" / "OpenSeeFace",
        root / "_internal" / "tools" / "OpenSeeFace",
        root.parent / "tools" / "vedio traker" / "OpenSeeFace",
        Path(__file__).resolve().parents[3] / "tools" / "vedio traker" / "OpenSeeFace",
    ]
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates = [
            exe_dir / "tools" / "OpenSeeFace",
            exe_dir / "_internal" / "tools" / "OpenSeeFace",
            *candidates,
        ]
    for c in candidates:
        if (c / "facetracker.py").is_file() or (c / "facetracker.exe").is_file():
            return c
    return candidates[0]


def _python_can_run_facetracker(python_exe: Path) -> bool:
    """True if *python_exe* can import OpenSeeFace's runtime deps."""
    if not python_exe.is_file():
        return False
    try:
        result = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import importlib.util as u; "
                "ok = all(u.find_spec(m) for m in ('PIL', 'cv2', 'numpy')); "
                "raise SystemExit(0 if ok else 1)",
            ],
            capture_output=True,
            timeout=12,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _tracker_python_candidates() -> list[Path]:
    """Candidate interpreters for spawning facetracker.py (never the frozen exe)."""
    root = app_root()
    frozen = getattr(sys, "frozen", False)
    exe = Path(sys.executable).resolve()
    out: list[Path] = []

    env = os.environ.get("SDANIME_TRACKER_PYTHON", "").strip()
    if env:
        out.append(Path(env))

    # Dev / monorepo venvs that usually have OpenCV + Pillow.
    out.extend(
        [
            root / ".venv-build" / "Scripts" / "python.exe",
            root / ".venv" / "Scripts" / "python.exe",
            root.parent / "pipeline" / "i1" / "torch_train" / ".venv" / "Scripts" / "python.exe",
        ]
    )

    if frozen:
        # Packaged exe: also walk parents so a local build next to the monorepo
        # can reuse real_stream_SDAnime/.venv-build or pipeline torch_train venv.
        search_roots = [exe.parent, root]
        for base in list(search_roots):
            search_roots.extend(list(base.parents)[:6])
        for base in search_roots:
            out.extend(
                [
                    base / ".venv-build" / "Scripts" / "python.exe",
                    base / ".venv" / "Scripts" / "python.exe",
                    base / "pipeline" / "i1" / "torch_train" / ".venv" / "Scripts" / "python.exe",
                    base / "real_stream_SDAnime" / ".venv-build" / "Scripts" / "python.exe",
                    base / "python.exe",
                ]
            )
        out.append(Path(sys.base_prefix) / "python.exe")
    else:
        out.append(exe)

    # De-dupe while preserving order; skip the packaged app itself.
    seen: set[Path] = set()
    unique: list[Path] = []
    for c in out:
        try:
            key = c.resolve()
        except OSError:
            key = c
        if key in seen:
            continue
        if frozen and key == exe:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def resolve_tracker_python() -> str:
    """Python that can run facetracker.py (not the frozen app exe)."""
    for candidate in _tracker_python_candidates():
        if _python_can_run_facetracker(candidate):
            return str(candidate.resolve())

    # Last resort: facetracker.exe if the build shipped one.
    osf = resolve_openseeface_dir()
    exe = osf / "facetracker.exe"
    if exe.is_file():
        return str(exe)

    # May still fail at spawn time; callers should surface the error.
    if not getattr(sys, "frozen", False):
        return sys.executable
    raise RuntimeError(
        "No Python with PIL/cv2/numpy found for OpenSeeFace. "
        "Set SDANIME_TRACKER_PYTHON to a working python.exe, "
        "or place facetracker.exe under tools/OpenSeeFace."
    )


def tracker_log_path() -> Path:
    return outputs_dir() / "facetracker.log"


@dataclass
class FaceFrame:
    timestamp: float
    face_id: int
    got_3d: bool
    fit_error: float
    right_eye_open: float
    left_eye_open: float
    euler: tuple[float, float, float]
    quat: tuple[float, float, float, float]
    mouth_open: float
    mouth_wide: float
    mouth_corner_up_l: float
    mouth_corner_up_r: float
    gaze_x: float
    gaze_y: float
    width: int = 0
    height: int = 0
    # 68 confidence scores
    confidences: tuple[float, ...] = field(default_factory=tuple)
    # 68 normalized (x,y) in 0..1 image space (y down)
    landmarks2d: tuple[tuple[float, float], ...] = field(default_factory=tuple)


def parse_face_packet(data: bytes, offset: int = 0) -> FaceFrame | None:
    if len(data) - offset < PACKET_FRAME_SIZE:
        return None
    o = offset
    timestamp = struct.unpack_from("<d", data, o)[0]
    o += 8
    face_id = struct.unpack_from("<i", data, o)[0]
    o += 4
    # OpenSeeFace packs camera resolution as two floats (see facetracker.py / OpenSee.cs).
    width_f, height_f = struct.unpack_from("<ff", data, o)
    o += 8
    width = int(round(float(width_f))) if math.isfinite(width_f) else 0
    height = int(round(float(height_f))) if math.isfinite(height_f) else 0
    right_eye_open = struct.unpack_from("<f", data, o)[0]
    o += 4
    left_eye_open = struct.unpack_from("<f", data, o)[0]
    o += 4
    got_3d = data[o] != 0
    o += 1
    fit_error = struct.unpack_from("<f", data, o)[0]
    o += 4
    quat = struct.unpack_from("<ffff", data, o)
    o += 16
    euler = struct.unpack_from("<fff", data, o)
    o += 12
    o += 12  # translation
    conf = struct.unpack_from("<" + "f" * 68, data, o)
    o += 68 * 4
    pts2 = struct.unpack_from("<" + "f" * (68 * 2), data, o)
    o += 68 * 2 * 4
    pts = struct.unpack_from("<" + "f" * (70 * 3), data, o)
    o += 70 * 3 * 4
    features = struct.unpack_from("<" + "f" * 14, data, o)

    w = float(width) if width > 0 else 1.0
    h = float(height) if height > 0 else 1.0
    landmarks: list[tuple[float, float]] = []
    for i in range(68):
        # facetracker packs (image_x, image_y) = (col, row) as floats.
        x = float(pts2[i * 2]) / w
        y = float(pts2[i * 2 + 1]) / h
        if not math.isfinite(x):
            x = 0.0
        if not math.isfinite(y):
            y = 0.0
        landmarks.append((_clamp(x, 0.0, 1.0), _clamp(y, 0.0, 1.0)))

    def pt(i: int) -> tuple[float, float, float]:
        base = i * 3
        return pts[base], pts[base + 1], pts[base + 2]

    def gaze_xy(pupil_i: int, center_i: int) -> tuple[float, float]:
        px, py, pz = pt(pupil_i)
        cx, cy, cz = pt(center_i)
        dx, dy, dz = px - cx, py - cy, pz - cz
        gx = _clamp(dx * 8.0, -1.0, 1.0)
        gy = _clamp(-dy * 8.0, -1.0, 1.0)
        _ = dz
        return gx, gy

    gx_r, gy_r = gaze_xy(66, 68)
    gx_l, gy_l = gaze_xy(67, 69)
    gaze_x = 0.5 * (gx_r + gx_l)
    gaze_y = 0.5 * (gy_r + gy_l)

    return FaceFrame(
        timestamp=timestamp,
        face_id=face_id,
        got_3d=got_3d,
        fit_error=float(fit_error),
        right_eye_open=float(right_eye_open),
        left_eye_open=float(left_eye_open),
        euler=(float(euler[0]), float(euler[1]), float(euler[2])),
        quat=(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])),
        mouth_open=float(features[12]),
        mouth_wide=float(features[13]),
        mouth_corner_up_l=float(features[8]),
        mouth_corner_up_r=float(features[10]),
        gaze_x=float(gaze_x),
        gaze_y=float(gaze_y),
        width=int(width),
        height=int(height),
        confidences=tuple(float(c) for c in conf),
        landmarks2d=tuple(landmarks),
    )


# iBUG / OpenSeeFace left↔right index pairs (center points omitted).
_LANDMARK_MIRROR_PAIRS: tuple[tuple[int, int], ...] = (
    *((i, 16 - i) for i in range(8)),
    *((17 + i, 26 - i) for i in range(5)),
    (31, 35),
    (32, 34),
    (36, 45),
    (37, 44),
    (38, 43),
    (39, 42),
    (40, 47),
    (41, 46),
    (48, 54),
    (49, 53),
    (50, 52),
    (55, 59),
    (56, 58),
    (60, 64),
    (61, 63),
    (65, 67),
)


def _swap_landmark_sides(values: list):
    if len(values) < 68:
        return values
    out = list(values)
    for a, b in _LANDMARK_MIRROR_PAIRS:
        out[a], out[b] = values[b], values[a]
    return out[:68]


def mirror_landmarks(
    points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
) -> list[tuple[float, float]]:
    """Flip X and swap L/R landmark indices so topology stays iBUG-ordered."""
    flipped = [(1.0 - float(x), float(y)) for x, y in points]
    return _swap_landmark_sides(flipped)


def mirror_confidences(
    confidences: list[float] | tuple[float, ...],
) -> list[float]:
    return _swap_landmark_sides([float(c) for c in confidences])


def osf_unity_rotation(face: FaceFrame) -> tuple[float, float, float]:
    ex, ey, ez = face.euler
    pitch = _wrap_deg(-(ex + 180.0))
    yaw = _wrap_deg(-ey)
    roll = _wrap_deg(ez - 90.0)
    return yaw, pitch, roll


class HeadAngleCalibrator:
    def __init__(self, samples_needed: int = 24) -> None:
        self.samples_needed = max(1, int(samples_needed))
        self._samples: list[tuple[float, float, float]] = []
        self.neutral: tuple[float, float, float] | None = None

    @property
    def ready(self) -> bool:
        return self.neutral is not None

    @property
    def progress(self) -> float:
        if self.neutral is not None:
            return 1.0
        return len(self._samples) / float(self.samples_needed)

    def reset(self) -> None:
        self._samples.clear()
        self.neutral = None

    def update(self, yaw: float, pitch: float, roll: float) -> tuple[float, float, float]:
        if self.neutral is None:
            self._samples.append((float(yaw), float(pitch), float(roll)))
            if len(self._samples) >= self.samples_needed:
                n = float(len(self._samples))
                self.neutral = (
                    sum(s[0] for s in self._samples) / n,
                    sum(s[1] for s in self._samples) / n,
                    sum(s[2] for s in self._samples) / n,
                )
            else:
                return 0.0, 0.0, 0.0
        ny, np_, nr = self.neutral
        return (
            _wrap_deg(yaw - ny),
            _wrap_deg(pitch - np_),
            _wrap_deg(roll - nr),
        )


def osf_face_to_vts_angles(
    face: FaceFrame,
    *,
    calibrator: HeadAngleCalibrator | None = None,
) -> tuple[float, float, float]:
    yaw, pitch, roll = osf_unity_rotation(face)
    if calibrator is not None:
        yaw, pitch, roll = calibrator.update(yaw, pitch, roll)
    return yaw, pitch, roll


def face_to_param_values(
    face: FaceFrame,
    *,
    mirror: bool = False,
    angle_gain: float = HEAD_ANGLE_GAIN,
    calibrator: HeadAngleCalibrator | None = None,
) -> list[float]:
    yaw, pitch, roll = osf_face_to_vts_angles(face, calibrator=calibrator)
    g = float(angle_gain)
    # Extra yaw gain: left/right head turns were under-represented in the character.
    yaw = _clamp(yaw * g * (YAW_ANGLE_GAIN / HEAD_ANGLE_GAIN), -ANGLE_CLAMP, ANGLE_CLAMP)
    pitch = _clamp(pitch * g, -ANGLE_CLAMP, ANGLE_CLAMP)
    roll = _clamp(roll * g, -ANGLE_CLAMP, ANGLE_CLAMP)

    eye_r = _clamp(face.right_eye_open, 0.0, 1.0)
    eye_l = _clamp(face.left_eye_open, 0.0, 1.0)
    mouth_open = _clamp(face.mouth_open, 0.0, 1.0)
    corner = 0.5 * (face.mouth_corner_up_l + face.mouth_corner_up_r)
    smile = _clamp(0.5 + 0.5 * corner + 0.15 * face.mouth_wide, 0.0, 1.0)
    mouth_form = _clamp(2.0 * smile - 1.0, -1.0, 1.0)

    mapping = {
        "ParamAngleX": yaw,
        "ParamAngleY": pitch,
        "ParamAngleZ": roll,
        "ParamMouthOpenY": mouth_open,
        "ParamMouthForm": mouth_form,
        "ParamEyeLOpen": _clamp(eye_l * 1.2, 0.0, 1.3),
        "ParamEyeROpen": _clamp(eye_r * 1.2, 0.0, 1.3),
        "ParamEyeBallX": _clamp(face.gaze_x, -1.0, 1.0),
        "ParamEyeBallY": _clamp(face.gaze_y, -1.0, 1.0),
        "FaceAngleX": yaw,
        "FaceAngleY": pitch,
        "FaceAngleZ": roll,
        "MouthOpen": mouth_open,
        "MouthSmile": smile,
        "EyeOpenLeft": _clamp(eye_l * 0.55, 0.0, 1.0),
        "EyeOpenRight": _clamp(eye_r * 0.55, 0.0, 1.0),
    }
    values = params_from_mapping(mapping)
    if mirror:
        values = flip_params_horizontal(values)
    out: list[float] = []
    for v in values:
        fv = float(v)
        if not math.isfinite(fv):
            fv = 0.0
        out.append(fv)
    return out


def values_to_pose_dict(values: list[float]) -> dict[str, float]:
    return {name: float(v) for name, v in zip(PARAM_NAMES, values)}


class SmoothedParams:
    _ANGLE_IDX = (0, 1, 2, 9, 10, 11)

    def __init__(self, alpha: float = 0.35, angle_alpha: float | None = None) -> None:
        self.alpha = float(alpha)
        self.angle_alpha = float(angle_alpha if angle_alpha is not None else alpha)
        self._values: list[float] | None = None

    def update(self, values: list[float]) -> list[float]:
        if self._values is None or len(self._values) != len(values):
            self._values = [float(v) for v in values]
            return list(self._values)
        out: list[float] = []
        for i, (old, new) in enumerate(zip(self._values, values)):
            a = self.angle_alpha if i in self._ANGLE_IDX else self.alpha
            out.append((1.0 - a) * old + a * float(new))
        self._values = out
        return list(self._values)

    def reset(self) -> None:
        self._values = None


class OpenSeeFaceReceiver:
    def __init__(self, host: str = "0.0.0.0", port: int = DEFAULT_UDP_PORT) -> None:
        self.host = host
        self.port = int(port)
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: FaceFrame | None = None
        self._last_recv = 0.0
        self.packets_received = 0

    @property
    def latest(self) -> FaceFrame | None:
        with self._lock:
            return self._latest

    def age_seconds(self) -> float:
        if self._last_recv <= 0:
            return math.inf
        return time.time() - self._last_recv

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self.packets_received = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.settimeout(0.5)
        self._thread = threading.Thread(target=self._loop, name="osf-udp", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        with self._lock:
            self._latest = None
        self._last_recv = 0.0

    def _loop(self) -> None:
        while not self._stop.is_set():
            sock = self._sock
            if sock is None:
                break
            try:
                data, _addr = sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) < PACKET_FRAME_SIZE or len(data) % PACKET_FRAME_SIZE != 0:
                continue
            face = parse_face_packet(data, 0)
            if face is None:
                continue
            with self._lock:
                self._latest = face
            self._last_recv = time.time()
            self.packets_received += 1


def _kill_orphaned_facetrackers() -> None:
    """Kill leftover facetracker.py processes from this project (camera locks)."""
    marker = str((resolve_openseeface_dir() / "facetracker.py").resolve()).lower()
    if sys.platform == "win32":
        try:
            listed = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    "CommandLine like '%facetracker.py%'",
                    "get",
                    "ProcessId,CommandLine",
                    "/FORMAT:LIST",
                ],
                text=True,
                errors="replace",
                timeout=8,
            )
        except (OSError, subprocess.SubprocessError):
            return
        pid: str | None = None
        cmd = ""
        for line in listed.splitlines():
            if line.startswith("CommandLine="):
                cmd = line.split("=", 1)[1].strip().lower()
            elif line.startswith("ProcessId="):
                pid = line.split("=", 1)[1].strip()
                if pid.isdigit() and marker in cmd.replace("/", "\\"):
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", pid, "/T", "/F"],
                            capture_output=True,
                            timeout=5,
                            check=False,
                        )
                    except (OSError, subprocess.TimeoutExpired):
                        pass
                pid = None
                cmd = ""
        return
    try:
        subprocess.run(["pkill", "-f", marker], check=False, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        pass


class OpenSeeFaceTrackerProcess:
    """Spawn facetracker with visualize forced off (never show the face)."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._log_fh = None
        self.camera_index: int | None = None
        self.last_cmd: list[str] = []

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def exit_code(self) -> int | None:
        if self._proc is None:
            return None
        return self._proc.poll()

    def start(
        self,
        camera_index: int,
        *,
        python_exe: str | None = None,
        fps: int = 30,
        port: int = DEFAULT_UDP_PORT,
        settle_seconds: float = 2.5,
    ) -> None:
        osf_dir = resolve_openseeface_dir()
        facetracker_py = osf_dir / "facetracker.py"
        facetracker_exe = osf_dir / "facetracker.exe"
        log_path = tracker_log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stop()
        _kill_orphaned_facetrackers()
        # Give DirectShow a moment to release the previous device handle.
        time.sleep(0.5)
        # Truncate cleanly so tails aren't polluted by stale writers.
        try:
            log_path.write_text("", encoding="utf-8")
        except OSError:
            pass
        self._log_fh = open(log_path, "w", encoding="utf-8", errors="replace")

        # Privacy: always -v 0 — never open the OpenCV face window.
        visualize = 0
        if facetracker_exe.is_file() and (
            python_exe is None or Path(python_exe).name.lower().startswith("facetracker")
        ):
            cmd = [
                str(facetracker_exe),
                "-c",
                str(int(camera_index)),
                "-F",
                str(int(fps)),
                "-i",
                DEFAULT_UDP_IP,
                "-p",
                str(int(port)),
                "-v",
                str(visualize),
                "-s",
                "0",
                "--model",
                "3",
                "--faces",
                "1",
            ]
        else:
            if not facetracker_py.is_file():
                raise FileNotFoundError(f"facetracker.py not found: {facetracker_py}")
            py = python_exe or resolve_tracker_python()
            if getattr(sys, "frozen", False) and Path(py).resolve() == Path(sys.executable).resolve():
                raise RuntimeError(
                    "Frozen app cannot spawn facetracker via its own exe. "
                    "Set SDANIME_TRACKER_PYTHON or ship tools/OpenSeeFace/facetracker.exe"
                )
            cmd = [
                py,
                "-u",
                str(facetracker_py),
                "-c",
                str(int(camera_index)),
                "-F",
                str(int(fps)),
                "-i",
                DEFAULT_UDP_IP,
                "-p",
                str(int(port)),
                "-v",
                str(visualize),
                "-s",
                "0",
                "--model",
                "3",
                "--faces",
                "1",
            ]

        self.last_cmd = cmd
        self.camera_index = int(camera_index)
        self._log_fh.write("CMD: " + " ".join(cmd) + "\n")
        self._log_fh.write(f"cwd: {osf_dir}\n")
        self._log_fh.write("privacy: visualize=0 (face window disabled)\n\n")
        self._log_fh.flush()
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(osf_dir),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        deadline = time.time() + max(0.5, float(settle_seconds))
        while time.time() < deadline:
            code = self._proc.poll()
            if code is not None:
                tail = self.read_log_tail()
                hint = _camera_fail_hint(tail, camera_index)
                raise RuntimeError(
                    f"OpenSeeFace exited immediately (code {code}) for camera {camera_index}.\n"
                    f"{hint}"
                    f"Log: {log_path}\n\n{tail}"
                )
            time.sleep(0.15)

    def read_log_tail(self, max_chars: int = 2000) -> str:
        try:
            if self._log_fh is not None:
                self._log_fh.flush()
            text = tracker_log_path().read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(no log)"
        text = text.strip()
        if len(text) > max_chars:
            return text[-max_chars:]
        return text or "(empty log)"

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        self.camera_index = None
        if proc is not None and proc.poll() is None:
            _terminate_process_tree(proc)
        if self._log_fh is not None:
            try:
                self._log_fh.close()
            except OSError:
                pass
            self._log_fh = None


def _camera_fail_hint(log_tail: str, camera_index: int) -> str:
    low = (log_tail or "").lower()
    if "no valid input" in low or "failed to start capture" in low:
        return (
            f"Camera {camera_index} could not be opened. "
            "Pick a real webcam (not nizima/Warudo/OBS virtual), "
            "close other apps using it, and ensure phone cams (DroidCam) are streaming.\n"
        )
    return ""


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    """Stop facetracker and any child capture helpers (Windows needs /T)."""
    if proc.poll() is not None:
        return
    pid = proc.pid
    if sys.platform == "win32" and pid:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        proc.terminate()
    except OSError:
        pass
    try:
        proc.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            proc.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _looks_virtual_camera(name: str) -> bool:
    lower = name.lower()
    needles = (
        "virtual",
        "obs",
        "nizima",
        "warudo",
        "manycam",
        "snap camera",
        "unity capture",
        "ndi",
    )
    return any(n in lower for n in needles)


def pick_default_camera(cameras: list[tuple[int, str]]) -> int:
    if not cameras:
        return 0
    for idx, name in cameras:
        if not _looks_virtual_camera(name):
            return idx
    return cameras[0][0]


def _list_cameras_dshow(osf_dir: Path) -> list[tuple[int, str]] | None:
    """Enumerate DirectShow devices in-process (no facetracker / PIL / cv2).

    Works inside the frozen SDAnimePose.exe where spawning a bare system
    Python often fails because that interpreter lacks OpenSeeFace deps.
    """
    if os.name != "nt":
        return None
    dll_dir = osf_dir / "dshowcapture"
    dll_path = dll_dir / (
        "dshowcapture_x86.dll"
        if sys.maxsize <= 2**32
        else "dshowcapture_x64.dll"
    )
    if not dll_path.is_file():
        alt = dll_dir / "dshowcapture_x64.dll"
        dll_path = alt if alt.is_file() else dll_dir / "dshowcapture_x86.dll"
    if not dll_path.is_file():
        return None
    try:
        import json
        from ctypes import c_char_p, c_int, c_void_p, cdll, create_string_buffer

        lib = cdll.LoadLibrary(str(dll_path))
        lib.create_capture.restype = c_void_p
        lib.get_json_length.argtypes = [c_void_p]
        lib.get_json.argtypes = [c_void_p, c_char_p, c_int]
        lib.destroy_capture.argtypes = [c_void_p]
        cap = lib.create_capture()
        if not cap:
            return None
        try:
            length = int(lib.get_json_length(cap))
            if length <= 0:
                return []
            buf = create_string_buffer(length)
            lib.get_json(cap, buf, length)
            raw = buf.value.decode("utf-8", "surrogateescape")
            info = json.loads(raw)
        finally:
            lib.destroy_capture(cap)
    except Exception:
        return None

    out: list[tuple[int, str]] = []
    if not isinstance(info, list):
        return None
    for cam in info:
        if not isinstance(cam, dict):
            continue
        idx = cam.get("index", cam.get("id"))
        name = cam.get("name")
        if idx is None or name is None:
            continue
        out.append((int(idx), str(name)))
    return out


def _parse_camera_lines(stdout: str) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    for line in (stdout or "").splitlines():
        text = line.strip()
        match = _CAM_LINE_RE.match(text)
        if match:
            out.append((int(match.group(1)), match.group(2).strip()))
    return out


def list_cameras(python_exe: str | None = None) -> list[tuple[int, str]]:
    """List cameras via in-process DirectShow, else OpenSeeFace facetracker (-l)."""
    osf_dir = resolve_openseeface_dir()

    # Preferred: no subprocess, works in packaged local mode.
    dshow = _list_cameras_dshow(osf_dir)
    if dshow is not None and len(dshow) > 0:
        return dshow

    facetracker_py = osf_dir / "facetracker.py"
    facetracker_exe = osf_dir / "facetracker.exe"

    runner: list[str] | None = None
    if python_exe and Path(python_exe).name.lower().startswith("facetracker"):
        runner = [python_exe]
    elif facetracker_exe.is_file() and not facetracker_py.is_file():
        runner = [str(facetracker_exe)]
    elif facetracker_py.is_file():
        try:
            py = python_exe or resolve_tracker_python()
        except RuntimeError as exc:
            if dshow is not None:
                return []
            return [(0, f"Camera 0 (no tracker python: {exc})")]
        if Path(py).name.lower().startswith("facetracker"):
            runner = [py]
        else:
            runner = [py, "-u", str(facetracker_py)]
    else:
        if dshow is not None:
            return []
        return [(0, "Camera 0")]

    last_err = ""
    try:
        result = subprocess.run(
            [*runner, "-l", "1"],
            cwd=str(osf_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        if dshow is not None:
            return []
        return [(0, f"Camera 0 (list failed: {exc})")]

    out = _parse_camera_lines(result.stdout or "")
    if out:
        return out

    # Fallback: names-only listing (-l 2), matching real_stream.
    try:
        result2 = subprocess.run(
            [*runner, "-l", "2"],
            cwd=str(osf_dir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=25,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        result2 = None

    if result2 is not None:
        names = [ln.strip() for ln in (result2.stdout or "").splitlines() if ln.strip()]
        names = [n for n in names if not n.lower().startswith("available cameras")]
        if names:
            return [(i, name) for i, name in enumerate(names)]
        last_err = (result2.stderr or result.stderr or "").strip()
    else:
        last_err = (result.stderr or "").strip()

    if dshow is not None:
        return []
    label = "Camera 0"
    if last_err:
        label = f"Camera 0 (list err: {last_err[:120]})"
    return [(0, label)]
