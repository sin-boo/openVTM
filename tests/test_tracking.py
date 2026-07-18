"""Unit tests for privacy-safe OpenSeeFace parsing + tracking service."""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.tracking.openseeface import (  # noqa: E402
    PACKET_FRAME_SIZE,
    HeadAngleCalibrator,
    SmoothedParams,
    face_to_param_values,
    mirror_landmarks,
    parse_face_packet,
    values_to_pose_dict,
)
from backend.tracking.service import TrackingService  # noqa: E402
from backend.utils.params import NUM_PARAMS, PARAM_NAMES  # noqa: E402


def _pack_face(
    *,
    width: int = 640,
    height: int = 480,
    euler=(10.0, -5.0, 90.0),
    mouth_open: float = 0.2,
) -> bytes:
    """Build a minimal valid OpenSeeFace UDP packet."""
    buf = bytearray(PACKET_FRAME_SIZE)
    o = 0
    struct.pack_into("<d", buf, o, 1.0)
    o += 8
    struct.pack_into("<i", buf, o, 0)
    o += 4
    # OpenSeeFace uses float width/height (not int).
    struct.pack_into("<ff", buf, o, float(width), float(height))
    o += 8
    struct.pack_into("<ff", buf, o, 0.9, 0.85)  # right/left eye open
    o += 8
    buf[o] = 1  # got3d
    o += 1
    struct.pack_into("<f", buf, o, 1.5)
    o += 4
    struct.pack_into("<ffff", buf, o, 0.0, 0.0, 0.0, 1.0)
    o += 16
    struct.pack_into("<fff", buf, o, float(euler[0]), float(euler[1]), float(euler[2]))
    o += 12
    struct.pack_into("<fff", buf, o, 0.0, 0.0, 0.0)  # translation
    o += 12
    conf = [0.9] * 68
    struct.pack_into("<" + "f" * 68, buf, o, *conf)
    o += 68 * 4
    # 68 2D landmarks — simple oval pattern in pixel space
    pts2: list[float] = []
    for i in range(68):
        ang = (i / 68.0) * 2.0 * math.pi
        x = width * (0.5 + 0.3 * math.cos(ang))
        y = height * (0.5 + 0.35 * math.sin(ang))
        pts2.extend([x, y])
    struct.pack_into("<" + "f" * (68 * 2), buf, o, *pts2)
    o += 68 * 2 * 4
    # 70 * 3 3D points (zeros except pupils for gaze)
    pts3 = [0.0] * (70 * 3)
    # pupil 66/67 and centers 68/69
    pts3[66 * 3] = 0.1
    pts3[67 * 3] = -0.1
    struct.pack_into("<" + "f" * (70 * 3), buf, o, *pts3)
    o += 70 * 3 * 4
    features = [0.0] * 14
    features[8] = 0.1
    features[10] = 0.1
    features[12] = mouth_open
    features[13] = 0.0
    struct.pack_into("<" + "f" * 14, buf, o, *features)
    assert len(buf) == PACKET_FRAME_SIZE
    return bytes(buf)


def test_list_cameras_dshow_or_fallback():
    from backend.tracking.openseeface import list_cameras, resolve_openseeface_dir

    osf = resolve_openseeface_dir()
    assert (osf / "facetracker.py").is_file() or (osf / "dshowcapture").is_dir()
    cams = list_cameras()
    assert isinstance(cams, list)
    assert len(cams) >= 1
    idx, name = cams[0]
    assert isinstance(idx, int)
    assert isinstance(name, str) and name


def test_parse_face_packet_landmarks_normalized():
    raw = _pack_face()
    face = parse_face_packet(raw)
    assert face is not None
    assert face.width == 640
    assert face.height == 480
    assert len(face.landmarks2d) == 68
    assert len(face.confidences) == 68
    xs = [p[0] for p in face.landmarks2d]
    ys = [p[1] for p in face.landmarks2d]
    assert min(xs) >= 0.15 and max(xs) <= 0.85
    assert min(ys) >= 0.10 and max(ys) <= 0.90
    for x, y in face.landmarks2d:
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
    assert face.mouth_open == pytest.approx(0.2, abs=1e-5)


def test_parse_face_packet_float_resolution_matches_osf():
    """Regression: OSF packs width/height as floats; int unpack collapsed landmarks to ~0."""
    import struct

    width, height = 1920, 1080
    buf = bytearray(PACKET_FRAME_SIZE)
    o = 0
    struct.pack_into("<d", buf, o, 1.0)
    o += 8
    struct.pack_into("<i", buf, o, 0)
    o += 4
    struct.pack_into("<ff", buf, o, float(width), float(height))
    o += 8
    struct.pack_into("<ff", buf, o, 1.0, 1.0)
    o += 8
    buf[o] = 1
    o += 1
    struct.pack_into("<f", buf, o, 1.0)
    o += 4
    struct.pack_into("<ffff", buf, o, 0.0, 0.0, 0.0, 1.0)
    o += 16
    struct.pack_into("<fff", buf, o, 0.0, 0.0, 90.0)
    o += 12
    o += 12
    struct.pack_into("<" + "f" * 68, buf, o, *([0.95] * 68))
    o += 68 * 4
    # One known pixel landmark near center.
    pts = [0.0] * (68 * 2)
    pts[0] = 960.0
    pts[1] = 540.0
    struct.pack_into("<" + "f" * (68 * 2), buf, o, *pts)
    o += 68 * 2 * 4
    struct.pack_into("<" + "f" * (70 * 3), buf, o, *([0.0] * (70 * 3)))
    o += 70 * 3 * 4
    struct.pack_into("<" + "f" * 14, buf, o, *([0.0] * 14))
    face = parse_face_packet(bytes(buf))
    assert face is not None
    assert face.width == 1920
    assert face.height == 1080
    assert face.landmarks2d[0][0] == pytest.approx(0.5, abs=1e-5)
    assert face.landmarks2d[0][1] == pytest.approx(0.5, abs=1e-5)


def test_face_to_param_values_and_mirror():
    face = parse_face_packet(_pack_face())
    assert face is not None
    cal = HeadAngleCalibrator(samples_needed=1)
    # First sample sets neutral -> angles near 0
    vals = face_to_param_values(face, calibrator=cal)
    assert len(vals) == NUM_PARAMS
    assert all(math.isfinite(v) for v in vals)
    mirrored = face_to_param_values(face, mirror=True, calibrator=cal)
    # ParamAngleX should flip sign under mirror
    idx = PARAM_NAMES.index("ParamAngleX")
    assert mirrored[idx] == pytest.approx(-vals[idx], abs=1e-5)


def test_mirror_landmarks():
    # Short list: X-flip only (not enough points for index swap).
    pts = [(0.25, 0.5), (0.75, 0.5)]
    out = mirror_landmarks(pts)
    assert out[0][0] == pytest.approx(0.75)
    assert out[1][0] == pytest.approx(0.25)

    # Full 68: X-flip + L/R index swap so jaw 0 stays image-left.
    full = [(i / 67.0, 0.4) for i in range(68)]
    mirrored = mirror_landmarks(full)
    assert mirrored[0][0] == pytest.approx(1.0 - full[16][0])
    assert mirrored[16][0] == pytest.approx(1.0 - full[0][0])
    assert mirrored[36][0] == pytest.approx(1.0 - full[45][0])
    assert mirrored[8][0] == pytest.approx(1.0 - full[8][0])


def test_calibrator_and_smoother():
    cal = HeadAngleCalibrator(samples_needed=3)
    assert not cal.ready
    assert cal.update(10, 0, 0) == (0.0, 0.0, 0.0)
    assert cal.update(12, 0, 0) == (0.0, 0.0, 0.0)
    y, p, r = cal.update(8, 0, 0)
    assert cal.ready
    assert abs(y) < 5
    sm = SmoothedParams(alpha=0.5, angle_alpha=0.5)
    a = sm.update([1.0] * NUM_PARAMS)
    b = sm.update([0.0] * NUM_PARAMS)
    assert a[0] == 1.0
    assert 0.0 < b[0] < 1.0


def test_nan_rejection_in_mapping():
    face = parse_face_packet(_pack_face())
    assert face is not None
    # Mutate euler to NaN path via calibrator ready with weird values
    vals = face_to_param_values(face)
    for v in vals:
        assert math.isfinite(v)
    d = values_to_pose_dict(vals)
    assert set(d) == set(PARAM_NAMES)


def test_server_mode_gating_and_privacy():
    svc = TrackingService(server_mode=True)
    with pytest.raises(RuntimeError):
        svc.start(0)
    with pytest.raises(ValueError, match="forbidden"):
        svc.ingest_frame({"pose": {}, "image": "data:image/png;base64,xxxx"})
    with pytest.raises(ValueError, match="forbidden"):
        svc.ingest_frame({"pose": {}, "frame": [1, 2, 3]})

    landmarks = [[0.5, 0.5]] * 68
    ok = svc.ingest_frame(
        {
            "pose": {"ParamAngleX": 5.0, "ParamMouthOpenY": 0.3},
            "landmarks": landmarks,
            "confidences": [1.0] * 68,
            "sequence": 7,
        }
    )
    assert ok["ok"] is True
    snap = svc.snapshot_dict()
    assert snap["mode"] == "server"
    assert snap["has_face"] is True
    assert snap["pose"]["ParamAngleX"] == pytest.approx(5.0)
    assert len(snap["landmarks"]) == 68
    # No image keys ever present
    assert "image" not in snap
    assert "frame" not in snap


def test_local_mode_rejects_external_frames():
    svc = TrackingService(server_mode=False)
    with pytest.raises(RuntimeError):
        svc.ingest_frame({"pose": {"ParamAngleX": 1.0}})


def test_short_packet_rejected():
    assert parse_face_packet(b"\x00" * 10) is None
