"""API tests for tracking mode gating + privacy."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.api as api_mod  # noqa: E402
from backend.tracking import set_server_mode  # noqa: E402


@pytest.fixture()
def client_local():
    api_mod.configure_runtime(server_mode=False)
    set_server_mode(False)
    with TestClient(api_mod.app) as client:
        # Re-assert after startup hook
        api_mod.configure_runtime(server_mode=False)
        yield client


@pytest.fixture()
def client_server():
    api_mod.configure_runtime(server_mode=True)
    set_server_mode(True)
    with TestClient(api_mod.app) as client:
        api_mod.configure_runtime(server_mode=True)
        yield client


def test_status_local_mode(client_local):
    api_mod.configure_runtime(server_mode=False)
    r = client_local.get("/api/status")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "local"
    assert body["defaults"]["prompt"] == ""
    assert "lowres" in body["defaults"]["negative"]


def test_status_server_mode(client_server):
    api_mod.configure_runtime(server_mode=True)
    r = client_server.get("/api/status")
    assert r.json()["mode"] == "server"
    assert r.json()["server_mode"] is True


def test_local_rejects_external_frame(client_local):
    api_mod.configure_runtime(server_mode=False)
    r = client_local.post("/api/tracking/frame", json={"pose": {"ParamAngleX": 1}})
    assert r.status_code == 400


def test_server_rejects_camera_start(client_server):
    api_mod.configure_runtime(server_mode=True)
    r = client_server.post("/api/tracking/start", json={})
    assert r.status_code == 400


def test_server_accepts_json_frame_and_rejects_image(client_server):
    api_mod.configure_runtime(server_mode=True)
    bad = client_server.post(
        "/api/tracking/frame",
        json={"pose": {}, "image": "data:image/png;base64,aaa"},
    )
    assert bad.status_code == 400

    landmarks = [[0.4, 0.5]] * 68
    ok = client_server.post(
        "/api/tracking/frame",
        json={
            "pose": {"ParamAngleX": -3.0, "ParamMouthOpenY": 0.4},
            "landmarks": landmarks,
            "confidences": [0.9] * 68,
            "sequence": 3,
        },
    )
    assert ok.status_code == 200
    st = client_server.get("/api/tracking/status")
    body = st.json()
    assert body["has_face"] is True
    assert body["pose"]["ParamAngleX"] == pytest.approx(-3.0)
    assert "image" not in body
    assert len(body["landmarks"]) == 68


def test_generate_request_empty_prompt_default():
    req = api_mod.GenerateRequest()
    assert req.prompt == ""
    assert "lowres" in req.negative
