"""Freeing what the singing models hold: by hand, and after a while without singing."""

from __future__ import annotations

import time

import numpy as np

from conftest import wait_job
from stellaria_backend.engine import render_client
from synth import write
from test_creation import _trained_model, import_media, put_canvas, render_node, source_node


def _analyze(client, tmp_path) -> None:
    """A pitch analysis goes through the render worker (it loads its models)."""
    ctx = client.ctx
    cid, mid = _trained_model(client, ctx)
    p = client.post("/api/projects", json={"name": "mem"}).json()
    sr = 44100
    t = np.arange(sr) / sr
    voice = 0.3 * np.sin(2 * np.pi * 220 * t)
    media_id = import_media(client, p["id"], write(tmp_path / "v.wav", np.stack([voice] * 2, axis=1), sr))
    put_canvas(client, p["id"], [source_node(media_id), render_node("nd_r", "mix", characterId=cid, modelId=mid)])
    job = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/pitch/analyze").json()["id"])
    assert job["status"] == "succeeded", job


def test_release_by_hand(client, tmp_path):
    status = client.get("/api/system/render").json()
    assert status == {"loaded": False, "busy": False, "device": None, "idleSeconds": None, "releaseAfterMin": 10}
    assert client.post("/api/system/render/release").json()["released"] is False  # nothing to free

    _analyze(client, tmp_path)
    status = client.get("/api/system/render").json()
    assert status["loaded"] is True and status["busy"] is False and status["device"] and status["idleSeconds"] >= 0
    out = client.post("/api/system/render/release").json()
    assert out["released"] is True and out["loaded"] is False
    assert client.ctx.render.loaded is False

    # the next job simply starts the worker again
    _analyze(client, tmp_path)
    assert client.get("/api/system/render").json()["loaded"] is True


def test_released_after_idle_minutes(client, tmp_path, monkeypatch):
    monkeypatch.setattr(render_client, "IDLE_CHECK_S", 0.05)
    assert client.put("/api/preferences", json={"gpuIdleReleaseMin": 0}).json()["gpuIdleReleaseMin"] == 0
    _analyze(client, tmp_path)
    render = client.ctx.render
    render._last_used -= 3600  # an hour without singing
    time.sleep(0.3)
    assert render.loaded is True  # 0 = keep them loaded

    client.put("/api/preferences", json={"gpuIdleReleaseMin": 5})
    render._last_used = time.monotonic() - 4 * 60
    time.sleep(0.3)
    assert render.loaded is True  # not yet
    render._last_used -= 2 * 60
    deadline = time.time() + 3
    while render.loaded and time.time() < deadline:
        time.sleep(0.05)
    assert render.loaded is False
    assert client.put("/api/preferences", json={"gpuIdleReleaseMin": -1}).status_code == 422
