"""API smoke tests: auth, system, preferences, CRUD, clips editing, projects, WebSocket events."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from conftest import TOKEN, wait_job
from synth import song, write


def test_auth_and_health(client):
    from fastapi.testclient import TestClient

    anon = TestClient(client.app)
    assert anon.get("/api/health").json() == {"ok": True, "version": "0.1.0"}
    r = anon.get("/api/system")
    assert r.status_code == 401 and r.json()["error"]["code"] == "auth.invalid_token"
    assert anon.get(f"/api/preferences?token={TOKEN}").status_code == 200
    assert anon.get("/api/preferences", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_system_and_preferences(client):
    info = client.get("/api/system").json()
    assert info["fakeEngine"] is True and info["canTrain"] is True and info["trainBlockers"] == []
    assert "星籁 库" in info["libraryPath"] and info["os"] in ("linux", "windows", "macos")
    assert info["torch"]["device"] == "cpu" and info["cpuCount"] >= 1 and info["ramTotal"] > 0
    gpus = client.get("/api/system/gpu").json()
    assert gpus and gpus[0]["name"] == "Stellaria Fake GPU" and gpus[0]["vramTotal"] > 0
    prefs = client.get("/api/preferences").json()
    assert prefs["language"] == "zh-CN" and prefs["gpuPolicy"] == "smart" and prefs["mirrorRegion"] == "auto"
    prefs = client.put("/api/preferences", json={"language": "en", "gpuPolicy": "cpu"}).json()
    assert prefs["language"] == "en" and prefs["gpuPolicy"] == "cpu" and prefs["glass"] is True
    bad = client.put("/api/preferences", json={"language": "fr"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "request.invalid"
    presets = client.get("/api/train/presets").json()
    assert [p["id"] for p in presets] == ["vram6", "vram8", "vram12", "vram24"]
    assert next(p for p in presets if p["recommended"])["id"] == "vram8"  # the fake GPU has 8 GB
    disk = client.get("/api/disk").json()
    assert disk["libraryTotal"] > 0 and disk["characters"] == []


def test_errors_have_the_contract_shape(client):
    r = client.get("/api/characters/chr_missing")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "character_not_found", "message": "Character 'chr_missing' not found", "detail": "", "params": {}}}
    r = client.post("/api/characters", json={})
    assert r.status_code == 422 and r.json()["error"]["code"] == "request.invalid"
    assert client.get("/api/nope").json()["error"]["code"] == "not_found"


def test_character_crud_and_avatar(client):
    c = client.post("/api/characters", json={"name": "星 A", "tags": ["alto"]}).json()
    assert c["stage"] == "empty" and c["health"] is None and c["modelIds"] == [] and c["color"].startswith("#")
    assert c["stats"] == {"rawCount": 0, "clipCount": 0, "activeDuration": 0.0, "trainCount": 0, "valCount": 0,
                          "trashedCount": 0}
    c = client.patch(f"/api/characters/{c['id']}", json={"notes": "n", "consent": {"source": "me", "authorized": True,
                                                                                    "note": ""}, "pinned": True}).json()
    assert c["notes"] == "n" and c["consent"]["authorized"] and c["pinned"] and c["name"] == "星 A"
    assert client.post(f"/api/characters/{c['id']}/opened").json()["lastOpenedAt"]
    buf = io.BytesIO()
    Image.new("RGB", (640, 320), (200, 40, 90)).save(buf, format="JPEG")
    c = client.put(f"/api/characters/{c['id']}/avatar", files={"file": ("a.jpg", buf.getvalue(), "image/jpeg")}).json()
    assert c["hasAvatar"] and c["avatarVersion"] == 1
    img = Image.open(io.BytesIO(client.get(f"/api/characters/{c['id']}/avatar?v=1").content))
    assert img.size == (256, 256)
    client.patch(f"/api/characters/{c['id']}", json={"archived": True})
    assert client.get("/api/characters?archived=0").json() == []
    assert len(client.get("/api/characters?archived=1").json()) == 1
    assert client.delete(f"/api/characters/{c['id']}").json() == {"ok": True}
    assert client.get("/api/characters").json() == []


def test_clip_editing(client, tmp_path):
    cid = client.post("/api/characters", json={"name": "Edit"}).json()["id"]
    files = [write(tmp_path / f"t{i}.wav", song(6, 220, seed=i, gap=0.2)) for i in range(4)]
    job = client.post(f"/api/characters/{cid}/raw", json={"paths": [str(p) for p in files]}).json()
    assert wait_job(client, job["id"])["status"] == "succeeded"
    job = client.post(f"/api/characters/{cid}/organize", json={"slice": {"enabled": False}}).json()
    assert wait_job(client, job["id"])["status"] == "succeeded"
    clips = client.get(f"/api/characters/{cid}/clips").json()
    assert len(clips) == 4
    first = clips[0]
    audio = client.get(f"/api/characters/{cid}/clips/{first['id']}/audio", headers={"Range": "bytes=0-99"})
    assert audio.status_code == 206 and len(audio.content) == 100
    peaks = client.get(f"/api/characters/{cid}/clips/{first['id']}/peaks?n=50").json()
    assert len(peaks["peaks"]) == 50 and 0 < max(peaks["peaks"]) <= 1 and abs(peaks["duration"] - 6) < 0.01

    parts = client.post(f"/api/characters/{cid}/clips/{first['id']}/split", json={"at": 2.5}).json()
    assert [p["name"] for p in parts] == [f"{first['name']}_a", f"{first['name']}_b"]
    assert abs(parts[0]["duration"] - 2.5) < 0.01 and abs(parts[1]["start"] - (first["start"] + 2.5)) < 0.01
    trashed = client.get(f"/api/characters/{cid}/clips?status=trashed").json()
    assert [c["id"] for c in trashed] == [first["id"]]
    assert client.get(f"/api/characters/{cid}").json()["health"]["stale"] is True

    renamed = client.patch(f"/api/characters/{cid}/clips", json={"ids": [parts[0]["id"]], "set": {"name": "改名"}}).json()
    assert renamed[0]["name"] == "改名" and renamed[0]["file"] == "clips/改名.wav"
    conflict = client.patch(f"/api/characters/{cid}/clips", json={"ids": [parts[1]["id"]], "set": {"name": "改名"}})
    assert conflict.status_code == 409 and conflict.json()["error"]["code"] == "clip.name_taken"
    bad = client.patch(f"/api/characters/{cid}/clips", json={"ids": [parts[1]["id"]], "set": {"name": "a/b"}})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "clip.bad_name"
    client.patch(f"/api/characters/{cid}/clips", json={"ids": [parts[1]["id"]], "set": {"split": "val", "reviewed": True}})

    renum = client.post(f"/api/characters/{cid}/clips/rename", json={"pattern": "{source}-{index:03}"}).json()
    pairs = sorted(c["name"].split("-") for c in renum)
    assert [p[0] for p in pairs] == ["t0", "t0", "t1", "t2", "t3"]  # {source} = raw file stem
    assert sorted(p[1] for p in pairs) == ["001", "002", "003", "004", "005"]
    renum = client.post(f"/api/characters/{cid}/clips/rename", json={"pattern": "take-{index:02d}"}).json()
    assert sorted(c["name"] for c in renum) == ["take-01", "take-02", "take-03", "take-04", "take-05"]
    for pattern in ("{name}_{index}", "{name}_{index:04}"):
        assert client.post(f"/api/characters/{cid}/clips/rename", json={"pattern": pattern}).status_code == 200
    bad = client.post(f"/api/characters/{cid}/clips/rename", json={"pattern": "{name}"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "organize.bad_pattern"
    renum = client.post(f"/api/characters/{cid}/clips/rename", json={"pattern": "take-{index:02d}"}).json()
    for c in renum:
        assert (client.ctx.characters.dir(cid) / c["file"]).is_file()
    assert client.post(f"/api/characters/{cid}/clips/empty-trash").json() == {"deleted": 1}
    picked = client.post(f"/api/characters/{cid}/clips/pick-val", json={"count": 2}).json()
    assert sum(c["split"] == "val" for c in picked) == 2

    raw_id = client.get(f"/api/characters/{cid}/raw").json()[0]["id"]
    client.delete(f"/api/characters/{cid}/raw/{raw_id}?withClips=1")
    assert len(client.get(f"/api/characters/{cid}/raw").json()) == 3
    assert all(c["rawId"] != raw_id for c in client.get(f"/api/characters/{cid}/clips?status=all").json())


def test_models_crud(client):
    a = client.post("/api/characters", json={"name": "A"}).json()["id"]
    b = client.post("/api/characters", json={"name": "B"}).json()["id"]
    m = client.post("/api/models", json={"characterIds": [a, b], "config": {"nLayers": 3}}).json()
    assert m["kind"] == "constellation" and m["name"] == "A & B" and m["status"] == "idle"
    assert m["characters"] == [{"characterId": a, "spkId": 1}, {"characterId": b, "spkId": 2}]
    assert m["config"]["nLayers"] == 3 and m["config"]["preset"] == "vram8" and m["targetSteps"] == 100000
    assert m["dataset"]["status"] == "none" and m["checkpoints"] == [] and m["diskUsage"]["total"] >= 0
    m = client.patch(f"/api/models/{m['id']}", json={"name": "Duo", "targetSteps": 5000, "config": {"lr": 0.001}}).json()
    assert m["name"] == "Duo" and m["targetSteps"] == 5000 and m["config"]["targetSteps"] == 5000 and m["config"]["lr"] == 0.001
    bad = client.patch(f"/api/models/{m['id']}", json={"config": {"ampDtype": "fp8"}})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "model.bad_config"
    assert [x["id"] for x in client.get(f"/api/models?characterId={a}").json()] == [m["id"]]
    assert client.get(f"/api/characters/{a}").json()["modelIds"] == [m["id"]]
    assert client.get(f"/api/models/{m['id']}/metrics").json() == []
    assert client.get(f"/api/models/{m['id']}/validation").json() is None
    assert client.get(f"/api/models/{m['id']}/log").json() == {"lines": []}
    r = client.post(f"/api/models/{m['id']}/train")
    assert r.status_code == 409 or wait_job(client, r.json()["id"])["error"]["code"] == "dataset.empty"
    assert client.delete(f"/api/models/{m['id']}").json() == {"ok": True}
    assert client.get(f"/api/characters/{a}").json()["modelIds"] == []


def test_projects_and_media(client, tmp_path):
    p = client.post("/api/projects", json={"name": "Song"}).json()
    assert p["canvas"] == {"nodes": [], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}
    stereo = np.stack([song(8, 220, seed=1), song(8, 330, seed=2)], axis=1)
    path = write(tmp_path / "歌 曲.flac", stereo)
    job = client.post(f"/api/projects/{p['id']}/media", json={"paths": [str(path)]}).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded"
    mid = done["result"]["mediaIds"][0]
    media = client.get(f"/api/projects/{p['id']}/media").json()
    assert media[0]["channels"] == 2 and media[0]["sampleRate"] == 44100 and media[0]["name"] == "歌 曲"
    pitch = client.get(f"/api/projects/{p['id']}/media/{mid}/pitch").json()
    assert pitch["hop"] == 0.01 and len(pitch["f0"]) > 700 and max(pitch["f0"]) > 150
    assert len(client.get(f"/api/projects/{p['id']}/media/{mid}/peaks?n=100").json()["peaks"]) == 100

    canvas = {"nodes": [{"id": "nd_src", "type": "source", "position": {"x": 0, "y": 0},
                         "data": {"mediaId": mid, "name": "歌 曲"}}], "edges": [], "viewport": {"x": 1, "y": 2, "zoom": 1}}
    timeline = {"tracks": [{"id": "trk_1", "name": "v", "kind": "vocal", "gainDb": -3, "pan": 0.2, "mute": False,
                            "solo": False, "clips": [{"id": "tc_1", "mediaId": mid, "nodeId": None, "start": 1.0,
                                                      "offset": 0.5, "duration": 4.0, "gainDb": 0, "fadeIn": 0.1,
                                                      "fadeOut": 0.2}]}], "length": 6.0}
    saved = client.put(f"/api/projects/{p['id']}", json={"canvas": canvas, "timeline": timeline}).json()
    assert saved["canvas"]["viewport"]["x"] == 1 and saved["timeline"]["length"] == 6.0
    summary = client.get("/api/projects").json()[0]
    assert summary["id"] == p["id"] and summary["renderCount"] == 0

    out = tmp_path / "mix out"
    out.mkdir()
    job = client.post(f"/api/projects/{p['id']}/mixdown", json={"format": "mp3", "path": str(out), "normalize": True}).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done
    assert done["result"]["path"].endswith("Song.mp3") and (out / "Song.mp3").stat().st_size > 10000
    mix = next(m for m in client.get(f"/api/projects/{p['id']}/media").json() if m["kind"] == "mixdown")
    assert abs(mix["duration"] - 6.0) < 0.01 and mix["channels"] == 2

    dup = client.post(f"/api/projects/{p['id']}/duplicate").json()
    assert dup["id"] != p["id"] and dup["name"] == "Song (copy)" and len(client.get(f"/api/projects/{dup['id']}/media").json()) == 2
    key = client.get(f"/api/projects/{p['id']}/suggest-key?mediaId={mid}&characterId=chr_nope")
    assert key.status_code == 404
    assert client.delete(f"/api/projects/{dup['id']}").json() == {"ok": True}


def test_websocket_events(client):
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        assert ws.receive_json() == {"type": "hello", "data": {"version": "0.1.0"}}
        ws.send_json({"type": "subscribe", "topics": ["gpu"]})
        client.post("/api/characters", json={"name": "WS"})
        seen = set()
        for _ in range(20):
            msg = ws.receive_json()
            seen.add(msg["type"])
            if {"character", "gpu"} <= seen:
                break
        assert {"character", "gpu"} <= seen
        client.put("/api/preferences", json={"glass": False})
        while (msg := ws.receive_json())["type"] != "preferences":
            pass
        assert msg["data"]["glass"] is False and "starfield" not in msg["data"]


def test_websocket_requires_token(client):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect) as err:
        with TestClient(client.app).websocket_connect("/api/events") as ws:
            ws.receive_json()
    assert err.value.code == 4401


def test_job_log_subscription(client, tmp_path):
    cid = client.post("/api/characters", json={"name": "L"}).json()["id"]
    path = write(tmp_path / "x.wav", song(6, 220))
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        ws.receive_json()
        job = client.post(f"/api/characters/{cid}/raw", json={"paths": [str(path)]}).json()
        ws.send_json({"type": "subscribe", "topics": [f"job.log:{job['id']}"]})
        job_events = []
        while True:
            msg = ws.receive_json()
            if msg["type"] == "job" and msg["data"]["id"] == job["id"]:
                job_events.append(msg["data"]["status"])
                if msg["data"]["status"] == "succeeded":
                    break
        assert job_events[-1] == "succeeded"
    log = client.get(f"/api/jobs/{job['id']}/log").json()
    assert isinstance(log["lines"], list)
    assert client.get("/api/jobs?active=1").json() == []
    assert client.get("/api/jobs").json()[0]["id"] == job["id"]


def test_preferences_store_only_explicit_choices(tmp_path):
    import json

    from stellaria_backend.events import EventHub
    from stellaria_backend.preferences import PreferencesStore

    path = tmp_path / "preferences.json"
    store = PreferencesStore(path, EventHub())
    assert store.get().gpu_policy == "smart"
    store.update({"language": "en", "unknown": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"language": "en"}  # defaults stay defaults
    # a file written by an older version holds every key: an explicit 'queue' cannot be told apart
    path.write_text(json.dumps({"language": "zh-CN", "gpuPolicy": "queue", "glass": True, "starfield": "off"}),
                    encoding="utf-8")
    old = PreferencesStore(path, EventHub())
    assert old.get().gpu_policy == "queue" and "starfield" not in old.get().wire()
    old.update({"glass": False})
    assert "starfield" not in json.loads(path.read_text(encoding="utf-8"))
