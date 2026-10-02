"""The whole product flow against ``--fake-engine`` (real subprocesses, real tfevents, no torch)."""

from __future__ import annotations

import base64
import io
import json
import sys
import time

import numpy as np
import psutil
import soundfile as sf

from conftest import TOKEN, make_ctx, wait_for, wait_job
from stellaria_backend.engine import dataset
from stellaria_backend.engine.checkpoints import latest_checkpoint
from stellaria_backend.engine.commands import engine_env, engine_script
from stellaria_backend.engine.config import default_config, merge_config
from stellaria_backend.library.views import model_view
from stellaria_backend.util import procs
from synth import song, write
from test_dataset_config import add_clips


def model_of(client, mid):
    return client.get(f"/api/models/{mid}").json()


def test_full_flow(client, material_dir, tmp_path):
    ctx = client.ctx
    ctx.options.fake_speed = 120.0

    # ---- material → health ---------------------------------------------------------------
    cid = client.post("/api/characters", json={"name": "星夜"}).json()["id"]
    job = client.post(f"/api/characters/{cid}/raw", json={"paths": [str(material_dir)]}).json()
    assert len(wait_job(client, job["id"])["result"]["rawIds"]) == 4
    job = client.post(f"/api/characters/{cid}/organize", json={}).json()
    done = wait_job(client, job["id"], timeout=240)
    assert done["status"] == "succeeded", done
    report = client.get(f"/api/characters/{cid}/health").json()
    assert not report["blocking"], report["issues"]
    assert {"data.little", "checks.music_unavailable"} <= {i["code"] for i in report["issues"]}
    character = client.get(f"/api/characters/{cid}").json()
    n_val = character["stats"]["valCount"]
    assert character["stage"] == "ready" and n_val == max(2, character["stats"]["clipCount"] // 20)

    # ---- model + training with pause/resume ----------------------------------------------
    m = client.post("/api/models", json={"characterIds": [cid], "config": {
        "targetSteps": 600, "intervalVal": 100, "intervalForceSave": 200, "encoder": "hubertsoft"}}).json()
    mid = m["id"]
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        ws.receive_json()
        job = client.post(f"/api/models/{mid}/train", json={}).json()
        assert job["type"] == "train" and job["usesGpu"] is True
        types = set()
        while not {"metrics", "validation", "model"} <= types:
            types.add(ws.receive_json()["type"])
    assert client.get(f"/api/characters/{cid}").json()["stage"] == "training"
    wait_for(lambda: model_of(client, mid)["step"] >= 150, 30)
    m = model_of(client, mid)
    assert m["status"] == "training" and m["speed"] and m["eta"] is not None and m["dataset"]["status"] == "ready"
    paused = client.post(f"/api/models/{mid}/pause").json()
    assert paused["status"] in ("stopping", "paused")
    done = wait_job(client, job["id"], timeout=30)
    assert done["status"] == "succeeded" and done["result"]["paused"] is True
    m = model_of(client, mid)
    pause_step = m["step"]
    assert m["status"] == "paused" and pause_step % 10 == 0
    kinds = {c["step"]: c["kind"] for c in m["checkpoints"]}
    expected = {s: "permanent" if s % 200 == 0 else "regular" if s % 100 == 0 else "pause" for s in kinds}
    assert kinds == expected and max(kinds) == pause_step
    assert pause_step % 100 == 0 or kinds[pause_step] == "pause"
    log = client.get(f"/api/models/{mid}/log?tail=50").json()["lines"]
    assert any("training stopped" in line for line in log)

    job = client.post(f"/api/models/{mid}/train", json={}).json()
    done = wait_job(client, job["id"], timeout=60)
    assert done["status"] == "succeeded" and done["result"]["done"] is True, done
    m = model_of(client, mid)
    assert m["status"] == "done" and m["step"] == 600 and m["trainingTime"] > 0
    steps = {c["step"]: c["kind"] for c in m["checkpoints"]}
    assert steps == {200: "permanent", 400: "permanent", 600: "permanent"}  # 500 deleted by the engine, pause pruned
    assert client.get(f"/api/characters/{cid}").json()["stage"] == "trained"

    # ---- metrics & validation -------------------------------------------------------------
    series = {s["tag"]: s["points"] for s in client.get(f"/api/models/{mid}/metrics").json()}
    assert {"train/loss", "train/lr", "validation/loss", "validation/mel_val_snr"} <= set(series)
    assert series["train/loss"][-1][0] == 600 and len(series["train/loss"]) >= 590
    assert series["train/loss"][0][1] > series["train/loss"][-1][1]  # decaying
    small = client.get(f"/api/models/{mid}/metrics?tags=train/loss&since=500&maxPoints=20").json()
    assert len(small) == 1 and all(p[0] > 500 for p in small[0]["points"]) and len(small[0]["points"]) <= 21
    assert client.get(f"/api/models/{mid}/validation/steps").json() == [100, 200, 300, 400, 500, 600]
    snap = client.get(f"/api/models/{mid}/validation").json()
    assert snap["step"] == 600 and len(snap["samples"]) == n_val
    sample = snap["samples"][0]
    gt = client.get("/api" + sample["gtUrl"])
    assert gt.status_code == 200 and gt.headers["content-type"] == "audio/wav"
    pred, sr = sf.read(io.BytesIO(client.get("/api" + sample["predUrl"]).content))
    assert sr == 44100 and len(pred) > 44100
    mel = client.get("/api" + sample["melUrl"]).json()
    assert mel["bins"] == 128 and len(base64.b64decode(mel["data"])) == mel["frames"] * 128
    assert client.get(f"/api/models/{mid}/validation?step=300").json()["step"] == 300

    # ---- releases & quick render ------------------------------------------------------------
    rel = client.post(f"/api/models/{mid}/releases", json={"step": 600, "name": "v1"}).json()
    assert rel["name"] == "v1" and rel["size"] > 0
    rel = client.patch(f"/api/models/{mid}/releases/{rel['id']}", json={"notes": "好听"}).json()
    assert rel["notes"] == "好听"
    m = model_of(client, mid)
    assert next(c for c in m["checkpoints"] if c["step"] == 600)["releaseId"] == rel["id"]
    probe = write(tmp_path / "试听.mp3", song(4, 200, seed=7))
    job = client.post(f"/api/models/{mid}/quick-render", json={"path": str(probe), "characterId": cid,
                                                              "params": {"key": 3}}).json()
    done = wait_job(client, job["id"], timeout=60)
    assert done["status"] == "succeeded", done
    url = done["result"]["url"]
    assert url == f"/models/{mid}/quick-render/{job['id']}/audio"
    out, sr = sf.read(io.BytesIO(client.get("/api" + url).content))
    assert sr == 44100 and abs(len(out) / sr - 4.0) < 0.05
    peaks = client.get("/api" + url.replace("/audio", "/peaks") + "?n=64").json()
    assert len(peaks["peaks"]) == 64 and abs(peaks["duration"] - 4.0) < 0.05

    # ---- creation: render node, add-on, separation, key ------------------------------------
    p = client.post("/api/projects", json={"name": "翻唱"}).json()
    song_path = write(tmp_path / "song.wav", song(6, 180, seed=8))
    media_id = wait_job(client, client.post(f"/api/projects/{p['id']}/media", json={"paths": [str(song_path)]}).json()["id"])["result"]["mediaIds"][0]
    render_data = {"input": {"nodeId": "nd_src", "port": "mix"}, "characterId": cid, "modelId": mid, "releaseId": rel["id"],
                   "params": {"key": 2, "formant": 0, "register": 0, "f0Extractor": "rmvpe", "steps": 20, "method": "euler",
                              "tStart": 0.0, "threshold": -60, "spkMix": None, "region": {"start": 1.0, "end": 4.0}},
                   "mediaId": None, "status": "draft", "jobId": None, "error": None, "rating": 0, "favorite": False, "label": "试唱"}
    canvas = {"nodes": [{"id": "nd_src", "type": "source", "position": {"x": 0, "y": 0}, "data": {"mediaId": media_id, "name": "song"}},
                        {"id": "nd_r", "type": "render", "position": {"x": 400, "y": 0}, "data": render_data}],
              "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}
    client.put(f"/api/projects/{p['id']}", json={"canvas": canvas})
    job = client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()
    done = wait_job(client, job["id"], timeout=60)
    assert done["status"] == "succeeded", done
    node = next(n for n in client.get(f"/api/projects/{p['id']}").json()["canvas"]["nodes"] if n["id"] == "nd_r")
    assert node["data"]["status"] == "done" and node["data"]["mediaId"] == done["result"]["mediaId"]
    assert node["data"]["rating"] == 0 and node["data"]["label"] == "试唱"
    rendered = next(x for x in client.get(f"/api/projects/{p['id']}/media").json() if x["kind"] == "render")
    assert abs(rendered["duration"] - 3.0) < 0.05
    # a stale canvas saved by the frontend must not undo the server-side update
    client.put(f"/api/projects/{p['id']}", json={"canvas": canvas})
    node = next(n for n in client.get(f"/api/projects/{p['id']}").json()["canvas"]["nodes"] if n["id"] == "nd_r")
    assert node["data"]["status"] == "done"
    assert client.get("/api/projects").json()[0]["renderCount"] == 1

    sep = client.post(f"/api/projects/{p['id']}/nodes/nd_src/separate")
    assert sep.status_code == 409 and sep.json()["error"]["code"] == "addon.missing"
    job = client.post("/api/resources/separator/install").json()
    assert job["type"] == "install_addon" and wait_job(client, job["id"])["status"] == "succeeded"
    assert next(r for r in client.get("/api/resources").json() if r["id"] == "separator")["status"] == "ready"
    job = client.post(f"/api/projects/{p['id']}/nodes/nd_src/separate").json()
    done = wait_job(client, job["id"], timeout=60)
    assert done["status"] == "succeeded", done
    stems = done["result"]["stems"]
    assert stems["status"] == "done" and stems["vocals"] and stems["inst"] and done["result"]["nodeId"] == "nd_src"
    nodes = client.get(f"/api/projects/{p['id']}").json()["canvas"]["nodes"]
    assert len(nodes) == 2 and next(n for n in nodes if n["id"] == "nd_src")["data"]["stems"] == stems

    job = client.post(f"/api/characters/{cid}/health", json={"advanced": True}).json()
    assert wait_job(client, job["id"], timeout=120)["status"] == "succeeded"
    report = client.get(f"/api/characters/{cid}/health").json()
    assert report["checks"]["music"] is True and "checks.music_unavailable" not in {i["code"] for i in report["issues"]}
    key = client.get(f"/api/projects/{p['id']}/suggest-key?mediaId={media_id}&characterId={cid}").json()
    assert isinstance(key["key"], int) and key["characterRange"]["median"] and key["confidence"] in ("high", "low")

    # ---- slim, packs -------------------------------------------------------------------------
    export = wait_job(client, client.post(f"/api/models/{mid}/export", json={"releaseId": rel["id"]}).json()["id"])
    assert export["status"] == "succeeded", export
    pack_bytes = client.get("/api" + export["result"]["downloadUrl"]).content
    pack = tmp_path / "导出 model.stella"
    pack.write_bytes(pack_bytes)
    job = client.post(f"/api/models/{mid}/slim", json={"dataset": True, "checkpoints": "keep_latest"}).json()
    assert wait_job(client, job["id"])["result"]["freedBytes"] > 0
    m = model_of(client, mid)
    assert [c["step"] for c in m["checkpoints"]] == [600] and m["dataset"]["status"] == "none"

    client.delete(f"/api/characters/{cid}?deleteModels=1")
    assert client.get("/api/models").json() == []
    job = client.post("/api/packs/import", files={"file": ("x.stella", pack_bytes, "application/zip")}).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done
    assert done["result"]["characterIds"] == [cid] and done["result"]["modelIds"] == [mid]
    imported = model_of(client, mid)
    assert imported["releases"][0]["id"] == rel["id"] and imported["status"] == "idle"
    assert client.get(f"/api/characters/{cid}").json()["name"] == "星夜"
    job = client.post(f"/api/models/{mid}/quick-render", json={"path": str(probe), "characterId": cid,
                                                              "releaseId": rel["id"]}).json()
    assert wait_job(client, job["id"], timeout=60)["status"] == "succeeded"
    assert "torch" not in sys.modules  # the API process never imports torch


def test_character_pack_roundtrip_with_material(client, tmp_path):
    cid = client.post("/api/characters", json={"name": "Pack 包"}).json()["id"]
    files = [write(tmp_path / f"p{i}.wav", song(5, 230, seed=20 + i, gap=0.2)) for i in range(3)]
    wait_job(client, client.post(f"/api/characters/{cid}/raw", json={"paths": [str(p) for p in files]}).json()["id"])
    wait_job(client, client.post(f"/api/characters/{cid}/organize", json={"slice": {"enabled": False}}).json()["id"])
    dest = tmp_path / "out dir"
    dest.mkdir()
    done = wait_job(client, client.post(f"/api/characters/{cid}/export", json={"includeMaterial": True, "path": str(dest)}).json()["id"])
    assert done["status"] == "succeeded" and done["result"]["path"].endswith(".stella")

    other = make_ctx(tmp_path / "second machine")
    from fastapi.testclient import TestClient

    from stellaria_backend.app import create_app

    with TestClient(create_app(other.options, ctx=other), headers={"Authorization": f"Bearer {TOKEN}"}) as c2:
        done = wait_job(c2, c2.post("/api/packs/import", json={"path": done["result"]["path"]}).json()["id"])
        assert done["status"] == "succeeded", done
        clips = c2.get(f"/api/characters/{cid}/clips").json()
        assert len(clips) == 3 and all((other.characters.dir(cid) / c["file"]).is_file() for c in clips)
        assert c2.get(f"/api/characters/{cid}").json()["name"] == "Pack 包"
        assert c2.get(f"/api/characters/{cid}/health").json() is not None


def test_reattach_to_running_training(tmp_path):
    """A training that outlives the backend is picked up again on the next start."""
    ctx = make_ctx(tmp_path, fake_speed=60.0)
    cid = ctx.characters.create("R").id
    add_clips(ctx, cid, 6)
    config = merge_config(default_config("vram8"), {"targetSteps": 150, "intervalVal": 50, "intervalForceSave": 100})
    rec = ctx.models.create("r", [cid], config)
    entries = dataset.plan(rec, ctx.characters)
    dataset.assemble(ctx.models.dataset_dir(rec.id), entries)
    ctx.training.write_engine_config(rec.id, entries)
    cmd, cwd = engine_script(ctx, "train_reflow.py", ["-c", str(ctx.models.config_path(rec.id))])
    log = open(ctx.models.log_path(rec.id), "ab")
    proc = procs.popen(cmd, cwd=cwd, env=engine_env(ctx), stdout=log, stderr=log)
    ctx.training.pid_path(rec.id).write_text(json.dumps({"pid": proc.pid, "createTime": psutil.Process(proc.pid).create_time()}))
    rec.status = "training"
    ctx.models.save(rec)
    ctx.stop(timeout=1)  # the "old" backend goes away without touching the process

    ctx2 = make_ctx(tmp_path)
    ctx2.start()
    job = wait_for(lambda: ctx2.training.active_job(rec.id), 5)
    wait_for(lambda: job.status == "running", 5)
    assert ctx2.models.get(rec.id).status == "training"
    wait_for(lambda: not job.active, 30)
    proc.wait(5)
    log.close()
    assert job.status == "succeeded" and job.result == {"done": True, "step": 150, "reattached": True}
    view = model_view(ctx2, rec.id)
    assert view.status == "done" and view.step == 150 and not ctx2.training.pid_path(rec.id).exists()
    ctx2.stop(timeout=2)


def test_shutdown_pauses_training(tmp_path):
    ctx = make_ctx(tmp_path, fake_speed=50.0)
    cid = ctx.characters.create("S").id
    add_clips(ctx, cid, 6)
    rec = ctx.models.create("s", [cid], merge_config(default_config("vram8"), {"targetSteps": 100000, "intervalVal": 1000}))
    job = ctx.training.start(rec.id)
    wait_for(lambda: ctx.training.runtime(rec.id) and ctx.training.runtime(rec.id).step > 20, 30)
    started = time.time()
    ctx.stop(timeout=20)
    assert time.time() - started < 20
    assert job.status in ("succeeded", "cancelled") and ctx.models.get(rec.id).status == "paused"
    assert latest_checkpoint(ctx.models.exp_dir(rec.id))[0] > 20
    assert not np.isnan(ctx.models.get(rec.id).training_time)


def test_render_while_training(client, tmp_path):
    """gpuPolicy decides: 'cpu' renders on the CPU at once, 'smart' next to the training on the
    (fake) GPU, 'queue' waits until the training pauses. releaseId null = latest checkpoint."""
    ctx = client.ctx
    ctx.options.fake_speed = 40.0
    cid = ctx.characters.create("Live").id
    add_clips(ctx, cid, 6)
    mid = client.post("/api/models", json={"characterIds": [cid], "config": {
        "targetSteps": 100000, "intervalVal": 40, "intervalForceSave": 400}}).json()["id"]
    train = client.post(f"/api/models/{mid}/train").json()
    wait_for(lambda: model_of(client, mid)["checkpoints"], 30)

    p = client.post("/api/projects", json={"name": "live"}).json()
    src = write(tmp_path / "in.wav", song(3, 220, seed=3))
    media_id = wait_job(client, client.post(f"/api/projects/{p['id']}/media", json={"paths": [str(src)]}).json()["id"])["result"]["mediaIds"][0]
    node = {"id": "nd_r", "type": "render", "position": {"x": 0, "y": 0}, "data": {
        "input": {"nodeId": "nd_s", "port": "mix"}, "characterId": cid, "modelId": mid, "releaseId": None, "params": {}, "mediaId": None,
        "status": "draft", "jobId": None, "error": None, "rating": 0, "favorite": False, "label": ""}}
    source = {"id": "nd_s", "type": "source", "position": {"x": 0, "y": 0}, "data": {"mediaId": media_id, "name": "in"}}
    client.put(f"/api/projects/{p['id']}", json={"canvas": {"nodes": [source, node], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}})

    for policy, device, stage in (("cpu", "cpu", "stage.render_cpu"), ("smart", "cuda", "stage.render")):
        client.put("/api/preferences", json={"gpuPolicy": policy})
        done = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()["id"], timeout=30)
        assert done["status"] == "succeeded", done
        assert done["result"]["device"] == device and done["stage"]["code"] == stage
        assert client.get(f"/api/jobs/{train['id']}").json()["status"] == "running"
    assert list((ctx.models.cache_dir(mid) / "render").glob("model_*.pt"))  # rendered from a stable copy

    client.put("/api/preferences", json={"gpuPolicy": "queue"})
    job = client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()
    wait_for(lambda: client.get(f"/api/jobs/{job['id']}").json()["status"] == "waiting_gpu", 5)
    client.post(f"/api/models/{mid}/pause")
    done = wait_job(client, job["id"], timeout=30)
    assert done["status"] == "succeeded" and done["result"]["device"] == "cuda"
    assert wait_job(client, train["id"])["result"]["paused"] is True
