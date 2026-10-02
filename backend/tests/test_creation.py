"""Creation v2: canvas migration, stems on sources, render inputs by port, mix jobs,
mix processors, media analysis/export, and local resource import."""

from __future__ import annotations

import hashlib
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf

from conftest import MANIFEST, TOKEN, make_ctx, wait_job
from stellaria_backend.creation.canvas import migrate_canvas
from synth import song, write
from test_dataset_config import add_clips

VIEWPORT = {"x": 0, "y": 0, "zoom": 1}


def import_media(client, pid: str, path: Path) -> str:
    job = client.post(f"/api/projects/{pid}/media", json={"paths": [str(path)]}).json()
    return wait_job(client, job["id"])["result"]["mediaIds"][0]


def put_canvas(client, pid: str, nodes: list[dict]) -> None:
    client.put(f"/api/projects/{pid}", json={"canvas": {"nodes": nodes, "edges": [], "viewport": VIEWPORT}})


def store_canvas(client, pid: str, nodes: list[dict]) -> None:
    """Write a canvas as stored state (server-owned fields included), bypassing the PUT merge."""
    ctx = client.ctx
    with ctx.projects.lock(pid):
        project = ctx.projects.get(pid)
        project.canvas = {"nodes": json.loads(json.dumps(nodes)), "edges": [], "viewport": VIEWPORT}
        migrate_canvas(project.canvas)
        ctx.projects.save(project)


def node_of(client, pid: str, node_id: str) -> dict:
    return next(n for n in client.get(f"/api/projects/{pid}").json()["canvas"]["nodes"] if n["id"] == node_id)


def source_node(media_id: str, node_id: str = "nd_s") -> dict:
    return {"id": node_id, "type": "source", "position": {"x": 0, "y": 0}, "data": {"mediaId": media_id, "name": "song"}}


def render_node(node_id: str, port: str, **data) -> dict:
    return {"id": node_id, "type": "render", "position": {"x": 300, "y": 0}, "data": {
        "input": {"nodeId": "nd_s", "port": port}, "characterId": data.pop("characterId", "chr_x"),
        "modelId": data.pop("modelId", "mdl_x"), "releaseId": None, "params": {}, "mediaId": None, "status": "draft",
        "jobId": None, "error": None, "rating": 0, "favorite": False, "label": "", "stale": False, "batchId": None, **data}}


# ---- migration ------------------------------------------------------------------------------------
V1_CANVAS = {
    "nodes": [
        {"id": "src", "type": "source", "position": {"x": 0, "y": 0}, "data": {"mediaId": "med_song", "name": "song"}},
        {"id": "stv", "type": "stem", "position": {"x": 300, "y": -90},
         "data": {"mediaId": "med_v", "sourceNodeId": "src", "stem": "vocals", "status": "done", "jobId": None}},
        {"id": "sti", "type": "stem", "position": {"x": 300, "y": 90},
         "data": {"mediaId": "med_i", "sourceNodeId": "src", "stem": "instrumental", "status": "done", "jobId": None}},
        {"id": "r1", "type": "render", "position": {"x": 600, "y": 0}, "data": {"inputNodeId": "stv", "characterId": "c",
                                                                             "modelId": "m", "status": "done"}},
        {"id": "r2", "type": "render", "position": {"x": 600, "y": 200}, "data": {"inputNodeId": "src"}},
        {"id": "mx", "type": "mix", "position": {"x": 900, "y": 0}, "data": {"vocals": [{"nodeId": "r1", "gainDb": 0, "pan": 0}]}},
        {"id": "nt", "type": "note", "position": {"x": 0, "y": 400}, "data": {"text": "hi", "color": "#fff"}},
    ],
    "edges": [{"id": "e1", "source": "src", "target": "stv"}, {"id": "e2", "source": "stv", "target": "r1"}],
    "viewport": VIEWPORT,
}


def test_migration_folds_stems_and_is_idempotent():
    canvas = json.loads(json.dumps(V1_CANVAS))
    assert migrate_canvas(canvas) is True
    nodes = {n["id"]: n for n in canvas["nodes"]}
    assert set(nodes) == {"src", "r1", "r2", "mx", "nt"} and canvas["edges"] == []
    assert nodes["src"]["data"]["stems"] == {"status": "done", "vocals": "med_v", "inst": "med_i", "drums": None, "bass": None,
                                             "other": None, "provided": [], "model": None, "jobId": None, "error": None}
    assert (nodes["src"]["data"]["bpm"], nodes["src"]["data"]["key"]) == (None, None)
    assert nodes["r1"]["data"]["input"] == {"nodeId": "src", "port": "vocals"} and "inputNodeId" not in nodes["r1"]["data"]
    assert nodes["r2"]["data"]["input"] == {"nodeId": "src", "port": "mix"}
    assert nodes["r1"]["data"]["stale"] is False and nodes["r1"]["data"]["batchId"] is None
    mix = nodes["mx"]["data"]
    assert (mix["processor"], mix["variants"], mix["outputs"], mix["selected"], mix["status"]) == ("builtin", 1, [], 0, "draft")
    assert mix["inputs"] == [{"id": "r1", "ref": {"nodeId": "r1", "port": "out"}, "role": "lead", "gainDb": 0.0, "pan": 0.0,
                              "mute": False, "solo": False}] and "vocals" not in mix and "backing" not in mix
    snapshot = json.dumps(canvas, sort_keys=True)
    assert migrate_canvas(canvas) is False and json.dumps(canvas, sort_keys=True) == snapshot


def test_only_vocals_stem_keeps_status(tmp_path):
    canvas = json.loads(json.dumps(V1_CANVAS))
    canvas["nodes"] = [n for n in canvas["nodes"] if n["id"] != "sti"]
    migrate_canvas(canvas)
    stems = next(n for n in canvas["nodes"] if n["id"] == "src")["data"]["stems"]
    assert stems["vocals"] == "med_v" and stems["inst"] is None and stems["status"] == "none"


def test_project_is_migrated_on_load_and_saved_once(client):
    p = client.post("/api/projects", json={"name": "旧"}).json()
    path = client.ctx.projects.dir(p["id"]) / "project.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["canvas"] = V1_CANVAS
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = client.get(f"/api/projects/{p['id']}").json()
    assert [n["type"] for n in loaded["canvas"]["nodes"]].count("stem") == 0 and loaded["canvas"]["edges"] == []
    stored = path.read_text(encoding="utf-8")
    assert '"stem"' not in stored and '"inputNodeId"' not in stored
    mtime = path.stat().st_mtime_ns
    client.get(f"/api/projects/{p['id']}")
    client.get("/api/projects")
    assert path.stat().st_mtime_ns == mtime  # nothing left to migrate: not saved again
    client.put(f"/api/projects/{p['id']}", json={"canvas": {**loaded["canvas"], "edges": [{"id": "x", "source": "a", "target": "b"}]}})
    assert json.loads(path.read_text(encoding="utf-8"))["canvas"]["edges"] == []


# ---- stems and render ports ---------------------------------------------------------------------
def _trained_model(client, ctx) -> tuple[str, str]:
    """A character + fake-trained model with a release (renders go through the fake worker)."""
    cid = ctx.characters.create("Port").id
    add_clips(ctx, cid, 6)
    config = {"targetSteps": 20, "intervalVal": 10, "intervalForceSave": 20}
    mid = client.post("/api/models", json={"characterIds": [cid], "config": config}).json()["id"]
    assert wait_job(client, client.post(f"/api/models/{mid}/train").json()["id"], timeout=60)["status"] == "succeeded"
    return cid, mid


def test_separate_into_stems_and_render_each_port(client, tmp_path):
    ctx = client.ctx
    cid, mid = _trained_model(client, ctx)
    p = client.post("/api/projects", json={"name": "ports"}).json()
    media_id = import_media(client, p["id"], write(tmp_path / "song.wav", np.stack([song(4, 220, seed=1)] * 2, axis=1)))
    put_canvas(client, p["id"], [source_node(media_id), render_node("nd_v", "vocals", characterId=cid, modelId=mid, stale=True)])

    missing = client.post(f"/api/projects/{p['id']}/nodes/nd_v/render")
    assert missing.status_code == 409
    assert missing.json()["error"]["code"] == "node.input_missing"

    wait_job(client, client.post("/api/resources/separator/install").json()["id"])
    job = client.post(f"/api/projects/{p['id']}/nodes/nd_s/separate").json()
    assert node_of(client, p["id"], "nd_s")["data"]["stems"]["status"] in ("queued", "running", "done")
    done = wait_job(client, job["id"], timeout=60)
    assert done["status"] == "succeeded", done
    stems = node_of(client, p["id"], "nd_s")["data"]["stems"]
    assert stems["status"] == "done" and stems["vocals"] and stems["inst"] and stems["jobId"] == job["id"]
    assert done["result"] == {"nodeId": "nd_s", "stems": stems, "device": done["result"]["device"]}
    project = client.get(f"/api/projects/{p['id']}").json()
    assert len(project["canvas"]["nodes"]) == 2 and project["canvas"]["edges"] == []  # no new nodes or edges
    kinds = {m["id"]: m["kind"] for m in client.get(f"/api/projects/{p['id']}/media").json()}
    assert kinds[stems["vocals"]] == kinds[stems["inst"]] == "stem"

    for port, expected in (("vocals", stems["vocals"]), ("inst", stems["inst"]), ("mix", media_id)):
        put_canvas(client, p["id"], [node_of(client, p["id"], "nd_s"),
                                     render_node("nd_r", port, characterId=cid, modelId=mid, stale=True)])
        done = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()["id"], timeout=60)
        assert done["status"] == "succeeded", done
        node = node_of(client, p["id"], "nd_r")["data"]
        assert node["status"] == "done" and node["stale"] is False
        out, _ = sf.read(ctx.projects.media_path(p["id"], node["mediaId"]))
        src, _ = sf.read(ctx.projects.media_path(p["id"], expected), always_2d=True)
        # the fake worker renders input.mean(channels) * -2 dB: the output proves which media was used
        assert np.allclose(out, src.mean(axis=1) * 10 ** (-2 / 20), atol=2e-3), port


# ---- mixing --------------------------------------------------------------------------------------
def _mix_project(client, tmp_path, takes: int = 2):
    p = client.post("/api/projects", json={"name": "混音"}).json()
    song_id = import_media(client, p["id"], write(tmp_path / "backing.wav", np.stack([song(6, 110, seed=2)] * 2, axis=1) * 0.5))
    vocal_ids = [import_media(client, p["id"], write(tmp_path / f"v{i}.wav", song(5, 220 + 40 * i, seed=10 + i)))
                 for i in range(takes)]
    nodes = [source_node(song_id)]
    for i, vid in enumerate(vocal_ids):
        nodes.append(render_node(f"nd_r{i}", "mix", status="done", mediaId=vid, label=f"voice {i}"))
    return p, song_id, vocal_ids, nodes


def mix_input(node_id: str, port: str = "out", role: str = "lead", gain: float = 0.0, pan: float = 0.0, **extra) -> dict:
    return {"id": f"{node_id}:{port}:{role}", "ref": {"nodeId": node_id, "port": port}, "role": role, "gainDb": gain,
            "pan": pan, "mute": False, "solo": False, **extra}


def mix_node(vocals: list[str], backing: tuple[str, str] | None = ("nd_s", "mix"), node_id: str = "nd_m", **data) -> dict:
    inputs = [mix_input(v, pan=-0.3 + 0.6 * i, id=f"in{i}") for i, v in enumerate(vocals)]
    if backing:
        inputs.append(mix_input(backing[0], backing[1], "backing", -2.0))
    return {"id": node_id, "type": "mix", "position": {"x": 600, "y": 0}, "data": {
        "inputs": inputs, "processor": "builtin", "preset": "natural",
        "params": {}, "variants": 1, "outputs": [], "selected": 0, "status": "draft", "jobId": None, "error": None,
        "stale": True, "label": "Duet", **data}}


def test_mix_with_builtin_makes_variants(client, tmp_path):
    p, _, _, nodes = _mix_project(client, tmp_path)
    put_canvas(client, p["id"], nodes + [mix_node(["nd_r0", "nd_r1"], variants=3)])
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        ws.receive_json()
        job = client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix").json()
        assert job["type"] == "mix" and job["usesGpu"] is False
        statuses = []
        while "done" not in statuses:
            msg = ws.receive_json()
            if msg["type"] == "project.nodes" and msg["data"]["nodes"][0]["id"] == "nd_m":
                statuses.append(msg["data"]["nodes"][0]["data"]["status"])
    assert statuses[0] == "queued" and "running" in statuses
    done = wait_job(client, job["id"], timeout=120)
    assert done["status"] == "succeeded", done
    outputs = done["result"]["outputs"]
    assert len(outputs) == 3 and outputs[0]["seed"] == 0 and len({o["seed"] for o in outputs}) == 3
    node = node_of(client, p["id"], "nd_m")["data"]
    assert node["outputs"] == outputs and node["selected"] == 0 and node["status"] == "done" and node["stale"] is False
    media = {m["id"]: m for m in client.get(f"/api/projects/{p['id']}/media").json()}
    for o in outputs:
        m = media[o["mediaId"]]
        assert m["kind"] == "mix" and m["channels"] == 2 and abs(m["duration"] - 6.0) < 0.05
        data, _ = sf.read(client.ctx.projects.media_path(p["id"], o["mediaId"]))
        assert np.isfinite(data).all() and np.max(np.abs(data)) <= 1.0
        assert o["lufs"] is None or -40 < o["lufs"] < 0


def test_mix_checks_its_inputs(client, tmp_path):
    p, _, _, nodes = _mix_project(client, tmp_path, takes=1)
    media_id = nodes[1]["data"]["mediaId"]
    nodes[1]["data"].update(status="draft", mediaId=None)
    store_canvas(client, p["id"], nodes + [mix_node(["nd_r0"])])
    r = client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix")
    assert r.status_code == 409 and r.json()["error"] == {"code": "node.input_missing", "message": r.json()["error"]["message"],
                                                          "detail": "", "params": {"nodeId": "nd_r0", "port": "out"}}
    nodes[1]["data"].update(status="done", mediaId=media_id)
    store_canvas(client, p["id"], nodes + [mix_node(["nd_r0"] * 5)])
    r = client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix")
    assert r.status_code == 422 and r.json()["error"]["code"] == "mix.too_many_vocals"
    assert r.json()["error"]["params"] == {"max": 4}
    store_canvas(client, p["id"], nodes + [mix_node(["nd_r0"], backing=("nd_s", "inst"))])
    r = client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix")
    assert r.status_code == 409 and r.json()["error"]["params"] == {"nodeId": "nd_s", "port": "inst"}
    store_canvas(client, p["id"], nodes + [mix_node([], backing=None)])
    assert client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix").json()["error"]["code"] == "mix.no_inputs"
    store_canvas(client, p["id"], nodes + [mix_node([])])
    assert client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix").json()["error"]["code"] == "mix.no_vocals"


def test_processors_list_builtin_first_then_plugins(client, tmp_path, monkeypatch):
    procs = client.get("/api/mix/processors").json()
    builtin = procs[0]
    assert builtin["id"] == "builtin" and builtin["kind"] == "builtin" and builtin["maxVocals"] == 4
    assert {p["id"] for p in builtin["presets"]} >= {"natural", "pop", "airy", "dry"}
    assert {p["id"] for p in builtin["params"]} >= {"vocalLevel", "reverb", "brightness", "width", "targetLufs", "backingGain"}

    plugin = {"id": "sodium", "name": "Sodium", "kind": "plugin", "presets": [], "params": [], "maxVocals": 1,
              "available": True, "error": None}
    monkeypatch.setattr(client.ctx, "plugins", types.SimpleNamespace(mix_processors=lambda: [plugin]), raising=False)
    assert [p["id"] for p in client.get("/api/mix/processors").json()] == ["builtin", "bus", "sodium"]

    calls = []

    def run_plugin(ctx, jctx, plugin_id, *, vocals, backing, input, output, params, names, on_progress):
        calls.append((plugin_id, [names[v] for v in vocals], names[backing], params))
        on_progress(0.5)
        sf.write(output, np.zeros((44100, 2), dtype=np.float32) + 0.1, 44100)
        return output

    package = types.ModuleType("stellaria_backend.plugins")
    runner = types.ModuleType("stellaria_backend.plugins.runner")
    runner.run_plugin = run_plugin
    monkeypatch.setitem(sys.modules, "stellaria_backend.plugins", package)
    monkeypatch.setitem(sys.modules, "stellaria_backend.plugins.runner", runner)
    p, _, _, nodes = _mix_project(client, tmp_path, takes=1)
    put_canvas(client, p["id"], nodes + [mix_node(["nd_r0"], processor="sodium", params={"x": 1})])
    done = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix").json()["id"], timeout=60)
    assert done["status"] == "succeeded", done
    assert calls == [("sodium", ["混音 - voice 0"], "混音 - backing", {"x": 1, "seed": 0})]
    assert node_of(client, p["id"], "nd_m")["data"]["outputs"][0]["mediaId"] == done["result"]["outputs"][0]["mediaId"]

    store_canvas(client, p["id"], nodes + [mix_node(["nd_r0"], processor="nope")])
    r = client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix")
    assert r.status_code == 409 and r.json()["error"]["code"] == "mix.processor_unavailable"


def test_media_analysis_and_export(client, tmp_path):
    p = client.post("/api/projects", json={"name": "Export"}).json()
    mid = import_media(client, p["id"], write(tmp_path / "a.wav", np.stack([song(5, 220, seed=4)] * 2, axis=1) * 0.3))
    analysis = client.get(f"/api/projects/{p['id']}/media/{mid}/analysis").json()
    assert set(analysis) >= {"bpm", "lufs", "peakDb", "duration"} and abs(analysis["duration"] - 5.0) < 0.01
    assert -40 < analysis["lufs"] < 0 and analysis["peakDb"] <= 0

    folder = tmp_path / "导出"
    folder.mkdir()
    done = wait_job(client, client.post(f"/api/projects/{p['id']}/media/{mid}/export",
                                        json={"format": "flac", "path": str(folder), "normalize": True}).json()["id"])
    assert done["status"] == "succeeded", done
    data, sr = sf.read(done["result"]["path"])
    assert done["result"]["path"].endswith("a.flac") and sr == 44100 and data.shape[1] == 2
    assert abs(np.max(np.abs(data)) - 10 ** (-1 / 20)) < 1e-3
    done = wait_job(client, client.post(f"/api/projects/{p['id']}/media/{mid}/export", json={"format": "mp3"}).json()["id"])
    url = done["result"]["downloadUrl"]
    assert url.startswith("/files/") and len(client.get("/api" + url).content) > 10000


# ---- local resource import -----------------------------------------------------------------------
def _resource_ctx(tmp_path, file_sha: str):
    manifest = json.loads((MANIFEST).read_text(encoding="utf-8"))
    manifest["resources"] = [
        {"id": "hubertsoft", "kind": "model", "required": False, "name": {"en": "H"}, "description": {"en": "d"},
         "size": None, "sha256": file_sha, "install": {"type": "file", "dest": "hubertsoft/h.pt"}, "sources": []},
        {"id": "nsf_hifigan", "kind": "model", "required": True, "name": {"en": "V"}, "description": {"en": "d"},
         "size": None, "sha256": None, "sources": [],
         "install": {"type": "zip", "dest": "nsf_hifigan", "files": {
             "pack/model.ckpt": "model.ckpt", "pack/config.json": "config.json", "pack/NOTICE.txt": "NOTICE.txt"}}},
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return make_ctx(tmp_path / "ctx", resources=path)


def test_resource_import_verifies_checksum(tmp_path):
    from fastapi.testclient import TestClient

    from stellaria_backend.app import create_app

    good = tmp_path / "模型 文件.pt"
    good.write_bytes(b"weights" * 1000)
    ctx = _resource_ctx(tmp_path, hashlib.sha256(good.read_bytes()).hexdigest())
    with TestClient(create_app(ctx.options, ctx=ctx), headers={"Authorization": f"Bearer {TOKEN}"}) as client:
        bad = tmp_path / "other.pt"
        bad.write_bytes(b"something else")
        done = wait_job(client, client.post("/api/resources/hubertsoft/import", json={"path": str(bad)}).json()["id"])
        assert done["status"] == "failed" and done["error"]["code"] == "resource.checksum"
        resource = next(r for r in client.get("/api/resources").json() if r["id"] == "hubertsoft")
        assert resource["status"] == "failed" and resource["error"]["code"] == "resource.checksum"
        assert not (ctx.options.data / "models" / "hubertsoft" / "h.pt").exists()

        job = client.post("/api/resources/hubertsoft/import", json={"path": str(good)}).json()
        assert job["type"] == "download"
        done = wait_job(client, job["id"])
        assert done["status"] == "succeeded", done
        assert (ctx.options.data / "models" / "hubertsoft" / "h.pt").read_bytes() == good.read_bytes()
        assert next(r for r in client.get("/api/resources").json() if r["id"] == "hubertsoft")["status"] == "ready"

        # an archive laid out like the official zip …
        archive = tmp_path / "nsf.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("pack/model.ckpt", b"ckpt")
            zf.writestr("pack/config.json", "{}")
            zf.writestr("pack/NOTICE.txt", "n")
        done = wait_job(client, client.post("/api/resources/nsf_hifigan/import", json={"path": str(archive)}).json()["id"])
        assert done["status"] == "succeeded", done
        client.delete("/api/resources/nsf_hifigan")
        # … or an old DDSP-SVC pretrain folder (no NOTICE file): pointing at a file inside it works too
        pretrain = tmp_path / "pretrain" / "nsf_hifigan"
        pretrain.mkdir(parents=True)
        (pretrain / "model.ckpt").write_bytes(b"ckpt")
        (pretrain / "config.json").write_text("{}", encoding="utf-8")
        done = wait_job(client, client.post("/api/resources/nsf_hifigan/import",
                                            json={"path": str(pretrain / "model.ckpt")}).json()["id"])
        assert done["status"] == "succeeded", done
        assert next(r for r in client.get("/api/resources").json() if r["id"] == "nsf_hifigan")["status"] == "ready"
        (pretrain / "config.json").unlink()
        client.delete("/api/resources/nsf_hifigan")
        done = wait_job(client, client.post("/api/resources/nsf_hifigan/import", json={"path": str(pretrain)}).json()["id"])
        assert done["status"] == "failed" and done["error"]["code"] == "resource.incomplete"
        assert "config.json" in done["error"]["detail"]



def test_resource_import_from_an_existing_ddsp_folder(tmp_path):
    from fastapi.testclient import TestClient

    from stellaria_backend.app import create_app

    manifest = json.loads((MANIFEST).read_text(encoding="utf-8"))
    for r in manifest["resources"]:
        if r["kind"] == "model":
            r["sha256"], r["size"] = None, None
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    ctx = make_ctx(tmp_path / "ctx", resources=path)

    # someone's DDSP-SVC bundle: models under pretrain/ (the older vocoder file is `model`), a
    # Python environment and a dataset that must not be searched
    root = tmp_path / "DDSP-SVC 整合包"
    (root / "pretrain" / "contentvec").mkdir(parents=True)
    (root / "pretrain" / "contentvec" / "pytorch_model.bin").write_bytes(b"cv" * 500)
    (root / "pretrain" / "rmvpe").mkdir()
    (root / "pretrain" / "rmvpe" / "model.pt").write_bytes(b"rm" * 500)
    (root / "pretrain" / "nsf_hifigan").mkdir()
    (root / "pretrain" / "nsf_hifigan" / "model").write_bytes(b"voc" * 500)
    (root / "pretrain" / "nsf_hifigan" / "config.json").write_text("{}", encoding="utf-8")
    (root / "pretrain" / "hubert").mkdir()
    (root / "pretrain" / "hubert" / "hubert-soft-0d54a1f4.pt").write_bytes(b"hs" * 500)
    decoy = root / "workenv" / "site" / "contentvec"
    decoy.mkdir(parents=True)
    (root / "workenv" / "pyvenv.cfg").write_text("home = x", encoding="utf-8")
    (decoy / "pytorch_model.bin").write_bytes(b"wrong")

    with TestClient(create_app(ctx.options, ctx=ctx), headers={"Authorization": f"Bearer {TOKEN}"}) as client:
        assert client.post("/api/resources/import-folder", json={"path": str(tmp_path / "nope")}).status_code == 422
        empty = tmp_path / "empty"
        empty.mkdir()
        assert client.post("/api/resources/import-folder", json={"path": str(empty)}).json()["found"] == []

        res = client.post("/api/resources/import-folder", json={"path": str(root)}).json()
        assert sorted(f["id"] for f in res["found"]) == ["contentvec", "hubertsoft", "nsf_hifigan", "rmvpe"]
        assert res["missing"] == []
        for f in res["found"]:
            assert wait_job(client, f["jobId"])["status"] == "succeeded"
        models = ctx.options.data / "models"
        cv = models / "contentvec" / "pytorch_model.bin"
        assert cv.read_bytes() == b"cv" * 500  # the real one, not the copy inside the environment
        assert (models / "nsf_hifigan" / "model.ckpt").read_bytes() == b"voc" * 500
        # same drive: linked, not copied
        assert cv.stat().st_ino == (root / "pretrain" / "contentvec" / "pytorch_model.bin").stat().st_ino
        separation = {r["id"] for r in manifest["resources"] if r.get("separation") or r.get("cleanup")}
        statuses = {r["id"]: r["status"] for r in client.get("/api/resources").json()
                    if r["kind"] == "model" and r["id"] not in separation}
        assert all(v == "ready" for v in statuses.values()), statuses
        # nothing left to import; pointing at one model's own folder works as well
        assert client.post("/api/resources/import-folder", json={"path": str(root)}).json() == {
            "found": [], "missing": [], "separation": {"found": [], "skipped": []}}
        client.delete("/api/resources/rmvpe")
        res = client.post("/api/resources/import-folder", json={"path": str(root / "pretrain" / "rmvpe")}).json()
        assert [f["id"] for f in res["found"]] == ["rmvpe"]


STUB_MIXER = r'''
import json, sys
import numpy as np, soundfile as sf
job = json.load(open(sys.argv[1], encoding="utf-8"))
print("not json: ignored", flush=True)
print(json.dumps({"progress": 0.5, "stage": "mix.reverb"}), flush=True)
print(json.dumps({"log": "seed %d, preset %s" % (job["seed"], job["preset"])}), flush=True)
if job["params"].get("fail"):
    print(json.dumps({"error": {"code": "mix.bad_input", "message": "nope"}}), flush=True)
    sys.exit(1)
lead = [i for i in job["inputs"] if i["role"] == "lead"]
backing = [i for i in job["inputs"] if i["role"] == "backing"]
assert len(lead) == 2 and backing[0]["gainDb"] == -2.0 and lead[1]["pan"] > 0 and all(i["offset"] == 0 for i in job["inputs"])
sf.write(job["output"], np.zeros((44100, 2), dtype=np.float32) + 0.05, 44100, subtype="FLOAT")
print(json.dumps({"result": {"lufs": -14.04, "peakDb": -1.0, "bpm": 96.0}}), flush=True)
'''


def test_builtin_mixer_cli_protocol(client, tmp_path, monkeypatch):
    """The JSON-lines protocol of ``python -m stellaria_backend.mixing.run`` (driven by a stub)."""
    from stellaria_backend.creation import mix

    stub = tmp_path / "stub_mixer.py"
    stub.write_text(STUB_MIXER, encoding="utf-8")
    monkeypatch.setattr(mix, "mixer_available", lambda: True)
    monkeypatch.setattr(mix, "mixer_command", lambda ctx, job_file: [sys.executable, str(stub), str(job_file)])
    p, _, _, nodes = _mix_project(client, tmp_path)
    put_canvas(client, p["id"], nodes + [mix_node(["nd_r0", "nd_r1"], variants=2, preset="airy")])
    job = client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix").json()
    done = wait_job(client, job["id"], timeout=60)
    assert done["status"] == "succeeded", done
    assert [o["lufs"] for o in done["result"]["outputs"]] == [-14.04, -14.04]
    log = client.get(f"/api/jobs/{job['id']}/log").json()["lines"]
    assert "seed 0, preset airy" in log and "not json: ignored" in log
    assert done["stage"]["code"] == "stage.mix.reverb" and done["stage"]["params"] == {"variant": 2, "variants": 2}

    store_canvas(client, p["id"], nodes + [mix_node(["nd_r0", "nd_r1"], params={"fail": True})])
    done = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_m/mix").json()["id"], timeout=60)
    assert done["status"] == "failed" and done["error"]["code"] == "mix.bad_input"
    node = node_of(client, p["id"], "nd_m")["data"]
    assert node["status"] == "failed" and node["error"]["code"] == "mix.bad_input"
