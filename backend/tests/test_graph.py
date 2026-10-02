"""Canvas v3 graph backend (docs/canvas-v3.md §1): migration, ports and offsets, /run for every
node type, process jobs (through a stand-in dsp CLI), stems import, staleness and the PUT
ownership rules, mix v2 inputs, analysis, diagnostics, CPU pools."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import TOKEN, wait_for, wait_job
from stellaria_backend.creation import process as process_mod
from stellaria_backend.creation.canvas import Graph, migrate_canvas
from synth import song, write
from test_creation import (VIEWPORT, _trained_model, import_media, mix_input, node_of, put_canvas, render_node, source_node,
                           store_canvas)

STUB = Path(__file__).with_name("dsp_stub.py")


@pytest.fixture
def gclient(client, monkeypatch):
    """A client whose process jobs run the stand-in dsp CLI."""
    monkeypatch.setattr(process_mod, "dsp_available", lambda: True)
    monkeypatch.setattr(process_mod, "dsp_command", lambda ctx, job_file: [sys.executable, str(STUB), str(job_file)])
    return client


def process_node(node_id: str, op: str, inputs: list[tuple[str, str]], params: dict | None = None, **data) -> dict:
    return {"id": node_id, "type": "process", "position": {"x": 900, "y": 0}, "data": {
        "op": op, "label": data.pop("label", ""), "inputs": [{"nodeId": n, "port": p} for n, p in inputs],
        "params": params or {}, "outputs": {}, "status": "draft", "jobId": None, "error": None, "stale": False,
        "bypass": False, "result": None, **data}}


def mix_v3(node_id: str, inputs: list[dict], processor: str = "builtin", **data) -> dict:
    return {"id": node_id, "type": "mix", "position": {"x": 1200, "y": 0}, "data": {
        "inputs": inputs, "processor": processor, "preset": "natural" if processor == "builtin" else None, "params": {},
        "variants": 1, "outputs": [], "selected": 0, "status": "draft", "jobId": None, "error": None, "stale": False,
        "label": data.pop("label", ""), **data}}


def run(client, pid: str, node_id: str, timeout: float = 90, **body) -> dict:
    r = client.post(f"/api/projects/{pid}/nodes/{node_id}/run", json=body or None)
    assert r.status_code == 200, r.json()
    done = wait_job(client, r.json()["id"], timeout=timeout)
    assert done["status"] == "succeeded", done
    return done


def media_of(client, pid: str) -> dict[str, dict]:
    return {m["id"]: m for m in client.get(f"/api/projects/{pid}/media").json()}


def stereo_song(path: Path, seconds: float = 6.0) -> Path:
    return write(path, np.stack([song(seconds, 220, seed=3), song(seconds, 165, seed=4)], axis=1) * 0.5)


# ---- migration --------------------------------------------------------------------------------------
V2_CANVAS = {
    "nodes": [
        {"id": "s", "type": "source", "position": {"x": 0, "y": 0}, "data": {"mediaId": "med_song", "name": "song", "stems": {
            "status": "done", "vocals": "med_v", "inst": "med_i", "jobId": None, "error": None}}},
        {"id": "r", "type": "render", "position": {"x": 0, "y": 0}, "data": {"input": {"nodeId": "s", "port": "vocals"},
                                                                         "status": "done", "mediaId": "med_r"}},
        {"id": "r2", "type": "render", "position": {"x": 0, "y": 0}, "data": {"input": {"nodeId": "r", "port": "mix"}}},
        {"id": "m", "type": "mix", "position": {"x": 0, "y": 0}, "data": {
            "vocals": [{"nodeId": "r", "gainDb": 1.5, "pan": -0.4}, {"nodeId": "r2", "gainDb": 0, "pan": 0.4}],
            "backing": {"nodeId": "s", "port": "inst", "gainDb": -3}, "processor": "builtin", "outputs": [], "selected": 0,
            "status": "draft", "jobId": None, "error": None, "stale": False, "label": "", "variants": 1, "preset": None,
            "params": {}}},
    ],
    "edges": [],
    "viewport": VIEWPORT,
}


def test_v2_to_v3_migration(client):
    canvas = json.loads(json.dumps(V2_CANVAS))
    assert migrate_canvas(canvas) is True
    nodes = {n["id"]: n["data"] for n in canvas["nodes"]}
    assert nodes["r"]["input"] == {"nodeId": "s", "port": "vocals"}
    assert nodes["r2"]["input"] == {"nodeId": "r", "port": "out"}  # v2 'mix' of a non-source node
    assert nodes["m"]["inputs"] == [
        {"id": "r", "ref": {"nodeId": "r", "port": "out"}, "role": "lead", "gainDb": 1.5, "pan": -0.4, "mute": False, "solo": False},
        {"id": "r2", "ref": {"nodeId": "r2", "port": "out"}, "role": "lead", "gainDb": 0.0, "pan": 0.4, "mute": False, "solo": False},
        {"id": "s:inst", "ref": {"nodeId": "s", "port": "inst"}, "role": "backing", "gainDb": -3.0, "pan": 0.0, "mute": False,
         "solo": False}]
    assert "vocals" not in nodes["m"] and "backing" not in nodes["m"]
    assert nodes["s"]["stems"]["provided"] == [] and nodes["s"]["stems"]["drums"] is None and nodes["s"]["bpm"] is None
    snapshot = json.dumps(canvas, sort_keys=True)
    assert migrate_canvas(canvas) is False and json.dumps(canvas, sort_keys=True) == snapshot

    graph = Graph(canvas)
    assert graph.media_of({"nodeId": "r", "port": "out"}) == "med_r" and graph.media_of({"nodeId": "s", "port": "inst"}) == "med_i"
    assert graph.media_of({"nodeId": "r", "port": "mix"}) == "med_r"  # legacy port still resolves

    # an old client's save is migrated too; media without offset read as 0
    p = client.post("/api/projects", json={"name": "v2"}).json()
    client.put(f"/api/projects/{p['id']}", json={"canvas": json.loads(json.dumps(V2_CANVAS))})
    stored = client.get(f"/api/projects/{p['id']}").json()["canvas"]["nodes"]
    assert next(n for n in stored if n["id"] == "m")["data"]["inputs"][2]["id"] == "s:inst"
    media_file = client.ctx.projects.dir(p["id"]) / "media.json"
    media_file.write_text(json.dumps([{"id": "med_old", "name": "old", "kind": "import", "duration": 1.0, "sampleRate": 44100,
                                       "channels": 1, "createdAt": "2025-01-01T00:00:00Z"}]), encoding="utf-8")
    assert client.get(f"/api/projects/{p['id']}/media").json()[0]["offset"] == 0.0


def test_bypass_resolution_and_missing_input():
    canvas = {"nodes": [
        {"id": "a", "type": "audio", "data": {"mediaId": "med_a", "name": "a", "role": "backing"}},
        {"id": "fx1", "type": "process", "data": {"op": "fx", "inputs": [{"nodeId": "a", "port": "out"}], "bypass": True,
                                                  "outputs": {"out": "med_old"}}},
        {"id": "p", "type": "process", "data": {"op": "pitch", "inputs": [{"nodeId": "fx1", "port": "out"}], "bypass": True}},
        {"id": "t", "type": "process", "data": {"op": "trim", "inputs": [{"nodeId": "p", "port": "out"}], "outputs": {}}},
        {"id": "x", "type": "process", "data": {"op": "fx", "inputs": [{"nodeId": "t", "port": "out"}], "bypass": True}},
    ], "edges": []}
    migrate_canvas(canvas)
    graph = Graph(canvas)
    assert graph.media_of({"nodeId": "p", "port": "out"}) == "med_a"  # through two bypassed nodes
    assert [r["nodeId"] for r in graph.chain({"nodeId": "p", "port": "out"})] == ["p", "fx1", "a"]
    with pytest.raises(Exception) as err:
        graph.resolve({"nodeId": "x", "port": "out"})
    assert err.value.code == "node.input_missing" and err.value.params == {"nodeId": "t", "port": "out"}
    assert {n["id"] for n in graph.dependents("a")} == {"fx1", "p", "t"}
    assert [n["id"] for n in graph.dependents("t")] == ["x"]


# ---- the whole graph ---------------------------------------------------------------------------------
def test_graph_end_to_end(gclient, tmp_path):
    """song → 4-stem split → bus(drums, bass, other) → take(vocals) → fx → mix(lead + bus) → export."""
    client, ctx = gclient, gclient.ctx
    cid, mid = _trained_model(client, ctx)
    wait_job(client, client.post("/api/resources/separator/install").json()["id"])
    p = client.post("/api/projects", json={"name": "图谱"}).json()
    pid = p["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "song.wav"))
    folder = tmp_path / "导出 folder"
    folder.mkdir()
    export_params = {"format": "flac", "sampleRate": 44100, "bitDepth": 24, "mp3Bitrate": 320, "loudness": None,
                     "folder": str(folder), "fileName": "{project} - {song} - {voice}", "keepPosition": False, "tags": None}
    nodes = [
        {**source_node(song_id), "data": {"mediaId": song_id, "name": "song"}},
        mix_v3("nd_bus", [mix_input("nd_s", "drums", "backing", 2.0), mix_input("nd_s", "bass", "backing"),
                          mix_input("nd_s", "other", "backing", -1.0)], processor="bus", label="伴奏"),
        render_node("nd_t", "vocals", characterId=cid, modelId=mid),
        process_node("nd_fx", "fx", [("nd_t", "out")], {"chain": [{"id": "s1", "kind": "builtin", "effect": "gain",
                                                                  "bypass": False, "mix": 1, "params": {"gainDb": -6}}]}),
        mix_v3("nd_m", [mix_input("nd_fx", "out", "lead"), mix_input("nd_bus", "out", "backing")]),
        process_node("nd_x", "export", [("nd_m", "out")], export_params),
    ]
    put_canvas(client, pid, nodes)

    r = client.post(f"/api/projects/{pid}/nodes/nd_t/run")
    assert r.status_code == 409 and r.json()["error"]["params"] == {"nodeId": "nd_s", "port": "vocals"}

    done = run(client, pid, "nd_s", stems=4)
    stems = node_of(client, pid, "nd_s")["data"]["stems"]
    assert stems["status"] == "done" and all(stems[k] for k in ("vocals", "inst", "drums", "bass", "other"))
    assert stems["model"] == "htdemucs" and stems["provided"] == [] and done["result"]["stems"] == stems
    media = media_of(client, pid)
    assert {media[stems[k]]["kind"] for k in ("vocals", "drums")} == {"stem"} and media[stems["drums"]]["offset"] == 0
    inst, _ = sf.read(ctx.projects.media_path(pid, stems["inst"]))
    parts = sum(sf.read(ctx.projects.media_path(pid, stems[k]))[0] for k in ("drums", "bass", "other"))
    assert np.allclose(inst, parts, atol=3e-4)  # inst = drums + bass + other

    done = run(client, pid, "nd_bus")
    bus = node_of(client, pid, "nd_bus")["data"]
    assert bus["status"] == "done" and len(bus["outputs"]) == 1 and done["type"] == "mix"
    assert media_of(client, pid)[bus["outputs"][0]["mediaId"]]["kind"] == "mix"

    run(client, pid, "nd_t")
    done = run(client, pid, "nd_fx")
    assert done["type"] == "process" and done["title"] == {"code": "job.process.fx", "params": {"name": media_of(client, pid)[
        node_of(client, pid, "nd_t")["data"]["mediaId"]]["name"]}}
    fx = node_of(client, pid, "nd_fx")["data"]
    assert fx["status"] == "done" and fx["stale"] is False and media_of(client, pid)[fx["outputs"]["out"]]["kind"] == "process"
    assert done["stage"]["code"] == "stage.dsp.fx"

    run(client, pid, "nd_m", timeout=180)
    mixed = node_of(client, pid, "nd_m")["data"]
    assert mixed["status"] == "done" and mixed["outputs"][0]["mediaId"]

    done = run(client, pid, "nd_x")
    out = Path(done["result"]["result"]["path"])
    assert out.parent == folder and out.name == "图谱 - song - Port.flac" and out.is_file()
    assert done["result"]["outputs"] == {} and done["result"]["result"]["downloadUrl"].startswith("/files/")
    log = "\n".join(client.get(f"/api/jobs/{done['id']}/log").json()["lines"])
    assert '"index": 1' in log and '"date": "' in log
    again = run(client, pid, "nd_x")
    assert Path(again["result"]["result"]["path"]).name == "图谱 - song - Port (2).flac"  # never overwrites
    assert node_of(client, pid, "nd_x")["data"]["result"]["path"] == again["result"]["result"]["path"]

    # re-running the take marks only its direct dependents stale
    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        ws.receive_json()
        run(client, pid, "nd_t")
        while True:
            msg = ws.receive_json()
            if msg["type"] == "project.nodes" and msg["data"]["nodes"][0]["data"].get("status") == "done":
                break
    assert [n["id"] for n in msg["data"]["nodes"]] == ["nd_t", "nd_fx"] and msg["data"]["nodes"][1]["data"]["stale"] is True
    assert node_of(client, pid, "nd_m")["data"]["stale"] is False
    assert node_of(client, pid, "nd_t")["data"]["stale"] is False


def test_offsets_trim_render_comp_split(gclient, tmp_path):
    client, ctx = gclient, gclient.ctx
    cid, mid = _trained_model(client, ctx)
    pid = client.post("/api/projects", json={"name": "offsets"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav"))
    params = {"key": 0, "formant": 0, "register": 0, "f0Extractor": "rmvpe", "steps": 10, "method": "euler", "tStart": 0,
              "threshold": -60, "spkMix": None, "region": {"start": 0.5, "end": 2.5}}
    nodes = [
        source_node(song_id),
        process_node("nd_tr", "trim", [("nd_s", "mix")], {"start": 1.0, "end": 3.0, "fadeIn": 0.01, "fadeOut": 0.01}),
        render_node("nd_t1", "out", characterId=cid, modelId=mid, input={"nodeId": "nd_tr", "port": "out"}),
        render_node("nd_t2", "mix", characterId=cid, modelId=mid, params=params),
        process_node("nd_c", "comp", [("nd_t1", "out"), ("nd_t2", "out")],
                     {"segments": [{"start": 1.0, "end": 2.0, "input": 0}, {"start": 2.0, "end": 2.5, "input": 1},
                                   {"start": 2.5, "end": 2.75, "input": -1}], "crossfade": 0.01}),
        process_node("nd_sp", "split", [("nd_s", "mix")], {"cuts": [2.0, 4.0], "fade": 0.01, "labels": []}),
        mix_v3("nd_m", [mix_input("nd_t1", "out", "lead"), mix_input("nd_s", "mix", "backing", -6)]),
    ]
    put_canvas(client, pid, nodes)
    run(client, pid, "nd_tr")
    trim = node_of(client, pid, "nd_tr")["data"]["outputs"]["out"]
    run(client, pid, "nd_t1")
    run(client, pid, "nd_t2")
    media = media_of(client, pid)
    t1, t2 = (node_of(client, pid, n)["data"]["mediaId"] for n in ("nd_t1", "nd_t2"))
    assert (media[trim]["offset"], media[t1]["offset"], media[t2]["offset"]) == (1.0, 1.0, 0.5)
    assert abs(media[t2]["duration"] - 2.0) < 0.01  # only the region is written

    run(client, pid, "nd_c")
    comp = media_of(client, pid)[node_of(client, pid, "nd_c")["data"]["outputs"]["out"]]
    assert comp["offset"] == 0.5 and abs(comp["duration"] - 2.25) < 0.01
    data, _ = sf.read(ctx.projects.media_path(pid, comp["id"]), always_2d=True)
    t1_audio, _ = sf.read(ctx.projects.media_path(pid, t1), always_2d=True)
    assert np.allclose(data[int(0.5 * 44100): int(0.5 * 44100) + 1000, 0], t1_audio[:1000, 0], atol=1e-5)  # aligned by offset
    assert np.max(np.abs(data[int(2.05 * 44100):])) == 0  # input -1 = silence

    run(client, pid, "nd_sp")
    outputs = node_of(client, pid, "nd_sp")["data"]["outputs"]
    media = media_of(client, pid)
    assert sorted(outputs) == ["part1", "part2", "part3"]
    assert [media[outputs[f"part{i}"]]["offset"] for i in (1, 2, 3)] == [0.0, 2.0, 4.0]

    done = run(client, pid, "nd_m", timeout=180)
    mixed = media_of(client, pid)[done["result"]["outputs"][0]["mediaId"]]
    assert mixed["offset"] == 0.0 and abs(mixed["duration"] - 6.0) < 0.05  # inputs placed at offset − min


def test_staleness_and_put_ownership(gclient, tmp_path):
    client = gclient
    pid = client.post("/api/projects", json={"name": "stale"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 3.0))
    nodes = [
        source_node(song_id),
        process_node("nd_a", "trim", [("nd_s", "mix")], {"start": 0.5, "end": 2.5, "fadeIn": 0, "fadeOut": 0}),
        process_node("nd_b", "fx", [("nd_a", "out")], {"chain": []}, bypass=True),
        process_node("nd_c", "trim", [("nd_b", "out")], {"start": 0.6, "end": 1.0, "fadeIn": 0, "fadeOut": 0}),
        process_node("nd_d", "trim", [("nd_s", "mix")], {"start": 0, "end": 1, "fadeIn": 0, "fadeOut": 0}),
    ]
    put_canvas(client, pid, nodes)
    r = client.post(f"/api/projects/{pid}/nodes/nd_b/run")
    assert r.status_code == 409 and r.json()["error"]["code"] == "node.not_runnable"
    r = client.post(f"/api/projects/{pid}/nodes/nd_c/run")
    assert r.status_code == 409 and r.json()["error"]["params"] == {"nodeId": "nd_a", "port": "out"}  # past the bypass

    with client.websocket_connect(f"/api/events?token={TOKEN}") as ws:
        ws.receive_json()
        run(client, pid, "nd_a")
        while True:
            msg = ws.receive_json()
            if msg["type"] == "project.nodes" and msg["data"]["nodes"][0]["data"].get("status") == "done":
                break
    changed = {n["id"]: n["data"] for n in msg["data"]["nodes"]}
    assert list(changed) == ["nd_a", "nd_b", "nd_c"] and changed["nd_c"]["stale"] is True and changed["nd_a"]["stale"] is False
    run(client, pid, "nd_c")  # reads nd_a's output through the bypassed fx
    assert node_of(client, pid, "nd_c")["data"]["stale"] is False

    # PUT: server-owned fields of existing nodes stay; stale can be set but not cleared
    canvas = client.get(f"/api/projects/{pid}").json()["canvas"]
    by_id = {n["id"]: n for n in canvas["nodes"]}
    a_outputs = by_id["nd_a"]["data"]["outputs"]
    by_id["nd_a"]["data"].update(status="draft", outputs={}, result={"x": 1}, jobId="job_fake", error={"code": "x"},
                                 params={"start": 0.25, "end": 2.5, "fadeIn": 0, "fadeOut": 0}, stale=True)
    by_id["nd_b"]["data"]["stale"] = False  # was set stale by the run
    by_id["nd_s"]["data"].update(mediaId="med_other", stems={"status": "done", "vocals": "x"}, bpm=99, key="C")
    new = process_node("nd_new", "trim", [("nd_s", "mix")], status="done", outputs={"out": "med_copied"})
    client.put(f"/api/projects/{pid}", json={"canvas": {**canvas, "nodes": [*by_id.values(), new]}})
    stored = {n["id"]: n["data"] for n in client.get(f"/api/projects/{pid}").json()["canvas"]["nodes"]}
    assert stored["nd_a"]["status"] == "done" and stored["nd_a"]["outputs"] == a_outputs and stored["nd_a"]["result"] is None
    assert stored["nd_a"]["error"] is None and stored["nd_a"]["jobId"] != "job_fake"
    assert stored["nd_a"]["params"]["start"] == 0.25 and stored["nd_a"]["stale"] is True  # client fields + stale OR
    assert stored["nd_b"]["stale"] is True
    assert stored["nd_s"]["mediaId"] == song_id and stored["nd_s"]["stems"]["vocals"] is None and stored["nd_s"]["bpm"] != 99
    assert stored["nd_new"]["status"] == "done" and stored["nd_new"]["outputs"] == {"out": "med_copied"}  # new nodes as sent


def test_run_endpoint_errors(gclient, tmp_path, monkeypatch):
    client, ctx = gclient, gclient.ctx
    pid = client.post("/api/projects", json={"name": "errors"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 2.0))
    nodes = [
        source_node(song_id),
        {"id": "nd_n", "type": "note", "position": {"x": 0, "y": 0}, "data": {"text": "hi", "color": "#fff"}},
        {"id": "nd_a", "type": "audio", "position": {"x": 0, "y": 0}, "data": {"mediaId": song_id, "name": "a", "role": "backing"}},
        process_node("nd_c", "comp", [("nd_s", "mix")], {"segments": [], "crossfade": 0.01}),
        process_node("nd_t", "trim", [], {"start": 0, "end": 1}),
        process_node("nd_x", "trim", [("nd_a", "out")], {"start": 0, "end": 1, "fadeIn": 0, "fadeOut": 0}),
    ]
    put_canvas(client, pid, nodes)
    for node_id in ("nd_n", "nd_a"):
        r = client.post(f"/api/projects/{pid}/nodes/{node_id}/run")
        assert r.status_code == 409 and r.json()["error"] == {"code": "node.not_runnable", "message": r.json()["error"]["message"],
                                                              "detail": "", "params": {"nodeId": node_id}}
    r = client.post(f"/api/projects/{pid}/nodes/nd_c/run")
    assert r.status_code == 422 and r.json()["error"]["code"] == "process.input_count"
    assert r.json()["error"]["params"] == {"min": 2, "max": 8}
    r = client.post(f"/api/projects/{pid}/nodes/nd_t/run")
    assert r.status_code == 409 and r.json()["error"]["code"] == "node.input_missing"
    assert client.post(f"/api/projects/{pid}/nodes/nd_zz/run").status_code == 404
    assert client.post(f"/api/projects/{pid}/nodes/nd_s/run", json={"stems": 3}).status_code == 422

    # busy while its job is active
    from stellaria_backend.schemas import Msg

    release = threading.Event()
    blocker = ctx.jobs.submit("process", Msg(code="job.process.trim"), lambda jctx: release.wait(10) and None)
    wait_for(lambda: ctx.jobs.get(blocker.id).status == "running", 5)
    ctx.node_updates.update(ctx, pid, "nd_x", status="running", jobId=blocker.id)
    r = client.post(f"/api/projects/{pid}/nodes/nd_x/run")
    assert r.status_code == 409 and r.json()["error"]["params"] == {"nodeId": "nd_x", "jobId": blocker.id}
    release.set()
    wait_for(lambda: not ctx.jobs.get(blocker.id).active, 5)
    run(client, pid, "nd_x")  # a finished job no longer blocks

    # the CLI's error line becomes the node's error (with params)
    store = client.get(f"/api/projects/{pid}").json()["canvas"]["nodes"]
    for n in store:
        if n["id"] == "nd_x":
            n["data"]["params"]["fail"] = True
    put_canvas(client, pid, store)
    done = wait_job(client, client.post(f"/api/projects/{pid}/nodes/nd_x/run").json()["id"])
    assert done["status"] == "failed" and done["error"]["code"] == "dsp.bad_params" and done["error"]["params"] == {"field": "fail"}
    node = node_of(client, pid, "nd_x")["data"]
    assert node["status"] == "failed" and node["error"] == {"code": "dsp.bad_params", "params": {"field": "fail"}}

    monkeypatch.setattr(process_mod, "dsp_available", lambda: False)
    r = client.post(f"/api/projects/{pid}/nodes/nd_x/run")
    assert r.status_code == 409 and r.json()["error"]["code"] == "dsp.unavailable"


def test_fx_vst_slots_and_crash(gclient, tmp_path, monkeypatch):
    client, ctx = gclient, gclient.ctx
    pid = client.post("/api/projects", json={"name": "vst"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 2.0))
    slot = {"id": "s1", "kind": "vst", "pluginId": "vst_proq", "state": None, "bypass": False, "mix": 1, "params": {"crash": True}}
    put_canvas(client, pid, [source_node(song_id), process_node("nd_fx", "fx", [("nd_s", "mix")], {"chain": [slot]})])
    if not ctx.vst.host_available():
        r = client.post(f"/api/projects/{pid}/nodes/nd_fx/run")
        assert r.status_code == 409 and r.json()["error"]["code"] == "vst.host_missing"
    plugins = {"vst_proq": {"id": "vst_proq", "name": "Pro-Q 3", "category": "effect", "path": "/x.vst3"},
               "vst_synth": {"id": "vst_synth", "name": "Synth", "category": "instrument", "path": "/y.vst3"}}

    def require(plugin_id):
        if plugin_id not in plugins:
            from stellaria_backend.vst.registry import not_found

            raise not_found(plugin_id)
        return plugins[plugin_id]

    monkeypatch.setattr(ctx, "vst", types.SimpleNamespace(require_host=lambda: None, require=require))
    done = wait_job(client, client.post(f"/api/projects/{pid}/nodes/nd_fx/run").json()["id"])
    assert done["status"] == "failed" and done["error"]["code"] == "vst.crashed" and done["error"]["params"] == {"name": "Pro-Q 3"}
    assert node_of(client, pid, "nd_fx")["data"]["error"] == {"code": "vst.crashed", "params": {"name": "Pro-Q 3"}}

    slot["params"] = {}
    put_canvas(client, pid, [source_node(song_id), process_node("nd_fx", "fx", [("nd_s", "mix")], {"chain": [slot]})])
    done = run(client, pid, "nd_fx")
    assert done["stage"]["code"] == "stage.dsp.fx"
    for plugin_id, code in (("vst_synth", "vst.instrument"), ("vst_gone", "vst.not_found")):
        put_canvas(client, pid, [source_node(song_id), process_node("nd_fx", "fx", [("nd_s", "mix")],
                                                                   {"chain": [{**slot, "pluginId": plugin_id}]})])
        r = client.post(f"/api/projects/{pid}/nodes/nd_fx/run")
        assert r.json()["error"]["code"] == code
    # bypassed VST slots are ignored
    put_canvas(client, pid, [source_node(song_id), process_node("nd_fx", "fx", [("nd_s", "mix")],
                                                               {"chain": [{**slot, "pluginId": "vst_gone", "bypass": True}]})])
    run(client, pid, "nd_fx")


# ---- song stems ---------------------------------------------------------------------------------------
def test_stems_import_and_split_keeps_provided(gclient, tmp_path):
    client = gclient
    wait_job(client, client.post("/api/resources/separator/install").json()["id"])
    pid = client.post("/api/projects", json={"name": "stems"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 3.0))
    official = stereo_song(tmp_path / "official inst.wav", 3.0)
    put_canvas(client, pid, [source_node(song_id),
                             process_node("nd_t", "trim", [("nd_s", "inst")], {"start": 0, "end": 1, "fadeIn": 0, "fadeOut": 0}),
                             process_node("nd_u", "trim", [("nd_s", "mix")], {"start": 0, "end": 1, "fadeIn": 0, "fadeOut": 0})])
    node = client.post(f"/api/projects/{pid}/nodes/nd_s/stems", json={"paths": [str(official)], "port": "inst"}).json()
    stems = node["data"]["stems"]
    assert node["id"] == "nd_s" and stems["inst"] and stems["provided"] == ["inst"] and stems["status"] == "none"
    media = media_of(client, pid)[stems["inst"]]
    assert media["kind"] == "import" and media["offset"] == 0 and media["name"] == "song · instrumental"
    stored = {n["id"]: n["data"] for n in client.get(f"/api/projects/{pid}").json()["canvas"]["nodes"]}
    assert stored["nd_t"]["stale"] is True and stored["nd_u"]["stale"] is False  # only readers of that port

    with open(tmp_path / "s.wav", "rb") as f:
        r = client.post(f"/api/projects/{pid}/nodes/nd_s/stems", files={"files": ("人声.wav", f, "audio/wav")},
                        data={"port": "vocals"})
    assert r.status_code == 200, r.json()
    stems = r.json()["data"]["stems"]
    assert stems["provided"] == ["inst", "vocals"] and stems["status"] == "done"
    r = client.post(f"/api/projects/{pid}/nodes/nd_s/run")
    assert r.status_code == 409 and r.json()["error"]["code"] == "node.not_runnable"  # everything was provided

    r = client.delete(f"/api/projects/{pid}/nodes/nd_s/stems/vocals")
    assert r.status_code == 200 and r.json()["data"]["stems"]["vocals"] is None
    inst_id = stems["inst"]
    run(client, pid, "nd_s", stems=4)
    stems = node_of(client, pid, "nd_s")["data"]["stems"]
    assert stems["inst"] == inst_id and stems["vocals"] and stems["drums"] and stems["provided"] == ["inst"]

    r = client.delete(f"/api/projects/{pid}/nodes/nd_s/stems/drums")
    assert r.status_code == 409 and r.json()["error"] == {"code": "stems.not_provided", "message": r.json()["error"]["message"],
                                                          "detail": "", "params": {"port": "drums"}}
    r = client.post(f"/api/projects/{pid}/nodes/nd_s/stems", json={"path": str(official), "port": "piano"})
    assert r.status_code == 422
    r = client.post(f"/api/projects/{pid}/nodes/nd_t/stems", json={"path": str(official), "port": "inst"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "node.not_source"


def test_process_separate_and_pitch(gclient, tmp_path):
    client = gclient
    wait_job(client, client.post("/api/resources/separator/install").json()["id"])
    pid = client.post("/api/projects", json={"name": "sep"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 3.0))
    put_canvas(client, pid, [source_node(song_id),
                             process_node("nd_tr", "trim", [("nd_s", "mix")], {"start": 1.0, "end": 3.0, "fadeIn": 0, "fadeOut": 0}),
                             process_node("nd_sep", "separate", [("nd_tr", "out")], {"model": "htdemucs", "stems": 4}, label="后半"),
                             process_node("nd_p", "pitch", [("nd_sep", "vocals")], {"semitones": 2, "cents": 0, "tempo": 2.0,
                                                                                    "preserveFormant": True, "quality": "fast"})])
    run(client, pid, "nd_tr")
    done = run(client, pid, "nd_sep")
    assert done["type"] == "process" and done["title"]["code"] == "job.process.separate" and done["usesGpu"] is True
    outputs = node_of(client, pid, "nd_sep")["data"]["outputs"]
    media = media_of(client, pid)
    assert sorted(outputs) == ["bass", "drums", "inst", "other", "vocals"]
    assert {media[m]["kind"] for m in outputs.values()} == {"process"} and {media[m]["offset"] for m in outputs.values()} == {1.0}
    assert media[outputs["vocals"]]["name"] == "后半 · vocals"
    run(client, pid, "nd_p")
    pitched = node_of(client, pid, "nd_p")["data"]
    assert media_of(client, pid)[pitched["outputs"]["out"]]["offset"] == 0.5 and pitched["result"] == {"engine": "builtin"}


# ---- mix v2 -------------------------------------------------------------------------------------------
def test_mix_job_v2_reference_bus_and_plugin(gclient, tmp_path, monkeypatch):
    from stellaria_backend.creation import mix

    client, ctx = gclient, gclient.ctx
    pid = client.post("/api/projects", json={"name": "ref"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 4.0))
    vocal_id = import_media(client, pid, write(tmp_path / "v.wav", song(4, 220, seed=8)))
    inst_id = import_media(client, pid, write(tmp_path / "i.wav", song(4, 110, seed=9)))
    take_id = import_media(client, pid, write(tmp_path / "t.wav", song(2, 260, seed=7)))
    stored = [
        {**source_node(song_id), "data": {"mediaId": song_id, "name": "song", "stems": {
            "status": "done", "vocals": vocal_id, "inst": inst_id, "jobId": None, "error": None}}},
        render_node("nd_t", "vocals", status="done", mediaId=take_id, label="甲"),
        process_node("nd_tr", "trim", [("nd_t", "out")], outputs={"out": take_id}, status="done"),
        mix_v3("nd_m", [mix_input("nd_tr", "out", "lead", 1.0, 0.2), mix_input("nd_s", "inst", "backing", -3.0),
                        mix_input("nd_s", "mix", "other", mute=True, id="muted")]),
    ]
    store_canvas(client, pid, stored)
    # place the take at 1.5 s on the song's timeline
    media_file = ctx.projects.dir(pid) / "media.json"
    items = json.loads(media_file.read_text("utf-8"))
    for m in items:
        if m["id"] == take_id:
            m["offset"] = 1.5
    media_file.write_text(json.dumps(items), "utf-8")

    jobs = []
    stub = tmp_path / "stub_mixer.py"
    stub.write_text("import json, sys, shutil\nimport numpy as np, soundfile as sf\n"
                    "job = json.load(open(sys.argv[1], encoding='utf-8'))\n"
                    f"json.dump(job, open({str(tmp_path / 'job.json')!r}, 'w'))\n"
                    "sf.write(job['output'], np.zeros((44100, 2), dtype=np.float32), 44100)\n"
                    "print(json.dumps({'result': {'lufs': -14.0, 'peakDb': -1.0, 'bpm': None}}))\n", encoding="utf-8")
    monkeypatch.setattr(mix, "mixer_available", lambda: True)
    monkeypatch.setattr(mix, "mixer_command", lambda c, job_file: (jobs.append(job_file), [sys.executable, str(stub), str(job_file)])[1])
    done = run(client, pid, "nd_m")
    job = json.loads((tmp_path / "job.json").read_text())
    assert [(i["role"], i["gainDb"], i["pan"], i["offset"], i["mute"]) for i in job["inputs"]] == [
        ("lead", 1.0, 0.2, 1.5, False), ("backing", -3.0, 0.0, 0.0, False), ("other", 0.0, 0.0, 0.0, True)]
    assert job["reference"] == {"vocals": str(ctx.projects.media_path(pid, vocal_id)),
                                "mix": str(ctx.projects.media_path(pid, song_id)), "offset": 0.0}
    assert job["processor"] == "builtin" and job["preset"] == "natural"
    assert media_of(client, pid)[done["result"]["outputs"][0]["mediaId"]]["offset"] == 0.0  # min of the playing inputs

    # bus: one variant, processor 'bus'
    store_canvas(client, pid, stored[:3] + [mix_v3("nd_m", [mix_input("nd_s", "inst", "backing", 2.0)], processor="bus",
                                                   variants=3)])
    done = run(client, pid, "nd_m")
    job = json.loads((tmp_path / "job.json").read_text())
    assert job["processor"] == "bus" and len(done["result"]["outputs"]) == 1 and job["reference"] is None

    # a plugin with maxVocals 1 gets the two vocals pre-summed, the backing summed, all placed
    plugin = {"id": "sodium", "name": "Sodium", "kind": "plugin", "presets": [], "params": [], "maxVocals": 1,
              "available": True, "error": None}
    monkeypatch.setattr(ctx, "plugins", types.SimpleNamespace(mix_processors=lambda: [plugin]), raising=False)
    seen = {}

    def run_plugin(ctx_, jctx, plugin_id, *, vocals, backing, input, output, params, names, on_progress):
        seen.update(vocals=[(names[v], sf.info(str(v)).frames) for v in vocals], backing=(names[backing], sf.info(str(backing)).frames))
        sf.write(output, np.zeros((44100, 2), dtype=np.float32) + 0.1, 44100)
        return output

    runner = types.ModuleType("stellaria_backend.plugins.runner")
    runner.run_plugin = run_plugin
    monkeypatch.setitem(sys.modules, "stellaria_backend.plugins.runner", runner)
    store_canvas(client, pid, stored[:3] + [mix_v3("nd_m", [mix_input("nd_tr", "out", "lead"), mix_input("nd_s", "vocals", "harmony"),
                                                            mix_input("nd_s", "inst", "backing")], processor="sodium")])
    done = run(client, pid, "nd_m")
    assert seen["vocals"] == [("ref - vocals", 4 * 44100)]  # take (1.5 s + 2 s) and stem (4 s), placed and summed
    assert seen["backing"] == ("ref - backing", 4 * 44100)
    assert media_of(client, pid)[done["result"]["outputs"][0]["mediaId"]]["offset"] == 0.0

    procs = client.get("/api/mix/processors").json()
    assert [p["id"] for p in procs] == ["builtin", "bus", "sodium"] and procs[1]["kind"] == "builtin" and procs[1]["presets"] == []


# ---- analysis, effects, diagnostics, preferences ---------------------------------------------------
def test_sections_key_and_song_analysis(gclient, tmp_path, monkeypatch):
    client, ctx = gclient, gclient.ctx
    calls = []
    analysis = types.ModuleType("stellaria_backend.dsp.analysis")

    def sections(path, *, vocals_path=None):
        calls.append((Path(path).name, Path(vocals_path).name if vocals_path else None))
        return {"bpm": 120.0, "beats": [0.5, 1.0], "downbeats": [0.5], "phrases": [{"start": 0.4, "end": 2.0}],
                "sections": [{"start": 0.0, "end": 3.0, "label": "verse"}]}

    analysis.sections = sections
    analysis.key = lambda path: "C#m"
    monkeypatch.setitem(sys.modules, "stellaria_backend.dsp.analysis", analysis)
    import stellaria_backend.dsp as dsp_pkg

    monkeypatch.setattr(dsp_pkg, "analysis", analysis, raising=False)

    pid = client.post("/api/projects", json={"name": "ana"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 3.0))
    cache = ctx.projects.media_path(pid, song_id).with_suffix(".analysis.json")
    # one analysis at a time in the background, and the first bpm detection may compile numba's cache
    wait_for(lambda: cache.is_file() and '"key"' in cache.read_text("utf-8"), 90)
    put_canvas(client, pid, [source_node(song_id)])
    node = node_of(client, pid, "nd_s")["data"]
    assert node["key"] == "C#m" and (node["bpm"] is None or node["bpm"] > 0)  # filled from the cache at once

    first = client.get(f"/api/projects/{pid}/media/{song_id}/sections").json()
    assert first["sections"][0]["label"] == "verse" and calls == [(ctx.projects.media_path(pid, song_id).name, None)]
    client.get(f"/api/projects/{pid}/media/{song_id}/sections")
    assert len(calls) == 1  # cached
    vocal_id = import_media(client, pid, write(tmp_path / "v.wav", song(3, 220, seed=8)))
    ctx.node_updates.update(ctx, pid, "nd_s", stems={**node["stems"], "vocals": vocal_id})
    client.get(f"/api/projects/{pid}/media/{song_id}/sections")
    assert calls[-1] == (ctx.projects.media_path(pid, song_id).name, ctx.projects.media_path(pid, vocal_id).name)
    assert client.get(f"/api/projects/{pid}/media/{song_id}/analysis").json()["key"] == "C#m"

    # a song node saved before its analysis finished gets bpm/key by event
    other = import_media(client, pid, stereo_song(tmp_path / "o.wav", 2.0))
    ctx.projects.media_path(pid, other).with_suffix(".analysis.json").unlink(missing_ok=True)
    put_canvas(client, pid, [node_of(client, pid, "nd_s"), {**source_node(other, "nd_o"), "data": {"mediaId": other, "name": "o"}}])
    wait_for(lambda: node_of(client, pid, "nd_o")["data"].get("key") == "C#m", 90)


def test_fx_effects_diagnostics_preferences_resources(client, monkeypatch, tmp_path):
    assert isinstance(client.get("/api/fx/effects").json(), list)
    fx = types.ModuleType("stellaria_backend.fx")
    fx.effect_specs = lambda: [{"id": "eq", "name": {"code": "fx.effect.eq"}, "category": "eq", "params": []}]
    monkeypatch.setitem(sys.modules, "stellaria_backend.fx", fx)
    assert client.get("/api/fx/effects").json()[0]["id"] == "eq"

    prefs = client.put("/api/preferences", json={"exportFolder": str(tmp_path / "exports here")}).json()
    assert prefs["exportFolder"] == str(tmp_path / "exports here")
    assert process_mod.export_folder(client.ctx) == tmp_path / "exports here"
    assert process_mod.export_folder(client.ctx, str(tmp_path / "x")) == tmp_path / "x"
    client.put("/api/preferences", json={"exportFolder": None})
    assert process_mod.export_folder(client.ctx) == client.ctx.options.library / "exports"

    (client.ctx.options.data / "logs").mkdir(exist_ok=True)
    (client.ctx.options.data / "logs" / "backend.log").write_text("hello log", encoding="utf-8")
    pid = client.post("/api/projects", json={"name": "diag"}).json()["id"]
    import_media(client, pid, stereo_song(tmp_path / "s.wav", 1.0))
    job = client.post("/api/system/diagnostics").json()
    assert job["type"] == "diagnostics" and job["title"]["code"] == "job.diagnostics"
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done
    with zipfile.ZipFile(done["result"]["path"]) as zf:
        names = set(zf.namelist())
        assert {"about.json", "system.json", "preferences.json", "resources.json", "jobs.json", "library.json",
                "logs/backend.log"} <= names
        assert not any(n.endswith((".wav", ".pt", ".flac")) for n in names)
        library = json.loads(zf.read("library.json"))
        assert library["projects"]["count"] == 1 and library["projects"]["items"][0]["media"] == 1
    assert done["result"]["downloadUrl"].startswith("/files/")

    resources = {r["id"]: r for r in client.get("/api/resources").json()}
    assert resources["contentvec"]["license"] == "MIT" and resources["contentvec"]["homepage"].startswith("https://")


def test_job_pools_and_uncapped_editor():
    from stellaria_backend.events import EventHub
    from stellaria_backend.jobs import JobManager
    from stellaria_backend.jobs.manager import POOL_DSP
    from stellaria_backend.schemas import Msg

    jobs = JobManager(EventHub(), max_cpu_jobs=1, pools={POOL_DSP: 2})
    gate = threading.Event()

    def hold(jctx):
        gate.wait(10)
        return {}

    title = Msg(code="t")
    dsp = [jobs.submit("process", title, hold, pool=POOL_DSP) for _ in range(3)]
    plain = [jobs.submit("import", title, hold) for _ in range(2)]
    editors = [jobs.submit("vst_editor", title, hold) for _ in range(4)]
    time.sleep(0.5)
    running = lambda js: sum(1 for j in js if j.status == "running")  # noqa: E731
    assert (running(dsp), running(plain), running(editors)) == (2, 1, 4)
    gate.set()
    wait_for(lambda: all(not j.active for j in dsp + plain + editors), 10)
    jobs.shutdown(timeout=2)


# ---- the real dsp CLI ----------------------------------------------------------------------------------
@pytest.mark.skipif(not process_mod.dsp_available(), reason="stellaria_backend.dsp is not installed")
def test_real_dsp_cli_through_the_graph(client, tmp_path):
    """trim → split → comp → concat → fx → export with ``python -m stellaria_backend.dsp.run``."""
    pid = client.post("/api/projects", json={"name": "真 dsp"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 6.0))
    nodes = [
        {**source_node(song_id), "data": {"mediaId": song_id, "name": "歌"}},
        process_node("nd_tr", "trim", [("nd_s", "mix")], {"start": 1.0, "end": 4.0, "fadeIn": 0.01, "fadeOut": 0.05}),
        process_node("nd_sp", "split", [("nd_tr", "out")], {"cuts": [1.0], "fade": 0.01, "labels": []}),
        process_node("nd_c", "comp", [("nd_sp", "part1"), ("nd_sp", "part2")],
                     {"segments": [{"start": 1.0, "end": 2.0, "input": 0}, {"start": 2.0, "end": 4.0, "input": 1}],
                      "crossfade": 0.02}),
        process_node("nd_cc", "concat", [("nd_sp", "part2"), ("nd_sp", "part1")], {"gap": 0.5, "crossfade": 0}),
        process_node("nd_fx", "fx", [("nd_c", "out")], {"chain": [
            {"id": "s1", "kind": "builtin", "effect": "eq", "bypass": False, "mix": 1, "params": {}},
            {"id": "s2", "kind": "builtin", "effect": "compressor", "bypass": False, "mix": 0.5, "params": {}}]}),
        process_node("nd_x", "export", [("nd_fx", "out")], {
            "format": "wav", "sampleRate": 44100, "bitDepth": 16, "mp3Bitrate": 320, "loudness": None, "folder": None,
            "fileName": "{project} - {song} - {label}", "keepPosition": True, "tags": None}, label="成品"),
    ]
    put_canvas(client, pid, nodes)
    for node_id in ("nd_tr", "nd_sp", "nd_c", "nd_cc", "nd_fx"):
        run(client, pid, node_id)
    media = media_of(client, pid)
    out = {n: node_of(client, pid, n)["data"]["outputs"] for n in ("nd_tr", "nd_sp", "nd_c", "nd_cc", "nd_fx")}
    assert media[out["nd_tr"]["out"]]["offset"] == 1.0 and abs(media[out["nd_tr"]["out"]]["duration"] - 3.0) < 0.01
    offsets = [media[out["nd_sp"][p]]["offset"] for p in ("part1", "part2")]
    assert offsets[0] == 1.0 and abs(offsets[1] - 2.0) < 0.01  # cuts may snap to a quiet sample nearby
    assert media[out["nd_c"]["out"]]["offset"] == 1.0 and abs(media[out["nd_c"]["out"]]["duration"] - 3.0) < 0.05
    assert media[out["nd_cc"]["out"]]["offset"] == 0.0 and abs(media[out["nd_cc"]["out"]]["duration"] - 3.5) < 0.05
    assert media[out["nd_fx"]["out"]]["offset"] == 1.0 and all(media[m]["kind"] == "process" for o in out.values() for m in o.values())

    done = run(client, pid, "nd_x")
    path = Path(done["result"]["result"]["path"])
    assert path.parent == client.ctx.options.library / "exports" and path.name == "真 dsp - 歌 - 成品.wav"
    data, sr = sf.read(str(path))
    assert sr == 44100 and abs(len(data) / sr - (1.0 + media[out["nd_fx"]["out"]]["duration"])) < 0.05  # keepPosition
    assert node_of(client, pid, "nd_x")["data"]["result"]["path"] == str(path)


@pytest.mark.skipif(not process_mod.dsp_available(), reason="stellaria_backend.dsp is not installed")
def test_real_sections_and_key(client, tmp_path):
    pid = client.post("/api/projects", json={"name": "ana"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 8.0))
    sections = client.get(f"/api/projects/{pid}/media/{song_id}/sections").json()
    assert set(sections) >= {"bpm", "beats", "downbeats", "phrases", "sections"}
    assert all(isinstance(sections[k], list) for k in ("beats", "downbeats", "phrases", "sections"))
    analysis = client.get(f"/api/projects/{pid}/media/{song_id}/analysis").json()
    assert "key" in analysis and (analysis["key"] is None or isinstance(analysis["key"], str))
    assert client.ctx.projects.media_path(pid, song_id).with_suffix(".sections.json").is_file()


def test_startup_resets_nodes_left_running(ctx):
    project = ctx.projects.create("中断")
    project.canvas["nodes"] = [
        {"id": "s", "type": "source", "position": {"x": 0, "y": 0},
         "data": {"mediaId": "m", "name": "song", "stems": {"status": "running", "jobId": "job_gone", "vocals": None, "inst": None}}},
        {"id": "r", "type": "render", "position": {"x": 0, "y": 0},
         "data": {"input": {"nodeId": "s", "port": "vocals"}, "status": "running", "jobId": "job_gone", "mediaId": None}},
        {"id": "r2", "type": "render", "position": {"x": 0, "y": 0},
         "data": {"input": {"nodeId": "s", "port": "vocals"}, "status": "queued", "jobId": "job_gone", "mediaId": "med_old"}},
        {"id": "d", "type": "render", "position": {"x": 0, "y": 0},
         "data": {"input": {"nodeId": "s", "port": "vocals"}, "status": "done", "jobId": "job_old", "mediaId": "med_done"}},
    ]
    ctx.projects.save(project)
    assert ctx.projects.reset_interrupted() == 3
    nodes = {n["id"]: n["data"] for n in ctx.projects.get(project.id).canvas["nodes"]}
    assert nodes["s"]["stems"]["status"] == "failed" and nodes["s"]["stems"]["error"]["code"] == "job.interrupted"
    assert nodes["r"]["status"] == "failed" and nodes["r"]["jobId"] is None
    assert nodes["r2"]["status"] == "done" and nodes["r2"]["stale"] is True and nodes["r2"]["mediaId"] == "med_old"
    assert nodes["d"]["status"] == "done" and nodes["d"]["jobId"] == "job_old"
    assert ctx.projects.reset_interrupted() == 0
