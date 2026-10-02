"""End-to-end with the REAL DDSP-SVC engine on CPU (slow; run with ``-m e2e``).

Needs torch in the interpreter running the tests and the pretrained files in
``$STELLARIA_E2E_MODELS``:
  rmvpe/model.pt
  nsf/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/{model.ckpt,config.json,NOTICE*.txt,STATEMENTS.txt}
  hubert-soft-0d54a1f4.pt
They are linked into ``<data>/models`` following the manifest's ``dest`` layout. ContentVec is
not needed: the model uses the ``hubertsoft`` encoder.
"""

from __future__ import annotations

import io
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import TOKEN, make_ctx, wait_for, wait_job
from stellaria_backend.audio.io import write_wav
from stellaria_backend.schemas import Clip
from stellaria_backend.util.timeutil import now_iso
from synth import song, write

pytestmark = pytest.mark.e2e

MODELS = Path(os.environ.get("STELLARIA_E2E_MODELS", "")).expanduser()
NSF = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02"


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def install_models(data: Path) -> None:
    models = data / "models"
    _link(MODELS / "rmvpe" / "model.pt", models / "rmvpe" / "model.pt")
    for name in ("model.ckpt", "config.json", "NOTICE.txt", "NOTICE.zh-CN.txt", "STATEMENTS.txt"):
        _link(MODELS / "nsf" / NSF / name, models / "nsf_hifigan" / name)
    _link(MODELS / "hubert-soft-0d54a1f4.pt", models / "hubertsoft" / "hubert-soft-0d54a1f4.pt")


def add_tone_clips(ctx, cid: str, n: int = 20) -> None:
    rng = np.random.default_rng(0)
    clips = []
    for i in range(n):
        seconds = float(rng.uniform(3.0, 4.0))
        name = f"tone_{i:02d}"
        audio = song(seconds, 180 + 9 * i, seed=100 + i, phrase_len=seconds, gap=0.0)
        write_wav(ctx.characters.clips_dir(cid) / f"{name}.wav", audio)
        clips.append(Clip(id=f"clp_e2e{i}", name=name, file=f"clips/{name}.wav", duration=round(len(audio) / 44100, 3),
                          split="val" if i < 2 else "train", created_at=now_iso()))
    ctx.characters.save_clips(cid, clips)


@pytest.mark.timeout(5400)
@pytest.mark.skipif(not (MODELS / "rmvpe" / "model.pt").is_file(), reason="set STELLARIA_E2E_MODELS")
def test_real_engine_on_cpu(tmp_path):
    from fastapi.testclient import TestClient

    from stellaria_backend.app import create_app
    from stellaria_backend.context import AppContext

    pytest.importorskip("torch")
    root = tmp_path / "端到端 e2e"
    options = make_ctx(root, fake=False).options  # only for the paths
    install_models(options.data)
    options.allow_cpu_train = True
    ctx = AppContext(options)  # real torch probe in a subprocess
    with TestClient(create_app(options, ctx=ctx), headers={"Authorization": f"Bearer {TOKEN}"}) as client:
        info = client.get("/api/system").json()
        assert info["torch"]["available"] and info["canTrain"] and not info["fakeEngine"]
        resources = {r["id"]: r["status"] for r in client.get("/api/resources").json()}
        assert resources["rmvpe"] == resources["nsf_hifigan"] == resources["hubertsoft"] == "ready"
        assert resources["contentvec"] == "missing"

        cid = client.post("/api/characters", json={"name": "音 E2E"}).json()["id"]
        add_tone_clips(ctx, cid)
        config = {"encoder": "hubertsoft", "f0Extractor": "rmvpe", "nLayers": 2, "nChans": 64, "nAuxLayers": 1,
                  "nAuxChans": 64, "batchSize": 4, "numWorkers": 0, "ampDtype": "fp32", "targetSteps": 30,
                  "intervalVal": 15, "intervalForceSave": 30}
        missing = client.post("/api/models", json={"characterIds": [cid], "config": {**config, "encoder": "contentvec768l12"}}).json()
        failed = wait_job(client, client.post(f"/api/models/{missing['id']}/train").json()["id"])
        assert failed["status"] == "failed" and failed["error"]["code"] == "engine.missing_model"
        assert "resource=contentvec" in failed["error"]["detail"]
        assert client.get(f"/api/models/{missing['id']}").json()["error"] == {"code": "engine.missing_model",
                                                                              "params": {"resource": "contentvec"}}

        mid = client.post("/api/models", json={"characterIds": [cid], "config": config}).json()["id"]
        started = time.time()
        job = client.post(f"/api/models/{mid}/train").json()
        done = wait_job(client, job["id"], timeout=1500)
        log = "\n".join(client.get(f"/api/jobs/{job['id']}/log?tail=80").json()["lines"])
        assert done["status"] == "succeeded", f"{done}\n{log}"
        print(f"first training run: {time.time() - started:.0f}s")
        m = client.get(f"/api/models/{mid}").json()
        assert m["status"] == "done" and m["step"] == 30 and m["dataset"]["status"] == "ready"
        # at step 30 the engine itself deletes model_15 (not a multiple of intervalForceSave)
        assert {c["step"]: c["kind"] for c in m["checkpoints"]} == {30: "permanent"}
        ds = ctx.models.dataset_dir(mid)
        assert (ds / "train" / "pitch_aug_dict.npy").is_file() and len(list((ds / "train" / "units").glob("*.npy"))) == 18

        series = {s["tag"]: s["points"] for s in client.get(f"/api/models/{mid}/metrics").json()}
        assert len(series["train/loss"]) == 30 and series["train/loss"][-1][0] == 30
        assert {"validation/loss", "validation/mel_val_snr", "train/lr"} <= set(series)
        assert client.get(f"/api/models/{mid}/validation/steps").json() == [15, 30]
        snap = client.get(f"/api/models/{mid}/validation").json()
        assert snap["step"] == 30 and len(snap["samples"]) == 2
        sample = snap["samples"][0]
        gt, sr = sf.read(io.BytesIO(client.get("/api" + sample["gtUrl"]).content))
        pred, _ = sf.read(io.BytesIO(client.get("/api" + sample["predUrl"]).content))
        assert sr == 44100 and abs(len(gt) - len(pred)) < 2048
        mel = client.get("/api" + sample["melUrl"]).json()
        assert mel["bins"] == 128 and mel["frames"] > 200

        # more steps, pause in the middle (STOP), resume to the end
        m = client.patch(f"/api/models/{mid}", json={"targetSteps": 60}).json()
        assert m["status"] == "paused" and m["targetSteps"] == 60
        job = client.post(f"/api/models/{mid}/train").json()
        wait_for(lambda: client.get(f"/api/models/{mid}").json()["step"] >= 34, 900, 0.5)
        client.post(f"/api/models/{mid}/pause")
        done = wait_job(client, job["id"], timeout=900)
        assert done["status"] == "succeeded" and done["result"]["paused"] is True, done
        m = client.get(f"/api/models/{mid}").json()
        pause_steps = [c["step"] for c in m["checkpoints"] if c["kind"] == "pause"]
        assert m["status"] == "paused" and len(pause_steps) == 1 and pause_steps[0] % 10 == 0 and pause_steps[0] > 30
        p = client.post("/api/projects", json={"name": "E2E"}).json()
        src = write(tmp_path / "输入 song.wav", song(5.0, 200, seed=5, gap=0.4))
        media_id = wait_job(client, client.post(f"/api/projects/{p['id']}/media", json={"paths": [str(src)]}).json()["id"])["result"]["mediaIds"][0]
        render_data = {
            "input": {"nodeId": "nd_src", "port": "mix"}, "characterId": cid, "modelId": mid, "releaseId": None, "mediaId": None,
            "params": {"key": 2, "formant": 0, "register": 0, "f0Extractor": "rmvpe", "steps": 10, "method": "euler",
                       "tStart": 0.0, "threshold": -60, "spkMix": None, "region": None},
            "status": "draft", "jobId": None, "error": None, "rating": 0, "favorite": False, "label": ""}
        canvas = {"nodes": [
            {"id": "nd_src", "type": "source", "position": {"x": 0, "y": 0}, "data": {"mediaId": media_id, "name": "in"}},
            {"id": "nd_r", "type": "render", "position": {"x": 300, "y": 0}, "data": render_data}],
            "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}
        client.put(f"/api/projects/{p['id']}", json={"canvas": canvas})

        # resume; meanwhile render the latest checkpoint of the training model on the CPU
        client.put("/api/preferences", json={"gpuPolicy": "cpu"})
        job = client.post(f"/api/models/{mid}/train").json()
        wait_for(lambda: client.get(f"/api/models/{mid}").json()["status"] == "training", 300, 0.5)
        live = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()["id"], timeout=600)
        assert live["status"] == "succeeded", live
        assert live["result"]["device"] == "cpu" and live["stage"]["code"] == "stage.render_cpu"
        assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "running"  # training was not disturbed
        assert list((ctx.models.cache_dir(mid) / "render").glob("model_*.pt"))
        done = wait_job(client, job["id"], timeout=1500)
        assert done["status"] == "succeeded" and done["result"]["done"] is True, done
        m = client.get(f"/api/models/{mid}").json()
        kinds = {c["step"]: c["kind"] for c in m["checkpoints"]}
        assert m["step"] == 60 and kinds == {30: "permanent", 60: "permanent"}, kinds  # pause checkpoint pruned
        series = {s["tag"]: s["points"] for s in client.get(f"/api/models/{mid}/metrics").json()}
        assert [p[0] for p in series["train/loss"]] == list(range(1, 61))

        # release + render through the persistent worker
        rel = client.post(f"/api/models/{mid}/releases", json={"step": 60, "name": "e2e"}).json()
        canvas["nodes"][1]["data"] = {**render_data, "releaseId": rel["id"]}
        client.put(f"/api/projects/{p['id']}", json={"canvas": canvas})
        for attempt in range(2):  # the second render reuses the loaded worker and the f0 cache
            started = time.time()
            done = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()["id"], timeout=600)
            assert done["status"] == "succeeded" and done["result"]["device"] == "cpu", done
            assert done["stage"]["code"] == "stage.render"  # nothing is training: no CPU fallback
            print(f"render {attempt}: {time.time() - started:.1f}s")
        out, sr = sf.read(io.BytesIO(client.get(f"/api/projects/{p['id']}/media/{done['result']['mediaId']}/audio").content))
        assert sr == 44100 and abs(len(out) / sr - 5.0) < 0.1 and np.sqrt(np.mean(out ** 2)) > 1e-3
        assert ctx.render.device == "cpu" and any((options.data / "cache" / "f0").glob("raw_rmvpe_*.npy"))

        # the pitch editor: the input's line (RMVPE through the worker) and a phrase sung an octave up
        a = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/pitch/analyze").json()["id"], timeout=600)
        assert a["status"] == "succeeded", a
        line = client.get(f"/api/projects/{p['id']}/nodes/nd_r/pitch").json()
        assert line["current"] and len(line["notes"]) >= 2 and abs(line["duration"] - 5.0) < 0.1
        edit = {"rate": 100, "segments": [{"start": 0.5, "values": [12.0] * 100}]}
        pv = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/pitch/preview",
                                          json={"start": 0.5, "end": 1.5, "pitchEdit": edit}).json()["id"], timeout=600)
        assert pv["status"] == "succeeded", pv
        sung, sr = sf.read(io.BytesIO(client.get("/api" + pv["result"]["url"]).content))
        assert sr == 44100 and abs(len(sung) / sr - 1.0) < 0.05 and np.sqrt(np.mean(sung ** 2)) > 1e-3
        job = client.post(f"/api/models/{mid}/quick-render", json={"path": str(src), "characterId": cid}).json()
        done = wait_job(client, job["id"], timeout=600)
        assert done["status"] == "succeeded", done

        # the trained checkpoint, imported as a model from elsewhere (real inspector), renders too
        ext = tmp_path / "外部 model"
        ext.mkdir()
        shutil.copy2(ctx.models.releases_dir(mid) / rel["id"] / "model.pt", ext / "model_60.pt")
        shutil.copy2(ctx.models.releases_dir(mid) / rel["id"] / "config.yaml", ext / "config.yaml")
        done = wait_job(client, client.post("/api/models/import-external", json={
            "ckptPaths": [str(ext / "model_60.pt")], "newCharacter": {"name": "外来"}}).json()["id"], timeout=600)
        assert done["status"] == "succeeded", done
        imported = done["result"]
        assert client.get(f"/api/models/{imported['modelId']}").json()["releases"][0]["name"] == "外来 · 60"
        canvas["nodes"][1]["data"] = {**render_data, "characterId": imported["characterId"], "modelId": imported["modelId"],
                                      "releaseId": imported["releaseIds"][0],
                                      "params": {**render_data["params"], "f0Min": 80, "f0Max": 800}}
        client.put(f"/api/projects/{p['id']}", json={"canvas": canvas})
        done = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()["id"], timeout=600)
        assert done["status"] == "succeeded", done

        # a base model made from the release seeds a new voice's first training run
        base = client.post("/api/base-models/from-release", json={"modelId": mid, "releaseId": rel["id"], "name": ""}).json()
        assert base["source"] == "release" and base["step"] == 60 and base["nChans"] == 64
        cid2 = client.post("/api/characters", json={"name": "Seeded"}).json()["id"]
        add_tone_clips(ctx, cid2)
        seeded = client.post("/api/models", json={"characterIds": [cid2], "config": {
            **config, "nChans": 128, "baseModelId": base["id"], "targetSteps": 5, "intervalVal": 5, "intervalForceSave": 5}}).json()
        assert seeded["config"]["nChans"] == 64
        job = client.post(f"/api/models/{seeded['id']}/train").json()
        done = wait_job(client, job["id"], timeout=1500)
        log = "\n".join(client.get(f"/api/jobs/{job['id']}/log?tail=400").json()["lines"])
        assert done["status"] == "succeeded" and done["result"] == {"done": True, "step": 5}, f"{done}\n{log}"
        assert f"seeded from base model {base['id']}: kept" in log and "dropped 0" in log
        engine_log = ctx.models.log_path(seeded["id"]).read_text("utf-8", errors="replace")
        assert "restoring model from" in engine_log and "model_0.pt" in engine_log  # the engine resumed from the seed
        m = client.get(f"/api/models/{seeded['id']}").json()
        assert {c["step"] for c in m["checkpoints"]} == {5}  # seed pruned

        # a small canvas v3 graph on the real engine: song → trim (dsp) → take (release) → fx → auto mix
        # (+ an imported backing track) → export; offsets keep everything on the song's timeline
        from stellaria_backend.creation.process import dsp_available

        if dsp_available():
            backing = write(tmp_path / "伴奏.wav", np.stack([song(5.0, 110, seed=6, gap=0.0)] * 2, axis=1) * 0.4)
            backing_id = wait_job(client, client.post(f"/api/projects/{p['id']}/media", json={"paths": [str(backing)]}).json()["id"])["result"]["mediaIds"][0]
            client.put("/api/preferences", json={"exportFolder": str(tmp_path / "导出")})

            def proc(node_id, op, inputs, params):
                return {"id": node_id, "type": "process", "position": {"x": 0, "y": 0}, "data": {
                    "op": op, "label": "", "inputs": inputs, "params": params, "outputs": {}, "status": "draft",
                    "jobId": None, "error": None, "stale": False, "bypass": False, "result": None}}

            graph = [
                canvas["nodes"][0],
                {"id": "nd_bk", "type": "audio", "position": {"x": 0, "y": 0}, "data": {"mediaId": backing_id, "name": "伴奏", "role": "backing"}},
                proc("nd_tr", "trim", [{"nodeId": "nd_src", "port": "mix"}], {"start": 1.0, "end": 4.0, "fadeIn": 0.01, "fadeOut": 0.05}),
                {"id": "nd_t", "type": "render", "position": {"x": 0, "y": 0},
                 "data": {**render_data, "input": {"nodeId": "nd_tr", "port": "out"}, "releaseId": rel["id"], "status": "draft",
                          "mediaId": None, "stale": False, "batchId": None}},
                proc("nd_fx", "fx", [{"nodeId": "nd_t", "port": "out"}], {"chain": [
                    {"id": "s1", "kind": "builtin", "effect": "eq", "bypass": False, "mix": 1, "params": {}}]}),
                {"id": "nd_m", "type": "mix", "position": {"x": 0, "y": 0}, "data": {
                    "inputs": [{"id": "lead", "ref": {"nodeId": "nd_fx", "port": "out"}, "role": "lead", "gainDb": 0, "pan": 0, "mute": False, "solo": False},
                               {"id": "bk", "ref": {"nodeId": "nd_bk", "port": "out"}, "role": "backing", "gainDb": -2, "pan": 0, "mute": False, "solo": False}],
                    "processor": "builtin", "preset": "natural", "params": {}, "variants": 1, "outputs": [], "selected": 0,
                    "status": "draft", "jobId": None, "error": None, "stale": False, "label": "成品"}},
                proc("nd_x", "export", [{"nodeId": "nd_m", "port": "out"}], {
                    "format": "flac", "sampleRate": 44100, "bitDepth": 24, "mp3Bitrate": 320, "loudness": -14, "folder": None,
                    "fileName": "{project} - {voice}", "keepPosition": False, "tags": None}),
            ]
            client.put(f"/api/projects/{p['id']}", json={"canvas": {**canvas, "nodes": graph}})
            for node_id in ("nd_tr", "nd_t", "nd_fx", "nd_m", "nd_x"):
                job = client.post(f"/api/projects/{p['id']}/nodes/{node_id}/run").json()
                done = wait_job(client, job["id"], timeout=600)
                assert done["status"] == "succeeded", done
            nodes = {n["id"]: n["data"] for n in client.get(f"/api/projects/{p['id']}").json()["canvas"]["nodes"]}
            media = {m["id"]: m for m in client.get(f"/api/projects/{p['id']}/media").json()}
            assert media[nodes["nd_tr"]["outputs"]["out"]]["offset"] == 1.0
            assert media[nodes["nd_t"]["mediaId"]]["offset"] == 1.0 and media[nodes["nd_fx"]["outputs"]["out"]]["offset"] == 1.0
            assert media[nodes["nd_m"]["outputs"][0]["mediaId"]]["offset"] == 0.0
            exported = Path(nodes["nd_x"]["result"]["path"])
            assert exported.parent == tmp_path / "导出" and exported.name == "E2E - 音 E2E.flac" and exported.stat().st_size > 10000

