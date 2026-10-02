"""External models (trained elsewhere), base models (底模) with seeding, the PATCH rules for
structural training settings, and f0Min/f0Max render params."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from conftest import BACKEND, ENGINE, make_ctx, wait_job
from stellaria_backend.engine.checkpoints import checkpoint_steps
from stellaria_backend.engine.modelfiles import inspect_files, run_inspector
from stellaria_backend.engine.rendering import parse_render_params, render_args
from stellaria_backend.errors import ApiError
from synth import song, write
from test_creation import import_media, node_of, put_canvas, render_node, source_node
from test_dataset_config import add_clips

TEMPLATE = ENGINE / "configs" / "reflow.yaml"


def engine_config(path: Path, **changes) -> Path:
    """The engine's reflow template with ``section__key=value`` changes (None removes the key)."""
    conf = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    for key, value in changes.items():
        section, name = key.split("__")
        if value is None:
            conf[section].pop(name, None)
        else:
            conf[section][name] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(conf, allow_unicode=True), encoding="utf-8")
    return path


def fake_ckpt(path: Path, step: int | None = None, empty: bool = False) -> Path:
    """A fake-engine checkpoint: a JSON header line ({global_step}) and some bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = {"fake": True, **({"global_step": step} if step is not None else {})}
    path.write_bytes(b"" if empty else json.dumps(header).encode() + b"\n" + os.urandom(256))
    return path


def import_external(client, body: dict) -> dict:
    job = client.post("/api/models/import-external", json=body).json()
    assert job["type"] == "import_model", job
    return wait_job(client, job["id"])


def import_base(client, tmp_path: Path, name: str = "Big base", step: int = 300000, **config) -> str:
    folder = tmp_path / f"base {name}"
    engine_config(folder / "config.yaml", **config)
    fake_ckpt(folder / "model_best.pt", step)
    job = client.post("/api/base-models/import", json={"ckptPath": str(folder / "model_best.pt"), "name": name}).json()
    assert job["type"] == "import_base"
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done
    return done["result"]["baseModelId"]


# ---- external models -------------------------------------------------------------------------------
def test_import_external_single_voice(client, tmp_path):
    ctx = client.ctx
    folder = tmp_path / "别人的 model"
    engine_config(folder / "config.yaml", model__glu_type=None, data__encoder="contentvec768l12", data__encoder_hop_size=320,
                  model__n_layers=3, model__n_chans=256, data__f0_extractor="fcpe")
    ckpts = [fake_ckpt(folder / "model_8000.pt", 8000), fake_ckpt(folder / "latest.pt", 12000)]
    done = import_external(client, {"ckptPaths": [str(p) for p in ckpts],
                                    "newCharacter": {"name": "外来 Voice", "color": "#8899aa"}})
    assert done["status"] == "succeeded", done
    res = done["result"]
    assert set(res) == {"characterId", "characterIds", "modelId", "releaseIds"} and len(res["releaseIds"]) == 2
    mid, cid = res["modelId"], res["characterId"]
    assert res["characterIds"] == [cid]

    model = client.get(f"/api/models/{mid}").json()
    assert model["kind"] == "single" and model["status"] == "done" and model["name"] == "外来 Voice"
    assert [(r["name"], r["step"], r["notes"]) for r in model["releases"]] == [("外来 Voice · 8000", 8000, ""),
                                                                               ("外来 Voice · 12000", 12000, "")]
    cfg = model["config"]
    assert (cfg["encoder"], cfg["nLayers"], cfg["nChans"], cfg["f0Extractor"], cfg["targetSteps"], cfg["baseModelId"]) == \
        ("contentvec768l12", 3, 256, "fcpe", 12000, None)
    assert ctx.models.get(mid).glu_type == "atanglu"  # config without glu_type = atanglu-trained (engine rule)
    exp = ctx.models.exp_dir(mid)
    assert set(checkpoint_steps(exp)) == {8000, 12000} and (exp / "config.yaml").is_file()
    for rel in model["releases"]:
        folder_ = ctx.models.releases_dir(mid) / rel["id"]
        assert (folder_ / "model.pt").is_file() and (folder_ / "config.yaml").is_file()

    char = client.get(f"/api/characters/{cid}").json()
    assert (char["name"], char["color"], char["stage"]) == ("外来 Voice", "#8899aa", "trained")
    assert client.get(f"/api/characters/{cid}/clips").json() == []

    # release notes travel everywhere a release does
    rel_id = res["releaseIds"][0]
    assert client.patch(f"/api/models/{mid}/releases/{rel_id}", json={"notes": "从别处导入"}).json()["notes"] == "从别处导入"
    listed = next(m for m in client.get("/api/models").json() if m["id"] == mid)
    assert listed["releases"][0]["notes"] == "从别处导入"
    assert json.loads((ctx.models.releases_dir(mid) / rel_id / "release.json").read_text("utf-8"))["notes"] == "从别处导入"

    # and the imported voice renders (fake engine)
    p = client.post("/api/projects", json={"name": "外来"}).json()
    media = import_media(client, p["id"], write(tmp_path / "in.wav", song(3, 220, seed=4)))
    put_canvas(client, p["id"], [source_node(media), render_node("nd_r", "mix", characterId=cid, modelId=mid,
                                                                  releaseId=rel_id)])
    done = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/render").json()["id"])
    assert done["status"] == "succeeded", done
    assert node_of(client, p["id"], "nd_r")["data"]["status"] == "done"


def test_import_external_multi_speaker(client, tmp_path):
    ctx = client.ctx
    first = ctx.characters.create("Alto").id
    folder = tmp_path / "trio"
    engine_config(folder / "config.yaml", model__n_spk=3)
    done = import_external(client, {"ckptPaths": [str(fake_ckpt(folder / "model_20000.pt"))],
                                    "characterId": first, "name": "Trio"})
    assert done["status"] == "succeeded", done
    res = done["result"]
    assert res["characterId"] == first and len(res["characterIds"]) == 3
    model = client.get(f"/api/models/{res['modelId']}").json()
    assert model["kind"] == "constellation" and model["releases"][0]["name"] == "Trio · 20000"
    assert [(c["characterId"], c["spkId"]) for c in model["characters"]] == [(c, i + 1) for i, c in enumerate(res["characterIds"])]
    names = [client.get(f"/api/characters/{c}").json()["name"] for c in res["characterIds"]]
    assert names == ["Alto", "Trio spk2", "Trio spk3"]
    assert ctx.models.get(res["modelId"]).glu_type == "softsign_glu"

    # characterIds maps every speaker explicitly
    a, b = ctx.characters.create("A").id, ctx.characters.create("B").id
    engine_config(tmp_path / "duo" / "config.yaml", model__n_spk=2)
    done = import_external(client, {"ckptPaths": [str(fake_ckpt(tmp_path / "duo" / "model_500.pt"))],
                                    "characterIds": [a, b]})
    assert done["status"] == "succeeded" and done["result"]["characterIds"] == [a, b]


@pytest.mark.parametrize("changes, reason", [
    ({"model__type": "Diffusion"}, "not_reflow"),
    ({"data__sampling_rate": 32000}, "sample_rate"),
    ({"vocoder__type": "hifigan"}, "vocoder"),
    ({"data__encoder": "cnhubertsoftfish"}, "encoder"),
    ({"data__encoder": "hubertsoft"}, "encoder"),  # encoder_out_channels still 768
])
def test_import_external_incompatible(client, tmp_path, changes, reason):
    folder = tmp_path / "bad"
    engine_config(folder / "config.yaml", **changes)
    before = (len(client.get("/api/models").json()), len(client.get("/api/characters").json()))
    done = import_external(client, {"ckptPaths": [str(fake_ckpt(folder / "model_100.pt", 100))],
                                    "newCharacter": {"name": "X"}})
    assert done["status"] == "failed" and done["error"]["code"] == "model.import_incompatible"
    assert done["error"]["params"] == {"reason": reason}
    assert (len(client.get("/api/models").json()), len(client.get("/api/characters").json())) == before


def test_import_external_bad_inputs(client, tmp_path):
    folder = tmp_path / "broken"
    engine_config(folder / "config.yaml")
    good, empty = fake_ckpt(folder / "model_100.pt", 100), fake_ckpt(folder / "model_200.pt", empty=True)
    done = import_external(client, {"ckptPaths": [str(good), str(empty)], "newCharacter": {"name": "Gone"}})
    assert done["status"] == "failed" and done["error"]["code"] == "model.import_bad_checkpoint"
    assert done["error"]["params"] == {"file": "model_200.pt"}
    assert not any(c["name"] == "Gone" for c in client.get("/api/characters").json())  # created voice removed again

    r = client.post("/api/models/import-external", json={"ckptPaths": [str(folder / "nope.pt")]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "import.not_found"
    r = client.post("/api/models/import-external", json={"ckptPaths": [str(good)], "configPath": str(tmp_path / "x.yaml")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "import.not_found"
    r = client.post("/api/models/import-external", json={"ckptPaths": [str(good)], "characterId": "chr_missing"})
    assert r.status_code == 404
    assert client.post("/api/models/import-external", json={"ckptPaths": []}).status_code == 422

    # steps: global_step, else the file name, else made up and unique
    other = tmp_path / "steps"
    engine_config(other / "config.yaml")
    paths = [fake_ckpt(other / "a.pt", 5000), fake_ckpt(other / "model_5000.pt"), other / "weights.pt"]
    paths[2].write_bytes(b"\x80\x02 not a header")
    done = import_external(client, {"ckptPaths": [str(p) for p in paths], "name": "Steps"})
    assert done["status"] == "succeeded", done
    model = client.get(f"/api/models/{done['result']['modelId']}").json()
    assert [r["step"] for r in model["releases"]] == [5000, 5001, 5002]


# ---- base models -----------------------------------------------------------------------------------
def test_base_models_crud(client, tmp_path):
    ctx = client.ctx
    bid = import_base(client, tmp_path, data__encoder="hubertsoft", data__encoder_out_channels=256, model__n_layers=3,
                      model__n_chans=256, model__glu_type="atanglu", model__n_spk=8)
    base = client.get(f"/api/base-models/{bid}").json()
    assert {k: base[k] for k in ("name", "notes", "source", "resourceId", "status", "encoder", "nLayers", "nChans",
                                 "nAuxLayers", "nAuxChans", "nSpk", "step")} == {
        "name": "Big base", "notes": "", "source": "local", "resourceId": None, "status": "ready", "encoder": "hubertsoft",
        "nLayers": 3, "nChans": 256, "nAuxLayers": 6, "nAuxChans": 512, "nSpk": 8, "step": 300000}
    assert base["size"] > 0 and base["createdAt"] and "gluType" not in base
    assert [b["id"] for b in client.get("/api/base-models").json()] == [bid]

    folder = tmp_path / "not reflow"
    engine_config(folder / "config.yaml", model__type="Diffusion")
    job = client.post("/api/base-models/import", json={"ckptPath": str(fake_ckpt(folder / "m.pt", 1)), "name": "x"}).json()
    failed = wait_job(client, job["id"])
    assert failed["error"]["code"] == "model.import_incompatible" and failed["error"]["params"] == {"reason": "not_reflow"}
    r = client.post("/api/base-models/import", json={"ckptPath": str(tmp_path / "none.pt"), "name": "x"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "import.not_found"

    patched = client.patch(f"/api/base-models/{bid}", json={"name": "多人底模", "notes": "300k steps"}).json()
    assert (patched["name"], patched["notes"]) == ("多人底模", "300k steps")

    # from one of the user's releases
    engine_config(tmp_path / "mine" / "config.yaml")
    imported = import_external(client, {"ckptPaths": [str(fake_ckpt(tmp_path / "mine" / "model_9000.pt", 9000))],
                                        "name": "Mine"})["result"]
    from_rel = client.post("/api/base-models/from-release", json={"modelId": imported["modelId"],
                                                                  "releaseId": imported["releaseIds"][0], "name": ""})
    assert from_rel.status_code == 200, from_rel.json()
    rel_base = from_rel.json()
    assert (rel_base["source"], rel_base["name"], rel_base["step"], rel_base["encoder"]) == \
        ("release", "Mine · 9000", 9000, "contentvec768l12tta2x")
    assert client.post("/api/base-models/from-release", json={"modelId": imported["modelId"], "releaseId": "rel_x",
                                                              "name": ""}).status_code == 404

    # a model that hasn't trained yet keeps its base from being deleted
    cid = ctx.characters.create("Uses base").id
    mid = client.post("/api/models", json={"characterIds": [cid], "config": {"baseModelId": bid}}).json()["id"]
    r = client.delete(f"/api/base-models/{bid}")
    assert r.status_code == 409 and r.json()["error"] == {
        "code": "base.in_use", "message": r.json()["error"]["message"], "detail": "", "params": {"models": "Uses base"}}
    client.delete(f"/api/models/{mid}")
    assert client.delete(f"/api/base-models/{bid}").json() == {"ok": True}
    assert client.get(f"/api/base-models/{bid}").status_code == 404
    assert [b["id"] for b in client.get("/api/base-models").json()] == [rel_base["id"]]


def test_manifest_base_models(tmp_path):
    from fastapi.testclient import TestClient

    from conftest import MANIFEST, TOKEN
    from stellaria_backend.app import create_app

    manifest = json.loads(MANIFEST.read_text("utf-8"))
    manifest["resources"].append({
        "id": "base_reflow_multi", "kind": "base", "required": False,
        "name": {"zh-CN": "多人底模", "en": "Multi-speaker base"}, "description": {"zh-CN": "说明", "en": "About"},
        "size": 123456, "sha256": None, "install": {"type": "file", "dest": "base/multi/model.pt"},
        "sources": [{"id": "example", "region": "global", "url": "https://example.invalid/model.pt"}],
        "base": {"encoder": "contentvec768l12", "nLayers": 6, "nChans": 512, "nAuxLayers": 4, "nAuxChans": 256,
                 "nSpk": 40, "step": 500000, "gluType": "atanglu"},
    })
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), "utf-8")
    ctx = make_ctx(tmp_path, resources=path)
    with TestClient(create_app(ctx.options, ctx=ctx), headers={"Authorization": f"Bearer {TOKEN}"}) as client:
        bases = client.get("/api/base-models").json()
        assert len(bases) == 1
        base = bases[0]
        assert {k: base[k] for k in ("id", "name", "source", "resourceId", "status", "nChans", "nSpk", "step", "size")} == {
            "id": "base_reflow_multi", "name": "多人底模", "source": "download", "resourceId": "base_reflow_multi",
            "status": "missing", "nChans": 512, "nSpk": 40, "step": 500000, "size": 123456}
        assert any(r["id"] == "base_reflow_multi" and r["kind"] == "base" for r in client.get("/api/resources").json())
        for r in (client.patch("/api/base-models/base_reflow_multi", json={"name": "x"}),
                  client.delete("/api/base-models/base_reflow_multi")):
            assert r.status_code == 409 and r.json()["error"]["code"] == "base.read_only"
        cid = ctx.characters.create("M").id
        model = client.post("/api/models", json={"characterIds": [cid], "config": {"baseModelId": "base_reflow_multi",
                                                                                   "nChans": 1024}}).json()
        assert (model["config"]["encoder"], model["config"]["nChans"], model["config"]["nAuxLayers"]) == ("contentvec768l12", 512, 4)
        assert client.post("/api/models", json={"characterIds": [cid], "config": {"baseModelId": "base_nope"}}).status_code == 404


def test_seeding_and_structural_rules(client, tmp_path):
    ctx = client.ctx
    ctx.options.fake_speed = 400.0
    bid = import_base(client, tmp_path, model__n_layers=3, model__n_chans=256, model__glu_type="atanglu")
    other = import_base(client, tmp_path, name="Other", model__n_layers=4, model__n_chans=384)
    cid = ctx.characters.create("Seeded").id
    add_clips(ctx, cid, 6)
    model = client.post("/api/models", json={"characterIds": [cid], "config": {
        "baseModelId": bid, "nLayers": 9, "targetSteps": 100, "intervalVal": 50, "intervalForceSave": 100}}).json()
    mid = model["id"]
    assert (model["config"]["nLayers"], model["config"]["nChans"], model["config"]["baseModelId"]) == (3, 256, bid)
    assert ctx.models.get(mid).glu_type == "atanglu"

    # before the first checkpoint everything may change; structure follows the base
    m = client.patch(f"/api/models/{mid}", json={"config": {"baseModelId": other, "nChans": 2048}}).json()
    assert (m["config"]["nLayers"], m["config"]["nChans"]) == (4, 384) and ctx.models.get(mid).glu_type == "softsign_glu"
    m = client.patch(f"/api/models/{mid}", json={"config": {"baseModelId": None, "nChans": 512}}).json()
    assert (m["config"]["nLayers"], m["config"]["nChans"], m["config"]["baseModelId"]) == (4, 512, None)
    assert ctx.models.get(mid).glu_type is None
    # a seed that was never trained on goes when the base changes
    seed = fake_ckpt(ctx.models.exp_dir(mid) / "model_0.pt", 0)
    client.patch(f"/api/models/{mid}", json={"config": {"baseModelId": bid}})
    assert not seed.exists()

    job = client.post(f"/api/models/{mid}/train").json()
    done = wait_job(client, job["id"], timeout=60)
    assert done["status"] == "succeeded", done
    log = "\n".join(client.get(f"/api/jobs/{job['id']}/log").json()["lines"])
    assert f"seeded from base model {bid}" in log
    assert "restoring model from" in ctx.models.log_path(mid).read_text("utf-8")  # the trainer resumed from the seed
    exp = ctx.models.exp_dir(mid)
    assert 0 not in checkpoint_steps(exp) and 100 in checkpoint_steps(exp)  # seed pruned once training saved

    # after the first checkpoint: non-structural fields only
    m = client.patch(f"/api/models/{mid}", json={"config": {"batchSize": 8, "lr": 0.0002, "f0Extractor": "fcpe",
                                                            "usePitchAug": False, "nLayers": 3}})
    assert m.status_code == 200 and m.json()["config"]["batchSize"] == 8
    ignored = client.patch(f"/api/models/{mid}", json={"config": {"nChans": 512, "encoder": "hubertsoft"}}).json()
    assert (ignored["config"]["nChans"], ignored["config"]["encoder"]) == (256, "contentvec768l12tta2x")  # from the base
    for change in ({"baseModelId": other}, {"baseModelId": None}):
        r = client.patch(f"/api/models/{mid}", json={"config": change})
        assert r.status_code == 409 and r.json()["error"]["code"] == "train.config_mismatch", change
    assert client.delete(f"/api/base-models/{bid}").json() == {"ok": True}  # trained: no longer needs the base


def test_seed_base_missing_fails_before_preprocessing(client, tmp_path):
    ctx = client.ctx
    bid = import_base(client, tmp_path)
    cid = ctx.characters.create("Lost base").id
    add_clips(ctx, cid, 6)
    mid = client.post("/api/models", json={"characterIds": [cid], "config": {"baseModelId": bid, "targetSteps": 50}}).json()["id"]
    import shutil

    shutil.rmtree(ctx.base_models.root / bid)
    done = wait_job(client, client.post(f"/api/models/{mid}/train").json()["id"])
    assert done["status"] == "failed" and done["error"]["code"] == "base_not_found"
    assert not checkpoint_steps(ctx.models.exp_dir(mid))


def test_structure_patch_without_base(client, tmp_path):
    ctx = client.ctx
    cid = ctx.characters.create("Plain").id
    mid = client.post("/api/models", json={"characterIds": [cid]}).json()["id"]
    assert client.patch(f"/api/models/{mid}", json={"config": {"nChans": 512}}).json()["config"]["nChans"] == 512
    fake_ckpt(ctx.models.exp_dir(mid) / "model_2000.pt", 2000)
    r = client.patch(f"/api/models/{mid}", json={"config": {"nChans": 256}})
    assert r.status_code == 409 and r.json()["error"]["code"] == "train.config_mismatch"
    assert client.patch(f"/api/models/{mid}", json={"config": {"nChans": 512, "intervalVal": 500}}).status_code == 200


# ---- render params ---------------------------------------------------------------------------------
def test_f0_range_params(client, tmp_path):
    ctx = client.ctx
    params = parse_render_params({"f0Min": 65, "f0Max": 880})
    assert (params.f0_min, params.f0_max) == (65, 880)
    for bad in ({"f0Min": 500, "f0Max": 400}, {"f0Min": "low"}):
        with pytest.raises(ApiError) as err:
            parse_render_params(bad)
        assert err.value.code == "request.invalid" and err.value.status == 422

    cid = ctx.characters.create("F0").id
    engine_config(tmp_path / "f0" / "config.yaml")
    res = import_external(client, {"ckptPaths": [str(fake_ckpt(tmp_path / "f0" / "model_10.pt", 10))],
                                   "characterId": cid})["result"]
    args = render_args(ctx, res["modelId"], cid, Path("m.pt"), params, Path("in.wav"), Path("out.wav"))
    assert (args["f0Min"], args["f0Max"]) == (65, 880)
    assert "f0Min" not in render_args(ctx, res["modelId"], cid, Path("m.pt"), parse_render_params({}), Path("i"), Path("o"))

    p = client.post("/api/projects", json={"name": "f0"}).json()
    media = import_media(client, p["id"], write(tmp_path / "in.wav", song(2, 220, seed=5)))
    put_canvas(client, p["id"], [source_node(media), render_node("nd_r", "mix", characterId=cid, modelId=res["modelId"],
                                                                  params={"f0Min": 900, "f0Max": 100})])
    r = client.post(f"/api/projects/{p['id']}/nodes/nd_r/render")
    assert r.status_code == 422 and r.json()["error"]["code"] == "request.invalid"


# ---- the real inspector (torch, in a subprocess) ---------------------------------------------------
TINY_CKPT = """
import sys
import torch
from stellaria_backend.engine.inspect_worker import build_model, load_yaml
model = build_model(load_yaml(sys.argv[1]))
torch.save({"global_step": int(sys.argv[3]), "model": model.state_dict(), "optimizer": {"state": {}}}, sys.argv[2])
"""

READ_CKPT = """
import json, sys, torch
ckpt = torch.load(sys.argv[1], map_location="cpu")
print(json.dumps({"keys": sorted(ckpt), "step": ckpt["global_step"], "tensors": len(ckpt["model"])}))
"""


def _torch_python(script: str, *args: str) -> str:
    env = {**os.environ, "PYTHONPATH": str(BACKEND)}
    out = subprocess.run([sys.executable, "-c", script, *args], cwd=ENGINE, env=env, capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is not installed")
def test_real_inspector_and_seed(tmp_path):
    tiny = dict(model__n_layers=1, model__n_chans=16, model__n_aux_layers=1, model__n_aux_chans=16,
                model__glu_type=None, model__use_pitch_aug=False, model__n_spk=2, data__encoder="contentvec768l12",
                data__encoder_hop_size=320)
    config = engine_config(tmp_path / "ext" / "config.yaml", **tiny)
    ckpt = tmp_path / "ext" / "weights.pt"
    _torch_python(TINY_CKPT, str(config), str(ckpt), "1234")
    ctx = make_ctx(tmp_path, fake=False)
    try:
        result = inspect_files(ctx, None, config, [ckpt])
        assert result["config"]["nChans"] == 16 and result["config"]["gluType"] is None
        assert result["checkpoints"][0]["ok"] and result["checkpoints"][0]["step"] == 1234

        wider = engine_config(tmp_path / "wider" / "config.yaml", **{**tiny, "model__n_chans": 32})
        with pytest.raises(ApiError) as err:
            inspect_files(ctx, None, wider, [ckpt])
        assert err.value.code == "model.import_bad_checkpoint" and err.value.params == {"file": "weights.pt"}
        assert "size mismatch" in err.value.detail
        glu = engine_config(tmp_path / "glu" / "config.yaml", **{**tiny, "model__glu_type": "softsign_glu"})
        inspect_files(ctx, None, glu, [ckpt])  # the activation has no weights: loads either way

        # seed a 3-voice model with pitch augmentation from the 2-voice base
        target = engine_config(tmp_path / "new" / "config.yaml", **{**tiny, "model__n_spk": 3, "model__use_pitch_aug": True})
        out = tmp_path / "new" / "model_0.pt"
        seeded = run_inspector(ctx, None, ["seed", "--base", str(ckpt), "--config", str(target), "--out", str(out)])
        assert seeded["speakerDropped"] is True and seeded["kept"] > 0
        assert any("spk_embed" in n for n in seeded["dropped"])
        assert any("spk_embed" in n for n in seeded["missing"]) and any("aug_shift" in n for n in seeded["missing"])
        info = json.loads(_torch_python(READ_CKPT, str(out)).strip().splitlines()[-1])
        assert info == {"keys": ["global_step", "model"], "step": 0, "tensors": seeded["kept"]}
    finally:
        ctx.stop(timeout=5)
