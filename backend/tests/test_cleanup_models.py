"""Vocal cleanup (dereverb) models: the shipped configs build their models and run, the catalog
and its default, the dereverb canvas op and organizing with dereverb (fake engine), and finding
cleanup checkpoints in an MSST folder."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import BACKEND, MANIFEST, wait_job
from stellaria_backend.engine.roformer.config import CONFIG_DIR, cleanup_kind, describe, load_config, parse_config
from synth import song, write
from test_creation import import_media, node_of, put_canvas, source_node
from test_graph import media_of, process_node, run, stereo_song
from test_separation_models import TINY_MEL, ckpt, fake_ready

CLEANUP_IDS = ["dereverb_mbr_anvuew", "dereverb_echo_mbr_fused"]
TINY_DEREVERB = TINY_MEL.replace("[vocals, other], target_instrument: other",
                                 "[noreverb, reverb], target_instrument: reverb")


def manifest_cleanup() -> dict[str, dict]:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {r["id"]: r for r in raw["resources"] if r.get("cleanup")}


# ---- configs and models ---------------------------------------------------------------------------
def test_cleanup_configs_describe_their_dry_output():
    resources = manifest_cleanup()
    assert list(resources) == CLEANUP_IDS
    for rid, rdef in resources.items():
        block = rdef["cleanup"]
        spec = describe(load_config(CONFIG_DIR / block["config"]))
        assert (spec.role, spec.arch, spec.target) == ("cleanup", block["arch"], 0)
        assert spec.outputs == [block["target"]] and (spec.sample_rate, spec.stereo) == (44100, True)
    # a model that returns the reverb: dry = mix − its output; vocal models stay separation models
    tiny = describe(parse_config(TINY_DEREVERB))
    assert (tiny.role, tiny.target) == ("cleanup", -1)
    assert describe(parse_config(TINY_MEL)).role == "separation"
    assert [cleanup_kind(n) for n in ("dereverb_echo_mbr_v2.ckpt", "denoise_mel_band_roformer_aufr33.ckpt",
                                      "de_big_reverb_mbr_ep_362.ckpt")] == ["dereverb_echo", "denoise", "dereverb"]


# These run in a fresh interpreter: the API test process must never import torch (test_fake_flow).
def _shipped_cleanup_models_build_and_demix() -> None:
    import dataclasses

    import torch

    from stellaria_backend.engine.roformer.inference import build_model, demix

    torch.manual_seed(0)
    mix = torch.randn(2, 22050) * 0.1
    for rid, rdef in manifest_cleanup().items():
        config = load_config(CONFIG_DIR / rdef["cleanup"]["config"])
        spec = describe(config)
        model = build_model(config, spec).eval()
        # the published checkpoint holds exactly these weights (fp32): its size matches the config
        weights = sum(t.numel() * t.element_size() for t in model.state_dict().values())
        assert abs(rdef["size"] - weights) / rdef["size"] < 0.002, rid
        small = dataclasses.replace(spec, chunk_size=11025, num_overlap=2)  # a few short chunks on the CPU
        out = demix(model, mix, small, torch.device("cpu"), 1)
        assert out.shape == (1, 2, 22050) and torch.isfinite(out).all(), rid
        del model


def _tiny_cleanup_through_the_worker(tmp_path: Path) -> None:
    import soundfile as sf
    import torch

    from stellaria_backend.engine.roformer.inference import RoformerSeparator, build_model

    config = parse_config(TINY_DEREVERB)
    model = build_model(config, describe(config)).eval()
    (tmp_path / "tiny.yaml").write_text(TINY_DEREVERB, encoding="utf-8")
    torch.save({"state_dict": model.state_dict()}, tmp_path / "tiny.ckpt")
    data = (np.random.default_rng(1).standard_normal((20000, 2)) * 0.1).astype(np.float32)
    dry = RoformerSeparator(tmp_path / "tiny.ckpt", tmp_path / "tiny.yaml", "cpu").extract(data, 44100)
    assert dry.shape == data.shape and np.isfinite(dry).all()

    # the worker: mono in, mono out, dry + wet = the input
    mono = song(1.0, 220, seed=5) * 0.5
    sf.write(tmp_path / "in.wav", mono, 44100)
    proc = subprocess.run(
        [sys.executable, "-m", "stellaria_backend.engine.separate_worker", "--input", str(tmp_path / "in.wav"),
         "--dry", str(tmp_path / "dry.wav"), "--wet", str(tmp_path / "wet.wav"),
         "--cleanup", str(tmp_path / "tiny.ckpt"), "--cleanup-config", str(tmp_path / "tiny.yaml"), "--device", "cpu"],
        cwd=BACKEND, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout.strip().splitlines()[-1])["type"] == "result"
    d, sr = sf.read(tmp_path / "dry.wav", always_2d=True)
    w, _ = sf.read(tmp_path / "wet.wav", always_2d=True)
    assert sr == 44100 and d.shape == w.shape == (len(mono), 1)
    assert np.abs(d[:, 0] + w[:, 0] - mono).max() < 1e-3


def test_cleanup_models_in_a_fresh_process(tmp_path):
    for module in ("torch", "einops", "rotary_embedding_torch", "librosa"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed")
    code = (f"import sys; sys.path[:0] = {[str(BACKEND), str(Path(__file__).parent)]!r}\n"
            "from pathlib import Path\nimport test_cleanup_models as t\n"
            "t._shipped_cleanup_models_build_and_demix()\n"
            f"t._tiny_cleanup_through_the_worker(Path({str(tmp_path)!r}))\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---- catalog, default, the dereverb op ---------------------------------------------------------------
def cleanup_by_id(client) -> tuple[dict[str, dict], str | None]:
    view = client.get("/api/resources/cleanup").json()
    return {m["id"]: m for m in view["models"]}, view["defaultModel"]


def test_catalog_default_and_the_dereverb_op(client, tmp_path):
    ctx = client.ctx
    models, default = cleanup_by_id(client)
    assert list(models) == CLEANUP_IDS and default is None
    anvuew = models["dereverb_mbr_anvuew"]
    assert [anvuew[k] for k in ("kind", "arch", "source", "resourceId", "sdr", "missing")] == \
        ["dereverb", "mel_band_roformer", "resource", "dereverb_mbr_anvuew", 19.17, "addon"]
    assert models["dereverb_echo_mbr_fused"]["kind"] == "dereverb_echo"
    assert anvuew["name"]["params"]["en"] == "Dereverb model (anvuew)"
    assert not set(CLEANUP_IDS) & {m["id"] for m in client.get("/api/resources/separation").json()["models"]}

    pid = client.post("/api/projects", json={"name": "dry"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 2.0))
    put_canvas(client, pid, [source_node(song_id),
                             process_node("nd_a", "dereverb", [("nd_s", "mix")], {"model": "auto"}, label="干"),
                             process_node("nd_x", "dereverb", [("nd_s", "mix")], {"model": "mel_roformer_kim"}),
                             process_node("nd_f", "dereverb", [("nd_s", "mix")], {"model": "dereverb_echo_mbr_fused"})])

    def error(node_id: str) -> tuple[int, str, dict]:
        r = client.post(f"/api/projects/{pid}/nodes/{node_id}/run")
        return r.status_code, r.json()["error"]["code"], r.json()["error"].get("params") or {}

    assert error("nd_a")[:2] == (409, "addon.missing")
    fake_ready(ctx, "separator")
    # nothing ready: 'automatic' names the recommended download
    assert error("nd_a") == (409, "cleanup.model_missing",
                             {"model": "dereverb_mbr_anvuew", "resource": "dereverb_mbr_anvuew"})
    assert error("nd_x") == (422, "cleanup.unknown_model", {"model": "mel_roformer_kim"})
    fake_ready(ctx, "dereverb_echo_mbr_fused")
    assert cleanup_by_id(client)[1] == "dereverb_echo_mbr_fused"
    fake_ready(ctx, "dereverb_mbr_anvuew")  # the recommended one wins
    assert cleanup_by_id(client)[1] == "dereverb_mbr_anvuew"

    put = client.put("/api/preferences", json={"cleanupModel": "dereverb_echo_mbr_fused"})
    assert put.status_code == 200 and put.json()["cleanupModel"] == "dereverb_echo_mbr_fused"
    assert cleanup_by_id(client)[1] == "dereverb_echo_mbr_fused"
    bad = client.put("/api/preferences", json={"cleanupModel": "htdemucs"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "cleanup.unknown_model"
    assert client.put("/api/preferences", json={"cleanupModel": "auto"}).json()["cleanupModel"] is None

    done = run(client, pid, "nd_a")
    assert done["title"]["code"] == "job.process.dereverb" and done["usesGpu"] is True
    data = node_of(client, pid, "nd_a")["data"]
    assert data["status"] == "done" and data["result"] == {"model": "dereverb_mbr_anvuew"}
    assert sorted(data["outputs"]) == ["dry", "reverb"]
    media = media_of(client, pid)
    dry, wet = media[data["outputs"]["dry"]], media[data["outputs"]["reverb"]]
    assert (dry["name"], wet["name"], dry["kind"], dry["channels"]) == ("干 · dry", "干 · reverb", "process", 2)
    assert abs(dry["duration"] - 2.0) < 0.01
    assert run(client, pid, "nd_f")["result"]["result"] == {"model": "dereverb_echo_mbr_fused"}


# ---- organizing ------------------------------------------------------------------------------------
def test_organize_with_dereverb(client, tmp_path):
    ctx = client.ctx
    cid = client.post("/api/characters", json={"name": "Dry"}).json()["id"]
    take = write(tmp_path / "take.wav", song(14, 240, seed=7))
    job = client.post(f"/api/characters/{cid}/raw", json={"paths": [str(take)]}).json()
    assert wait_job(client, job["id"])["status"] == "succeeded"

    def organize(**body) -> dict:
        return wait_job(client, client.post(f"/api/characters/{cid}/organize", json=body).json()["id"], timeout=120)

    assert client.get(f"/api/characters/{cid}/organize/defaults").json()["dereverb"] is False

    fake_ready(ctx, "separator")
    failed = organize(dereverb=True)
    assert failed["status"] == "failed" and failed["error"]["code"] == "cleanup.model_missing"

    fake_ready(ctx, "dereverb_echo_mbr_fused")
    calls = []
    cleanup = ctx.separation.cleanup

    def spy(jctx, src, dry, wet=None, **kw):
        calls.append(Path(src).name)
        return cleanup(jctx, src, dry, wet, **kw)

    ctx.separation.cleanup = spy  # type: ignore[method-assign]
    done = organize(dereverb=True)
    assert done["status"] == "succeeded", done
    first = len(client.get(f"/api/characters/{cid}/clips").json())
    assert first >= 1 and calls == ["input.wav"]
    # songs: separate first, then clean up the vocals
    done = organize(separateVocals=True, dereverb=True, rawIds=done["result"]["rawIds"])
    assert done["status"] == "succeeded", done
    assert calls == ["input.wav", "vocals.wav"] and len(client.get(f"/api/characters/{cid}/clips").json()) > first


# ---- importing from an MSST folder -----------------------------------------------------------------
def test_import_finds_cleanup_checkpoints(client, tmp_path):
    ctx = client.ctx
    fake_ready(ctx, "separator")
    fused = ctx.resources.definition("dereverb_echo_mbr_fused")
    fused.size, fused.sha256 = None, None  # a small stand-in for the published checkpoint
    root = tmp_path / "MSST-WebUI"
    configs, pretrain = root / "configs" / "single_stem_models", root / "pretrain" / "single_stem_models"
    configs.mkdir(parents=True)
    anvuew = (CONFIG_DIR / "dereverb_mbr_anvuew.yaml").read_text(encoding="utf-8")
    fused_cfg = (CONFIG_DIR / "dereverb_echo_mbr_fused.yaml").read_text(encoding="utf-8")
    names = {"less": "dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt",
             "denoise": "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
             "fused": "dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt",
             "crowd": "mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt"}
    (configs / f"{names['less']}.yaml").write_text(anvuew, encoding="utf-8")
    (configs / f"{names['denoise']}.yaml").write_text(fused_cfg, encoding="utf-8")  # [dry, other] like the real one
    (configs / f"{names['fused']}.yaml").write_text(fused_cfg, encoding="utf-8")
    (configs / f"{names['crowd']}.yaml").write_text(fused_cfg.replace("- dry", "- crowd").replace(": dry", ": crowd"),
                                                    encoding="utf-8")
    for name in names.values():
        ckpt(pretrain / name)

    res = client.post("/api/resources/separation/import", json={"path": str(root)}).json()
    found = {Path(f["path"]).name: f for f in res["found"]}
    assert sorted(found) == sorted([names["less"], names["denoise"], names["fused"]])
    assert [found[names["fused"]][k] for k in ("resourceId", "role")] == ["dereverb_echo_mbr_fused", "cleanup"]
    assert (found[names["less"]]["role"], found[names["less"]]["kind"]) == ("cleanup", "dereverb")
    assert found[names["denoise"]]["kind"] == "denoise"
    assert [s["reason"]["code"] for s in res["skipped"]] == ["separation.skip.no_vocals"]
    for f in res["found"]:
        assert wait_job(client, f["jobId"])["status"] == "succeeded"

    models, default = cleanup_by_id(client)
    local = {m["kind"]: m for m in models.values() if m["source"] == "local"}
    assert sorted(local) == ["denoise", "dereverb"] and all(m["ready"] for m in local.values())
    assert default == "dereverb_echo_mbr_fused" and models[default]["ready"]
    separation = {m["id"] for m in client.get("/api/resources/separation").json()["models"]}
    assert not {m["id"] for m in local.values()} & separation
    ctx.resources.delete("dereverb_echo_mbr_fused")
    assert cleanup_by_id(client)[1] == local["dereverb"]["id"]  # 'automatic' never picks a denoise model

    pid = client.post("/api/projects", json={"name": "imported"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 1.0))
    put_canvas(client, pid, [source_node(song_id),
                             process_node("nd_n", "dereverb", [("nd_s", "mix")], {"model": local["denoise"]["id"]})])
    assert run(client, pid, "nd_n")["result"]["result"] == {"model": local["denoise"]["id"]}

    assert client.delete(f"/api/resources/separation/{local['dereverb']['id']}").status_code == 404
    view = client.delete(f"/api/resources/cleanup/{local['dereverb']['id']}").json()
    assert local["dereverb"]["id"] not in {m["id"] for m in view["models"]} and view["defaultModel"] is None
