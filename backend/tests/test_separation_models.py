"""Vocal separation models: MSST config reading, the vendored Roformer models and their chunked
inference, the model catalog / default / validation, and importing from an MSST folder."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import BACKEND, wait_job
from stellaria_backend.engine.roformer.config import (
    CALLABLE_PARAMS, CONFIG_DIR, PARAMS, ConfigProblem, describe, load_config, parse_config,
)
from test_creation import import_media, node_of, put_canvas, source_node
from test_graph import process_node, run, stereo_song

TINY_BS = """
audio: {chunk_size: 8820, sample_rate: 44100, num_channels: 2}
model:
  dim: 16
  depth: 1
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  freqs_per_bands: !!python/tuple [64, 65]
  dim_head: 8
  heads: 2
  flash_attn: true
  stft_n_fft: 256
  stft_hop_length: 147
  stft_win_length: 256
  mask_estimator_depth: 1
  multi_stft_resolutions_window_sizes: !!python/tuple [256, 128]
training: {instruments: [vocals, other], target_instrument: vocals}
inference: {batch_size: 2, num_overlap: 2}
"""

TINY_MEL = """
audio: {chunk_size: 8820, sample_rate: 44100}
model:
  dim: 16
  depth: 1
  stereo: true
  num_stems: 1
  time_transformer_depth: 1
  freq_transformer_depth: 1
  num_bands: 8
  dim_head: 8
  heads: 2
  flash_attn: false
  sample_rate: 44100
  stft_n_fft: 256
  stft_hop_length: 147
  stft_win_length: 256
  mask_estimator_depth: 1
training: {instruments: [vocals, other], target_instrument: other}
inference: {num_overlap: 4}
"""


def fake_ready(ctx, *rids: str) -> None:
    """Mark resources installed the way the fake engine's install does."""
    for rid in rids:
        marker = ctx.resources.models_dir / ".fake" / rid
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("fake", encoding="utf-8")


def ckpt(path: Path, size: int = (1 << 20) + 10) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(os.urandom(size))
    return path


# ---- configs ---------------------------------------------------------------------------------------
def test_yaml_loader_reads_msst_tags():
    config = parse_config("""
model:
  freqs_per_bands: !!python/tuple [2, 2, 4]
  nested: !!python/tuple
  - 1
  - !!python/tuple [2, 3]
odd: !!python/object/apply:collections.OrderedDict {a: 1}
local: !custom 5
""")
    assert config["model"]["freqs_per_bands"] == (2, 2, 4)
    assert config["model"]["nested"] == (1, (2, 3))
    assert config["odd"] == {"a": 1} and config["local"] == "5"
    with pytest.raises(ConfigProblem) as err:
        parse_config("- just\n- a list\n")
    assert err.value.code == "separation.skip.bad_config"
    with pytest.raises(ConfigProblem):
        parse_config("model: [unclosed\n")


def test_shipped_configs_and_architecture_detection():
    bs = describe(load_config(CONFIG_DIR / "bs_roformer_viperx.yaml"))
    assert (bs.arch, bs.outputs, bs.target, bs.chunk_size, bs.num_overlap, bs.stereo) == \
        ("bs_roformer", ["vocals"], 0, 352800, 2, True)
    mel = describe(load_config(CONFIG_DIR / "mel_roformer_kim.yaml"))
    assert (mel.arch, mel.target, mel.sample_rate) == ("mel_band_roformer", 0, 44100)

    tiny = parse_config(TINY_MEL)
    assert describe(tiny).target == -1  # an instrumental model: vocals = mix − its output

    def problem(config: dict, hint: str | None = None) -> tuple[str, dict]:
        with pytest.raises(ConfigProblem) as err:
            describe(config, hint)
        return err.value.code, err.value.params

    karaoke = parse_config(TINY_MEL.replace("[vocals, other], target_instrument: other", "[karaoke, other]")
                           .replace("num_stems: 1", "num_stems: 2"))
    assert problem(karaoke) == ("separation.skip.no_vocals", {"instruments": "karaoke, other"})
    mdx = {"model": {"num_subbands": 4, "act": "gelu"}, "training": {"instruments": ["vocals", "other"]}}
    assert problem(mdx) == ("separation.skip.arch", {"arch": "mdx23c"})
    assert problem(parse_config(TINY_BS), "htdemucs") == ("separation.skip.arch", {"arch": "htdemucs"})
    hyper = parse_config(TINY_BS.replace("heads: 2", "heads: 2\n  hyperace_dim: 4"))
    assert problem(hyper) == ("separation.skip.params", {"params": "hyperace_dim"})
    stems = parse_config(TINY_BS.replace(", target_instrument: vocals", ""))
    assert problem(stems)[0] == "separation.skip.bad_config"  # 1 stem but two instruments
    assert problem({"audio": {}})[0] == "separation.skip.bad_config"


# ---- vendored models + inference ------------------------------------------------------------------
# These run in a fresh interpreter: the API test process must never import torch (test_fake_flow).
def _params_match_the_vendored_classes() -> None:
    from stellaria_backend.engine.roformer.bs_roformer import BSRoformer
    from stellaria_backend.engine.roformer.mel_band_roformer import MelBandRoformer

    for cls, arch in ((BSRoformer, "bs_roformer"), (MelBandRoformer, "mel_band_roformer")):
        accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
        assert accepted == set(PARAMS[arch]) | CALLABLE_PARAMS, arch


def _tiny_roformers_run_and_overlap_add_is_exact(tmp_path: Path) -> None:
    import torch

    from stellaria_backend.engine.roformer.inference import (
        CheckpointMismatch, RoformerSeparator, build_model, demix,
    )

    torch.manual_seed(0)
    mix = torch.randn(2, 30000) * 0.1
    # windows and counters: a model that returns its input gives the input back
    spec = describe(parse_config(TINY_BS))

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x.unsqueeze(1)

    progress: list[float] = []
    out = demix(Identity(), mix, spec, torch.device("cpu"), 3, progress.append)
    assert out.shape == (1, 2, 30000) and torch.allclose(out[0], mix, atol=1e-6) and progress[-1] == 1.0
    short = demix(Identity(), mix[:, :3000], spec, torch.device("cpu"), 1)
    assert torch.allclose(short[0], mix[:, :3000], atol=1e-6)

    for name, text in (("bs", TINY_BS), ("mel", TINY_MEL)):
        config = parse_config(text)
        model = build_model(config, describe(config)).eval()
        (tmp_path / f"{name}.yaml").write_text(text, encoding="utf-8")
        torch.save({"state_dict": model.state_dict()}, tmp_path / f"{name}.ckpt")
        sep = RoformerSeparator(tmp_path / f"{name}.ckpt", tmp_path / f"{name}.yaml", "cpu")
        data = mix.numpy().T.copy()
        vocals = sep.extract(data, 44100)
        assert vocals.shape == data.shape and vocals.dtype == np.float32 and np.isfinite(vocals).all()
        if name == "mel":  # the model returns the accompaniment: vocals + it == the mix
            accompaniment = demix(sep.model, mix, sep.spec, torch.device("cpu"), 1)[0].numpy().T
            assert np.allclose(vocals + accompaniment, data, atol=1e-5)
        resampled = sep.extract(data[:16000], 32000)
        assert resampled.shape == (16000, 2)

    with pytest.raises(CheckpointMismatch):  # strict loading
        RoformerSeparator(tmp_path / "bs.ckpt", tmp_path / "mel.yaml", "cpu")


def test_vendored_roformers_in_a_fresh_process(tmp_path):
    for module in ("torch", "einops", "rotary_embedding_torch", "librosa"):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed")
    code = (f"import sys; sys.path[:0] = {[str(BACKEND), str(Path(__file__).parent)]!r}\n"
            "from pathlib import Path\nimport test_separation_models as t\n"
            "t._params_match_the_vendored_classes()\n"
            f"t._tiny_roformers_run_and_overlap_add_is_exact(Path({str(tmp_path)!r}))\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---- catalog, default, validation ----------------------------------------------------------------
def models_by_id(client) -> tuple[dict[str, dict], str]:
    view = client.get("/api/resources/separation").json()
    return {m["id"]: m for m in view["models"]}, view["defaultModel"]


def test_catalog_default_preference_and_model_ids(client, tmp_path):
    ctx = client.ctx
    models, default = models_by_id(client)
    assert list(models)[:4] == ["htdemucs", "htdemucs_ft", "mel_roformer_kim", "bs_roformer_viperx"]
    assert default == "htdemucs" and all(m["missing"] == "addon" for m in models.values())
    kim = models["mel_roformer_kim"]
    assert (kim["engine"], kim["arch"], kim["source"], kim["resourceId"], kim["quality"], kim["sdr"]) == \
        ("roformer", "mel_band_roformer", "resource", "mel_roformer_kim", "high", 10.98)
    assert kim["name"]["params"]["en"] == "Mel-Band-Roformer vocal model (Kim)"

    fake_ready(ctx, "separator")
    models, default = models_by_id(client)
    assert default == "htdemucs" and models["htdemucs_ft"]["ready"] and models["bs_roformer_viperx"]["missing"] == "model"
    fake_ready(ctx, "bs_roformer_viperx")
    assert models_by_id(client)[1] == "bs_roformer_viperx"
    fake_ready(ctx, "mel_roformer_kim")  # the best one wins
    assert models_by_id(client)[1] == "mel_roformer_kim"

    put = client.put("/api/preferences", json={"separationModel": "htdemucs_ft"})
    assert put.status_code == 200 and put.json()["separationModel"] == "htdemucs_ft"
    assert models_by_id(client)[1] == "htdemucs_ft"
    bad = client.put("/api/preferences", json={"separationModel": "spleeter"})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "separation.unknown_model"
    assert client.put("/api/preferences", json={"separationModel": "auto"}).json()["separationModel"] is None
    ctx.resources.delete("mel_roformer_kim")
    assert models_by_id(client)[1] == "bs_roformer_viperx"

    # process ops and song nodes take any ready model id; the fake engine runs them all
    pid = client.post("/api/projects", json={"name": "sep"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 2.0))
    put_canvas(client, pid, [source_node(song_id),
                             process_node("nd_x", "separate", [("nd_s", "mix")], {"model": "spleeter", "stems": 2}),
                             process_node("nd_m", "separate", [("nd_s", "mix")], {"model": "mel_roformer_kim", "stems": 2}),
                             process_node("nd_b", "separate", [("nd_s", "mix")], {"model": "bs_roformer_viperx", "stems": 4}),
                             process_node("nd_a", "separate", [("nd_s", "mix")], {"model": "auto", "stems": 2})])
    r = client.post(f"/api/projects/{pid}/nodes/nd_x/run")
    assert r.status_code == 422 and r.json()["error"]["code"] == "separation.unknown_model"
    r = client.post(f"/api/projects/{pid}/nodes/nd_m/run")
    assert r.status_code == 409 and r.json()["error"]["code"] == "separation.model_missing"
    assert r.json()["error"]["params"] == {"model": "mel_roformer_kim", "resource": "mel_roformer_kim"}
    done = run(client, pid, "nd_b")
    assert done["result"]["result"] == {"model": "bs_roformer_viperx", "stems": 4}
    assert sorted(node_of(client, pid, "nd_b")["data"]["outputs"]) == ["bass", "drums", "inst", "other", "vocals"]
    assert run(client, pid, "nd_a")["result"]["result"]["model"] == "bs_roformer_viperx"
    done = run(client, pid, "nd_s", model="htdemucs_ft")
    assert done["result"]["stems"]["model"] == "htdemucs_ft"

    # an add-on installed before Roformer support cannot run Roformer models
    ctx.options.fake_engine = False
    try:
        ctx.separation.catalog.runtime_ready = lambda: False  # type: ignore[method-assign]
        ctx.separation.catalog.addon_ready = lambda: True  # type: ignore[method-assign]
        assert models_by_id(client)[0]["bs_roformer_viperx"]["missing"] == "addon_update"
        assert models_by_id(client)[1] == "htdemucs"
    finally:
        ctx.options.fake_engine = True


# ---- importing from an MSST folder ----------------------------------------------------------------
def msst_folder(root: Path) -> Path:
    """An MSST-GUI checkout: checkpoints in pretrain/, configs in configs/, the GUI's model list."""
    configs = root / "configs"
    configs.mkdir(parents=True)
    bs = (CONFIG_DIR / "bs_roformer_viperx.yaml").read_text(encoding="utf-8")
    (configs / "model_bs_roformer_ep_368_sdr_12.9628.yaml").write_text(bs, encoding="utf-8")
    (configs / "config_vocals_mel_band_roformer_kim.yaml").write_text(
        (CONFIG_DIR / "mel_roformer_kim.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    (configs / "config_karaoke.yaml").write_text(TINY_MEL.replace("vocals, other], target_instrument: other",
                                                                  "karaoke, other], target_instrument: karaoke"))
    (configs / "mdx.yaml").write_text("model: {num_subbands: 4}\ntraining: {instruments: [vocals, other]}\n")
    vocal = root / "pretrain" / "vocal_models"
    ckpt(vocal / "model_bs_roformer_ep_368_sdr_12.9628.ckpt")  # config found by name
    ckpt(vocal / "MelBandRoformer_kim.ckpt")  # the downloadable model under the GUI's name
    ckpt(vocal / "kim_ft.ckpt")  # config from the GUI's list
    ckpt(root / "pretrain" / "kara_models" / "karaoke.ckpt")
    ckpt(root / "pretrain" / "other" / "mdx.ckpt")
    ckpt(root / "pretrain" / "orphan.ckpt")
    ckpt(root / "pretrain" / "tiny.ckpt", 100)  # too small to be a separation model
    (root / "env").mkdir()
    (root / "env" / "pyvenv.cfg").write_text("home = x", encoding="utf-8")
    ckpt(root / "env" / "hidden.ckpt")
    (root / "model_config_en.json").write_text(json.dumps({
        "config_paths": {"kim_ft.ckpt": ["configs/config_vocals_mel_band_roformer_kim.yaml", "configs/missing-fast.yaml"],
                         "karaoke.ckpt": ["configs/config_karaoke.yaml"]},
        "model_types": {"kim_ft.ckpt": "mel_band_roformer", "karaoke.ckpt": "mel_band_roformer"},
    }), encoding="utf-8")
    return root


def test_import_from_an_msst_folder(client, tmp_path):
    ctx = client.ctx
    fake_ready(ctx, "separator")
    root = msst_folder(tmp_path / "MSST GUI 整合包")
    r = client.post("/api/resources/separation/import", json={"path": str(tmp_path / "nope")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "import.not_found"

    res = client.post("/api/resources/separation/import", json={"path": str(root)}).json()
    found = {Path(f["path"]).name: f for f in res["found"]}
    assert sorted(found) == ["MelBandRoformer_kim.ckpt", "kim_ft.ckpt", "model_bs_roformer_ep_368_sdr_12.9628.ckpt"]
    assert found["MelBandRoformer_kim.ckpt"]["resourceId"] == "mel_roformer_kim"
    assert found["kim_ft.ckpt"]["arch"] == "mel_band_roformer"
    assert found["model_bs_roformer_ep_368_sdr_12.9628.ckpt"]["arch"] == "bs_roformer"
    skipped = {Path(s["path"]).name: s["reason"] for s in res["skipped"]}
    assert skipped == {
        "karaoke.ckpt": {"code": "separation.skip.no_vocals", "params": {"instruments": "karaoke, other"}},
        "mdx.ckpt": {"code": "separation.skip.arch", "params": {"arch": "mdx23c"}},
        "orphan.ckpt": {"code": "separation.skip.no_config", "params": {}},
    }
    for f in res["found"]:
        assert wait_job(client, f["jobId"])["status"] == "succeeded"

    models, default = models_by_id(client)
    assert default == "mel_roformer_kim" and models["mel_roformer_kim"]["ready"]
    local = {m["id"]: m for m in models.values() if m["source"] == "local"}
    assert sorted(m["name"]["params"]["en"] for m in local.values()) == ["kim_ft", "model_bs_roformer_ep_368_sdr_12.9628"]
    bs_id = next(i for i, m in local.items() if m["arch"] == "bs_roformer")
    assert all(m["ready"] and m["quality"] == "high" for m in local.values())
    stored = Path(local[bs_id]["path"])
    source = root / "pretrain" / "vocal_models" / stored.name
    assert stored.stat().st_ino == source.stat().st_ino  # same drive: linked, not copied
    assert (stored.parent / "config.yaml").read_text(encoding="utf-8").startswith("# From Music-Source")

    # again: nothing new; the folder import of the resources page reports the same
    again = client.post("/api/resources/import-folder", json={"path": str(root)}).json()
    assert again["separation"]["found"] == []
    assert sorted(s["reason"]["code"] for s in again["separation"]["skipped"]) == [
        "separation.skip.arch", "separation.skip.duplicate", "separation.skip.duplicate", "separation.skip.duplicate",
        "separation.skip.no_config", "separation.skip.no_vocals"]

    # one checkpoint with its config
    config = root / "configs" / "config_vocals_mel_band_roformer_kim.yaml"
    single = client.post("/api/resources/separation/import", json={
        "path": str(root / "pretrain" / "orphan.ckpt"), "config": str(config)}).json()
    assert [Path(f["path"]).name for f in single["found"]] == ["orphan.ckpt"] and single["skipped"] == []
    orphan_id = single["found"][0]["id"]
    assert wait_job(client, single["found"][0]["jobId"])["status"] == "succeeded"

    # an imported model separates like any other (fake engine)
    pid = client.post("/api/projects", json={"name": "local"}).json()["id"]
    song_id = import_media(client, pid, stereo_song(tmp_path / "s.wav", 2.0))
    put_canvas(client, pid, [source_node(song_id)])
    assert run(client, pid, "nd_s", model=bs_id)["result"]["stems"]["model"] == bs_id

    client.put("/api/preferences", json={"separationModel": orphan_id})
    assert models_by_id(client)[1] == orphan_id
    r = client.delete("/api/resources/separation/htdemucs")
    assert r.status_code == 409 and r.json()["error"]["code"] == "separation.read_only"
    assert client.delete("/api/resources/separation/sep_nothing").status_code == 404
    view = client.delete(f"/api/resources/separation/{orphan_id}").json()
    assert orphan_id not in {m["id"] for m in view["models"]} and view["defaultModel"] == "mel_roformer_kim"
    assert not (ctx.options.data / "separation-models" / orphan_id).exists()
