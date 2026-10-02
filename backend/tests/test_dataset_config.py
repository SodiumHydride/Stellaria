"""Dataset assembly (single and multi-character), fingerprints, engine config generation."""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from conftest import ENGINE
from stellaria_backend.engine import config as engine_config
from stellaria_backend.engine import dataset
from stellaria_backend.errors import ApiError
from stellaria_backend.schemas import Clip, TrainConfig
from stellaria_backend.audio.io import write_wav
from stellaria_backend.util.timeutil import now_iso


def add_clips(ctx, cid: str, n: int, val: int = 2, prefix: str = "c") -> list[Clip]:
    clips = []
    for i in range(n):
        name = f"{prefix}_{i:03d}"
        write_wav(ctx.characters.clips_dir(cid) / f"{name}.wav", np.zeros(44100 * 3, dtype=np.float32) + 0.01)
        clips.append(Clip(id=f"clp_{prefix}{i}", name=name, file=f"clips/{name}.wav", duration=3.0,
                          split="val" if i < val else "train", created_at=now_iso()))
    ctx.characters.save_clips(cid, clips)
    return clips


def test_single_character_dataset(ctx):
    cid = ctx.characters.create("单人").id
    add_clips(ctx, cid, 6)
    rec = ctx.models.create("m", [cid], TrainConfig())
    entries = dataset.plan(rec, ctx.characters)
    ds = ctx.models.dataset_dir(rec.id)
    manifest = dataset.assemble(ds, entries)
    train = sorted(p.name for p in (ds / "train" / "audio").iterdir())
    val = sorted(p.name for p in (ds / "val" / "audio").iterdir())
    assert train == [f"c_{i:03d}.wav" for i in range(2, 6)] and val == ["c_000.wav", "c_001.wav"]
    src = ctx.characters.clips_dir(cid) / "c_002.wav"
    assert os.path.samefile(src, ds / "train" / "audio" / "c_002.wav")  # hard link
    assert manifest["hardLinks"] == 6 and manifest["clipCount"] == 6
    assert dataset.status(rec, ctx.characters, ds)[0] == "ready"
    clips = ctx.characters.clips(cid)
    clips[3].status = "trashed"
    ctx.characters.save_clips(cid, clips)
    assert dataset.status(rec, ctx.characters, ds)[0] == "stale"
    assert dataset.fingerprint(dataset.plan(rec, ctx.characters)) != manifest["fingerprint"]


def test_constellation_gets_numbered_speaker_folders(ctx):
    a = ctx.characters.create("Alpha").id
    b = ctx.characters.create("Beta").id
    add_clips(ctx, a, 4, prefix="a")
    add_clips(ctx, b, 5, prefix="b")
    rec = ctx.models.create("duo", [a, b], TrainConfig())
    assert [(mc.character_id, mc.spk_id) for mc in rec.characters] == [(a, 1), (b, 2)]
    ds = ctx.models.dataset_dir(rec.id)
    dataset.assemble(ds, dataset.plan(rec, ctx.characters))
    assert sorted(p.name for p in (ds / "train" / "audio").iterdir()) == ["1", "2"]
    assert len(list((ds / "train" / "audio" / "2").iterdir())) == 3
    assert sorted(p.name for p in (ds / "val" / "audio" / "1").iterdir()) == ["a_000.wav", "a_001.wav"]


def test_empty_dataset_is_rejected(ctx):
    cid = ctx.characters.create("Empty").id
    rec = ctx.models.create("m", [cid], TrainConfig())
    with pytest.raises(ApiError) as err:
        dataset.validate(dataset.plan(rec, ctx.characters))
    assert err.value.code == "dataset.empty"


def test_engine_config_generation(tmp_path):
    template = engine_config.load_template(ENGINE)
    cfg = engine_config.merge_config(engine_config.default_config("vram12"),
                                     {"encoder": "hubertsoft", "targetSteps": 1234, "useFusedKernels": "auto"})
    assert cfg.batch_size == 64 and cfg.cache_device == "cuda"
    model_dir = tmp_path / "模型 dir"
    conf = engine_config.build_engine_config(
        template, model_dir=model_dir, cfg=cfg, target_steps=1234, n_spk=2, encoder_ckpt=tmp_path / "h.pt",
        vocoder_ckpt=tmp_path / "nsf" / "model.ckpt", device="cuda", fused_available=True, duration=2.0)
    assert conf["data"]["encoder"] == "hubertsoft" and conf["data"]["encoder_out_channels"] == 256
    assert conf["data"]["encoder_hop_size"] == 320 and conf["data"]["encoder_ckpt"] == str(tmp_path / "h.pt")
    assert conf["data"]["train_path"] == str(model_dir / "dataset" / "train")
    assert conf["data"]["valid_path"] == str(model_dir / "dataset" / "val")
    assert conf["env"]["expdir"] == str(model_dir / "exp") and conf["model"]["n_spk"] == 2
    assert conf["train"]["max_steps"] == 1234 and conf["train"]["use_fused_kernels"] is True
    assert conf["vocoder"]["ckpt"] == str(tmp_path / "nsf" / "model.ckpt") and conf["device"] == "cuda"
    assert conf["train"]["batch_size"] == 64 and conf["train"]["cache_device"] == "cuda"
    cv = engine_config.merge_config(cfg, {"encoder": "contentvec768l12tta2x"})
    cpu = engine_config.build_engine_config(
        template, model_dir=model_dir, cfg=cv, target_steps=10, n_spk=1, encoder_ckpt=tmp_path / "c.bin",
        vocoder_ckpt=tmp_path / "v.ckpt", device="cpu", fused_available=True, duration=1.5)
    assert cpu["data"]["encoder_out_channels"] == 768 and cpu["data"]["encoder_hop_size"] == 160
    assert cpu["train"]["use_fused_kernels"] is False and cpu["train"]["amp_dtype"] == "fp32"
    assert cpu["train"]["cache_device"] == "cpu" and cpu["data"]["duration"] == 1.5
    assert template["train"]["max_steps"] is None  # the template itself is untouched
    path = tmp_path / "out" / "config.yaml"
    engine_config.write_yaml(path, conf)
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["data"]["train_path"].endswith("train")


def test_presets_and_defaults():
    ps = {p.id: p for p in engine_config.presets(8.0)}
    assert ps["vram8"].recommended and ps["vram6"].config == {"batchSize": 24, "cacheDevice": "cpu"}
    assert ps["vram24"].config == {"batchSize": 96, "cacheDevice": "cuda"}
    assert engine_config.recommended_preset(None) == "vram6" and engine_config.recommended_preset(24) == "vram24"
    d = engine_config.default_config("vram8")
    assert (d.target_steps, d.amp_dtype, d.encoder, d.f0_extractor) == (100000, "fp16", "contentvec768l12tta2x", "rmvpe")
    assert (d.n_layers, d.n_chans, d.n_aux_layers, d.n_aux_chans, d.use_pitch_aug) == (6, 1024, 6, 512, True)
    assert engine_config.training_duration(5.0) == 2.0 and engine_config.training_duration(1.6) == 1.5
