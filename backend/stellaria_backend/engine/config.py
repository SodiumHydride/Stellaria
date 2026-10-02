"""Train presets and generation of the engine's ``config.yaml`` from a ``TrainConfig``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..errors import ApiError
from ..schemas import TrainConfig, TrainPreset

PRESETS: list[tuple[str, int, dict[str, Any]]] = [
    ("vram6", 6, {"batchSize": 24, "cacheDevice": "cpu"}),
    ("vram8", 8, {"batchSize": 48, "cacheDevice": "cpu"}),
    ("vram12", 12, {"batchSize": 64, "cacheDevice": "cuda"}),
    ("vram24", 24, {"batchSize": 96, "cacheDevice": "cuda"}),
]

# fields that change the network's shape: checkpoints trained with other values cannot resume
# (usePitchAug only adds an embedding the engine loads non-strictly, so it may change)
ARCH_FIELDS = ("encoder", "n_layers", "n_chans", "n_aux_layers", "n_aux_chans")

ENCODER_RESOURCE = {"contentvec768l12tta2x": "contentvec", "contentvec768l12": "contentvec", "hubertsoft": "hubertsoft"}


def recommended_preset(vram_gb: float | None) -> str:
    if vram_gb is None:
        return "vram6"
    if vram_gb >= 22:
        return "vram24"
    if vram_gb >= 11:
        return "vram12"
    if vram_gb >= 7.5:
        return "vram8"
    return "vram6"


def presets(vram_gb: float | None) -> list[TrainPreset]:
    rec = recommended_preset(vram_gb)
    return [TrainPreset(id=pid, vram=vram, config=dict(cfg), recommended=pid == rec) for pid, vram, cfg in PRESETS]


def default_config(preset_id: str) -> TrainConfig:
    cfg = TrainConfig(preset=preset_id)
    for pid, _, values in PRESETS:
        if pid == preset_id:
            cfg = cfg.model_copy(update=TrainConfig.model_validate(values).model_dump(include={"batch_size", "cache_device"}))
    return cfg


def merge_config(base: TrainConfig, patch: dict[str, Any] | None) -> TrainConfig:
    if not patch:
        return base
    try:
        return TrainConfig.model_validate({**base.wire(), **patch})
    except Exception as exc:  # noqa: BLE001
        raise ApiError("model.bad_config", "Invalid training configuration", status=422, detail=str(exc)) from exc


def encoder_dims(encoder: str) -> tuple[int, int]:
    """(encoder_out_channels, encoder_hop_size) for an encoder name."""
    out_channels = 256 if encoder == "hubertsoft" else 768
    hop = 160 if encoder == "contentvec768l12tta2x" else 320
    return out_channels, hop


def training_duration(shortest_clip: float) -> float:
    """Crop length used during training; must stay below the shortest clip (engine rule)."""
    if shortest_clip >= 2.1:
        return 2.0
    return max(0.5, round(int((shortest_clip - 0.1) * 10) / 10, 1))


def build_engine_config(
    template: dict[str, Any],
    *,
    model_dir: Path,
    cfg: TrainConfig,
    target_steps: int,
    n_spk: int,
    encoder_ckpt: Path,
    vocoder_ckpt: Path,
    device: str,
    fused_available: bool,
    duration: float,
    glu_type: str | None = None,
) -> dict[str, Any]:
    conf = yaml.safe_load(yaml.safe_dump(template))  # deep copy
    out_channels, hop = encoder_dims(cfg.encoder)
    data = conf["data"]
    data.update(
        f0_extractor=cfg.f0_extractor, encoder=cfg.encoder, encoder_ckpt=str(encoder_ckpt),
        encoder_out_channels=out_channels, encoder_hop_size=hop, duration=duration,
        train_path=str(model_dir / "dataset" / "train"), valid_path=str(model_dir / "dataset" / "val"),
    )
    conf["model"].update(
        n_layers=cfg.n_layers, n_chans=cfg.n_chans, n_aux_layers=cfg.n_aux_layers, n_aux_chans=cfg.n_aux_chans,
        use_pitch_aug=cfg.use_pitch_aug, n_spk=max(1, n_spk),
    )
    if glu_type:  # imported/base models keep the activation they were trained with
        conf["model"]["glu_type"] = glu_type
    conf["device"] = device
    conf["vocoder"]["ckpt"] = str(vocoder_ckpt)
    conf["env"]["expdir"] = str(model_dir / "exp")
    on_gpu = device == "cuda"
    fused = fused_available and on_gpu if cfg.use_fused_kernels == "auto" else bool(cfg.use_fused_kernels) and on_gpu
    conf["train"].update(
        batch_size=cfg.batch_size, lr=cfg.lr,
        # autocast on CPU is slow and partly unsupported; fp16/bf16 only make sense on CUDA
        amp_dtype=cfg.amp_dtype if on_gpu else "fp32",
        cache_device=cfg.cache_device if on_gpu else "cpu",
        num_workers=cfg.num_workers, interval_val=cfg.interval_val,
        interval_force_save=cfg.interval_force_save, max_steps=target_steps, use_fused_kernels=fused,
    )
    return conf


def load_template(engine_root: Path) -> dict[str, Any]:
    with open(engine_root / "configs" / "reflow.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: Path, conf: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, allow_unicode=True, sort_keys=False)
    tmp.replace(path)


def read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None


def arch_of_engine_config(conf: dict[str, Any]) -> dict[str, Any]:
    """The ARCH_FIELDS as recorded in an engine config (``exp/config.yaml``)."""
    return {
        "encoder": conf["data"]["encoder"], "n_layers": conf["model"]["n_layers"],
        "n_chans": conf["model"]["n_chans"], "n_aux_layers": conf["model"]["n_aux_layers"],
        "n_aux_chans": conf["model"]["n_aux_chans"],
    }
