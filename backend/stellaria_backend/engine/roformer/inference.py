"""Roformer inference: build the model its MSST config describes, load the checkpoint with a
strict state dict, and separate a whole song in overlapping chunks like MSST's ``demix``:
chunks of ``chunk_size`` every ``chunk_size // num_overlap`` samples, a linear fade over a
tenth of a chunk at both ends (none at the song's edges), reflect padding, batches on the
device, float32 accumulation on the CPU. fp16 autocast only on CUDA (when the config uses AMP).
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from .config import RoformerSpec, describe, load_config, model_params

Progress = Callable[[float], None]


class CheckpointMismatch(RuntimeError):
    """The checkpoint's weights do not fit the model its config builds."""


def build_model(config: dict[str, Any], spec: RoformerSpec) -> torch.nn.Module:
    if spec.arch == "bs_roformer":
        from .bs_roformer import BSRoformer as model_class
    else:
        from .mel_band_roformer import MelBandRoformer as model_class
    return model_class(**model_params(config))


def state_dict_of(checkpoint: Any) -> dict[str, torch.Tensor]:
    """The weights in a checkpoint file's object (bare state dicts and the wrappers MSST knows)."""
    for key in ("state", "state_dict", "model_state_dict"):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
    if not isinstance(checkpoint, dict):
        raise CheckpointMismatch(f"unexpected checkpoint content: {type(checkpoint).__name__}")
    return {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in checkpoint.items()}


def load_weights(model: torch.nn.Module, path: Path | str) -> None:
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state_dict_of(checkpoint), strict=True)
    except RuntimeError as exc:
        raise CheckpointMismatch(str(exc)) from exc


def fade_window(chunk_size: int, fade_size: int) -> torch.Tensor:
    window = torch.ones(chunk_size)
    if fade_size > 0:
        window[:fade_size] = torch.linspace(0, 1, fade_size)
        window[-fade_size:] = torch.linspace(1, 0, fade_size)
    return window


def demix(model: torch.nn.Module, mix: torch.Tensor, spec: RoformerSpec, device: torch.device, batch_size: int,
          progress: Progress | None = None) -> torch.Tensor:
    """(channels, samples) float32 → (outputs, channels, samples) float32, on the CPU."""
    chunk = spec.chunk_size
    fade = chunk // 10
    step = max(1, chunk // spec.num_overlap)
    border = chunk - step
    length = mix.shape[-1]
    padded = length > 2 * border and border > 0
    if padded:
        mix = F.pad(mix, (border, border), mode="reflect")
    total = mix.shape[-1]
    starts = list(range(0, total, step))
    window = fade_window(chunk, fade)
    result = torch.zeros((len(spec.outputs), *mix.shape), dtype=torch.float32)
    counter = torch.zeros(total, dtype=torch.float32)
    amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and spec.use_amp \
        else contextlib.nullcontext()
    with torch.inference_mode(), amp:
        for first in range(0, len(starts), batch_size):
            batch = starts[first:first + batch_size]
            parts = []
            for start in batch:
                part = mix[:, start:start + chunk]
                size = part.shape[-1]
                mode = "reflect" if size > chunk // 2 else "constant"
                parts.append(F.pad(part, (0, chunk - size), mode=mode) if size < chunk else part)
            out = model(torch.stack(parts).to(device)).float().cpu()
            if out.ndim == 3:  # a single output without its stem axis
                out = out.unsqueeze(1)
            for j, start in enumerate(batch):
                size = min(chunk, total - start)
                w = window.clone()
                if start == 0:
                    w[:fade] = 1
                if start + step >= total:
                    w[-fade:] = 1
                result[..., start:start + size] += out[j, ..., :size] * w[:size]
                counter[start:start + size] += w[:size]
            if progress is not None:
                progress(min(len(starts), first + batch_size) / len(starts))
    estimated = result / counter.clamp(min=1e-8)
    if padded:
        estimated = estimated[..., border:-border]
    return torch.nan_to_num(estimated, nan=0.0)


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)) or \
        "out of memory" in str(exc).lower()


class RoformerSeparator:
    """One loaded model; ``extract`` runs any number of songs or clips through it and returns
    what the model is for: the vocals (a separation model) or the dry voice (a cleanup model)."""

    def __init__(self, checkpoint: Path | str, config_path: Path | str, device: str) -> None:
        config = load_config(config_path)
        self.spec = describe(config)
        self.device = torch.device(device)
        model = build_model(config, self.spec)
        load_weights(model, checkpoint)
        self.model = model.to(self.device).eval()
        self.batch_size = min(self.spec.batch_size, 4) if self.device.type == "cuda" else 1

    def _demix(self, mix: torch.Tensor, progress: Progress | None) -> torch.Tensor:
        try:
            return demix(self.model, mix, self.spec, self.device, self.batch_size, progress)
        except RuntimeError as exc:  # CUDA OOM with batches: retry one chunk at a time
            if self.device.type != "cuda" or self.batch_size == 1 or not _is_oom(exc):
                raise
            torch.cuda.empty_cache()
            self.batch_size = 1
            return demix(self.model, mix, self.spec, self.device, 1, progress)

    def extract(self, data: np.ndarray, sr: int, progress: Progress | None = None) -> np.ndarray:
        """``data`` (samples, 2) float32 at ``sr`` → the vocals / the dry voice, same shape and rate."""
        spec = self.spec
        x = data
        if spec.sample_rate != sr:
            import soxr

            x = soxr.resample(x, sr, spec.sample_rate).astype(np.float32)
        mix = torch.from_numpy(np.ascontiguousarray(x.T, dtype=np.float32))
        if spec.stereo:
            out = self._demix(mix, progress)
        else:  # a mono model: each channel on its own
            def part(c: int) -> Progress | None:
                return None if progress is None else (lambda v: progress((c + v) / mix.shape[0]))

            out = torch.stack([self._demix(mix[c:c + 1], part(c))[:, 0] for c in range(mix.shape[0])], dim=1)
        kept = out[spec.target] if spec.target >= 0 else mix - out[0]
        result = kept.numpy().T
        if spec.sample_rate != sr:
            import soxr

            result = soxr.resample(result, spec.sample_rate, sr)
            result = result[:data.shape[0]]
            if result.shape[0] < data.shape[0]:
                result = np.pad(result, ((0, data.shape[0] - result.shape[0]), (0, 0)))
        return np.ascontiguousarray(result, dtype=np.float32)
