"""Running an effect chain (``FxSlot[]``): built-in slots here, ``vst`` slots through
``stellaria_backend.vst.processing.apply`` (imported only when a chain has a VST slot)."""

from __future__ import annotations

import importlib
from typing import Any, Callable

import numpy as np

from ..errors import ApiError
from .effects import MAX_TAIL_S, RUNNERS, TAIL_FLOOR_DB
from .specs import EFFECTS

VST_MODULE = "stellaria_backend.vst.processing"


def _vst_apply() -> Callable[..., np.ndarray]:
    try:
        module = importlib.import_module(VST_MODULE)
    except ModuleNotFoundError as e:
        if e.name and VST_MODULE.startswith(e.name):
            raise ApiError("vst.host_missing", "The VST host add-on is not installed", status=409) from None
        raise
    return module.apply


def _as_audio(x: Any) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ApiError("fx.bad_audio", "Audio must be (channels, samples)", status=422)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def blend(dry: np.ndarray, wet: np.ndarray, mix: float) -> np.ndarray:
    """``dry·(1-mix) + wet·mix`` with channels broadcast (mono ↔ stereo) and lengths zero-padded."""
    if mix >= 1.0:
        return wet
    channels = max(dry.shape[0], wet.shape[0])
    n = max(dry.shape[-1], wet.shape[-1])
    out = np.zeros((channels, n), dtype=np.float32)
    out[:, : dry.shape[-1]] += dry * np.float32(1.0 - mix)       # (1, m) broadcasts over 2 channels
    out[:, : wet.shape[-1]] += wet * np.float32(mix)
    return out


def trim_tail(x: np.ndarray, n: int, sr: float) -> np.ndarray:
    """Cut what follows the original ``n`` samples after the last sample above −90 dBFS
    (never shorter than ``n``, never more than 6 s longer)."""
    limit = n + int(MAX_TAIL_S * sr)
    x = x[:, :limit]
    if x.shape[-1] <= n:
        return x
    tail = np.abs(x[:, n:]).max(axis=0)
    loud = np.flatnonzero(tail > 10.0 ** (TAIL_FLOOR_DB / 20.0))
    return x[:, : n + (int(loud[-1]) + 1 if loud.size else 0)]


def apply_chain(x: np.ndarray, sr: int, chain: list[dict], *, bpm: float | None,
                vst_plugins: dict[str, dict]) -> np.ndarray:
    """Run ``chain`` over ``x`` (float32 ``(channels, samples)``) and return float32
    ``(channels', samples')``: effects may turn mono into stereo, and reverb/delay tails extend the
    end by up to 6 s (trimmed at −90 dBFS). Bypassed slots are skipped; ``mix`` blends dry/wet."""
    x = _as_audio(x)
    n0 = x.shape[-1]
    ctx: dict[str, Any] = {"bpm": bpm, "max_n": n0 + int(MAX_TAIL_S * sr)}
    for i, slot in enumerate(chain or []):
        if not isinstance(slot, dict):
            raise ApiError("fx.bad_slot", f"Effect slot {i + 1} is invalid", status=422, params={"index": i + 1})
        if slot.get("bypass"):
            continue
        try:
            mix = float(slot.get("mix", 1.0))
        except (TypeError, ValueError):
            mix = 1.0
        mix = min(max(mix if np.isfinite(mix) else 1.0, 0.0), 1.0)
        if mix <= 0.0:
            continue
        kind = slot.get("kind", "builtin")
        ctx["n"] = x.shape[-1]
        if kind == "builtin":
            effect = EFFECTS.get(str(slot.get("effect")))
            if effect is None:
                raise ApiError("fx.unknown_effect", f"Unknown effect '{slot.get('effect')}'", status=422,
                               params={"effect": str(slot.get("effect"))})
            wet = RUNNERS[effect.id](x, float(sr), effect.parse(slot.get("params")), ctx)
        elif kind == "vst":
            plugin_id = slot.get("pluginId")
            plugin = (vst_plugins or {}).get(plugin_id) if isinstance(plugin_id, str) else None
            if plugin is None:
                raise ApiError("vst.not_found", f"VST plugin '{plugin_id}' is not available", status=404,
                               params={"name": str(slot.get("name") or plugin_id)})
            wet = _as_audio(_vst_apply()(x, sr, slot, plugin))
        else:
            raise ApiError("fx.bad_slot", f"Effect slot {i + 1} has an unknown kind '{kind}'", status=422,
                           params={"index": i + 1})
        x = blend(x, _as_audio(wet), mix)
    return trim_tail(x, n0, sr).astype(np.float32, copy=False)
