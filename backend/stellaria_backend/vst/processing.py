"""Run a VST3 / Audio Unit plugin on audio. Imported inside the ``dsp`` subprocess only (never in
the API process): this is where pedalboard gets imported.

``apply(x, sr, slot, plugin)``: ``x`` is float32 ``(channels, samples)``; ``slot`` is a ``vst``
``FxSlot`` (``state`` base64 or null, ``params`` overrides by parameter id); ``plugin`` the
``VstPlugin`` wire dict. Steps: load by path (+ ``pluginName``, cached per process via
``load``), restore ``state`` (or the plugin's default state when null, so a plugin reused by
several slots never inherits the previous slot's settings), apply ``params``, process with a
reset and up to 6 s of tail (trimmed at −90 dBFS) at 44.1 kHz — or 48 kHz with resampling when
the plugin rejects 44.1 kHz. pedalboard compensates the latency the plugin reports, so the
result stays aligned with the input (no extra shift here). Failures raise ``ApiError`` with
``vst.*`` codes; ``mix`` (dry/wet) and ``bypass`` are the caller's business.
"""

from __future__ import annotations

import base64
import importlib.util
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import ApiError

RATES = (44100, 48000)
TAIL_SECONDS = 6.0
TAIL_FLOOR_DB = -90.0

_lock = threading.Lock()
_loaded: dict[tuple[str, str], tuple[Any, bytes | None]] = {}  # (path, pluginName) → (plugin, default state)


def host_available() -> bool:
    return importlib.util.find_spec("pedalboard") is not None


def _name(plugin: dict[str, Any]) -> str:
    return str(plugin.get("name") or plugin.get("pluginName") or Path(str(plugin.get("path") or "plugin")).stem)


def _key(plugin: dict[str, Any]) -> tuple[str, str]:
    return os.path.normcase(os.path.abspath(str(plugin.get("path") or ""))), str(plugin.get("pluginName") or "")


def _load_failed(name: str, reason: str) -> ApiError:
    return ApiError("vst.load_failed", f"'{name}' could not be loaded: {reason}", status=422, detail=reason,
                    params={"name": name, "reason": reason[:300]})


def load(plugin: dict[str, Any]) -> Any:
    """The pedalboard plugin for a ``VstPlugin`` wire dict, loaded once per process."""
    name = _name(plugin)
    key = _key(plugin)
    with _lock:
        cached = _loaded.get(key)
    if cached is not None:
        return cached[0]
    if not host_available():
        raise ApiError("vst.host_missing", "The VST host add-on (pedalboard) is not installed", status=409)
    path = str(plugin.get("path") or "")
    if not path or not os.path.exists(path):
        raise ApiError("vst.not_found", f"'{name}' is not installed any more: {path}", status=404,
                       params={"name": name})
    import pedalboard

    try:
        instance = pedalboard.load_plugin(path, plugin_name=plugin.get("pluginName") or None,
                                          initialization_timeout=10.0)
    except Exception as exc:  # noqa: BLE001 - ImportError/RuntimeError/… from the plugin
        raise _load_failed(name, f"{type(exc).__name__}: {exc}") from exc
    try:
        default = bytes(instance.raw_state)
    except Exception:  # noqa: BLE001 - plugins without state support
        default = None
    with _lock:
        _loaded.setdefault(key, (instance, default))
        return _loaded[key][0]


def clear_cache() -> None:
    with _lock:
        _loaded.clear()


def _is_instrument(plugin: dict[str, Any], instance: Any | None = None) -> bool:
    if plugin.get("category") == "instrument":
        return True
    if instance is not None:
        return bool(getattr(instance, "is_instrument", False)) and not bool(getattr(instance, "is_effect", True))
    return False


def _restore(instance: Any, state: str | None, default: bytes | None, name: str) -> None:
    try:
        data = base64.b64decode(state) if state else default
    except (ValueError, TypeError) as exc:
        raise _load_failed(name, f"state is not base64: {exc}") from exc
    if data is None:
        return
    try:
        instance.raw_state = data
    except Exception as exc:  # noqa: BLE001
        raise _load_failed(name, f"could not restore the state: {exc}") from exc


def _apply_params(instance: Any, params: dict[str, Any]) -> None:
    """Overrides by parameter id (pedalboard's attribute name); values that do not fit are skipped,
    numbers are clamped to the parameter's range."""
    if not params:
        return
    parameters = instance.parameters
    for pid, value in params.items():
        parameter = parameters.get(pid)
        if parameter is None or value is None:
            continue
        try:
            if parameter.type is float:
                number = float(value)
                if parameter.min_value is not None:
                    number = max(float(parameter.min_value), number)
                if parameter.max_value is not None:
                    number = min(float(parameter.max_value), number)
                value = number
            elif parameter.type is bool:
                value = value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "on", "yes")
            else:
                value = str(value)
                if value not in [str(v) for v in parameter.valid_values]:
                    continue
            setattr(instance, pid, value)
        except (TypeError, ValueError):
            continue


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """(channels, samples) float32 resampling (soxr when available, else polyphase)."""
    if sr_in == sr_out:
        return x
    try:
        import soxr

        return np.ascontiguousarray(soxr.resample(x.T, sr_in, sr_out, quality="HQ").T, dtype=np.float32)
    except ImportError:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(int(sr_in), int(sr_out))
        return resample_poly(x, sr_out // g, sr_in // g, axis=1).astype(np.float32)


def _run(instance: Any, audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        return np.asarray(instance.process(audio, float(sr), reset=True), dtype=np.float32)
    except Exception:
        if audio.shape[0] != 1:
            raise
        # stereo-only plugins: feed a mono signal as two channels
        return np.asarray(instance.process(np.repeat(audio, 2, axis=0), float(sr), reset=True), dtype=np.float32)


def _process(instance: Any, audio: np.ndarray, sr: int, name: str) -> np.ndarray:
    rates = [sr] if sr in RATES else []
    rates += [r for r in RATES if r not in rates]
    errors = []
    for rate in rates:
        data = resample(audio, sr, rate)
        try:
            out = _run(instance, data, rate)
        except Exception as exc:  # noqa: BLE001 - try the next rate
            errors.append(f"{rate} Hz: {type(exc).__name__}: {exc}")
            continue
        out = resample(out, rate, sr)
        n = audio.shape[1]
        if out.shape[1] < n:
            out = np.pad(out, ((0, 0), (0, n - out.shape[1])))
        return out[:, :n]
    raise _load_failed(name, "processing failed (" + "; ".join(errors) + ")")


def trim_tail(y: np.ndarray, n: int, floor_db: float = TAIL_FLOOR_DB) -> np.ndarray:
    """Keep the first ``n`` samples plus the tail up to its last sample above ``floor_db``."""
    if y.shape[1] <= n:
        return y
    floor = 10 ** (floor_db / 20)
    loud = np.nonzero(np.max(np.abs(y[:, n:]), axis=0) > floor)[0]
    return y[:, : n + (int(loud[-1]) + 1 if len(loud) else 0)]


def apply(x: np.ndarray, sr: int, slot: dict[str, Any], plugin: dict[str, Any]) -> np.ndarray:
    """Process ``x`` (float32, channels × samples) through the slot's plugin; returns float32
    (channels × samples), possibly longer than ``x`` by the plugin's tail (≤ 6 s)."""
    name = _name(plugin)
    if _is_instrument(plugin):
        raise ApiError("vst.instrument", f"'{name}' is an instrument, not an effect", status=422,
                       params={"name": name})
    instance = load(plugin)
    if _is_instrument(plugin, instance):
        raise ApiError("vst.instrument", f"'{name}' is an instrument, not an effect", status=422,
                       params={"name": name})
    with _lock:
        default = _loaded.get(_key(plugin), (None, None))[1]
    _restore(instance, slot.get("state"), default, name)
    _apply_params(instance, dict(slot.get("params") or {}))
    audio = np.asarray(x, dtype=np.float32)
    if audio.ndim == 1:
        audio = audio[None, :]
    n = audio.shape[1]
    padded = np.concatenate([audio, np.zeros((audio.shape[0], int(round(TAIL_SECONDS * sr))), np.float32)], axis=1)
    out = _process(instance, padded, int(sr), name)
    if not np.all(np.isfinite(out)):
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(trim_tail(out, n), dtype=np.float32)
