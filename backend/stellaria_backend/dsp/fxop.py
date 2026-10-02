"""Effect-chain op: runs ``stellaria_backend.fx.apply_chain``.

Consecutive built-in slots go to ``apply_chain`` together (exactly the whole-chain semantics: one
tail budget, one pass); every VST slot gets a call of its own, preceded by a progress line naming
it: a plugin that crashes takes the whole process down, and the last line on stdout is then
``{"progress": …, "stage": "dsp.fx.vst", "slot": i, "plugin": "<name>"}``, which the job runner
maps to ``vst.crashed {name}``. ``apply_chain`` handles bypass, dry/wet ``mix`` and VST loading.
"""

from __future__ import annotations

import numpy as np

from .common import SR, DspError, bad_params, read_audio, seconds
from .edit import Output

MAX_TAIL_S = 6.0
TAIL_FLOOR = 10 ** (-90 / 20)
TAIL_FADE_S = 0.05


def trim_tail(y: np.ndarray, n: int) -> np.ndarray:
    """Keep the input length ``n``; of anything longer (reverb/delay tails) keep up to the last
    sample above -90 dBFS, at most 6 s, faded out over 50 ms when the cap cuts audible audio."""
    if y.shape[-1] <= n:
        if y.shape[-1] == n:
            return y
        out = np.zeros((y.shape[0], n), dtype=np.float32)
        out[:, : y.shape[-1]] = y
        return out
    cap = n + int(MAX_TAIL_S * SR)
    tail = np.abs(y[:, n:cap]).max(axis=0)
    loud = np.flatnonzero(tail > TAIL_FLOOR)
    end = n + (int(loud[-1]) + 1 if loud.size else 0)
    y = np.ascontiguousarray(y[:, :end], dtype=np.float32)
    if end == cap and y.shape[-1] > n:
        k = min(int(TAIL_FADE_S * SR), end - n)
        y[:, end - k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)
    return y


def _slot_name(slot: dict, plugins: dict) -> tuple[str, str]:
    """(kind, display name) of a slot for progress/log lines."""
    if slot.get("kind") == "vst":
        plugin = plugins.get(str(slot.get("pluginId"))) or {}
        return "vst", str(plugin.get("name") or slot.get("pluginId") or "VST")
    return "builtin", str(slot.get("effect") or "effect")


def fx(job, ctx) -> tuple[list[Output], dict]:
    """``{chain: FxSlot[]}`` → port ``out`` at the input offset (tails may extend the end)."""
    chain = job.params.get("chain", [])
    if not isinstance(chain, list) or not all(isinstance(s, dict) for s in chain):
        raise bad_params("fx.chain", "'chain' must be a list of effect slots")
    try:
        from ..fx import apply_chain
    except ImportError as e:  # pragma: no cover - the fx package ships with the backend
        raise DspError("dsp.failed", f"The effects engine is not available: {e}") from None
    src = job.inputs[0]
    ctx.progress(0.02, "dsp.read")
    x = read_audio(src.path)
    n = x.shape[-1]
    active = [(i, s) for i, s in enumerate(chain) if not s.get("bypass")]
    skipped = len(chain) - len(active)
    if skipped:
        ctx.log(f"fx: {skipped} bypassed slot(s) skipped")
    groups: list[list[tuple[int, dict]]] = []          # runs of built-in slots; each VST slot alone
    for i, slot in active:
        if slot.get("kind") == "vst" or not groups or groups[-1][0][1].get("kind") == "vst":
            groups.append([(i, slot)])
        else:
            groups[-1].append((i, slot))
    y = x
    done = 0
    for group in groups:
        i, slot = group[0]
        fraction = 0.1 + 0.8 * done / max(1, len(active))
        if slot.get("kind") == "vst":
            _, name = _slot_name(slot, job.vst_plugins)
            ctx.log(f"fx: slot {i + 1}/{len(chain)}: plugin '{name}'")
            # the line naming the plugin is the last one before it runs (crash attribution)
            ctx.progress(fraction, "dsp.fx.vst", slot=i, plugin=name)
        else:
            names = ", ".join(_slot_name(s, job.vst_plugins)[1] for _, s in group)
            ctx.log(f"fx: slot{'s' if len(group) > 1 else ''} {', '.join(str(j + 1) for j, _ in group)}"
                    f"/{len(chain)}: {names}")
            ctx.progress(fraction, "dsp.fx", slot=i, effect=_slot_name(slot, job.vst_plugins)[1])
        y = apply_chain(y, SR, [s for _, s in group], bpm=job.bpm, vst_plugins=job.vst_plugins)
        y = np.asarray(y, dtype=np.float32)
        if y.ndim == 1:
            y = y[None, :]
        if not np.isfinite(y).all():
            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
            ctx.log(f"fx: slot {i + 1} produced non-finite samples; replaced with silence")
        done += len(group)
    ctx.progress(0.92, "dsp.fx")
    y = trim_tail(y, n)
    tail = y.shape[-1] - n
    if tail:
        ctx.log(f"fx: tail {tail / SR:.2f} s")
    return [Output("out", y, src.offset)], {"slots": len(active), "tail": seconds(tail)}
