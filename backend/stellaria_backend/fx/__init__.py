"""Built-in effects library (``docs/canvas-v3.md`` §3.1, §6), reusing the mixer's DSP blocks.

    effect_specs() -> list[dict]                    # EffectSpec wire dicts (GET /fx/effects)
    apply_chain(x, sr, chain, *, bpm, vst_plugins)  # float32 (channels, samples) → processed

Importing this package is cheap (numpy only); the DSP modules load on first use, and the VST
host (``stellaria_backend.vst.processing``) only when a chain contains a ``vst`` slot.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .specs import CATEGORIES, EFFECTS, effect_specs

__all__ = ["CATEGORIES", "EFFECTS", "apply_chain", "effect_specs"]


def apply_chain(x: np.ndarray, sr: int, chain: list[dict], *, bpm: float | None,
                vst_plugins: dict[str, dict[str, Any]]) -> np.ndarray:
    """Run an ``FxSlot[]`` chain over ``x`` (float32 ``(channels, samples)``); see :mod:`.chain`."""
    from .chain import apply_chain as _apply

    return _apply(x, sr, chain, bpm=bpm, vst_plugins=vst_plugins)
