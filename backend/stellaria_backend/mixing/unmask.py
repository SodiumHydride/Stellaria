"""Unmasking: a dynamic EQ on the backing that makes room for the voice while it sings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import filters

PRESENCE_BAND = (1000.0, 4000.0)
BODY_BAND = (200.0, 400.0)


@dataclass
class Bands:
    presence: np.ndarray        # zero-phase 1–4 kHz band of the backing
    body: np.ndarray            # zero-phase 200–400 Hz band


def _band(x: np.ndarray, lo: float, hi: float, sr: float) -> np.ndarray:
    # 12 dB/oct sections run forwards and backwards: 24 dB/oct, zero phase, -6 dB at the edges
    return filters.zero_phase(filters.cascade(filters.highpass(lo, sr), filters.lowpass(hi, sr)), x)


def split(backing: np.ndarray, sr: float) -> Bands:
    """The two bands the dynamic EQ works on (independent of the voice, so this can run early)."""
    return Bands(_band(backing, *PRESENCE_BAND, sr), _band(backing, *BODY_BAND, sr))


def unmask(backing: np.ndarray, key: np.ndarray, sr: float, amount: float, density: float,
           bands: Bands | None = None) -> tuple[np.ndarray, dict]:
    """Dip the backing at 1–4 kHz by up to ``3·amount`` dB while ``key`` (0..1 per sample, the
    voice's presence) is up, and at 200–400 Hz by up to ``1.5·amount·density`` dB (``density``:
    0..1, how much of the time the voice sings). The bands are zero-phase, so ``x + (g-1)·band``
    is exact: with the key at 0 the backing passes unchanged."""
    if amount <= 0 or backing.shape[-1] == 0:
        return backing, {"maxDb": 0.0, "lowDb": 0.0}
    bands = bands or split(backing, sr)
    k = np.asarray(key, dtype=np.float32)
    g1 = np.power(np.float32(10.0), np.float32(-3.0 * amount / 20.0) * k) - np.float32(1.0)
    out = backing + g1 * bands.presence
    low_db = 1.5 * amount * float(np.clip(density, 0.0, 1.0))
    if low_db > 0.05:
        g2 = np.power(np.float32(10.0), np.float32(-low_db / 20.0) * k) - np.float32(1.0)
        out += g2 * bands.body
    return out.astype(np.float32, copy=False), {"maxDb": 3.0 * amount, "lowDb": low_db}
