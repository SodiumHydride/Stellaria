"""Doubler: two re-timed, slightly detuned copies of a voice, one per side (ADT-style)."""

from __future__ import annotations

import math

import numpy as np

from . import dynamics as dyn
from . import filters


def _wobble(n: int, sr: float, rate_hz: float, rng: np.random.Generator, block: int = 64) -> np.ndarray:
    """Smooth pseudo-random modulation in -1..1 (two incommensurate sines), computed every
    ``block`` samples and interpolated (it moves far slower than that)."""
    t = np.arange(-(-n // block) + 1, dtype=np.float64) * (block / sr)
    p1, p2 = rng.uniform(0, 2 * math.pi, 2)
    knots = 0.7 * np.sin(2 * math.pi * rate_hz * t + p1) + 0.3 * np.sin(2 * math.pi * rate_hz * 2.37 * t + p2)
    return dyn.upsample(knots, block, n).astype(np.float64)


def _segment_voices(x: np.ndarray, sr: float, detune_cents: float, delay_ms: float,
                    rng: np.random.Generator) -> np.ndarray:
    n = x.size
    out = np.zeros((2, n), dtype=np.float32)
    rate = 0.23
    depth = (2.0 ** (abs(detune_cents) / 1200.0) - 1.0) / (2 * math.pi * rate * 1.25)   # seconds
    pos = np.arange(n, dtype=np.int64)
    padded = np.concatenate([np.zeros(2, np.float32), x])            # index -1 and -2 read silence
    for side, base in enumerate((delay_ms * 0.85, delay_ms * 1.25)):
        d = ((base / 1000.0 + depth * _wobble(n, sr, rate * (1.0 + 0.17 * side), rng)) * sr).astype(np.float32)
        whole = np.ceil(np.maximum(d, 0.0))                          # read x[n - whole] and x[n - whole + 1]
        frac = (whole - d).astype(np.float32)                        # weight of the later sample
        i0 = np.maximum(pos - whole.astype(np.int64), -2) + 2
        a = padded[i0]
        b = padded[np.minimum(i0 + 1, n + 1)]
        out[side] = a + (b - a) * frac
    return out


def _runs(mask: np.ndarray, margin: int) -> list[tuple[int, int]]:
    on = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(on.astype(np.int8)))
    runs = []
    for a, b in zip(edges[::2], edges[1::2]):
        a, b = max(0, a - margin), min(mask.size, b + margin)
        if runs and a <= runs[-1][1]:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    return runs


def voices(x: np.ndarray, sr: float, *, detune_cents: float = 10.0, delay_ms: float = 20.0,
           seed: int = 0, where: np.ndarray | None = None) -> np.ndarray:
    """``(2, n)`` float32: the left and right doubles of a mono voice (the dry voice not included).

    Each side is ``x`` read through a slowly modulated delay (``delay_ms`` ± a few ms); the rate of
    change of the delay detunes it by up to about ``detune_cents``. High-passed at 150 Hz and
    low-passed at 9 kHz so the doubles stay behind the lead. Only computed where ``where`` (bool
    per sample; default: where ``x`` is not silent) is set, plus 100 ms around it."""
    x = np.asarray(x, dtype=np.float32)
    n = x.size
    out = np.zeros((2, n), dtype=np.float32)
    if n == 0:
        return out
    rng = np.random.default_rng(1009 + seed)
    mask = np.abs(x) > 1e-6 if where is None else np.asarray(where, dtype=bool)
    band = filters.cascade(filters.highpass(150.0, sr), filters.lowpass(9000.0, sr))
    for a, b in _runs(mask, int(0.1 * sr)):
        out[:, a:b] = filters.apply(band, _segment_voices(x[a:b], sr, detune_cents, delay_ms, rng))
    return out
