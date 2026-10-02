"""Dynamics: RMS compressor, split-band de-esser, soft clipper, look-ahead peak limiter.

Everything is vectorized. Detectors run at a *block rate* (one value per 8–32 samples, block
energies are exact) and the resulting gains are linearly interpolated back to the sample rate.

The only non-linear recursion, the attack/release ballistics, uses the "decoupled" peak detector
of Giannoulis, Massberg & Reiss (JAES 2012): ``y1[n] = max(x[n], r·y1[n-1])`` (instant attack,
exponential release) followed by a one-pole attack filter. The max-recursion has a closed form —
``log y1[n] = n·log r + max_{k≤n}(log x[k] − k·log r)`` — so it is a cumulative maximum
(:func:`release_hold`) and needs no Python loop.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, signal

from . import filters

EPS = 1e-12
LIMITER_MARGIN_DB = 0.01


# ----------------------------------------------------------------------------- helpers

def coef(time_s: float, rate: float) -> float:
    """One-pole smoothing coefficient for a time constant at ``rate`` values per second."""
    if time_s <= 0 or rate <= 0:
        return 0.0
    return math.exp(-1.0 / (time_s * rate))


def one_pole(x: np.ndarray, c: float) -> np.ndarray:
    """``y[n] = (1-c)·x[n] + c·y[n-1]`` starting at ``y[-1] = x[0]`` (no start-up transient)."""
    x = np.asarray(x, dtype=np.float64)
    if c <= 0 or x.size == 0:
        return x.copy()
    zi = np.array([c * x[0]])
    y, _ = signal.lfilter([1.0 - c], [1.0, -c], x, zi=zi)
    return y


def release_hold(x: np.ndarray, r: float, chunk: int = 1 << 16) -> np.ndarray:
    """``y[n] = max(x[n], r·y[n-1])`` for ``x ≥ 0``, ``0 ≤ r < 1``, vectorized (see module doc).

    Processed in chunks so ``n·log r`` stays small; the carry from the previous chunk enters as
    ``y_prev · r^(k+1)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    if r <= 0:
        return np.maximum(x, 0.0)
    lr = math.log(r)
    out = np.empty_like(x)
    carry = 0.0
    for start in range(0, x.size, chunk):
        seg = x[start: start + chunk]
        k = np.arange(seg.size, dtype=np.float64)
        with np.errstate(divide="ignore"):
            u = np.log(np.maximum(seg, 0.0)) - k * lr
        u = np.maximum.accumulate(u) + k * lr
        y = np.exp(u)
        if carry > 0:
            y = np.maximum(y, carry * np.exp((k + 1) * lr))
        out[start: start + seg.size] = y
        carry = float(y[-1])
    return out


def release_hold_reference(x: np.ndarray, r: float) -> np.ndarray:
    """Plain loop version of :func:`release_hold` (tests only)."""
    out = np.empty(len(x))
    prev = 0.0
    for i, v in enumerate(np.asarray(x, dtype=np.float64)):
        prev = max(float(v), r * prev)
        out[i] = prev
    return out


def ballistics(gr_db: np.ndarray, attack_s: float, release_s: float, rate: float) -> np.ndarray:
    """Smooth a gain-reduction curve (dB, ≥ 0): decoupled peak detector + one-pole attack."""
    held = release_hold(np.maximum(gr_db, 0.0), coef(release_s, rate))
    return one_pole(held, coef(attack_s, rate))


def block_mean_square(x: np.ndarray, block: int) -> np.ndarray:
    """Mean square per block (the last one zero-padded), averaged over channels →
    ``(ceil(n / block),)`` float64."""
    xc = x[None, :] if x.ndim == 1 else x
    channels, n = xc.shape
    full = n // block
    out = np.empty(full + (1 if n % block else 0))
    if full:
        r = xc[:, : full * block].reshape(channels, full, block)
        out[:full] = np.einsum("cbk,cbk->b", r, r) / (block * channels)
    if n % block:
        out[full] = float(np.sum(np.square(xc[:, full * block:], dtype=np.float64))) / (block * channels)
    return out


def block_max_abs(env: np.ndarray, block: int) -> np.ndarray:
    n = env.shape[-1]
    blocks = -(-n // block)
    pad = blocks * block - n
    if pad:
        env = np.concatenate([env, np.zeros(pad, dtype=env.dtype)])
    return env.reshape(blocks, block).max(axis=-1)


def upsample(values: np.ndarray, block: int, n: int) -> np.ndarray:
    """Linear interpolation from knots at block starts ``b·block`` to ``n`` samples (float32)."""
    v = np.asarray(values, dtype=np.float32)
    nxt = np.concatenate([v[1:], v[-1:]])
    frac = (np.arange(block, dtype=np.float32) / block)[None, :]
    out = (v[:, None] + (nxt - v)[:, None] * frac).reshape(-1)
    return out[:n]


def uniform_interp(values: np.ndarray, hop: float, n: int) -> np.ndarray:
    """``values`` given at the centres ``(i + 0.5)·hop`` of a uniform grid → ``n`` samples (float32),
    linear between centres, held before the first and after the last (no searching)."""
    v = np.asarray(values, dtype=np.float32)
    if v.size == 0 or n == 0:
        return np.zeros(n, dtype=np.float32)
    step = max(1, int(round(hop)))
    half = min(n, step // 2)
    rest = n - half
    y = upsample(v, step, min(rest, v.size * step))                     # knot i at sample half + i·step
    if y.size < rest:
        y = np.concatenate([y, np.full(rest - y.size, v[-1], dtype=np.float32)])
    return np.concatenate([np.full(half, v[0], dtype=np.float32), y])


def db_to_gain(db_values: np.ndarray) -> np.ndarray:
    return np.power(10.0, np.asarray(db_values) / 20.0)


def active_mask(level_db: np.ndarray, rel_db: float = -40.0, floor_db: float = -80.0) -> np.ndarray:
    """Values within ``rel_db`` of the loud end (95th percentile) and above ``floor_db``."""
    if level_db.size == 0:
        return np.zeros(0, dtype=bool)
    top = float(np.percentile(level_db, 95))
    return level_db > max(top + rel_db, floor_db)


def active_rms_db(x: np.ndarray, sr: float, rel_db: float = -40.0) -> float | None:
    """RMS (dBFS) over the 20 ms frames within ``rel_db`` of the loud end; ``None`` if silent."""
    frame = max(1, int(0.02 * sr))
    ms = block_mean_square(x, frame)
    level = 10.0 * np.log10(ms + EPS)
    mask = active_mask(level, rel_db)
    if not mask.any():
        return None
    return float(10.0 * np.log10(ms[mask].mean() + EPS))


# ----------------------------------------------------------------------------- compressor

def gain_computer(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    """Static soft-knee curve → gain reduction in dB (≥ 0)."""
    over = np.asarray(level_db, dtype=np.float64) - threshold_db
    slope = 1.0 - 1.0 / max(ratio, 1.0)
    gr = np.where(over > 0, slope * over, 0.0)
    if knee_db > 0:
        half = knee_db / 2.0
        in_knee = np.abs(over) <= half
        gr = np.where(in_knee, slope * (over + half) ** 2 / (2.0 * knee_db), gr)
        gr = np.where(over < -half, 0.0, gr)
    return gr


def detector_db(x: np.ndarray, sr: float, rms_s: float = 0.01, block: int = 32) -> np.ndarray:
    """RMS level (dB) at block rate: exact block energies smoothed by a one-pole of ``rms_s``."""
    ms = block_mean_square(x, block)
    rate = sr / block
    return 10.0 * np.log10(one_pole(ms, coef(rms_s, rate)) + EPS)


def compress(x: np.ndarray, sr: float, *, threshold_db: float, ratio: float = 4.0, knee_db: float = 4.5,
             attack_s: float = 0.019, release_s: float = 0.4, rms_s: float = 0.01, makeup_db: float = 0.0,
             block: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """RMS compressor (linked over channels). Returns ``(output, gain_reduction_db_per_block)``."""
    n = x.shape[-1]
    if n == 0:
        return x.copy(), np.zeros(0)
    rate = sr / block
    level = detector_db(x, sr, rms_s, block)
    gr = ballistics(gain_computer(level, threshold_db, ratio, knee_db), attack_s, release_s, rate)
    gain = upsample(db_to_gain(makeup_db - gr), block, n)
    return (x * gain).astype(np.float32), gr


# ----------------------------------------------------------------------------- de-esser

def deess(x: np.ndarray, sr: float, *, freq: float = 6500.0, max_reduction_db: float = 6.0, ratio: float = 4.0,
          threshold_db: float | None = None, above_median_db: float = 6.0, relative_db: float = 15.0,
          attack_s: float = 0.0015, release_s: float = 0.06, rms_s: float = 0.003,
          block: int = 16) -> tuple[np.ndarray, dict]:
    """Split-band de-esser: only the band above ``freq`` is turned down.

    ``hb`` is a zero-phase high-pass of ``x`` (4th-order Butterworth run forwards and backwards,
    -6 dB at ``freq``), so ``x - hb`` is its exact, phase-coherent complement; the output is
    ``x + (g - 1)·hb`` (exactly ``x`` while ``g = 1``). A causal IIR split would not do: near the
    split its phase shift makes ``x - k·hb`` *louder* than ``x``. The gain reduction
    follows the band's RMS envelope above a threshold taken from the band's statistics over the
    sung parts (median + ``above_median_db``) and is only applied where the band is within
    ``relative_db`` of the full-band level (sibilants are loud *relative* to the voice, bright
    vowels are not), capped at ``max_reduction_db``."""
    n = x.shape[-1]
    if n == 0:
        return x.copy(), {"threshold": None, "maxReduction": 0.0}
    hb = filters.zero_phase(filters.highpass4(freq, sr), x)
    rate = sr / block
    c = coef(rms_s, rate)
    hb_db = 10.0 * np.log10(one_pole(block_mean_square(hb, block), c) + EPS)
    fb_db = 10.0 * np.log10(one_pole(block_mean_square(x, block), c) + EPS)
    active = active_mask(fb_db)
    if threshold_db is None:
        if not active.any():
            return x.copy(), {"threshold": None, "maxReduction": 0.0}
        threshold_db = float(np.median(hb_db[active])) + above_median_db
    over = np.minimum(hb_db - threshold_db, hb_db - (fb_db - relative_db))
    gr = np.clip(over * (1.0 - 1.0 / max(ratio, 1.0)), 0.0, max_reduction_db)
    gr = ballistics(gr, attack_s, release_s, rate)
    g = upsample(db_to_gain(-gr), block, n)
    out = x + (g - 1.0) * hb
    return out.astype(np.float32), {"threshold": threshold_db, "maxReduction": float(gr.max(initial=0.0))}


# ----------------------------------------------------------------------------- clipper / limiter

def soft_clip(x: np.ndarray, ceiling: float, softness: float = 0.2) -> np.ndarray:
    """Soft clipper: linear up to ``ceiling·(1 - softness)``, then a tanh knee that approaches
    (never exceeds) ``ceiling``. ``softness = 0`` is a hard clip."""
    ceiling = float(ceiling)
    knee = ceiling * (1.0 - min(max(softness, 0.0), 0.99))
    ax = np.abs(x)
    over = ax > knee
    if not over.any():
        return x.copy()
    y = x.copy()
    if knee >= ceiling:
        y[over] = np.sign(x[over]) * ceiling
        return y
    span = ceiling - knee
    y[over] = np.sign(x[over]) * (knee + span * np.tanh((ax[over] - knee) / span))
    return y


def limiter_gain(env: np.ndarray, sr: float, *, ceiling_db: float = -1.0, drive_db: float = 0.0,
                 lookahead_s: float = 0.0015, release_s: float = 0.08, block: int = 8) -> np.ndarray:
    """Gain reduction (dB, ≥ 0, one knot per ``block`` samples) that keeps ``env·drive`` at or
    under the ceiling.

    Look-ahead: the required reduction of each block is spread over the ``L`` blocks before it
    (``maximum_filter1d`` over the next ``L`` blocks, then two causal moving averages whose
    combined span is ``L`` blocks); this keeps every sample under the ceiling while the gain
    ramps down smoothly in advance. Release: decoupled peak detector (exponential)."""
    peaks = block_max_abs(np.asarray(env, dtype=np.float32), block).astype(np.float64)
    # aim 0.01 dB under the ceiling so float32 rounding never lands a hair above it
    need = np.maximum(20.0 * np.log10(peaks + EPS) + drive_db - (ceiling_db - LIMITER_MARGIN_DB), 0.0)
    la = max(2, int(round(lookahead_s * sr / block)))
    # window [b-1, b+la]: covers the interpolation between knot b and b+1 as well
    size = la + 2
    h = ndimage.maximum_filter1d(need, size=size, origin=-(size // 2) + 1, mode="nearest")
    h = release_hold(h, coef(release_s, sr / block))
    m = la // 2 + 1                      # two boxcars of m: combined causal span 2m-1 ≤ la+1
    for _ in range(2):
        h = ndimage.uniform_filter1d(h, size=m, origin=(m - 1) // 2, mode="nearest")
    return h


def apply_block_gain(x: np.ndarray, gr_db: np.ndarray, block: int, gain_db: float = 0.0) -> np.ndarray:
    g = upsample(db_to_gain(gain_db - gr_db), block, x.shape[-1])
    return (x * g).astype(np.float32)


def limit(x: np.ndarray, sr: float, *, ceiling_db: float = -1.0, drive_db: float = 0.0,
          lookahead_s: float = 0.0015, release_s: float = 0.08, env: np.ndarray | None = None,
          block: int = 8) -> tuple[np.ndarray, float]:
    """Look-ahead limiter on the 2× oversampled peak estimate. Returns ``(output, max_gr_db)``."""
    from .loudness import peak_envelope

    if x.shape[-1] == 0:
        return x.copy(), 0.0
    env = peak_envelope(x) if env is None else env
    gr = limiter_gain(env, sr, ceiling_db=ceiling_db, drive_db=drive_db, lookahead_s=lookahead_s,
                      release_s=release_s, block=block)
    return apply_block_gain(x, gr, block, drive_db), float(gr.max(initial=0.0))


# ----------------------------------------------------------------------------- expander / gate

def expand(x: np.ndarray, sr: float, *, threshold_db: float | None = None, range_db: float | None = None,
           hold_s: float = 0.08, attack_s: float = 0.002, release_s: float = 0.06, lookahead_s: float = 0.005,
           rms_s: float = 0.005, block: int = 32) -> tuple[np.ndarray, dict]:
    """Downward expander for the gaps between phrases (hiss and artefacts of converted vocals).

    The gate is open while the RMS level is above the threshold, plus ``hold_s`` after it falls
    below and ``lookahead_s`` before it rises (so consonant onsets are never cut). When closed
    the signal is turned down by ``range_db``; it opens within ``attack_s`` and closes over
    ``release_s``. Defaults come from the level statistics: threshold 6 dB over the noise floor
    (at most 18 dB under the median sung level), range 6..12 dB — more when the floor is high."""
    n = x.shape[-1]
    if n == 0:
        return x.copy(), {"threshold": None, "range": 0.0}
    rate = sr / block
    level = detector_db(x, sr, rms_s, block)
    active = active_mask(level)
    if not active.any():
        return x.copy(), {"threshold": None, "range": 0.0}
    median = float(np.median(level[active]))
    audible = level[level > -100.0]
    floor = float(np.percentile(audible, 10)) if audible.size else -120.0
    if threshold_db is None:
        threshold_db = min(max(floor + 6.0, median - 40.0), median - 18.0)
    if range_db is None:
        range_db = float(np.clip(12.0 - (median - floor - 30.0) * 6.0 / 15.0, 6.0, 12.0))
    hold = max(0, int(round(hold_s * rate)))
    la = max(0, int(round(lookahead_s * rate)))
    size = hold + la + 1
    is_open = (level > threshold_db).astype(np.float64)
    # window [b - hold, b + la]: open if the level was above the threshold recently or soon will be
    is_open = ndimage.maximum_filter1d(is_open, size=size, origin=hold - size // 2, mode="nearest")
    opening = ballistics(is_open * range_db, attack_s, release_s, rate)     # rises fast, falls slowly
    gr = np.clip(range_db - opening, 0.0, range_db)
    y = (x * upsample(db_to_gain(-gr), block, n)).astype(np.float32)
    return y, {"threshold": float(threshold_db), "range": float(range_db), "floor": floor, "median": median,
               "closed": float(np.mean(gr > 0.5 * range_db))}


# ----------------------------------------------------------------------------- saturation

def _halfband(factor: int, half_len: int = 6) -> np.ndarray:
    """Anti-imaging/anti-aliasing FIR for ``resample_poly`` (``resample_poly`` scales it by ``up``):
    ``2·half_len·factor + 1`` taps instead of the default ``20·factor + 1`` — about half the work,
    with a stop band that is plenty under a saturator."""
    return signal.firwin(2 * half_len * factor + 1, 1.0 / factor, window=("kaiser", 7.0)).astype(np.float32)

def saturate(x: np.ndarray, sr: float, drive_db: float, *, mode: str = "tube", oversample: int = 2) -> np.ndarray:
    """Waveshaper with unity small-signal gain, run 2× oversampled (``resample_poly``).

    ``tanh``: symmetric (odd harmonics); ``tube``: biased tanh (adds even harmonics), followed by a
    5 Hz DC blocker. Louder ``drive_db`` = earlier, rounder peak compression."""
    n = x.shape[-1]
    if n == 0:
        return x.copy()
    g = 10.0 ** (drive_db / 20.0)
    up = (signal.resample_poly(x, oversample, 1, axis=-1, window=_halfband(oversample)) if oversample > 1
          else np.asarray(x, dtype=np.float64))
    if mode == "tanh":
        y = np.tanh(g * up) / g
    else:
        b = 0.25
        tb = math.tanh(b)
        y = (np.tanh(g * up + b) - tb) / (g * (1.0 - tb * tb))
    if oversample > 1:
        y = signal.resample_poly(y, 1, oversample, axis=-1, window=_halfband(oversample))[..., :n]
    y = np.asarray(y, dtype=np.float32)
    if mode != "tanh":
        y = filters.apply(filters.highpass1(5.0, sr), y)
    return y
