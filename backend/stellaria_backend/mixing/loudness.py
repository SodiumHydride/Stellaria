"""ITU-R BS.1770-4 integrated loudness (K-weighting + gating) and peak estimates.

Arrays are channel-first: ``(n,)`` mono or ``(channels, n)``. The K-weighting filter is designed
for the actual sample rate from the analogue prototype parameters (the libebur128 derivation),
so 44.1 kHz and 48 kHz measurements agree with the standard's 48 kHz coefficients.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

# Pre-filter (high shelf) and RLB high-pass of BS.1770 as analogue prototypes.
_SHELF = (1681.974450955533, 3.999843853973347, 0.7071752369554196)
_RLB = (38.13547087602444, 0.5003270373238773)

ABS_GATE = -70.0
REL_GATE = -10.0
SILENCE_LUFS = -70.0   # what callers report instead of -inf
EPS = 1e-12


def k_weighting(sr: float) -> np.ndarray:
    """K-weighting (pre-filter shelf + RLB high-pass) as SOS for ``sr``.

    Bilinear transform of the analogue prototypes exactly as libebur128 derives them; at 48 kHz
    this reproduces the coefficient tables of BS.1770-4 to ~1e-12."""
    f0, gain, q = _SHELF
    k = math.tan(math.pi * f0 / sr)
    vh = 10.0 ** (gain / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = [(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0,
             1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    f0, q = _RLB
    k = math.tan(math.pi * f0 / sr)
    a0 = 1.0 + k / q + k * k
    rlb = [1.0, -2.0, 1.0, 1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0]
    return np.array([shelf, rlb], dtype=np.float64)


def _channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x[None, :] if x.ndim == 1 else x


def channel_weights(channels: int) -> np.ndarray:
    """BS.1770 channel weights: 1.0 for L/R/C, 1.41 for the surround pair of a 5-channel layout."""
    w = np.ones(channels)
    if channels == 5:
        w[3:] = 1.41
    return w


def k_weight(x: np.ndarray, sr: float) -> np.ndarray:
    """K-weighted copy, channel-first (float32 in → float32; within 0.001 LU of float64)."""
    from scipy import signal

    from .filters import dithered

    xc = _channels(x)
    if xc.dtype == np.float32:
        return signal.sosfilt(k_weighting(sr).astype(np.float32), dithered(xc), axis=-1)
    return signal.sosfilt(k_weighting(sr), xc.astype(np.float64, copy=False), axis=-1)


def block_energies(xk: np.ndarray, sr: float) -> np.ndarray:
    """Mean square per channel of 400 ms blocks with 75 % overlap → ``(channels, blocks)``.

    ``xk`` is K-weighted audio. A signal shorter than one block yields one block over all of it."""
    xk = _channels(xk)
    n = xk.shape[-1]
    block = int(round(0.4 * sr))
    hop = int(round(0.1 * sr))
    if n == 0:
        return np.zeros((xk.shape[0], 0))
    if n < block:
        return np.mean(np.square(xk, dtype=np.float64), axis=-1, keepdims=True)
    count = 1 + (n - block) // hop
    if block == 4 * hop:
        # exact per-hop sums, then four consecutive hops per block
        hops = count + 3
        r = xk[:, : hops * hop].reshape(xk.shape[0], hops, hop)
        sq = np.einsum("chk,chk->ch", r, r).astype(np.float64)
        cs = np.concatenate([np.zeros((xk.shape[0], 1)), np.cumsum(sq, axis=-1)], axis=-1)
        sums = cs[:, 4: 4 + count] - cs[:, :count]
    else:
        cs = np.concatenate([np.zeros((xk.shape[0], 1)), np.cumsum(np.square(xk, dtype=np.float64), axis=-1)], axis=-1)
        starts = hop * np.arange(count)
        sums = cs[:, starts + block] - cs[:, starts]
    return np.maximum(sums, 0.0) / block


def gated_loudness(z: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Integrated loudness (LUFS) from block energies ``z`` (channels × blocks); ``-inf`` if silent."""
    if z.size == 0:
        return float("-inf")
    w = channel_weights(z.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    power = (w[:, None] * z).sum(axis=0)
    with np.errstate(divide="ignore"):
        lj = -0.691 + 10.0 * np.log10(np.maximum(power, 0.0))
    gate = lj > ABS_GATE
    if not gate.any():
        return float("-inf")
    abs_level = -0.691 + 10.0 * math.log10(float(power[gate].mean()))
    gate &= lj > abs_level + REL_GATE
    if not gate.any():
        return float("-inf")
    return -0.691 + 10.0 * math.log10(float(power[gate].mean()))


def integrated_loudness(x: np.ndarray, sr: float) -> float:
    """BS.1770-4 integrated loudness in LUFS (``-inf`` for silence)."""
    xc = _channels(x)
    if xc.shape[-1] == 0:
        return float("-inf")
    return gated_loudness(block_energies(k_weight(xc, sr), sr))


def integrated_weighted(xk: np.ndarray, sr: float) -> float:
    """Integrated loudness of an already K-weighted signal."""
    xk = _channels(xk)
    return gated_loudness(block_energies(xk, sr)) if xk.shape[-1] else float("-inf")


def report_lufs(value: float) -> float:
    """Finite, rounded LUFS for JSON (silence reads as -70)."""
    if not math.isfinite(value):
        return SILENCE_LUFS
    return round(max(value, SILENCE_LUFS), 2)


def db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), EPS))


def sample_peak(x: np.ndarray) -> float:
    x = np.asarray(x)
    return float(np.max(np.abs(x))) if x.size else 0.0


def sample_peak_db(x: np.ndarray) -> float:
    return max(db(sample_peak(x)), -120.0)


def _fraction_kernel(frac: float, taps: int = 16) -> np.ndarray:
    """Windowed-sinc interpolator for the point ``frac`` (0..1) of the way to the next sample."""
    k = np.arange(taps) - (taps // 2 - 1)        # -7..8 for 16 taps
    h = np.sinc(k - frac) * np.kaiser(taps, 6.0)
    return (h / h.sum()).astype(np.float64)


_KERNELS = {
    2: [_fraction_kernel(0.5)],
    4: [_fraction_kernel(0.25), _fraction_kernel(0.5), _fraction_kernel(0.75)],
}


def peak_envelope(x: np.ndarray, oversample: int = 4) -> np.ndarray:
    """Per-sample peak of a 2× or 4× oversampled reconstruction, maximum over channels → ``(n,)``.

    The oversampled signal is the original samples interleaved with windowed-sinc interpolations
    at ¼, ½, ¾ (4×) or ½ (2×) of the way to the next sample, so the envelope at ``n`` is the
    largest of ``|x[n]|`` and those in-between values (BS.1770-style true-peak estimate)."""
    xc = _channels(x)
    if xc.shape[-1] == 0:
        return np.zeros(0, dtype=np.float32)
    env = np.abs(xc)
    for kernel in _KERNELS[4 if oversample >= 4 else 2]:
        # correlate1d centres the kernel; origin -1 makes taps -7..8 line up with x[n-7]..x[n+8]
        mid = ndimage.correlate1d(xc, kernel.astype(xc.dtype), axis=-1, mode="constant", origin=-1)
        np.maximum(env, np.abs(mid), out=env)
    return env.max(axis=0) if env.shape[0] > 1 else env[0]


def true_peak_db(x: np.ndarray, oversample: int = 4) -> float:
    """Peak in dBFS of the oversampled estimate (dBTP; ≥ the sample peak)."""
    env = peak_envelope(x, oversample)
    return max(db(float(env.max())) if env.size else -120.0, -120.0)


def short_term(x: np.ndarray, sr: float, window_s: float = 3.0, hop_s: float = 0.5,
               weighted: bool = False) -> np.ndarray:
    """Short-term loudness (LUFS, ungated) of ``window_s`` windows centred every ``hop_s`` seconds;
    frame ``i`` is centred at ``(i + 0.5) * hop_s``. Silence reads -inf. ``weighted``: ``x`` is
    already K-weighted."""
    xk = x if weighted else k_weight(x, sr)
    xk = xk[None, :] if xk.ndim == 1 else xk
    hop = max(1, int(round(hop_s * sr)))
    n = xk.shape[-1]
    frames = -(-n // hop)
    if frames == 0:
        return np.zeros(0)
    full = n // hop
    per_channel = np.zeros((xk.shape[0], frames))
    if full:
        r = xk[:, : full * hop].reshape(xk.shape[0], full, hop)
        per_channel[:, :full] = np.einsum("chk,chk->ch", r, r)
    if n % hop:
        per_channel[:, full] = np.sum(np.square(xk[:, full * hop:], dtype=np.float64), axis=-1)
    hop_energy = (channel_weights(xk.shape[0])[:, None] * per_channel).sum(axis=0)
    k = max(1, int(round(window_s / hop_s)))
    ms = ndimage.uniform_filter1d(hop_energy, size=k, mode="constant") / hop
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(np.maximum(ms, 0.0))
