"""RBJ "Audio EQ Cookbook" biquads as second-order sections for ``scipy.signal.sosfilt``.

Every designer returns a ``(1, 6)`` float64 array ``[b0, b1, b2, 1, a1, a2]``; stack several with
:func:`cascade` and run them in one call with :func:`apply`. Audio inside the mixer is laid out
channel-first (``(n,)`` mono or ``(channels, n)``), and filtering runs along the last axis, which
is several times faster than filtering an interleaved ``(n, channels)`` array.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy import signal

BUTTERWORTH_Q = 1.0 / math.sqrt(2.0)
# Q values of the two sections of a 4th-order Butterworth filter.
BUTTERWORTH4_Q = (0.5411961001461969, 1.3065629648763766)


def _w0(f0: float, sr: float) -> float:
    f = min(max(float(f0), 1.0), 0.49 * float(sr))
    return 2.0 * math.pi * f / float(sr)


def _sos(b0: float, b1: float, b2: float, a0: float, a1: float, a2: float) -> np.ndarray:
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64)


def peaking(f0: float, gain_db: float, q: float, sr: float) -> np.ndarray:
    """Bell: ``gain_db`` at ``f0``; bandwidth set by ``q``."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = _w0(f0, sr)
    alpha = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    return _sos(1 + alpha * a, -2 * c, 1 - alpha * a, 1 + alpha / a, -2 * c, 1 - alpha / a)


def low_shelf(f0: float, gain_db: float, sr: float, q: float = BUTTERWORTH_Q) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = _w0(f0, sr)
    alpha = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    sa = 2.0 * math.sqrt(a) * alpha
    return _sos(
        a * ((a + 1) - (a - 1) * c + sa), 2 * a * ((a - 1) - (a + 1) * c), a * ((a + 1) - (a - 1) * c - sa),
        (a + 1) + (a - 1) * c + sa, -2 * ((a - 1) + (a + 1) * c), (a + 1) + (a - 1) * c - sa,
    )


def high_shelf(f0: float, gain_db: float, sr: float, q: float = BUTTERWORTH_Q) -> np.ndarray:
    a = 10.0 ** (gain_db / 40.0)
    w0 = _w0(f0, sr)
    alpha = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    sa = 2.0 * math.sqrt(a) * alpha
    return _sos(
        a * ((a + 1) + (a - 1) * c + sa), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - sa),
        (a + 1) - (a - 1) * c + sa, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - sa,
    )


def highpass(f0: float, sr: float, q: float = BUTTERWORTH_Q) -> np.ndarray:
    """12 dB/oct high-pass (Butterworth at the default ``q``)."""
    w0 = _w0(f0, sr)
    alpha = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    return _sos((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + alpha, -2 * c, 1 - alpha)


def lowpass(f0: float, sr: float, q: float = BUTTERWORTH_Q) -> np.ndarray:
    """12 dB/oct low-pass (Butterworth at the default ``q``)."""
    w0 = _w0(f0, sr)
    alpha = math.sin(w0) / (2.0 * q)
    c = math.cos(w0)
    return _sos((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + alpha, -2 * c, 1 - alpha)


def highpass1(f0: float, sr: float) -> np.ndarray:
    """6 dB/oct (first-order) high-pass, bilinear transform with pre-warping."""
    k = math.tan(_w0(f0, sr) / 2.0)
    return _sos(1.0, -1.0, 0.0, 1.0 + k, k - 1.0, 0.0)


def lowpass1(f0: float, sr: float) -> np.ndarray:
    """6 dB/oct (first-order) low-pass, bilinear transform with pre-warping."""
    k = math.tan(_w0(f0, sr) / 2.0)
    return _sos(k, k, 0.0, 1.0 + k, k - 1.0, 0.0)


def highpass4(f0: float, sr: float) -> np.ndarray:
    """24 dB/oct Butterworth high-pass (two RBJ sections)."""
    return cascade(*(highpass(f0, sr, q) for q in BUTTERWORTH4_Q))


def lowpass4(f0: float, sr: float) -> np.ndarray:
    """24 dB/oct Butterworth low-pass (two RBJ sections)."""
    return cascade(*(lowpass(f0, sr, q) for q in BUTTERWORTH4_Q))


def cascade(*sections: np.ndarray) -> np.ndarray:
    parts = [np.atleast_2d(s) for s in sections if s is not None and len(s)]
    if not parts:
        return np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    return np.vstack(parts)


# fixed noise at -360 dBFS (deterministic)
_DITHER = (np.random.default_rng(0x5EED).standard_normal(1 << 16) * 1e-18).astype(np.float32)


def dithered(x: np.ndarray) -> np.ndarray:
    """``x`` plus noise at -360 dBFS. Digital silence lets float32 IIR states decay into the
    denormal range, which makes ``sosfilt`` ~5x slower; the dither keeps them normal and is far
    below anything audible or measurable."""
    if x.dtype != np.float32 or x.shape[-1] == 0:
        return x
    n = x.shape[-1]
    return x + (_dither(n) if n >= 1 << 20 else np.resize(_DITHER, n))


@lru_cache(maxsize=4)
def _dither(n: int) -> np.ndarray:
    d = np.resize(_DITHER, n)
    d.setflags(write=False)
    return d


def apply(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter along the last axis. float32 audio is filtered in float32 (3-4x faster than the
    float64 path; ≥ 95 dB SNR for the mixer's filters), float64 stays float64."""
    if x.shape[-1] == 0:
        return x.copy()
    if x.dtype == np.float32:
        return signal.sosfilt(np.asarray(sos, dtype=np.float32), dithered(x), axis=-1)
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float64), axis=-1)


def zero_phase(sos: np.ndarray, x: np.ndarray, pad: int = 2048) -> np.ndarray:
    """Forward-backward filtering along the last axis (zero phase, squared magnitude), with an odd
    extension of ``pad`` samples at both ends like ``filtfilt``. Unlike ``sosfiltfilt`` it stays
    in float32 for float32 audio (about 2.5x faster)."""
    n = x.shape[-1]
    if n == 0:
        return x.copy()
    f32 = x.dtype == np.float32
    sos = np.asarray(sos, dtype=np.float32 if f32 else np.float64)
    x = dithered(x) if f32 else np.asarray(x, dtype=np.float64)
    p = min(pad, n - 1)
    if p > 0:
        left = 2 * x[..., :1] - x[..., p:0:-1]
        right = 2 * x[..., -1:] - x[..., -2: -p - 2: -1]
        x = np.concatenate([left, x, right], axis=-1)
    y = signal.sosfilt(sos, x, axis=-1)
    y = signal.sosfilt(sos, y[..., ::-1], axis=-1)[..., ::-1]
    return y[..., p: p + n]                  # a (reversed-stride) view: callers only do elementwise math


def response_db(sos: np.ndarray, freqs, sr: float) -> np.ndarray:
    """Magnitude response in dB at ``freqs`` (Hz)."""
    _, h = signal.sosfreqz(sos, worN=np.asarray(freqs, dtype=np.float64), fs=sr)
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-12))
