"""Synthesized stereo reverb impulse responses and the vocal "space" bus.

An IR is early reflections (sparse taps, independent for L and R) plus a late tail of
exponentially decaying noise, split into three bands with their own decay times (lows ring a bit
longer, highs are damped), with a build-up ramp. L and R use independent noise, so the tail is
decorrelated. Everything linear on the reverb return — the type blend, pre-delay, return EQ,
stereo width and the optional tempo-synced delay — is folded into one stereo IR, so the whole
space costs one FFT convolution of the mono send.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import signal

from . import filters

IR_KINDS = ("room", "plate", "chamber", "hall")


@dataclass(frozen=True)
class IrShape:
    decay_scale: float      # RT60 relative to the space's decay time
    build_s: float          # tail build-up time constant
    er_start_s: float
    er_end_s: float
    er_count: int
    er_energy: float        # share of the IR's energy in the early reflections
    low_mult: float         # RT60 multiplier below ~500 Hz
    high_mult: float        # RT60 multiplier above ~4 kHz
    lowpass_hz: float       # brightness of the whole IR
    seed: int


SHAPES: dict[str, IrShape] = {
    "room": IrShape(0.55, 0.004, 0.003, 0.035, 22, 0.40, 1.05, 0.55, 9000, 11),
    "plate": IrShape(1.00, 0.0015, 0.001, 0.008, 6, 0.06, 0.90, 0.85, 14000, 23),
    "chamber": IrShape(0.80, 0.008, 0.006, 0.050, 16, 0.25, 1.10, 0.65, 10000, 37),
    "hall": IrShape(1.25, 0.025, 0.012, 0.090, 26, 0.18, 1.30, 0.55, 8500, 41),
}


def _band_split(noise: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    low = filters.apply(filters.lowpass4(500.0, sr), noise)
    high = filters.apply(filters.highpass4(4000.0, sr), noise)
    return low, noise - low - high, high


def synth_ir(kind: str, rt60: float, sr: float) -> np.ndarray:
    """Stereo IR ``(2, n)`` float32 with unit energy (mean over channels) for ``kind``."""
    shape = SHAPES[kind]
    rt = max(0.15, rt60 * shape.decay_scale)
    length = int(min(8.0, rt * 1.15 + shape.er_end_s) * sr)
    rng = np.random.default_rng(shape.seed)
    t = np.arange(length) / sr

    # late tail: three bands, each with its own exponential decay (-60 dB at its RT60)
    noise = rng.standard_normal((2, length))
    low, mid, high = _band_split(noise, sr)
    env = lambda mult: np.exp(-6.907755 * t / (rt * mult))  # noqa: E731
    tail = low * env(shape.low_mult) + mid * env(1.0) + high * env(shape.high_mult)
    onset = shape.er_start_s
    ramp = np.clip((t - onset) / max(shape.build_s, 1e-4), 0.0, None)
    tail *= (1.0 - np.exp(-ramp)) * (t >= onset)

    # early reflections: sparse taps with decaying amplitudes, independent per channel
    er = np.zeros((2, length))
    if shape.er_count:
        for ch in range(2):
            times = np.sort(rng.uniform(shape.er_start_s, shape.er_end_s, shape.er_count))
            idx = np.minimum((times * sr).astype(int), length - 1)
            amp = rng.uniform(0.4, 1.0, shape.er_count) * np.exp(-3.0 * (times - shape.er_start_s) / max(rt, 0.3))
            amp *= rng.choice([-1.0, 1.0], shape.er_count)
            np.add.at(er[ch], idx, amp)
        er = filters.apply(filters.lowpass(shape.lowpass_hz * 0.8, sr), er)

    tail_e = float(np.mean(np.sum(tail ** 2, axis=-1))) or 1.0
    er_e = float(np.mean(np.sum(er ** 2, axis=-1)))
    ir = tail * math.sqrt((1.0 - shape.er_energy) / tail_e)
    if er_e > 0:
        ir += er * math.sqrt(shape.er_energy / er_e)
    ir = filters.apply(filters.lowpass(shape.lowpass_hz, sr), ir)
    return normalize_energy(ir).astype(np.float32)


def normalize_energy(ir: np.ndarray) -> np.ndarray:
    e = float(np.mean(np.sum(np.square(ir, dtype=np.float64), axis=-1)))
    return ir / math.sqrt(e) if e > 0 else ir


def set_width(ir: np.ndarray, width: float) -> np.ndarray:
    """Mid/side width of a stereo IR: 0 = mono, 1 = unchanged."""
    mid = (ir[0] + ir[1]) * 0.5
    side = (ir[0] - ir[1]) * 0.5 * float(np.clip(width, 0.0, 1.0))
    return np.stack([mid + side, mid - side])


@dataclass
class DelaySpec:
    time_s: float
    feedback: float = 0.35
    level_db: float = -12.0
    repeats: int = 5
    highpass_hz: float = 300.0
    lowpass_hz: float = 4500.0
    ping_pong: bool = True


def delay_ir(spec: DelaySpec, sr: float) -> np.ndarray:
    """Filtered ping-pong delay as a stereo IR: each repeat runs through the filters once more."""
    step = int(round(spec.time_s * sr))
    if step <= 0 or spec.repeats <= 0:
        return np.zeros((2, 1), dtype=np.float32)
    flt = filters.cascade(filters.highpass(spec.highpass_hz, sr), filters.lowpass(spec.lowpass_hz, sr))
    tap_len = min(step, int(0.05 * sr))
    pulse = np.zeros(tap_len)
    pulse[0] = 1.0
    out = np.zeros((2, step * spec.repeats + tap_len))
    level = 10.0 ** (spec.level_db / 20.0)
    for k in range(1, spec.repeats + 1):
        pulse = signal.sosfilt(flt, pulse)
        start = k * step
        side = (0.8 if k % 2 else 0.2) if spec.ping_pong else 0.5    # ping-pong: left, right, left, ...
        gain = level * spec.feedback ** (k - 1)
        out[0, start: start + tap_len] += pulse * gain * math.sqrt(side)
        out[1, start: start + tap_len] += pulse * gain * math.sqrt(1.0 - side)
    return out.astype(np.float32)


@dataclass
class SpaceSpec:
    weights: dict[str, float]
    decay_s: float
    predelay_s: float
    width: float
    return_highpass_hz: float = 200.0
    return_lowpass_hz: float = 8000.0
    bus_cut_db: float = -4.0          # bell at 1.45 kHz
    bus_cut_hz: float = 1450.0
    bus_boost_db: float = 5.0         # wide bell at 2.3 kHz
    bus_boost_hz: float = 2300.0
    delay: DelaySpec | None = None


def space_ir(spec: SpaceSpec, sr: float) -> np.ndarray:
    """The whole vocal space as one stereo IR ``(2, n)`` (unit-energy reverb + delay taps)."""
    parts = []
    for kind in IR_KINDS:
        w = float(spec.weights.get(kind, 0.0))
        if w > 0:
            parts.append(math.sqrt(w) * synth_ir(kind, spec.decay_s, sr))
    if not parts:
        reverb = np.zeros((2, 1), dtype=np.float32)
    else:
        length = max(p.shape[-1] for p in parts)
        reverb = np.zeros((2, length))
        for p in parts:
            reverb[:, : p.shape[-1]] += p
        eq = filters.cascade(
            filters.highpass(spec.return_highpass_hz, sr),
            filters.lowpass(spec.return_lowpass_hz, sr),
            filters.peaking(spec.bus_cut_hz, spec.bus_cut_db, 1.0, sr),
            filters.peaking(spec.bus_boost_hz, spec.bus_boost_db, 0.5, sr),
        )
        reverb = normalize_energy(set_width(filters.apply(eq, reverb), spec.width))
        pre = int(round(max(spec.predelay_s, 0.0) * sr))
        reverb = np.concatenate([np.zeros((2, pre)), reverb], axis=-1)
    if spec.delay is not None:
        d = delay_ir(spec.delay, sr)
        length = max(reverb.shape[-1], d.shape[-1])
        out = np.zeros((2, length))
        out[:, : reverb.shape[-1]] += reverb
        out[:, : d.shape[-1]] += d
        reverb = out
    return reverb.astype(np.float32)


HALF_RATE_MAX_HZ = 9500.0        # returns low-passed at or below this run at half the sample rate


def convolve(send: np.ndarray, ir: np.ndarray, n: int | None = None, *, sr: float | None = None,
             cutoff_hz: float | None = None,
             post: Callable[[np.ndarray, float], np.ndarray] | None = None) -> np.ndarray:
    """Mono ``send`` (n,) through a stereo IR (2, m) → ``(2, n)`` float32 (FFT convolution,
    overlap-add: cheaper than one huge FFT when the song is much longer than the IR).

    With ``sr`` and a ``cutoff_hz`` (the IR's low-pass) at or below 9.5 kHz the convolution runs at
    half the sample rate (``resample_poly`` down and up again): nothing above ~10 kHz is lost
    that the IR would not have removed anyway, and it is about twice as fast. ``post(wet, rate)``
    turns the wet signal into the stereo result at the rate it was made at (``ir`` may then have
    more rows than two: one pass of the send serves them all)."""
    n = send.shape[-1] if n is None else n
    if send.size == 0 or not np.any(ir):
        return np.zeros((2, n), dtype=np.float32)
    from scipy import fft

    x = np.asarray(send, dtype=np.float32)[None, :]
    h = np.asarray(ir, dtype=np.float32)
    h = h[:, : int(np.flatnonzero(np.abs(h).max(axis=0))[-1]) + 1]    # trailing silence costs time
    half = sr is not None and cutoff_hz is not None and cutoff_hz <= HALF_RATE_MAX_HZ and sr >= 40000
    if half:
        x = signal.resample_poly(x, 1, 2, axis=-1).astype(np.float32)
        h = (signal.resample_poly(h, 1, 2, axis=-1) * 2.0).astype(np.float32)   # each tap now covers 2 samples
    with fft.set_workers(-1):
        wet = signal.oaconvolve(x, h, axes=-1)
    if post is not None:
        wet = post(wet, (sr or 0.0) / 2.0 if half else (sr or 0.0))
    if half:
        wet = signal.resample_poly(wet, 2, 1, axis=-1)
    out = np.zeros((2, n), dtype=np.float32)
    m = min(n, wet.shape[-1])
    out[:, :m] = wet[:, :m]
    return out


# ---- modulation -----------------------------------------------------------------------------
EARLY_S = 0.08          # after the first reflection: kept clean
MOD_MAX_MS = 0.9        # delay swing at modulation 1 (≈ ±6 cents of drift)


def modulate(wet: np.ndarray, sr: float, depth_ms: float, seed: int = 7) -> np.ndarray:
    """A slowly varying delay on each channel (two incommensurate LFOs, different per channel):
    the gentle pitch drift that keeps an algorithmic reverb's tail alive and stops a static tail
    from sounding grainy on long notes. ``(c, n)`` → same shape.

    Read with 4-point Lagrange interpolation (much less treble loss than linear), in blocks that
    stay in cache."""
    if depth_ms <= 0 or wet.shape[-1] == 0:
        return wet
    n = wet.shape[-1]
    depth = depth_ms / 1000.0 * sr
    rng = np.random.default_rng(seed)
    rates = ((0.43, 0.97), (0.51, 1.13), (0.47, 1.05), (0.55, 1.21))
    step = 64                                            # LFOs are smooth: compute them sparsely
    block = 512 * step
    tt = np.arange(n // step + 2) * (step / sr)
    ramp = (np.arange(step, dtype=np.float32) / step)[None, :]
    base = np.arange(block, dtype=np.int32)
    pad = int(math.ceil(2.0 * depth)) + 4
    third, half = np.float32(1.0 / 3.0), np.float32(0.5)
    out = np.empty(wet.shape, dtype=np.float32)
    for c in range(wet.shape[0]):
        p1, p2 = rng.uniform(0.0, 2.0 * math.pi, 2)
        r1, r2 = rates[c % len(rates)]
        coarse = ((np.sin(2.0 * math.pi * r1 * tt + p1) + 0.6 * np.sin(2.0 * math.pi * r2 * tt + p2)) / 1.6).astype(np.float32)
        x = np.concatenate([np.zeros(pad, dtype=np.float32), wet[c].astype(np.float32), np.zeros(2, dtype=np.float32)])
        for s0 in range(0, n, block):
            m = min(block, n - s0)
            seg = coarse[s0 // step: s0 // step + (m + step - 1) // step + 1]
            lfo = (seg[:-1, None] + (seg[1:] - seg[:-1])[:, None] * ramp).ravel()[:m]
            delay = np.float32(depth) * (np.float32(1.0) + lfo) + np.float32(2.0)   # 2 … 2·depth + 2 behind
            whole = np.ceil(delay)
            t = whole - delay                                # read position = i - whole + t, t in [0, 1)
            i0 = base[:m] + np.int32(s0 + pad) - whole.astype(np.int32)
            tm1, tp1, tm2 = t - np.float32(1.0), t + np.float32(1.0), t - np.float32(2.0)
            # Lagrange weights for the taps at -1, 0, +1, +2
            a = tm1 * tm2
            b = tp1 * t
            out[c, s0:s0 + m] = (-(t * a) * (half * third) * x[i0 - 1] + (tp1 * a) * half * x[i0]
                                 - (b * tm2) * half * x[i0 + 1] + (b * tm1) * (half * third) * x[i0 + 2])
    return out


def split_early(ir: np.ndarray, sr: float, early_s: float = EARLY_S, fade_s: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """``ir`` = early + late, crossfaded ``fade_s`` around ``early_s`` after its first sound."""
    energy = np.abs(ir).max(axis=0)
    first = int(np.argmax(energy > energy.max() * 1e-3)) if energy.any() else 0
    cut = first + int(early_s * sr)
    fade = max(1, int(fade_s * sr))
    ramp = np.zeros(ir.shape[-1], dtype=np.float32)
    if cut < ir.shape[-1]:
        a = max(0, cut - fade)
        ramp[a:cut] = np.linspace(0.0, 1.0, cut - a, endpoint=False)
        ramp[cut:] = 1.0
    late = (ir * ramp).astype(np.float32)
    return (ir - late).astype(np.float32), late


def convolve_modulated(send: np.ndarray, ir: np.ndarray, n: int, *, sr: float, cutoff_hz: float | None,
                       modulation: float) -> np.ndarray:
    """``convolve`` with the tail modulated by ``modulation`` (0..1); 0 is exactly ``convolve``."""
    if modulation <= 0:
        return convolve(send, ir, n, sr=sr, cutoff_hz=cutoff_hz)
    early, late = split_early(ir, sr)
    depth = MOD_MAX_MS * float(modulation)
    # early and late through one convolution pass (the send is transformed once), then the late
    # part's two channels get the drift
    both = np.concatenate([early, late], axis=0)
    return convolve(send, both, n, sr=sr, cutoff_hz=cutoff_hz,
                    post=lambda wet, rate: wet[:2] + modulate(wet[2:], rate, depth))
