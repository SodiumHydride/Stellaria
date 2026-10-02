"""Following the original song: balance, level riding, spectral match, reverb estimate."""

from __future__ import annotations

import math

import numpy as np
from scipy import signal

from . import activity, loudness
from . import dynamics as dyn

THIRD_OCTAVES = 1000.0 * 2.0 ** (np.arange(-10, 13) / 3.0)      # 100 Hz … 16 kHz
MATCH_MAX_DB = 4.0
RIDE_MAX_DB = 4.0


def align(ref: np.ndarray, ref_offset: float, out_offset: float, n: int, sr: float) -> np.ndarray:
    """``ref`` (channels, m) starting at ``ref_offset`` on the song timeline → the ``n`` samples
    that start at ``out_offset`` (zero-padded where the reference does not reach)."""
    ref = ref[None, :] if ref.ndim == 1 else ref
    out = np.zeros((ref.shape[0], n), dtype=np.float32)
    shift = int(round((out_offset - ref_offset) * sr))       # index in ref of the output's first sample
    src0, dst0 = max(shift, 0), max(-shift, 0)
    m = min(ref.shape[-1] - src0, n - dst0)
    if m > 0:
        out[:, dst0: dst0 + m] = ref[:, src0: src0 + m]
    return out


def balance_db(vocal: np.ndarray, backing: np.ndarray, sr: float) -> float | None:
    """Integrated loudness of ``vocal`` minus that of ``backing`` (LU), ``None`` if either is silent."""
    lv = loudness.integrated_loudness(vocal, sr)
    lb = loudness.integrated_loudness(backing, sr)
    if not (math.isfinite(lv) and math.isfinite(lb)):
        return None
    return lv - lb


# ----------------------------------------------------------------------------- spectrum

def band_levels(x: np.ndarray, sr: float, seg: int = 8192, max_segments: int = 300) -> np.ndarray:
    """Long-term level (dB) of each 1/3-octave band 100 Hz–16 kHz: mean Hann-windowed power
    spectrum (mono sum) of up to ``max_segments`` evenly spread, non-silent 8192-sample segments."""
    mono = x.mean(axis=0) if x.ndim == 2 else x
    mono = np.asarray(mono, dtype=np.float32)
    k = mono.size // seg
    if k == 0:
        mono = np.pad(mono, (0, seg - mono.size))
        k = 1
    segs = mono[: k * seg].reshape(k, seg)
    energy = np.einsum("ij,ij->i", segs, segs)
    loud = np.flatnonzero(energy > energy.max() * 1e-5) if energy.max() > 0 else np.arange(k)
    pick = loud[:: max(1, -(-loud.size // max_segments))]
    spec = np.abs(np.fft.rfft(segs[pick] * np.hanning(seg).astype(np.float32), axis=-1)) ** 2
    p = spec.mean(axis=0)
    f = np.fft.rfftfreq(seg, 1.0 / sr)
    out = np.empty(THIRD_OCTAVES.size)
    for i, fc in enumerate(THIRD_OCTAVES):
        sel = (f >= fc / 2 ** (1 / 6)) & (f < fc * 2 ** (1 / 6))
        out[i] = 10.0 * math.log10(float(p[sel].sum()) + 1e-20)
    return out


def match_curve(ours_db: np.ndarray, target_db: np.ndarray, amount: float) -> np.ndarray:
    """Per-band correction (dB): shape of ``target - ours`` without its overall level, smoothed
    across neighbouring bands, clamped to ±4 dB and scaled by ``amount``."""
    diff = target_db - ours_db
    core = (THIRD_OCTAVES >= 200) & (THIRD_OCTAVES <= 5000)
    diff = diff - float(np.median(diff[core]))
    diff = np.convolve(np.pad(diff, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    return np.clip(diff, -MATCH_MAX_DB, MATCH_MAX_DB) * float(np.clip(amount, 0.0, 1.0))


def match_fir(curve_db: np.ndarray, sr: float, taps: int = 2049) -> np.ndarray:
    """Linear-phase FIR with the per-band ``curve_db`` (interpolated on a log-frequency axis)."""
    nyq = sr / 2.0
    grid = np.linspace(0.0, nyq, 1025)
    logf = np.log2(np.maximum(grid, 1.0))
    gain_db = np.interp(logf, np.log2(THIRD_OCTAVES), curve_db)    # flat beyond the outer bands
    return signal.firwin2(taps, grid, 10.0 ** (gain_db / 20.0), fs=sr).astype(np.float32)


def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Zero-delay application of a linear-phase FIR (the group delay is removed)."""
    delay = (h.size - 1) // 2
    y = signal.oaconvolve(x, h if x.ndim == 1 else h[None, :], axes=-1)
    return y[..., delay: delay + x.shape[-1]].astype(np.float32)


# ----------------------------------------------------------------------------- level riding

def ride(vocal: np.ndarray, backing: np.ndarray, ref_vocal_k: np.ndarray, ref_backing_k: np.ndarray, sr: float,
         amount: float, hop_s: float = 0.5) -> tuple[np.ndarray, dict]:
    """Gain (dB per sample, for ``vocal``) that makes the vocal-minus-backing short-term loudness
    (3 s windows) follow the original's around both static levels; clamped to ±4 dB, smoothed,
    0 where either song is not singing, scaled by ``amount``. The reference stems come
    K-weighted (:func:`loudness.k_weight`)."""
    n = vocal.shape[-1]
    v = loudness.short_term(vocal, sr, hop_s=hop_s)
    b = loudness.short_term(backing, sr, hop_s=hop_s)
    vo = loudness.short_term(ref_vocal_k, sr, hop_s=hop_s, weighted=True)
    bo = loudness.short_term(ref_backing_k, sr, hop_s=hop_s, weighted=True)
    frames = min(v.size, b.size, vo.size, bo.size)
    v, b, vo, bo = v[:frames], b[:frames], vo[:frames], bo[:frames]
    finite = np.isfinite(v) & np.isfinite(b) & np.isfinite(vo) & np.isfinite(bo)
    if finite.sum() < 4:
        return np.zeros(n, dtype=np.float32), {"frames": 0}
    top_v = np.percentile(v[finite], 95)
    top_vo = np.percentile(vo[finite], 95)
    active = finite & (v > top_v - 20.0) & (vo > top_vo - 25.0) & (b > -70.0) & (bo > -70.0)
    if active.sum() < 4:
        return np.zeros(n, dtype=np.float32), {"frames": int(active.sum())}
    d_ours = v - b
    d_orig = vo - bo
    dev = (d_orig - np.median(d_orig[active])) - (d_ours - np.median(d_ours[active]))
    dev = np.where(active, np.clip(dev, -RIDE_MAX_DB, RIDE_MAX_DB), 0.0)
    dev = np.convolve(np.pad(dev, 2, mode="edge"), np.ones(5) / 5.0, mode="valid") * float(np.clip(amount, 0, 1))
    gain_db = dyn.uniform_interp(dev, hop_s * sr, n)
    return gain_db, {"frames": int(active.sum()), "maxDb": float(np.max(np.abs(dev)))}


# ----------------------------------------------------------------------------- reverb estimate

# Tail-to-phrase ratio (dB) of a vocal through the mixer's own space at reverb amount r is about
# A(decay) + 20*log10(r); A was measured on synthetic vocals for the presets' decay times (see
# tests/test_mixing_v2.py::test_reverb_estimate_round_trip).
TAIL_INTERCEPTS = ((0.7, -35.0), (1.6, -20.3), (2.8, -14.3))
DRY_TAIL_DB = -55.0


def tail_ratio_db(vocal: np.ndarray, sr: float) -> float | None:
    """Median over phrase ends (followed by ≥ 0.5 s of gap) of the energy 80–400 ms after the end
    relative to the last 300 ms of the phrase, in the 300 Hz–5 kHz band; ``None`` without ends."""
    mono = vocal.mean(axis=0) if vocal.ndim == 2 else vocal
    sos = signal.butter(4, [300.0, 5000.0], "bandpass", fs=sr, output="sos")
    band = signal.sosfilt(sos, mono.astype(np.float64))
    act = activity.detect(mono, sr, below_median_db=12.0, min_gap_s=0.15)
    phrases = act.phrases()
    ratios = []
    for i, (start, end) in enumerate(phrases):
        nxt = phrases[i + 1][0] if i + 1 < len(phrases) else mono.size / sr
        if nxt - end < 0.5 or end - start < 0.5:
            continue
        e = int(end * sr)
        pre = band[max(0, e - int(0.3 * sr)): e]
        post = band[e + int(0.08 * sr): e + int(0.4 * sr)]
        if pre.size < 100 or post.size < 100:
            continue
        ratios.append(10.0 * math.log10((np.mean(post ** 2) + 1e-20) / (np.mean(pre ** 2) + 1e-20)))
    return float(np.median(ratios)) if ratios else None


def reverb_for_tail(ratio_db: float, decay_s: float) -> float:
    """Reverb amount (0..1, the mixer's ``reverb`` scale) that gives a ``ratio_db`` tail with a
    space of ``decay_s`` seconds."""
    if ratio_db < DRY_TAIL_DB:
        return 0.03
    xs = [math.log10(d) for d, _ in TAIL_INTERCEPTS]
    a = float(np.interp(math.log10(max(decay_s, 0.1)), xs, [v for _, v in TAIL_INTERCEPTS]))
    return float(np.clip(10.0 ** ((ratio_db - a) / 20.0), 0.03, 0.8))
