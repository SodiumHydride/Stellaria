"""Master: glue compressor → soft clipper → 4× true-peak look-ahead limiter → loudness target."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import dynamics as dyn
from . import loudness

LogFn = Callable[[str], None]


def _db(x: float) -> float:
    return 20.0 * math.log10(max(float(x), 1e-12))


def _gain(db_value: float) -> float:
    return 10.0 ** (float(db_value) / 20.0)


@dataclass
class MasterSpec:
    target_lufs: float = -14.0
    ceiling_db: float = -1.0            # dBTP (4× oversampled estimate)
    limiter_release_s: float = 0.08
    glue: bool = True
    glue_ratio: float = 2.0
    glue_attack_s: float = 0.03
    glue_release_s: float = 0.3
    glue_knee_db: float = 6.0
    clip_db: float = 3.6                # 0 disables the clipper
    clip_softness: float = 0.2


@dataclass
class MasterResult:
    audio: np.ndarray
    lufs: float
    peak_db: float                      # sample peak
    true_peak_db: float                 # 4× oversampled estimate


def glue(x: np.ndarray, sr: float, m: MasterSpec) -> tuple[np.ndarray, float]:
    """Slow 2:1 bus compressor (30 ms RMS) with its threshold at the 60th percentile of the loud
    parts: holds the mix together by 1–2 dB on its louder half."""
    det = dyn.detector_db(x, sr, rms_s=0.03, block=64)
    active = dyn.active_mask(det)
    if not active.any():
        return x, 0.0
    threshold = float(np.percentile(det[active], 60))
    y, gr = dyn.compress(x, sr, threshold_db=threshold, ratio=m.glue_ratio, knee_db=m.glue_knee_db,
                         attack_s=m.glue_attack_s, release_s=m.glue_release_s, rms_s=0.03, block=64)
    return y, float(np.mean(gr[active[: gr.size]])) if gr.size else 0.0


def master(x: np.ndarray, sr: float, m: MasterSpec, log: LogFn = lambda _m: None,
           progress: Callable[[float], None] = lambda _p: None) -> MasterResult:
    """Returns the mastered audio with its loudness and peaks.

    The mix is set to the target loudness and glued; the clipper then takes up to ``clip_db`` (or
    half of the peak reduction still needed, whichever is less) off the highest peaks. The
    limiter's drive is searched with the secant method on a K-weighted copy (K-weighting commutes
    with the slowly varying limiter gain closely enough), the result is measured exactly, and a
    final trim never pushes above the ceiling (if it would need headroom that is not there, the
    limiter runs again with more drive)."""
    l0 = loudness.integrated_loudness(x, sr)
    if not math.isfinite(l0):
        log("master: silent mix")
        return MasterResult(np.zeros_like(x), float("-inf"), -120.0, -120.0)
    x = (x * _gain(m.target_lufs - l0)).astype(np.float32)
    glue_db = 0.0
    if m.glue:
        x, glue_db = glue(x, sr, m)
        x *= np.float32(_gain(glue_db))      # roughly back to the target; the drive search does the rest
    peak = loudness.sample_peak_db(x)
    clip = float(np.clip(0.5 * (peak - m.ceiling_db), 0.0, m.clip_db))
    if clip > 0.05:
        x = dyn.soft_clip(x, _gain(peak - clip), m.clip_softness)
    else:
        clip = 0.0
    env = loudness.peak_envelope(x, oversample=4)
    xk = loudness.k_weight(x, sr)
    progress(0.2)

    block = 8
    n = x.shape[-1]
    lim = dict(ceiling_db=m.ceiling_db, lookahead_s=0.0015, release_s=m.limiter_release_s, block=block)

    def gains(drive: float) -> np.ndarray:
        gr = dyn.limiter_gain(env, sr, drive_db=drive, **lim)
        return dyn.upsample(dyn.db_to_gain(drive - gr), block, n)

    def estimate(drive: float) -> float:
        return loudness.gated_loudness(loudness.block_energies(xk * gains(drive), sr))

    target = m.target_lufs + 0.05
    d0, f0 = 0.0, estimate(0.0) - target
    d1, f1 = d0, f0
    passes = 1
    if f0 < -0.1:
        d1 = -f0
        f1 = estimate(d1) - target
        passes += 1
        while abs(f1) > 0.08 and passes < 7:
            slope = float(np.clip((f1 - f0) / (d1 - d0), 0.1, 1.5)) if d1 != d0 else 1.0
            d0, f0 = d1, f1
            d1 = float(np.clip(d1 - f1 / slope, -12.0, 36.0))
            f1 = estimate(d1) - target
            passes += 1
            progress(0.2 + 0.1 * passes)
    xk = None                                                        # free it before the exact passes

    for attempt in range(3):
        g = gains(d1)
        y = (x * g).astype(np.float32)
        lufs = loudness.integrated_loudness(y, sr)
        headroom = m.ceiling_db - _db(float(np.max(env * g)))       # 4x-oversampled peak of y
        short = m.target_lufs - lufs
        if short <= headroom + 0.02 or d1 >= 36.0 or attempt == 2:
            break
        d1 = min(36.0, d1 + (short - max(headroom, 0.0)) + 0.02)     # needs more limiting
        passes += 1
    progress(0.9)
    trim = min(m.target_lufs - lufs, max(headroom, 0.0))
    y *= np.float32(_gain(trim))
    lufs += trim
    # y = x·g with a smooth g, so its 4x-oversampled peaks are env·g (no second interpolation pass)
    true_peak = m.ceiling_db - headroom + trim
    if true_peak > m.ceiling_db:                                       # numerical safety net
        fix = m.ceiling_db - true_peak - 0.001
        y *= np.float32(_gain(fix))
        lufs += fix
        true_peak += fix
    peak = loudness.sample_peak_db(y)
    log(f"master: glue avg -{glue_db:.1f} dB, clip {clip:.1f} dB, limiter drive {d1:+.1f} dB ({passes} passes), "
        f"trim {trim:+.2f} dB → {lufs:.2f} LUFS, {true_peak:.2f} dBTP")
    return MasterResult(y, lufs, peak, true_peak)


def normalize(x: np.ndarray, sr: float, target_lufs: float, ceiling_db: float = -1.0,
              log: LogFn = lambda _m: None) -> MasterResult:
    """Gain to ``target_lufs``; a true-peak limiter only where the gain would cross the ceiling."""
    return master(x, sr, MasterSpec(target_lufs=target_lufs, ceiling_db=ceiling_db, glue=False, clip_db=0.0), log)
