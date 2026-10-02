"""Taming harsh resonances ("去刺耳"): a spectral dynamic EQ that only cuts.

Each short-time spectrum is smoothed twice on a log-frequency scale: finely (a sixth of an octave,
wide enough to merge neighbouring harmonics, so a sung note's partials are not mistaken for
resonances) and coarsely (an octave: the spectrum's overall shape). Where the fine one bulges above
the coarse one — the whistling, metallic or nasal resonances synthetic voices often have — the
spectrum is pulled down in proportion to how far it sticks out,
inside a band (2–10 kHz by default) and never by more than ``max_db``. Gain reductions move with
a fast attack and a slower release, are smoothed across neighbouring bins, and only ever cut, so
quiet parts and the overall tone stay. The same idea as the resonance suppressors mixers use, from
the published principle (adaptive spectral dynamic EQ), not from any product's code.

Overlapping frames (75 %, sqrt-Hann analysis and synthesis) are processed in chunks, so a whole
song never needs its full spectrogram in memory. Stereo is linked: both channels get the cut found
on their summed power, so the image never wanders.
"""

from __future__ import annotations

import math

import numpy as np

CHUNK_FRAMES = 384


def _fft_size(sr: float) -> int:
    return int(2 ** round(math.log2(max(256.0, 0.046 * sr))))  # ≈ 46 ms (2048 at 44.1/48 kHz)


def _gaussian(points: np.ndarray, octaves: float) -> np.ndarray:
    """Row-normalised Gaussian weights between log-frequency points (FWHM ``octaves``)."""
    sigma = octaves / 2.355
    w = np.exp(-0.5 * ((points[:, None] - points[None, :]) / sigma) ** 2)
    return (w / w.sum(axis=1, keepdims=True)).astype(np.float32)


def _log_grid(freqs: np.ndarray, lo: float, hi: float, per_octave: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A log-frequency grid over [lo, hi]: its points (in octaves), the bins → grid averaging
    matrix and the grid → bins interpolation matrix (both dense, small)."""
    octs = np.arange(math.log2(lo), math.log2(hi) + 1e-9, 1.0 / per_octave)
    logf = np.log2(np.maximum(freqs, 1.0))
    # bins → grid: triangular weights one grid step wide (at least the nearest bin)
    dist = np.abs(octs[:, None] - logf[None, :]) * per_octave
    down = np.clip(1.0 - dist, 0.0, None)
    nearest = np.argmin(dist, axis=1)
    down[np.arange(octs.size), nearest] = np.maximum(down[np.arange(octs.size), nearest], 1.0)
    down /= down.sum(axis=1, keepdims=True)
    # grid → bins: linear interpolation in log frequency
    up = np.zeros((freqs.size, octs.size))
    pos = np.clip((logf - octs[0]) * per_octave, 0.0, octs.size - 1.0)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, octs.size - 1)
    frac = pos - i0
    up[np.arange(freqs.size), i0] += 1.0 - frac
    up[np.arange(freqs.size), i1] += frac
    return octs, down.astype(np.float32), up.astype(np.float32)


def smooth_resonances(x: np.ndarray, sr: float, *, amount: float = 0.5, low_hz: float = 2000.0,
                      high_hz: float = 10000.0, max_db: float = 6.0, attack_s: float = 0.005,
                      release_s: float = 0.08) -> tuple[np.ndarray, dict]:
    """``x`` (n,) or (channels, n) → (same shape float32, ``{"maxReduction", "avgReduction"}``).

    ``amount`` 0..1 scales how hard sticking-out peaks are pulled toward the envelope (0 = off)."""
    x = np.asarray(x, dtype=np.float32)
    if amount <= 0 or max_db <= 0 or x.shape[-1] == 0:
        return x.copy(), {"maxReduction": 0.0, "avgReduction": 0.0}
    mono = x.ndim == 1
    chans = x[None, :] if mono else x
    n = chans.shape[-1]
    n_fft = _fft_size(sr)
    hop = n_fft // 4
    win = np.sqrt(np.hanning(n_fft + 1)[:-1]).astype(np.float32)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    nyq = sr / 2.0
    lo, hi = max(20.0, min(low_hz, nyq * 0.9)), max(40.0, min(high_hz, nyq * 0.98))
    if hi <= lo:
        return x.copy(), {"maxReduction": 0.0, "avgReduction": 0.0}
    # work on the band plus half an octave either side (the envelope needs context)
    band = np.flatnonzero((freqs >= lo / 2.0) & (freqs <= min(hi * 2.0, nyq)))
    fb = freqs[band]
    # the spectrum is judged on a log-frequency grid (24 points per octave): much cheaper than
    # bin by bin, and the scale resonances live on
    octs, down, up = _log_grid(fb, float(fb[0]), float(fb[-1]))
    fine = _gaussian(octs, 1.0 / 6.0)
    coarse = _gaussian(octs, 1.0)
    # full strength inside [lo, hi], fading out over the half octave around it
    taper = np.clip(np.minimum(octs - math.log2(lo / 1.41), math.log2(hi * 1.41) - octs) / 0.5, 0.0, 1.0).astype(np.float32)
    frame_s = hop / sr
    att = math.exp(-frame_s / max(attack_s, 1e-4))
    rel = math.exp(-frame_s / max(release_s, 1e-4))
    strength = 1.5 * float(amount)
    threshold_db = 2.0
    # sqrt-Hann in and out = Hann; periodic Hann at 75 % overlap sums to 2
    norm = np.float32(0.5)

    from scipy import fft as sfft

    b0, b1 = int(band[0]), int(band[-1]) + 1
    up_t = np.ascontiguousarray(up.T)
    pad = np.pad(chans, ((0, 0), (n_fft, n_fft + hop)))
    frames = 1 + (pad.shape[-1] - n_fft) // hop
    out = np.zeros_like(pad)
    stats_max = 0.0
    stats_sum = 0.0
    stats_count = 0
    windows = [np.lib.stride_tricks.sliding_window_view(sig, n_fft)[::hop] for sig in pad]
    state = np.zeros(octs.size, dtype=np.float32)
    for f0 in range(0, frames, CHUNK_FRAMES):
        f1 = min(frames, f0 + CHUNK_FRAMES)
        specs = [sfft.rfft(w[f0:f1] * win, axis=-1) for w in windows]          # complex64
        # stereo-linked: judged on the channels' summed power, the same cut on each (no image shift)
        power = sum(sp[:, b0:b1].real ** 2 + sp[:, b0:b1].imag ** 2 for sp in specs) / len(specs)
        grid = power @ down.T
        detail = 10.0 * np.log10(grid @ fine.T + 1e-12)
        env = 10.0 * np.log10(grid @ coarse.T + 1e-12)
        loud = detail.max(axis=1, keepdims=True) > -80.0
        target = np.clip((detail - env - threshold_db) * strength, 0.0, max_db) * taper * loud
        gr = np.empty_like(target)
        for k in range(target.shape[0]):  # attack / release per grid point
            coef = np.where(target[k] > state, att, rel)
            state = target[k] + coef * (state - target[k])
            gr[k] = state
        # gains on the grid, interpolated to the bins (linear gain: cheaper than per-bin powers)
        gain = np.power(np.float32(10.0), gr * np.float32(-0.05)) @ up_t
        stats_max = max(stats_max, float(gr.max(initial=0.0)))
        stats_sum += float(gr.sum())
        stats_count += gr.size
        for c, sp in enumerate(specs):
            sp[:, b0:b1] *= gain
            y = sfft.irfft(sp, n=n_fft, axis=-1) * (win * norm)
            # overlap-add: every 4th frame tiles without overlap, so each phase is one long add
            for phase in range(min(4, y.shape[0])):
                tiles = y[phase::4]
                start = (f0 + phase) * hop
                out[c, start:start + tiles.size] += tiles.ravel()
    y = out[:, n_fft:n_fft + n]
    info = {"maxReduction": round(stats_max, 2), "avgReduction": round(stats_sum / max(1, stats_count), 3)}
    return (y[0] if mono else y).astype(np.float32), info
