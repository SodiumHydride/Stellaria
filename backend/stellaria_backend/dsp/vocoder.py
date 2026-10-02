"""Built-in pitch shifter / time stretcher (the pitch op's fallback engine).

Pitch shifting is a time stretch by ``r = α / tempo`` (``α = 2^(semitones/12)``) followed by
resampling by ``α`` (soxr), so the output is exactly ``round(n / tempo)`` samples long and starts
where the input starts (no latency).

The stretch is a phase vocoder (periodic Hann, analysis frames centred at ``round(m·Hs/r)``,
synthesis frames at ``m·Hs``, weighted overlap-add):

* **Instantaneous frequency** of each bin from the phase advance between consecutive analysis
  frames (their actual distance, which varies by a sample because the centres are rounded).
* **Identity phase locking** (Laroche & Dolson, "Improved phase vocoder time-scale modification
  of audio", 1999): spectral peaks (local maxima over ±2 bins) have their phase propagated with
  their instantaneous frequency; every other bin keeps its analysis phase offset to its nearest
  peak. Each partial stays vertically coherent, which removes most of the "phasiness" of a plain
  phase vocoder.
* **Stereo**: the phase propagation runs on the mid (sum) spectrum and all channels are rotated by
  the same per-bin angle, so inter-channel phase differences — the stereo image — are preserved.
* **Transient preservation**: onsets are detected up front (high-frequency log-spectral flux with
  an adaptive threshold, refined to ~1.5 ms on the high-passed envelope). Frames centred before an
  onset have the attack gated out of their analysis window, so no frame smears it backwards (no
  pre-echo); the first frame at the onset resets the phase of every peak whose magnitude rose by
  more than 6 dB to its analysis phase, re-creating the attack's vertical phase structure, while
  steady partials (pads, bass, held notes) keep their propagated phase and do not click. The
  overlap-add is normalised by the actual (gated) window sum.
* **Formants** (optional): the cepstrally smoothed log envelope ``E`` of each analysis frame gives a
  per-bin gain ``E(α·f) / E(f)`` (clamped to ±18 dB) applied before resampling, so the resampled
  output keeps the input's spectral envelope.

Pure numpy/scipy (+ soxr), float32, processed in chunks of frames so memory stays bounded.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import numpy as np

ProgressFn = Callable[[float], None]

PRESETS = {"high": (4096, 512), "fast": (2048, 512)}   # (FFT size, synthesis hop)
CHUNK_FRAMES = 192
EPS = 1e-9
RESET_RISE = math.log(10 ** (6.0 / 20.0))              # a peak is reset when it rose by > 6 dB
GATE_MARGIN = 96                                       # samples before the onset estimate
GATE_RAMP = 64
FORMANT_MAX_DB = 18.0

# onset detection
ONSET_FFT, ONSET_HOP = 1024, 256
ONSET_MIN_HZ = 150.0
ONSET_MEDIAN_S = 0.6
ONSET_RATIO, ONSET_OFFSET, ONSET_ABS = 1.8, 0.10, 0.22
ONSET_MIN_GAP_S = 0.05

_WORKERS = max(1, min(8, os.cpu_count() or 1))


def _hann(n: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def _hop_for(n_fft: int, hop: int, r: float) -> int:
    """Synthesis hop: the preset's, made smaller for strong compression so the analysis hop
    ``hop / r`` never exceeds a quarter of the window (phase unwrapping stays unambiguous)."""
    h = hop
    while h > 64 and h > n_fft * r / 4.0:
        h //= 2
    return h


def resample_ratio(x: np.ndarray, alpha: float) -> np.ndarray:
    """Resample ``(C, n)`` so content at frequency ``f`` ends up at ``α·f`` (length ``n / α``)."""
    if abs(alpha - 1.0) < 1e-12 or x.shape[-1] == 0:
        return x
    n = x.shape[-1]
    if n < 64:                                    # a handful of samples: linear interpolation will do
        m = max(1, int(round(n / alpha)))
        pos = np.arange(m) * alpha
        return np.stack([np.interp(pos, np.arange(n), ch, right=0.0) for ch in x]).astype(np.float32)
    try:
        import soxr

        y = soxr.resample(np.ascontiguousarray(x.T), 44100.0 * alpha, 44100.0, quality="VHQ")
        return np.ascontiguousarray(y.T, dtype=np.float32)
    except ImportError:  # pragma: no cover - soxr is a runtime requirement
        from fractions import Fraction

        from scipy import signal

        frac = Fraction(1.0 / alpha).limit_denominator(2000)
        return signal.resample_poly(x, frac.numerator, frac.denominator, axis=-1).astype(np.float32)


def pitch_shift(x: np.ndarray, semitones: float, tempo: float = 1.0, *, quality: str = "high",
                preserve_formant: bool = False, sr: float = 44100.0,
                progress: ProgressFn | None = None) -> np.ndarray:
    """Pitch-shift by ``semitones`` and change the tempo by ``tempo`` (2 = twice as fast).

    ``x`` is float32 ``(channels, n)``; returns ``(channels, round(n / tempo))``."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]
    n = x.shape[-1]
    target = int(round(n / tempo))
    alpha = 2.0 ** (semitones / 12.0)
    if n == 0 or target == 0:
        return np.zeros((x.shape[0], target), dtype=np.float32)
    formant = alpha if preserve_formant and abs(alpha - 1.0) > 1e-9 else None
    y = stretch(x, alpha / tempo, quality=quality, formant_alpha=formant, sr=sr, progress=progress)
    y = resample_ratio(y, alpha)
    if y.shape[-1] >= target:
        return np.ascontiguousarray(y[:, :target])
    out = np.zeros((x.shape[0], target), dtype=np.float32)
    out[:, : y.shape[-1]] = y
    return out


def detect_onsets(mono: np.ndarray, sr: float = 44100.0) -> np.ndarray:
    """Onset sample positions (ascending) of percussive/attack events in a mono signal."""
    from scipy import fft as sfft
    from scipy import ndimage

    n = mono.shape[-1]
    if n < ONSET_FFT * 2:
        return np.zeros(0, dtype=np.int64)
    padded = np.concatenate([np.zeros(ONSET_FFT // 2, np.float32), mono.astype(np.float32),
                             np.zeros(ONSET_FFT, np.float32)])
    frames = np.lib.stride_tricks.sliding_window_view(padded, ONSET_FFT)[::ONSET_HOP]
    win = _hann(ONSET_FFT)
    k0 = int(math.ceil(ONSET_MIN_HZ * ONSET_FFT / sr))
    mag = np.empty((frames.shape[0], ONSET_FFT // 2 + 1 - k0), dtype=np.float32)
    step = 4096
    for s in range(0, frames.shape[0], step):        # bounded memory for long songs
        mag[s: s + step] = np.abs(sfft.rfft(frames[s: s + step] * win, axis=-1, workers=_WORKERS)[:, k0:])
    # log magnitude with a floor 50 dB under the loudest bin: noise between partials cannot rise
    logm = np.log(mag + max(float(mag.max()), 1e-9) * 10 ** (-50 / 20))
    # rise against the max of the two previous frames (vibrato does not count as an onset)
    prev = np.maximum(np.concatenate([logm[:1], logm[:-1]]), np.concatenate([logm[:1], logm[:1], logm[:-2]]))
    flux = np.maximum(logm - prev, 0.0).mean(axis=1)
    size = max(3, int(round(ONSET_MEDIAN_S * sr / ONSET_HOP)) | 1)
    med = ndimage.median_filter(flux, size=size, mode="nearest")
    local_max = flux >= ndimage.maximum_filter1d(flux, size=5, mode="nearest")
    picks = np.flatnonzero((flux > ONSET_RATIO * med + ONSET_OFFSET) & (flux > ONSET_ABS) & local_max)
    if picks.size == 0:
        return np.zeros(0, dtype=np.int64)
    # refine: steepest rise of the high-passed envelope (32-sample blocks) in the frame's window
    hp = np.abs(np.diff(mono.astype(np.float32), prepend=np.float32(0.0)))
    block = 32
    nb = n // block
    env = hp[: nb * block].reshape(nb, block).mean(axis=1)
    env = np.convolve(env, np.ones(3, np.float32) / 3, mode="same")
    out: list[int] = []
    min_gap = int(ONSET_MIN_GAP_S * sr)
    for p in picks:
        centre = p * ONSET_HOP
        lo = max(0, (centre - ONSET_FFT // 2 - ONSET_HOP) // block)
        hi = min(nb - 1, (centre + ONSET_FFT // 4) // block)
        if hi - lo < 3:
            continue
        seg = np.log(env[lo: hi + 1] + 1e-7)
        onset = int((lo + 1 + int(np.argmax(np.diff(seg)))) * block)
        if out and onset - out[-1] < min_gap:
            continue
        out.append(onset)
    return np.asarray(out, dtype=np.int64)


def stretch(x: np.ndarray, r: float, *, quality: str = "high", formant_alpha: float | None = None,
            transients: bool = True, sr: float = 44100.0, progress: ProgressFn | None = None) -> np.ndarray:
    """Time-stretch ``(C, n)`` float32 by ``r`` → ``(C, round(n·r))`` (pitch unchanged)."""
    from scipy import fft as sfft

    channels, n = x.shape
    out_len = int(round(n * r))
    if out_len == 0:
        return np.zeros((channels, 0), dtype=np.float32)
    if abs(r - 1.0) < 1e-12 and formant_alpha is None:
        return x.copy()
    n_fft, hop0 = PRESETS.get(quality, PRESETS["high"])
    hop = _hop_for(n_fft, hop0, r)
    overlap = n_fft // hop                       # frames covering each output sample
    bins = n_fft // 2 + 1
    half = n_fft // 2
    win = _hann(n_fft)
    synth_win = (win * np.float32(hop / float(np.sum(win.astype(np.float64) ** 2)))).astype(np.float32)

    m_first = -overlap                           # frames before 0 complete the overlap-add at the start
    m_last = int(math.ceil((out_len + half) / hop)) + 1
    centres = np.round(np.arange(m_first, m_last + 1) * (hop / r)).astype(np.int64)
    pad = n_fft + int(math.ceil(overlap * hop / r))
    xp = np.zeros((channels, n + 2 * pad + n_fft), dtype=np.float32)
    xp[:, pad: pad + n] = x
    windows = np.lib.stride_tricks.sliding_window_view(xp, n_fft, axis=-1)

    # Onsets → per frame: where the next onset cuts its analysis window (in_cut; n_fft = none),
    # where it cuts its output (out_pre), where the previous onset starts in its output (out_post;
    # 0 = none), and the phase-reset frame of each onset. Output cuts use the onset's position as
    # the reset frame places it, so every frame splices at the same output time.
    if transients:
        attacks = detect_onsets(x.mean(axis=0) if channels > 1 else x[0], sr)
    else:
        attacks = np.zeros(0, dtype=np.int64)
    onsets = attacks - GATE_MARGIN
    frame_count = len(centres)
    in_cut = np.full(frame_count, n_fft, dtype=np.int64)
    out_pre = np.full(frame_count, n_fft, dtype=np.int64)
    out_post = np.zeros(frame_count, dtype=np.int64)
    reset_frames = np.zeros(frame_count, dtype=bool)
    reset_shift = np.zeros(frame_count, dtype=np.float64)
    first = np.searchsorted(centres, onsets, side="left")          # reset frame of each onset
    keep = first < frame_count
    first, onsets, attacks = first[keep], onsets[keep], attacks[keep]
    if onsets.size:
        uniq = np.concatenate([[True], np.diff(first) > 0])            # one onset per reset frame
        first, onsets, attacks = first[uniq], onsets[uniq], attacks[uniq]
        reset_frames[first] = True
        # the reset frame would put the attack at m·Hs + (attack - centre); a linear phase on the
        # reset bins moves it to attack·r, where it belongs (later frames keep it there)
        ideal = np.round(attacks * r)
        reset_shift[first] = ideal - ((m_first + first) * hop + (attacks - centres[first]))
        out_pos = ideal - GATE_MARGIN                                   # output gate of each onset
        frame_idx = np.arange(frame_count)
        out_centre = (m_first + frame_idx) * hop
        nxt = np.searchsorted(first, frame_idx, side="right")          # next onset (reset frame > i)
        j = np.minimum(nxt, onsets.size - 1)
        ahead = nxt < onsets.size
        d_in = onsets[j] - centres
        in_cut = np.where(ahead & (d_in < half), d_in + half, n_fft).astype(np.int64)
        d_out = out_pos[j] - out_centre
        out_pre = np.where(ahead & (d_out < half), d_out + half, n_fft).astype(np.int64)
        prv = np.maximum(nxt - 1, 0)
        d_out = out_pos[prv] - out_centre
        out_post = np.where((nxt > 0) & (d_out > -half), d_out + half, 0).astype(np.int64)
    gated_any = bool((in_cut < n_fft).any() or (out_pre < n_fft).any() or (out_post > 0).any())

    omega = 2.0 * np.pi * np.arange(bins) / n_fft
    k_idx = np.arange(bins, dtype=np.int32)
    formant = _FormantGain(n_fft, bins, formant_alpha, sr) if formant_alpha else None
    origin = m_first * hop - half                # buffer index 0 = this output time
    y = np.zeros((channels, (m_last - m_first + 1) * hop + n_fft), dtype=np.float32)
    wsum = np.zeros(y.shape[-1], dtype=np.float32) if gated_any else None

    phs_prev: np.ndarray | None = None
    prev_pha: np.ndarray | None = None
    prev_logm: np.ndarray | None = None
    prev_centre = 0
    offsets = np.arange(n_fft)
    total = len(centres)

    for c0 in range(0, total, CHUNK_FRAMES):
        cs = centres[c0: c0 + CHUNK_FRAMES]
        count = len(cs)
        frames = windows[:, cs + pad - half] * win                                   # (C, F, N)
        mask_in = mask_out = None
        cut_in, cut_pre, cut_post = in_cut[c0: c0 + count], out_pre[c0: c0 + count], out_post[c0: c0 + count]
        gated_rows = np.flatnonzero((cut_in < n_fft) | (cut_pre < n_fft) | (cut_post > 0))
        if gated_rows.size:
            # complementary ramps centred on the cuts: the attack leaves earlier frames (analysis
            # and output), and later frames' output does not reach back before the onset
            g = offsets[None, :]
            mask_in = _ramp(g, cut_in[gated_rows, None], down=True)
            mask_out = (_ramp(g, cut_pre[gated_rows, None], down=True) *
                        _ramp(g, cut_post[gated_rows, None], down=False))
            frames[:, gated_rows] *= mask_in[None]
        spec = sfft.rfft(frames, axis=-1, workers=_WORKERS)                          # complex64
        mid = spec[0] if channels == 1 else (spec.sum(axis=0) * np.float32(1.0 / channels))
        mag = np.abs(mid)
        pha = np.angle(mid).astype(np.float64)
        logm = np.log(mag + EPS)

        if prev_pha is None:
            p_pha = np.concatenate([pha[:1], pha[:-1]])
            p_logm = np.concatenate([logm[:1], logm[:-1]])
            dt = np.concatenate([[hop / r], np.diff(cs)]).astype(np.float64)
        else:
            p_pha = np.concatenate([prev_pha[None], pha[:-1]])
            p_logm = np.concatenate([prev_logm[None], logm[:-1]])
            dt = np.diff(np.concatenate([[prev_centre], cs])).astype(np.float64)
        dt = np.maximum(dt, 1.0)

        # phase advance over one synthesis hop from the instantaneous frequency (float64, wrapped)
        dphi = pha - p_pha - omega[None, :] * dt[:, None]
        dphi -= 2.0 * np.pi * np.round(dphi / (2.0 * np.pi))
        advance = np.mod(omega * hop, 2.0 * np.pi)[None, :] + dphi * (hop / dt)[:, None]

        # spectral peaks (local maxima over ±2 bins) and every bin's nearest peak
        peak = np.zeros(mag.shape, dtype=bool)
        core = mag[:, 2:-2]
        peak[:, 2:-2] = (core > mag[:, 1:-3]) & (core >= mag[:, 3:-1]) & (core > mag[:, :-4]) & (core >= mag[:, 4:])
        peak[:, 0] = True
        left = np.where(peak, k_idx, np.int32(-1))
        np.maximum.accumulate(left, axis=1, out=left)
        right = np.where(peak[:, ::-1], k_idx[::-1], np.int32(bins))
        np.minimum.accumulate(right, axis=1, out=right)
        right = right[:, ::-1]
        use_right = (right - k_idx) < (k_idx - left)
        use_right &= right < bins
        near = np.where(use_right, right, left)
        flat = near + (np.arange(count, dtype=np.int32) * np.int32(bins))[:, None]
        step = np.take(advance, flat) - np.take(pha, flat)    # φs(k) = φa(k) + φs_prev(p) + step(p)
        resets, shifts = reset_frames[c0: c0 + count], reset_shift[c0: c0 + count]

        base = pha + step
        phs = np.empty_like(pha)
        for f in range(count):
            if phs_prev is None:
                cur = pha[f].copy()
            else:
                cur = base[f] + phs_prev[near[f]]           # phases grow slowly; wrapped per chunk
                if resets[f]:
                    sel = (logm[f] - p_logm[f])[near[f]] > RESET_RISE
                    cur[sel] = pha[f][sel] - omega[sel] * shifts[f]
            phs[f] = cur
            phs_prev = cur
        phs -= 2.0 * np.pi * np.floor((phs + np.pi) / (2.0 * np.pi))
        phs_prev = phs[-1]

        theta = (phs - pha).astype(np.float32)
        rot = np.empty(theta.shape, dtype=np.complex64)
        rot.real = np.cos(theta)
        rot.imag = np.sin(theta)
        if formant is not None:
            rot *= formant.gain(logm)
        out = sfft.irfft(spec * rot[None], n=n_fft, axis=-1, workers=_WORKERS)
        out *= synth_win
        if mask_out is not None:
            out[:, gated_rows] *= mask_out[None]
        start = c0 * hop
        _overlap_add(y, out, start, hop, overlap)
        if wsum is not None:
            weights = np.empty((count, n_fft), dtype=np.float32)
            weights[:] = win * synth_win
            if mask_out is not None:
                weights[gated_rows] *= mask_in * mask_out
            _overlap_add(wsum[None], weights[None], start, hop, overlap)

        prev_pha, prev_logm, prev_centre = pha[-1], logm[-1], int(cs[-1])
        if progress is not None:
            progress(min(1.0, (c0 + count) / total))

    begin = -origin
    result = y[:, begin: begin + out_len]
    if wsum is not None:
        w = wsum[begin: begin + out_len]
        fix = np.abs(w - 1.0) > 1e-3
        if fix.any():
            result[:, fix] /= np.maximum(w[fix], 0.25)
    return np.ascontiguousarray(result)


def _ramp(i: np.ndarray, cut: np.ndarray, *, down: bool) -> np.ndarray:
    """1 → 0 (``down``) or 0 → 1 over ``GATE_RAMP`` samples centred on ``cut``."""
    t = (i - cut) / GATE_RAMP + 0.5
    return np.clip(1.0 - t if down else t, 0.0, 1.0).astype(np.float32)


def _overlap_add(y: np.ndarray, frames: np.ndarray, start: int, hop: int, overlap: int) -> None:
    """Add ``frames`` ``(C, F, N)`` at ``start + f·hop`` into ``y`` ``(C, L)``: every ``overlap``-th
    frame tiles without overlap, so it is ``overlap`` vectorised adds instead of ``F``."""
    channels, count, n_fft = frames.shape
    for j in range(min(overlap, count)):
        part = frames[:, j::overlap]
        s = start + j * hop
        y[:, s: s + part.shape[1] * n_fft] += part.reshape(channels, -1)


class _FormantGain:
    """Per-bin gain ``E(α·k) / E(k)`` from the cepstrally smoothed log magnitude of each frame."""

    def __init__(self, n_fft: int, bins: int, alpha: float, sr: float) -> None:
        self.n_fft = n_fft
        self.lifter = max(8, int(sr / 1100.0))          # keeps formants, drops harmonics of f0 ≤ ~1.1 kHz
        src = np.arange(bins) * alpha
        self.i0 = np.minimum(np.floor(src).astype(np.int64), bins - 1)
        self.i1 = np.minimum(self.i0 + 1, bins - 1)
        self.frac = np.clip(src - self.i0, 0.0, 1.0)
        self.lim = math.log(10 ** (FORMANT_MAX_DB / 20.0))

    def gain(self, logm: np.ndarray) -> np.ndarray:
        from scipy import fft as sfft

        cep = sfft.irfft(logm, n=self.n_fft, axis=-1, workers=_WORKERS)
        cep[:, self.lifter: self.n_fft - self.lifter + 1] = 0.0
        env = sfft.rfft(cep, axis=-1, workers=_WORKERS).real
        shifted = env[:, self.i0] * (1.0 - self.frac) + env[:, self.i1] * self.frac
        return np.exp(np.clip(shifted - env, -self.lim, self.lim)).astype(np.float32)
