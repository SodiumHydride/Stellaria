"""Tempo detection (librosa) and tempo-synced times (note values)."""

from __future__ import annotations

import math

import numpy as np

BPM_SR = 22050
FALLBACK_BPM = 90.0
MAX_TIMING_BPM = 100.0
MAX_ANALYSIS_S = 240.0          # beat tracking looks at most at this much of the middle of a song

# note values as fractions of a whole note: straight and dotted, 2 bars … 1/128
NOTE_VALUES: tuple[float, ...] = tuple(sorted(
    {f for k in range(-1, 8) for f in (2.0 ** -k, 1.5 * 2.0 ** -k)}, reverse=True))


def detect_bpm(x: np.ndarray, sr: float) -> float | None:
    """Tempo of ``x`` (``(n,)`` or ``(channels, n)``) in BPM, or ``None`` when there is no beat.

    Runs exactly the tempo estimate of ``librosa.beat.beat_track`` on a 22.05 kHz mono copy:
    median-aggregated onset strength → ``librosa.feature.tempo`` (tempogram + log-normal prior
    around 120 BPM), which is what ``beat_track`` returns as its tempo. The dynamic-programming
    beat placement ``beat_track`` adds afterwards is not needed and is skipped: its numba code
    costs ~2 s of import per process and ~20 s of JIT compilation on a cold numba cache."""
    from scipy import signal

    mono = np.asarray(x, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=0)
    if mono.size < sr * 3:
        return None
    max_len = int(MAX_ANALYSIS_S * sr)
    if mono.size > max_len:
        start = (mono.size - max_len) // 2
        mono = mono[start: start + max_len]
    if float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))) < 1e-4:
        return None
    g = math.gcd(int(sr), BPM_SR)
    y = signal.resample_poly(mono, BPM_SR // g, int(sr) // g).astype(np.float32) if int(sr) != BPM_SR else mono
    try:
        import librosa.feature
        import librosa.onset

        onset = librosa.onset.onset_strength(y=y, sr=BPM_SR, hop_length=512, aggregate=np.median)
        if not np.any(onset > 0) or not np.all(np.isfinite(onset)):
            return None
        tempo = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset, sr=BPM_SR, hop_length=512,
                                                          start_bpm=120.0))[0])
    except Exception:  # noqa: BLE001 - analysis must never fail a mix
        return None
    if not math.isfinite(tempo) or tempo < 30.0 or tempo > 300.0:
        return None
    return tempo


def timing_bpm(bpm: float | None) -> float:
    """BPM used for synced times: the detected tempo halved while above 100, else 90."""
    if bpm is None or not math.isfinite(bpm) or bpm <= 0:
        return FALLBACK_BPM
    while bpm > MAX_TIMING_BPM:
        bpm /= 2.0
    return float(bpm)


def note_seconds(bpm: float, fraction: float) -> float:
    """Duration of a note value (fraction of a whole note, 4/4) at ``bpm``."""
    return 240.0 / bpm * fraction


def synced(bpm: float, lo: float, hi: float) -> float:
    """The note value (straight or dotted) closest to the window ``[lo, hi]`` seconds.

    Inside the window the one nearest its geometric centre wins; when none fits, the one with the
    smallest log-distance to the window."""
    centre = math.sqrt(lo * hi)
    best, best_key = None, None
    for f in NOTE_VALUES:
        d = note_seconds(bpm, f)
        outside = 0.0 if lo <= d <= hi else min(abs(math.log(d / lo)), abs(math.log(d / hi)))
        key = (outside, abs(math.log(d / centre)))
        if best_key is None or key < best_key:
            best, best_key = d, key
    return float(best)
