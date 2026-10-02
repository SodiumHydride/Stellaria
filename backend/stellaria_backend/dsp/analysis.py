"""Song analysis for the canvas editors (docs/canvas-v3.md §1.6): beats, downbeats, vocal phrases,
labelled sections, and the musical key. Pure functions on file paths, no FastAPI, no torch.

* **Tempo** is ``mixing.tempo.detect_bpm`` (librosa's tempo estimate without ``beat_track``'s numba
  beat placement, which costs seconds of JIT per process — see that module).
* **Beats**: dynamic-programming beat tracker (Ellis, "Beat tracking by dynamic programming", 2007
  — the algorithm behind ``librosa.beat.beat_track``) on the same onset envelope, in numpy.
* **Downbeats**: a bar-position HMM over the beats (Viterbi, 4/4 or 3/4) whose downbeat state is
  rewarded by beat-level accents — onset strength, bass onsets and harmonic (chroma) change, the
  strongest bar-level periodicity of pop music; it follows a phase change when the tracker
  skips or adds a beat.
* **Phrases**: vocal activity (frame RMS against the loud end, breaths bridged) on the vocals stem
  when given, else on the input; when the input is a full mix whose "activity" never stops, phrases
  are 4-bar groups from the downbeats inside each section.
* **Sections**: Foote novelty on a beat-synchronous self-similarity matrix of time-delay-embedded
  chroma and loudness, peaks snapped to downbeats; segments are clustered by aligned diagonal
  similarity; labels are best-effort (chorus = loudest repeated cluster, verse = repeated cluster
  before the choruses, intro/outro/bridge by position and repetition), ``None`` when unsure.
* **Key**: Krumhansl–Schmuckler (Krumhansl–Kessler profiles) on a tuning-corrected,
  energy-weighted chroma; ``None`` without a clear tonality.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

ASR = 22050
HOP = 512                          # onset envelope frames (23.2 ms)
CHROMA_FFT, CHROMA_HOP = 8192, 2048
FMIN, FMAX = 55.0, 2000.0
TIGHTNESS = 100.0
ONSET_LAG_S = 0.03                 # librosa's flux envelope peaks ~30 ms after a percussive onset
LABELS = ("intro", "verse", "chorus", "bridge", "outro")

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
MAJOR_NAMES = ("C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
MINOR_NAMES = ("Cm", "C#m", "Dm", "Ebm", "Em", "Fm", "F#m", "Gm", "G#m", "Am", "Bbm", "Bm")
KEY_MIN_CORR = 0.5


# ----------------------------------------------------------------------------- loading & features

def _load(path: Path | str) -> np.ndarray:
    from ..mixing.audioio import read, to_mono
    from .common import check_input

    x, _ = read(check_input(path), sr=ASR)
    return to_mono(x).astype(np.float32)


def _onset_envelope(y: np.ndarray) -> np.ndarray:
    """Median-aggregated spectral-flux onset strength (librosa, as ``mixing.tempo`` uses it)."""
    if y.size < HOP * 4:
        return np.zeros(max(1, y.size // HOP), dtype=np.float32)
    import librosa.onset

    env = librosa.onset.onset_strength(y=y, sr=ASR, hop_length=HOP, aggregate=np.median)
    return np.nan_to_num(env.astype(np.float32))


def _spectra(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One STFT pass (``CHROMA_FFT``/``CHROMA_HOP``, centred, in blocks so long files stay light):
    ``(band magnitudes FMIN..FMAX (frames, bins), their frequencies, frame power, bass magnitude sum)``."""
    from scipy import fft as sfft

    n_fft, hop = CHROMA_FFT, CHROMA_HOP
    freqs = np.fft.rfftfreq(n_fft, 1.0 / ASR)
    band = (freqs >= FMIN) & (freqs <= FMAX)
    low = (freqs >= 30) & (freqs <= 160)
    pad = np.concatenate([np.zeros(n_fft // 2, np.float32), y, np.zeros(n_fft // 2 + n_fft, np.float32)])
    frames = np.lib.stride_tricks.sliding_window_view(pad, n_fft)[::hop][: y.size // hop + 1]
    win = np.hanning(n_fft).astype(np.float32)
    count = frames.shape[0]
    mag_band = np.empty((count, int(band.sum())), dtype=np.float32)
    power = np.empty(count)
    bass = np.empty(count)
    for s in range(0, count, 256):
        mag = np.abs(sfft.rfft(frames[s: s + 256] * win, axis=-1))
        mag_band[s: s + 256] = mag[:, band]
        power[s: s + 256] = np.mean(mag.astype(np.float64) ** 2, axis=1) * 2.0 / n_fft
        bass[s: s + 256] = mag[:, low].sum(axis=1)
    return mag_band, freqs[band], power, bass


def _tuning(mag: np.ndarray, freqs: np.ndarray) -> float:
    """Global tuning offset in semitones (-0.5..0.5) from the magnitude-weighted circular mean of
    spectral peaks' deviation from equal temperament (A4 = 440 Hz)."""
    m = mag
    f = freqs
    peak = (m[:, 1:-1] > m[:, :-2]) & (m[:, 1:-1] >= m[:, 2:])
    peak &= m[:, 1:-1] > (m.max(axis=1, keepdims=True) * 0.1)
    rows, cols = np.nonzero(peak)
    if cols.size < 20:
        return 0.0
    cols = cols + 1
    a, b, c = m[rows, cols - 1], m[rows, cols], m[rows, cols + 1]
    denom = a - 2 * b + c
    delta = np.where(np.abs(denom) > 1e-12, 0.5 * (a - c) / np.where(denom == 0, 1, denom), 0.0)
    fp = f[cols] + np.clip(delta, -0.5, 0.5) * (f[1] - f[0])
    semis = 12.0 * np.log2(np.maximum(fp, 1e-6) / 440.0)
    dev = semis - np.round(semis)
    angle = np.angle(np.sum(b * np.exp(2j * np.pi * dev)))
    return float(angle / (2 * np.pi))


def _chroma(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(chroma (frames, 12) magnitude-weighted, rms_db (frames,), bass_flux (frames,))`` at
    ``CHROMA_HOP`` frames; chroma bins map to pitch classes with a triangular semitone weight
    around the tuning-corrected semitone centres."""
    mag, freqs, power, bass = _spectra(y)
    tune = _tuning(mag, freqs)
    semis = 12.0 * np.log2(freqs / 440.0) + 69.0 - tune
    nearest = np.round(semis)
    weight = np.maximum(0.0, 1.0 - np.abs(semis - nearest) / 0.5)
    mapping = np.zeros((freqs.size, 12), dtype=np.float32)
    mapping[np.arange(freqs.size), nearest.astype(np.int64) % 12] = weight
    chroma = mag @ mapping
    rms_db = 10.0 * np.log10(power + 1e-12)
    lm = np.log(bass + 1e-6)
    bass_flux = np.maximum(np.diff(lm, prepend=lm[:1]), 0.0)
    return chroma.astype(np.float32), rms_db.astype(np.float32), bass_flux.astype(np.float32)


# ----------------------------------------------------------------------------- beats

def track_beats(env: np.ndarray, bpm: float, *, tightness: float = TIGHTNESS) -> np.ndarray:
    """Beat frame indices (onset-envelope frames) by Ellis' dynamic programming."""
    n = env.size
    if n < 8 or not bpm or bpm <= 0:
        return np.zeros(0, dtype=np.int64)
    std = float(env.std())
    if std <= 0:
        return np.zeros(0, dtype=np.int64)
    period = 60.0 * ASR / HOP / bpm
    onset = env / std
    p = int(round(period))
    k = np.arange(-p, p + 1)
    window = np.exp(-0.5 * (k * 32.0 / period) ** 2)
    local = np.convolve(onset, window, mode="same")
    lo, hi = int(round(2 * period)), max(1, int(round(period / 2)))
    lags = np.arange(hi, lo + 1)                                  # τ = t - lag
    penalty = -tightness * np.log(lags / period) ** 2
    cum = np.zeros(n)
    back = np.full(n, -1, dtype=np.int64)
    first = True
    thresh = 0.01 * float(local.max())
    for t in range(n):
        valid = lags <= t
        if valid.any() and not (first and local[t] < thresh):
            cand = cum[t - lags[valid]] + penalty[valid]
            j = int(np.argmax(cand))
            cum[t] = local[t] + cand[j]
            back[t] = t - int(lags[valid][j])
        else:
            cum[t] = local[t]
        if first and local[t] >= thresh:
            first = False
    # last beat: the last local maximum of the cumulative score above half its median
    is_max = np.zeros(n, dtype=bool)
    is_max[1:-1] = (cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:])
    if not is_max.any():
        return np.zeros(0, dtype=np.int64)
    med = float(np.median(cum[is_max]))
    cands = np.flatnonzero(is_max & (cum >= 0.5 * med))
    beats = [int(cands[-1]) if cands.size else int(np.argmax(cum))]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    beats = np.array(beats[::-1], dtype=np.int64)
    # trim weak beats at both ends (silence/fade): smoothed local score under half its RMS
    w = np.hanning(5)
    smooth = np.convolve(local[beats], w / w.sum(), mode="same")
    level = 0.5 * math.sqrt(float(np.mean(smooth ** 2))) if smooth.size else 0.0
    keep = np.flatnonzero(smooth >= level)
    if keep.size:
        beats = beats[keep[0]: keep[-1] + 1]
    return beats


def _beat_accents(beat_t: np.ndarray, env: np.ndarray, chroma: np.ndarray, bass: np.ndarray) -> np.ndarray:
    """z-scored downbeat evidence per beat: onset strength, bass onset, harmonic change."""
    env_t = np.arange(env.size) * HOP / ASR
    ch_t = np.arange(chroma.shape[0]) * CHROMA_HOP / ASR

    def at(series: np.ndarray, times: np.ndarray, t: float, width: float) -> float:
        sel = np.abs(times - t) <= width
        return float(series[sel].max()) if sel.any() else 0.0

    onset = np.array([at(env, env_t, t, 0.05) for t in beat_t])
    low = np.array([at(bass, ch_t, t, 0.07) for t in beat_t])
    norm = chroma / (np.linalg.norm(chroma, axis=1, keepdims=True) + 1e-9)
    change = np.zeros(beat_t.size)
    for i, t in enumerate(beat_t):
        before = norm[(ch_t >= t - 0.6) & (ch_t < t - 0.05)]
        after = norm[(ch_t >= t + 0.05) & (ch_t < t + 0.6)]
        if len(before) and len(after):
            a, b = before.mean(axis=0), after.mean(axis=0)
            change[i] = 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def z(v: np.ndarray) -> np.ndarray:
        s = float(v.std())
        return (v - float(v.mean())) / s if s > 1e-9 else np.zeros_like(v)

    return z(onset) + z(low) + 1.5 * z(change)


def _extend_grid(beat_t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Continue the beat grid (median inter-beat interval) over a quiet intro/outro the tracker
    left out, within the part of the file that is not silent."""
    beat_t = beat_t[beat_t >= 0]
    if beat_t.size < 8:
        return beat_t
    ibi = float(np.median(np.diff(beat_t)))
    hop = int(0.05 * ASR)
    n = y.size // hop
    if ibi <= 0 or n == 0:
        return beat_t
    level = 20 * np.log10(np.sqrt(np.mean(y[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1)) + 1e-9)
    audible = np.flatnonzero(level > float(np.max(level)) - 45.0)
    first, last = audible[0] * hop / ASR, (audible[-1] + 1) * hop / ASR
    head = np.arange(beat_t[0] - ibi, first - ibi * 0.25, -ibi)[::-1]
    tail = np.arange(beat_t[-1] + ibi, last + ibi * 0.25 - 1e-9, ibi)
    tail = tail[tail < y.size / ASR]
    return np.concatenate([head[head >= 0], beat_t, tail])


def _bar_phase(accent: np.ndarray, meter: int, switch: float = 6.0) -> tuple[np.ndarray, float]:
    """Viterbi over bar positions (0 = downbeat): returns (position per beat, contrast)."""
    k = accent.size
    states = meter
    score = np.zeros(states)
    score[0] = 0.0
    back = np.zeros((k, states), dtype=np.int64)
    emit = np.zeros((k, states))
    emit[:, 0] = accent
    score = emit[0].copy()
    for i in range(1, k):
        nxt = np.full(states, -np.inf)
        arg = np.zeros(states, dtype=np.int64)
        for s in range(states):
            # from (s-1) at no cost, from anything else at a cost
            prev = score - switch
            prev[(s - 1) % states] += switch
            j = int(np.argmax(prev))
            nxt[s], arg[s] = prev[j], j
        score = nxt + emit[i]
        back[i] = arg
    path = np.zeros(k, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    for i in range(k - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    down = path == 0
    contrast = float(accent[down].mean() - accent[~down].mean()) if down.any() and (~down).any() else 0.0
    return path, contrast


def downbeat_indices(accent: np.ndarray) -> tuple[np.ndarray, int]:
    """Indices of downbeats among the beats and the meter (4, or 3 when clearly better)."""
    if accent.size < 6:
        return np.arange(0, accent.size, 4), 4
    p4, c4 = _bar_phase(accent, 4)
    p3, c3 = _bar_phase(accent, 3)
    if c3 > c4 + 0.5:
        return np.flatnonzero(p3 == 0), 3
    return np.flatnonzero(p4 == 0), 4


# ----------------------------------------------------------------------------- phrases

def vocal_phrases(y: np.ndarray, *, gap_s: float = 0.3, min_s: float = 0.3) -> tuple[list[tuple[float, float]], float]:
    """Phrases (start, end) of vocal activity and the active fraction of the signal."""
    hop = int(0.02 * ASR)
    n = y.size // hop
    if n < 5:
        return [], 0.0
    frames = y[: n * hop].reshape(n, hop).astype(np.float64)
    db = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    top = float(np.percentile(db, 95))
    if top < -60.0:
        return [], 0.0
    active = db > max(top - 30.0, -50.0)
    from scipy import ndimage

    active = ndimage.binary_closing(active, structure=np.ones(int(gap_s / 0.02) + 1, bool))
    active = ndimage.binary_opening(active, structure=np.ones(3, bool))
    labels, count = ndimage.label(active)
    out: list[tuple[float, float]] = []
    for sl in ndimage.find_objects(labels):
        a, b = sl[0].start * hop / ASR, sl[0].stop * hop / ASR
        if b - a >= min_s:
            out.append((a, b))
    return out, float(active.mean())


# ----------------------------------------------------------------------------- sections

def _embed(feat: np.ndarray, m: int = 4, lag: int = 2) -> np.ndarray:
    """Time-delay embedding: row ``i`` is rows ``i, i+lag, … i+(m-1)·lag`` (zero-padded at the end),
    so a beat's feature describes the next two bars — chord *progressions*, not single chords."""
    k, d = feat.shape
    pad = np.vstack([feat, np.zeros(((m - 1) * lag, d), feat.dtype)])
    return np.hstack([pad[i * lag: i * lag + k] for i in range(m)])


def _ssm(units_chroma: np.ndarray, units_loud: np.ndarray) -> np.ndarray:
    c = _embed(units_chroma / (np.linalg.norm(units_chroma, axis=1, keepdims=True) + 1e-9))
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    s_chroma = np.clip(c @ c.T, 0.0, 1.0)
    loud = units_loud[:, None] - units_loud[None, :]
    s_loud = np.exp(-np.abs(loud) / 4.0)                          # 4 dB scale
    # loudness carries real weight: verses and choruses often share their chords
    return 0.6 * s_chroma + 0.4 * s_loud


def _novelty(s: np.ndarray, half: int) -> np.ndarray:
    k = s.shape[0]
    g = np.arange(-half, half) + 0.5
    taper = np.exp(-0.5 * (g / (0.5 * half)) ** 2)
    sign = np.sign(g)
    kernel = np.outer(sign, sign) * np.outer(taper, taper)
    pad = np.pad(s, half, mode="constant")
    nov = np.zeros(k)
    for i in range(k):
        nov[i] = float(np.sum(pad[i: i + 2 * half, i: i + 2 * half] * kernel))
    edge = int(0.75 * half)                      # the zero padding fakes a change at both ends
    nov[:edge] = 0.0
    nov[max(edge, k - edge):] = 0.0
    nov = np.maximum(nov, 0.0)
    return nov / (nov.max() + 1e-12)


def _segment_similarity(s: np.ndarray, a: tuple[int, int], b: tuple[int, int], shifts: int = 2) -> float:
    la, lb = a[1] - a[0], b[1] - b[0]
    length = min(la, lb)
    if length <= 0:
        return 0.0
    best = 0.0
    for d in range(-shifts, shifts + 1):
        i0, j0 = a[0], b[0] + d
        idx = np.arange(length)
        ok = (j0 + idx >= b[0]) & (j0 + idx < b[1])
        if ok.sum() < max(2, length // 2):
            continue
        best = max(best, float(np.mean(s[i0 + idx[ok], j0 + idx[ok]])))
    return best


def _clusters(bounds: list[tuple[int, int]], s: np.ndarray) -> list[int]:
    """Cluster id per segment: segments whose aligned diagonal similarity is high repeat each other."""
    count = len(bounds)
    parent = list(range(count))
    if count < 2:
        return parent
    sims = np.zeros((count, count))
    for i in range(count):
        for j in range(i + 1, count):
            sims[i, j] = sims[j, i] = _segment_similarity(s, bounds[i], bounds[j])
    upper = sims[np.triu_indices(count, 1)]
    thresh = max(0.78, float(np.mean(upper) + 0.5 * np.std(upper)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(count):
        for j in range(i + 1, count):
            if sims[i, j] >= thresh:
                parent[find(i)] = find(j)
    return [find(i) for i in range(count)]


def _label(bounds: list[tuple[int, int]], s: np.ndarray, loud: np.ndarray) -> list[str | None]:
    """Best-effort labels; ``None`` wherever the structure does not say.

    chorus: the loudest repeated cluster · intro/outro: a first/last segment that is quieter than
    the song or does not repeat inside it · verse: the repeated cluster with most members before
    the first chorus · bridge: a one-off segment after the first chorus that is not the last."""
    count = len(bounds)
    labels: list[str | None] = [None] * count
    if count < 2:
        return labels
    seg_loud = np.array([float(np.mean(loud[a:b])) for a, b in bounds])
    median = float(np.median(seg_loud))
    ids = _clusters(bounds, s)
    members: dict[int, list[int]] = {}
    for i, c in enumerate(ids):
        members.setdefault(c, []).append(i)
    inner = set(range(1, count - 1))

    def repeats(i: int) -> bool:                    # repeats somewhere other than at the very ends
        return any(j != i and j in inner for j in members[ids[i]])

    if seg_loud[0] < median - 2.0 or not repeats(0):
        labels[0] = "intro"
    if seg_loud[-1] < median - 2.0 or not repeats(count - 1):
        labels[-1] = "outro"
    repeated = [[i for i in m if labels[i] is None] for m in members.values()]
    repeated = [m for m in repeated if len(m) >= 2]
    if not repeated:
        return labels
    chorus = max(repeated, key=lambda m: (float(np.mean(seg_loud[m])), len(m)))
    if float(np.mean(seg_loud[chorus])) < median - 1.0:
        return labels
    for i in chorus:
        labels[i] = "chorus"
    first_chorus = min(chorus)
    rest = [m for m in repeated if m is not chorus]
    if rest:
        verse = max(rest, key=lambda m: (sum(1 for i in m if i < first_chorus), len(m), -min(m)))
        if any(i < first_chorus for i in verse) or len(verse) >= 2:
            for i in verse:
                labels[i] = "verse"
    for i in range(first_chorus + 1, count - 1):
        if labels[i] is None and len(members[ids[i]]) == 1:
            labels[i] = "bridge"
    return labels


def _snap(t: float, grid: np.ndarray, tolerance: float) -> float:
    if grid.size == 0:
        return t
    j = int(np.argmin(np.abs(grid - t)))
    return float(grid[j]) if abs(grid[j] - t) <= tolerance else t


def sections(path: Path | str, *, vocals_path: Path | str | None = None) -> dict[str, Any]:
    """``{bpm, beats, downbeats, phrases: [{start, end}], sections: [{start, end, label}]}`` (s)."""
    from ..mixing import tempo

    y = _load(path)
    duration = y.size / ASR
    if duration < 1.0 or float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) < 1e-5:
        return {"bpm": None, "beats": [], "downbeats": [], "phrases": [],
                "sections": [{"start": 0.0, "end": round(duration, 3), "label": None}] if duration > 0 else []}
    bpm = tempo.detect_bpm(y, ASR)
    env = _onset_envelope(y)
    chroma, rms_db, bass = _chroma(y)
    ch_t = np.arange(chroma.shape[0]) * CHROMA_HOP / ASR

    beat_t = np.zeros(0)
    downbeat_t = np.zeros(0)
    meter = 4
    if bpm:
        beat_t = _extend_grid(track_beats(env, bpm) * HOP / ASR - ONSET_LAG_S, y)
        if beat_t.size >= 4:
            accent = _beat_accents(beat_t, env, chroma, bass)
            idx, meter = downbeat_indices(accent)
            downbeat_t = beat_t[idx]

    # analysis units: beats (or 0.5 s frames without a beat)
    if beat_t.size >= 16:
        edges = np.concatenate([[0.0], (beat_t[:-1] + beat_t[1:]) / 2.0, [duration]])
        half = max(4, 4 * meter)                 # 4 bars each side
    else:
        edges = np.arange(0.0, duration + 0.5, 0.5)
        edges[-1] = duration
        half = 16
    units = len(edges) - 1
    which = np.clip(np.searchsorted(edges, ch_t, side="right") - 1, 0, units - 1)
    u_chroma = np.zeros((units, 12))
    u_loud = np.full(units, -120.0)
    np.add.at(u_chroma, which, chroma)
    counts = np.bincount(which, minlength=units)
    power = np.zeros(units)
    np.add.at(power, which, 10 ** (rms_db / 10.0))
    nz = counts > 0
    u_loud[nz] = 10 * np.log10(power[nz] / counts[nz] + 1e-12)
    if (~nz).any():                                # empty units take their neighbour's features
        idx = np.maximum.accumulate(np.where(nz, np.arange(units), 0))
        u_chroma, u_loud = u_chroma[idx], u_loud[idx]

    s = _ssm(u_chroma, u_loud)
    nov = _novelty(s, min(half, max(2, units // 4)))
    min_gap = max(2, half)                         # sections are at least 4 bars long
    peaks = [i for i in range(1, units - 1) if nov[i] >= nov[i - 1] and nov[i] > nov[i + 1]]
    thresh = max(0.25, float(np.median(nov) + 0.75 * np.std(nov)))
    peaks = sorted((i for i in peaks if nov[i] >= thresh), key=lambda i: -nov[i])
    chosen: list[int] = []
    for i in peaks:
        if all(abs(i - j) >= min_gap for j in chosen) and i >= min_gap // 2 and units - i >= min_gap // 2:
            chosen.append(i)
    chosen.sort()
    bound_units = [0, *chosen, units]
    bounds = list(zip(bound_units[:-1], bound_units[1:]))
    labels = _label(bounds, s, u_loud)
    bar = 60.0 / bpm * meter if bpm else 2.0
    sec_list = []
    for (a, b), lab in zip(bounds, labels):
        start = 0.0 if a == 0 else _snap(float(edges[a]), downbeat_t, bar / 2)
        end = duration if b == units else _snap(float(edges[b]), downbeat_t, bar / 2)
        sec_list.append({"start": start, "end": end, "label": lab})
    for prev, cur in zip(sec_list, sec_list[1:]):  # snapping keeps them contiguous
        cur["start"] = prev["end"]
    sec_list = [g for g in sec_list if g["end"] - g["start"] > 1e-3]

    # phrases
    phrases: list[tuple[float, float]] = []
    if vocals_path is not None:
        phrases, _ = vocal_phrases(_load(vocals_path))
    else:
        found, active = vocal_phrases(y)
        if len(found) >= 2 and active < 0.85:
            phrases = found
        elif downbeat_t.size >= 2:                 # a full mix: 4-bar phrases inside each section
            for g in sec_list:
                inside = downbeat_t[(downbeat_t >= g["start"] - 1e-6) & (downbeat_t < g["end"] - 1e-6)]
                starts = [g["start"], *[float(t) for t in inside[4::4] if t - g["start"] > bar * 0.5]]
                for a, b in zip(starts, [*starts[1:], g["end"]]):
                    phrases.append((a, b))
        else:
            phrases = [(g["start"], g["end"]) for g in sec_list]

    def r(v: float) -> float:
        return round(float(v), 3)

    return {
        "bpm": round(float(bpm), 2) if bpm else None,
        "beats": [r(t) for t in beat_t],
        "downbeats": [r(t) for t in downbeat_t],
        "phrases": [{"start": r(a), "end": r(b)} for a, b in phrases],
        "sections": [{"start": r(g["start"]), "end": r(g["end"]), "label": g["label"]} for g in sec_list],
    }


# ----------------------------------------------------------------------------- key

def key_from_chroma(profile: np.ndarray) -> tuple[str | None, float]:
    """Best Krumhansl–Kessler key for a 12-bin chroma profile and its correlation."""
    v = np.asarray(profile, dtype=np.float64)
    if v.sum() <= 0 or np.allclose(v, v.mean()):
        return None, 0.0
    best, best_r = None, -1.0
    for tonic in range(12):
        for prof, names in ((MAJOR, MAJOR_NAMES), (MINOR, MINOR_NAMES)):
            r = float(np.corrcoef(v, np.roll(prof, tonic))[0, 1])
            if r > best_r:
                best, best_r = names[tonic], r
    return best, best_r


def key(path: Path | str) -> str | None:
    """``"C#m"``/``"F"``-style key of the audio at ``path``, or None without a clear tonality."""
    y = _load(path)
    if y.size < ASR or float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) < 1e-5:
        return None
    chroma, rms_db, _ = _chroma(y)
    frame_norm = chroma / (chroma.sum(axis=1, keepdims=True) + 1e-9)
    weight = 10 ** (rms_db / 20.0)
    weight[rms_db < float(np.max(rms_db)) - 50.0] = 0.0
    profile = (frame_norm * weight[:, None]).sum(axis=0)
    name, corr = key_from_chroma(profile)
    return name if corr >= KEY_MIN_CORR else None
