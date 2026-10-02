"""Pitch lines for the pitch editor, and the user's edits applied to them when singing.

Times are seconds from the start of the render node's input. An edit is only what the user
changed: segments of semitone offsets on a regular grid (``rate`` per second), applied to the
extracted f0 before the key shift — so changing the key, the extractor or the voice keeps them.

No torch here: the backend builds the editor's payload (notes, key) from the f0 the render worker
extracted, and the worker applies edits with :func:`offsets`.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

EDIT_RATE = 100                    # offsets per second when an edit doesn't say
MIN_RATE, MAX_RATE = 20, 400       # the editor stores at its analysis frame rate (≈ 86 /s)
MAX_EDIT_SECONDS = 60 * 30
MAX_OFFSET = 24.0                  # semitones
# the analysis grid: DDSP-SVC's 512-sample hop at 44.1 kHz (the render cache reuses it)
ANALYSIS_HOP_S = 512 / 44100

NOTE_SPLIT_ST = 0.75               # a pitch this far from the running note starts a new one…
NOTE_SPLIT_FRAMES = 4              # …once it stays there this many frames
NOTE_MIN_S = 0.06                  # shorter pieces merge into a neighbour

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


# ---- edits ---------------------------------------------------------------------------------
def parse(raw: Any) -> dict | None:
    """A well-formed edit ``{"rate": r, "segments": [{"start": s, "values": [st, …]}]}`` or None
    (empty / not an edit). Values are clamped to ±24 semitones; non-numbers become 0."""
    if not isinstance(raw, dict):
        return None
    rate = raw.get("rate", EDIT_RATE)
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not MIN_RATE <= rate <= MAX_RATE:
        raise ValueError(f"pitch edit rate must be {MIN_RATE}–{MAX_RATE} per second")
    segments = []
    total = 0
    for seg in raw.get("segments") or []:
        if not isinstance(seg, dict):
            raise ValueError("pitch edit segment must be an object")
        start = seg.get("start")
        values = seg.get("values")
        if not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0 or not isinstance(values, list):
            raise ValueError("pitch edit segment needs a start ≥ 0 and values")
        clean = []
        for v in values:
            x = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else 0.0
            clean.append(max(-MAX_OFFSET, min(MAX_OFFSET, x)))
        total += len(clean)
        if total > rate * MAX_EDIT_SECONDS:
            raise ValueError("pitch edit is too long")
        if clean:
            segments.append({"start": float(start), "values": clean})
    return {"rate": float(rate), "segments": segments} if segments else None


def offsets(edit: dict | None, n_frames: int, frame_s: float, start_s: float = 0.0) -> np.ndarray | None:
    """Semitone offset of each of ``n_frames`` frames (frame ``i`` at ``start_s + i * frame_s``
    on the input's timeline), or None when nothing is edited there. Segments are interpolated
    linearly; later segments win where they overlap."""
    if not edit or not edit.get("segments") or n_frames <= 0:
        return None
    t = start_s + np.arange(n_frames) * frame_s
    out = np.zeros(n_frames)
    touched = False
    step = 1.0 / edit.get("rate", EDIT_RATE)
    for seg in edit["segments"]:
        values = np.asarray(seg["values"], dtype=np.float64)
        if values.size == 0:
            continue
        times = seg["start"] + np.arange(values.size) * step
        inside = (t >= times[0]) & (t <= times[-1])
        if not inside.any():
            continue
        out[inside] = np.interp(t[inside], times, values)
        touched = True
    return out if touched and np.any(out) else None


def apply(f0: np.ndarray, edit: dict | None, frame_s: float, start_s: float = 0.0) -> np.ndarray:
    """``f0`` (Hz per frame) with the edit applied."""
    shift = offsets(edit, len(f0), frame_s, start_s)
    if shift is None:
        return f0
    return f0 * np.power(2.0, shift / 12.0)


# ---- f0 helpers ----------------------------------------------------------------------------
def interp_unvoiced(f0: np.ndarray, f0_min: float) -> np.ndarray:
    """Exactly ``F0_Extractor.extract(uv_interp=True)`` from its raw output (0 = unvoiced)."""
    f0 = np.array(f0, dtype=np.float64)
    uv = f0 == 0
    if len(f0[~uv]) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    f0[f0 < f0_min] = f0_min
    return f0


def to_midi(f0: np.ndarray) -> np.ndarray:
    """MIDI note numbers (fractional), NaN where unvoiced."""
    f0 = np.asarray(f0, dtype=np.float64)
    out = np.full(f0.shape, np.nan)
    voiced = f0 > 0
    out[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)
    return out


def _median3(x: np.ndarray) -> np.ndarray:
    if x.size < 3:
        return x.copy()
    padded = np.concatenate([x[:1], x, x[-1:]])
    stacked = np.stack([padded[:-2], padded[1:-1], padded[2:]])
    if not np.isnan(stacked).any():
        return np.median(stacked, axis=0)
    with warnings.catch_warnings():  # all-NaN columns (unvoiced) stay NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stacked, axis=0)


def notes(midi: np.ndarray, frame_s: float) -> list[dict]:
    """Notes like the blobs of a pitch editor: voiced runs split where the pitch settles somewhere
    else. Each ``{"start", "end"}`` (seconds), ``"pitch"`` (median MIDI), ``"first"``/``"last"``
    (frame indices, inclusive)."""
    smooth = _median3(midi)
    voiced = ~np.isnan(smooth)
    found: list[list[int]] = []
    i, n = 0, len(smooth)
    while i < n:
        if not voiced[i]:
            i += 1
            continue
        j = i
        while j < n and voiced[j]:
            j += 1
        # split the run [i, j) where the pitch moves away from the current note and stays away
        start = i
        center = smooth[i]
        away = 0
        for k in range(i, j):
            if abs(smooth[k] - center) > NOTE_SPLIT_ST:
                away += 1
                if away >= NOTE_SPLIT_FRAMES:
                    cut = k - away + 1
                    if cut > start:
                        found.append([start, cut - 1])
                    start = cut
                    center = float(np.median(smooth[start:k + 1]))
                    away = 0
            else:
                away = 0
                center = float(np.median(smooth[start:k + 1])) if (k - start) % 4 == 0 else center
        found.append([start, j - 1])
        i = j
    # pieces too short to be notes join the adjacent piece before them (or the one after)
    min_frames = max(1, int(round(NOTE_MIN_S / frame_s)))
    merged: list[list[int]] = []
    for a, b in found:
        if merged and merged[-1][1] == a - 1 and (b - a + 1 < min_frames or merged[-1][1] - merged[-1][0] + 1 < min_frames):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        seg = midi[a:b + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size == 0:
            continue
        out.append({"start": round(a * frame_s, 4), "end": round((b + 1) * frame_s, 4),
                    "pitch": round(float(np.median(seg)), 3), "first": a, "last": b})
    return out


def estimate_key(found: list[dict]) -> dict | None:
    """Krumhansl–Schmuckler on the notes' pitch classes weighted by length:
    ``{"tonic": 0..11 (C = 0), "mode": "major" | "minor", "confidence": 0..1}``."""
    hist = np.zeros(12)
    for note in found:
        hist[int(round(note["pitch"])) % 12] += note["end"] - note["start"]
    if hist.sum() <= 0:
        return None
    best: tuple[float, int, str] | None = None
    scores = []
    for mode, profile in (("major", MAJOR), ("minor", MINOR)):
        for tonic in range(12):
            r = float(np.corrcoef(hist, np.roll(profile, tonic))[0, 1])
            scores.append(r)
            if best is None or r > best[0]:
                best = (r, tonic, mode)
    assert best is not None
    ranked = sorted(scores, reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else 0.0
    return {"tonic": best[1], "mode": best[2], "confidence": round(max(0.0, min(1.0, best[0] * 0.5 + margin * 2)), 3)}


def analysis(f0: np.ndarray, frame_s: float) -> dict:
    """The editor's view of an input: its pitch line (MIDI per frame, null where unvoiced), notes
    and key."""
    midi = to_midi(f0)
    found = notes(midi, frame_s)
    return {
        "frameS": frame_s,
        "duration": round(len(f0) * frame_s, 4),
        "midi": [None if math.isnan(v) else round(float(v), 3) for v in midi],
        "notes": [{k: n[k] for k in ("start", "end", "pitch")} for n in found],
        "key": estimate_key(found),
    }
