"""Media analysis for ``GET /projects/{id}/media/{mediaId}/analysis``."""

from __future__ import annotations

from pathlib import Path

from . import loudness, tempo
from .audioio import read


def analyze(path: Path | str) -> dict:
    """``{"bpm": float | None, "lufs": float, "peakDb": float, "duration": float}``.

    ``bpm`` is the detected tempo as is (not halved; ``None`` without a clear beat), ``lufs`` the
    BS.1770-4 integrated loudness at the file's own sample rate (-70 for silence), ``peakDb`` the
    sample peak in dBFS (-120 for silence), ``duration`` in seconds."""
    x, sr = read(path, sr=None)
    if x.shape[0] > 2:
        x = x[:2]
    bpm = tempo.detect_bpm(x, sr)
    return {
        "bpm": round(bpm, 2) if bpm else None,
        "lufs": loudness.report_lufs(loudness.integrated_loudness(x, sr)),
        "peakDb": round(loudness.sample_peak_db(x), 2),
        "duration": round(x.shape[-1] / sr, 3) if sr else 0.0,
    }
