"""Pitch curves of project media (parselmouth, cached) and key suggestions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..audio.io import read_wav, resample
from ..quality.analysis import PITCH_HOP, pitch_track
from ..schemas import PitchRange


def media_pitch(audio_path: Path) -> dict[str, Any]:
    """``{hop, f0}`` (seconds, Hz with 0 = unvoiced), cached next to the media file."""
    cache = audio_path.with_suffix(".pitch.json")
    if cache.is_file() and cache.stat().st_mtime >= audio_path.stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    x, sr = read_wav(audio_path)
    f0 = pitch_track(resample(x, sr, 16000))
    payload = {"hop": PITCH_HOP, "f0": [round(float(v), 1) for v in f0]}
    cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def summarize(f0: list[float] | np.ndarray) -> tuple[PitchRange | None, float]:
    """(p5/p95/median range, voiced seconds) of a pitch curve."""
    arr = np.asarray(f0, dtype=np.float64)
    voiced = arr[arr > 0]
    seconds = len(voiced) * PITCH_HOP
    if len(voiced) < 20:
        return None, seconds
    return PitchRange(low=round(float(np.percentile(voiced, 5)), 2), high=round(float(np.percentile(voiced, 95)), 2),
                      median=round(float(np.median(voiced)), 2)), seconds


def suggest_key(source: PitchRange | None, voiced_seconds: float, character: PitchRange | None) -> dict[str, Any]:
    """Semitone shift that moves the source's median onto the character's median (±12 max)."""
    empty = {"low": None, "high": None, "median": None}
    if source is None or character is None:
        return {"key": 0, "sourceRange": source.wire() if source else empty,
                "characterRange": character.wire() if character else empty, "confidence": "low"}
    key = int(round(12 * math.log2(character.median / source.median)))
    key = max(-12, min(12, key))
    factor = 2 ** (key / 12)
    margin = 2 ** (3 / 12)
    fits = source.low * factor >= character.low / margin and source.high * factor <= character.high * margin
    confidence = "high" if voiced_seconds >= 10 and fits else "low"
    return {"key": key, "sourceRange": source.wire(), "characterRange": character.wire(), "confidence": confidence}
