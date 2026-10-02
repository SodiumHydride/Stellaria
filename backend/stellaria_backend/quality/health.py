"""Health rules and scoring (pure functions; the job that gathers data lives in ``checker``).

Issue codes are stable and localized by the frontend:
``data.too_little`` (error) · ``data.little`` · ``data.more_is_better`` · ``clips.too_short`` (error)
· ``clips.too_long`` · ``clips.clipping|silent|noisy|outlier|music|duplicate|f0_failed``
· ``val.too_few`` (error) · ``val.flagged`` · ``pitch.narrow`` · ``checks.music_unavailable``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..schemas import (
    Clip, HealthChecks, HealthFix, HealthIssue, HealthPitch, HealthReport, HealthTotals, PitchBin,
)
from ..util.timeutil import now_iso
from .analysis import PITCH_HOP

TOO_LONG_S = 30.0
CLIPPING_RATIO = 0.001
SILENCE_RATIO = 0.6
SILENT_RMS_DB = -45.0
NOISY_SNR_DB = 20.0
OUTLIER_SCORE = 3.5
OUTLIER_MIN_CLIPS = 8
MUSIC_RATIO = 0.15
F0_FAILED_VOICED = 0.1
PITCH_NARROW_SEMITONES = 12

# per-flag issue: (code, score weight, max penalty)
FLAG_RULES: dict[str, tuple[str, float, float]] = {
    "clipping": ("clips.clipping", 1.0, 15),
    "silent": ("clips.silent", 0.8, 10),
    "noisy": ("clips.noisy", 0.8, 15),
    "outlier": ("clips.outlier", 0.6, 10),
    "music": ("clips.music", 1.0, 15),
    "duplicate": ("clips.duplicate", 0.5, 5),
    "f0_failed": ("clips.f0_failed", 0.8, 10),
}


@dataclass
class HealthInputs:
    clips: list[Clip]  # all clips (trashed ones are ignored)
    f0: dict[str, np.ndarray] = field(default_factory=dict)  # clip id → f0 track
    checks: HealthChecks = field(default_factory=HealthChecks)
    music_available: bool = False
    min_duration: float = 2.0


def compute_flags(clip: Clip, duplicate: bool, min_duration: float = 2.0) -> list[str]:
    flags: list[str] = []
    if clip.duration < min_duration:
        flags.append("too_short")
    if clip.duration > TOO_LONG_S:
        flags.append("too_long")
    m = clip.metrics
    if m is not None:
        if m.clipping_ratio >= CLIPPING_RATIO:
            flags.append("clipping")
        if m.silence_ratio > SILENCE_RATIO or m.rms_db < SILENT_RMS_DB:
            flags.append("silent")
        if m.snr_db is not None and m.snr_db < NOISY_SNR_DB:
            flags.append("noisy")
        if m.outlier_score is not None and m.outlier_score > OUTLIER_SCORE:
            flags.append("outlier")
        if m.music_ratio is not None and m.music_ratio > MUSIC_RATIO:
            flags.append("music")
        if m.voiced_ratio is not None and m.voiced_ratio < F0_FAILED_VOICED:
            flags.append("f0_failed")
    if duplicate:
        flags.append("duplicate")
    return flags


def pitch_summary(f0_tracks: list[np.ndarray]) -> HealthPitch:
    voiced = np.concatenate([f[f > 0] for f in f0_tracks]) if f0_tracks else np.zeros(0)
    if len(voiced) < 10:
        return HealthPitch()
    midi = np.round(69 + 12 * np.log2(voiced / 440.0)).astype(int)
    values, counts = np.unique(midi, return_counts=True)
    return HealthPitch(
        histogram=[PitchBin(midi=int(v), seconds=round(float(c) * PITCH_HOP, 2)) for v, c in zip(values, counts)],
        low=round(float(np.percentile(voiced, 5)), 2),
        high=round(float(np.percentile(voiced, 95)), 2),
        median=round(float(np.median(voiced)), 2),
    )


def level_for(score: int, blocking: bool) -> str:
    if blocking:
        return "blocked"
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "good"
    if score >= 55:
        return "fair"
    return "poor"


def evaluate(inputs: HealthInputs) -> HealthReport:
    active = [c for c in inputs.clips if c.status == "active"]
    issues: list[HealthIssue] = []
    penalty = 0.0

    def add(code: str, severity: str, params: dict | None = None, clip_ids: list[str] | None = None,
            fix: str | None = None, fix_params: dict | None = None, cost: float = 0.0) -> None:
        nonlocal penalty
        issues.append(HealthIssue(
            code=code, severity=severity, params=params or {}, clip_ids=clip_ids or [],
            fix=HealthFix(action=fix, params=fix_params or {}) if fix else None,
        ))
        penalty += cost

    duration = sum(c.duration for c in active)
    minutes = round(duration / 60.0, 1)
    if duration < 5 * 60:
        add("data.too_little", "error", {"minutes": minutes}, fix="add_material", cost=40)
    elif duration < 15 * 60:
        add("data.little", "warning", {"minutes": minutes}, fix="add_material", cost=15)
    elif duration < 30 * 60:
        add("data.more_is_better", "info", {"minutes": minutes}, fix="add_material", cost=5)

    n = max(1, len(active))
    too_short = [c.id for c in active if "too_short" in c.flags]
    if too_short:
        add("clips.too_short", "error", {"count": len(too_short), "minDuration": inputs.min_duration},
            too_short, fix="trash_clips", cost=min(10, 100 * len(too_short) / n))
    too_long = [c.id for c in active if "too_long" in c.flags]
    if too_long:
        add("clips.too_long", "info", {"count": len(too_long), "maxSeconds": TOO_LONG_S}, too_long, cost=2)
    for flag, (code, weight, cap) in FLAG_RULES.items():
        ids = [c.id for c in active if flag in c.flags]
        if ids:
            ratio = len(ids) / n
            add(code, "warning", {"count": len(ids), "ratio": round(ratio, 3)}, ids, fix="trash_clips",
                cost=min(cap, weight * 100 * ratio))

    val = [c for c in active if c.split == "val"]
    if len(val) < 2:
        add("val.too_few", "error", {"count": len(val), "min": 2}, fix="pick_val", cost=10)
    flagged_val = [c.id for c in val if c.flags]
    if flagged_val:
        add("val.flagged", "warning", {"count": len(flagged_val)}, flagged_val, fix="pick_val", cost=5)

    pitch = pitch_summary([inputs.f0[c.id] for c in active if c.id in inputs.f0])
    if pitch.low and pitch.high:
        span = 12 * math.log2(pitch.high / pitch.low)
        if span < PITCH_NARROW_SEMITONES:
            add("pitch.narrow", "info", {"semitones": round(span, 1)}, cost=3)

    if not inputs.music_available:
        add("checks.music_unavailable", "info", {}, fix="install_addon", fix_params={"resource": "separator"})

    blocking = any(i.severity == "error" for i in issues)
    score = int(round(max(0.0, min(100.0, 100.0 - penalty))))
    return HealthReport(
        score=score, level=level_for(score, blocking), blocking=blocking, checked_at=now_iso(),
        totals=HealthTotals(
            clips=len(active), duration=round(duration, 3),
            train_duration=round(sum(c.duration for c in active if c.split == "train"), 3),
            val_count=len(val),
        ),
        pitch=pitch, issues=issues, checks=inputs.checks,
    )
