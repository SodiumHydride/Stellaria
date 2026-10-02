"""The health check job: analyze clips, detect outliers/duplicates/music, flag, score."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..jobs import JobContext
from ..schemas import HealthChecks, HealthReport, msg
from .analysis import analyze_file, outlier_scores
from .health import OUTLIER_MIN_CLIPS, HealthInputs, compute_flags, evaluate
from .valpick import apply_val, pick_val

if TYPE_CHECKING:
    from ..context import AppContext


def run_health(
    ctx: "AppContext",
    jctx: JobContext,
    cid: str,
    *,
    advanced: bool = False,
    repick_val: bool = False,
    val_count: int = 10,
    min_duration: float = 2.0,
    span: tuple[float, float] = (0.0, 1.0),
) -> HealthReport:
    chars = ctx.characters
    lo, hi = span
    clips = chars.clips(cid)
    active = [c for c in clips if c.status == "active"]
    cache_dir = chars.cache_dir(cid) / "analysis"
    analyses = {}
    music_span = (lo + (hi - lo) * 0.7, hi) if advanced else (hi, hi)
    analyze_hi = music_span[0]
    for i, clip in enumerate(active):
        jctx.check_cancelled()
        jctx.progress(lo + (analyze_hi - lo) * i / max(1, len(active)),
                      msg("stage.analyze", index=i + 1, total=len(active)))
        analyses[clip.id] = analyze_file(chars.clip_path(cid, clip), cache_dir, clip.id, force=advanced)

    ids = [c.id for c in active]
    outliers: dict[str, float] = {}
    if len(ids) >= OUTLIER_MIN_CLIPS:
        scores = outlier_scores(np.stack([analyses[i].mfcc for i in ids]))
        outliers = {i: round(float(s), 3) for i, s in zip(ids, scores)}

    music_available = ctx.separation.available
    previous_music = {c.id: c.metrics.music_ratio for c in active if c.metrics and c.metrics.music_ratio is not None}
    music: dict[str, float] = {}
    if music_available and advanced and active:
        jctx.progress(music_span[0], msg("stage.music"))
        by_path = {str(chars.clip_path(cid, c)): c.id for c in active}
        ratios = ctx.separation.measure(jctx, [chars.clip_path(cid, c) for c in active], music_span)
        music = {by_path[p]: r for p, r in ratios.items() if p in by_path}
    else:
        music = previous_music
    music_checked = music_available and bool(active) and all(c.id in music for c in active)

    seen_hashes: set[str] = set()
    with chars.lock(cid):
        clips = chars.clips(cid)  # re-read: the user may have edited clips meanwhile
        for clip in clips:
            analysis = analyses.get(clip.id)
            if clip.status != "active" or analysis is None:
                continue
            metrics = analysis.metrics.model_copy()
            metrics.outlier_score = outliers.get(clip.id)
            metrics.music_ratio = music.get(clip.id) if music_available else None
            clip.metrics = metrics
            duplicate = analysis.pcm_hash in seen_hashes
            seen_hashes.add(analysis.pcm_hash)
            sticky = ["f0_failed"] if "f0_failed" in clip.flags else []  # set by preprocessing
            clip.flags = sorted(set(compute_flags(clip, duplicate, min_duration) + sticky))
        changed_split = apply_val(clips, pick_val(clips, val_count)) if repick_val else []
        chars.save_clips(cid, clips, bump=bool(changed_split))
        revision = chars.get(cid).clips_revision
        report = evaluate(HealthInputs(
            clips=clips, f0={i: a.f0 for i, a in analyses.items()},
            checks=HealthChecks(basic=True, pitch=True, outlier=bool(outliers), music=music_checked),
            music_available=music_available, min_duration=min_duration,
        ))
        chars.save_health(cid, report, revision)
    ctx.hub.publish("health", {"characterId": cid, "report": report.wire()})
    ctx.emit_clips(cid)
    return report
