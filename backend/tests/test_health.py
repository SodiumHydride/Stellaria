"""Health rules and scoring (pure), clipping detection, outlier scores, validation picking."""

from __future__ import annotations

import numpy as np

from stellaria_backend.quality.analysis import clipping_ratio, outlier_scores, snr_estimate
from stellaria_backend.quality.health import HealthInputs, compute_flags, evaluate, level_for
from stellaria_backend.quality.valpick import pick_val, val_target
from stellaria_backend.schemas import Clip, ClipMetrics, HealthChecks
from synth import note

GOOD = dict(peak_db=-3.0, rms_db=-20.0, silence_ratio=0.1, clipping_ratio=0.0, snr_db=45.0, f0_low=180.0,
            f0_high=420.0, f0_median=260.0, voiced_ratio=0.85, outlier_score=1.0, music_ratio=None)


def make_clips(n: int, duration: float = 8.0, val: int = 2, **metric_overrides) -> list[Clip]:
    clips = []
    for i in range(n):
        m = ClipMetrics(**{**GOOD, "f0_median": 200.0 + i, **metric_overrides})
        clips.append(Clip(id=f"clp_{i}", name=f"c{i}", file=f"clips/c{i}.wav", duration=duration,
                          split="val" if i < val else "train", metrics=m, created_at="2026-01-01T00:00:00Z"))
    return clips


def flagged(clips: list[Clip]) -> list[Clip]:
    for c in clips:
        c.flags = compute_flags(c, duplicate=False)
    return clips


def report_for(clips, music_available=True, f0=None):
    return evaluate(HealthInputs(clips=clips, f0=f0 or {}, checks=HealthChecks(basic=True), music_available=music_available))


def codes(report) -> dict[str, str]:
    return {i.code: i.severity for i in report.issues}


def test_amount_of_material_rules():
    assert codes(report_for(flagged(make_clips(30))))["data.too_little"] == "error"  # 4 min
    r = report_for(flagged(make_clips(60)))  # 8 min
    assert codes(r)["data.little"] == "warning" and not r.blocking
    assert codes(report_for(flagged(make_clips(150))))["data.more_is_better"] == "info"  # 20 min
    r = report_for(flagged(make_clips(300)))  # 40 min, clean
    assert r.score == 100 and r.level == "excellent" and r.issues == []
    assert r.totals.clips == 300 and r.totals.val_count == 2
    assert abs(r.totals.duration - 2400) < 1e-6 and abs(r.totals.train_duration - 298 * 8) < 1e-6


def test_issue_params_and_blocking():
    r = report_for(flagged(make_clips(30)))
    issue = next(i for i in r.issues if i.code == "data.too_little")
    assert issue.params == {"minutes": 4.0} and issue.fix.action == "add_material"
    assert r.blocking and r.level == "blocked"


def test_flag_rules_and_penalties():
    clips = flagged(make_clips(300))
    clips[5].metrics.clipping_ratio = 0.01
    clips[6].metrics.snr_db = 10.0
    clips[7].metrics.silence_ratio = 0.9
    clips[8].metrics.outlier_score = 6.0
    clips[9].metrics.music_ratio = 0.5
    clips[10].metrics.voiced_ratio = 0.02
    clips[11].duration = 1.5
    clips[12].duration = 45.0
    clips[0].metrics.snr_db = 5.0  # a flagged validation clip
    flagged(clips)
    assert clips[5].flags == ["clipping"] and clips[6].flags == ["noisy"] and clips[7].flags == ["silent"]
    assert clips[8].flags == ["outlier"] and clips[9].flags == ["music"] and clips[10].flags == ["f0_failed"]
    assert clips[11].flags == ["too_short"] and clips[12].flags == ["too_long"]
    r = report_for(clips)
    c = codes(r)
    for code in ("clips.clipping", "clips.noisy", "clips.silent", "clips.outlier", "clips.music", "clips.f0_failed"):
        assert c[code] == "warning"
        issue = next(i for i in r.issues if i.code == code)
        assert issue.fix.action == "trash_clips" and issue.clip_ids
    assert c["clips.too_short"] == "error" and c["clips.too_long"] == "info" and c["val.flagged"] == "warning"
    assert next(i for i in r.issues if i.code == "val.flagged").fix.action == "pick_val"
    assert r.blocking
    clips[11].status = "trashed"
    r2 = report_for(clips)
    assert not r2.blocking and r2.level in ("excellent", "good") and r2.score < 100


def test_val_and_music_and_pitch_issues():
    clips = flagged(make_clips(300, val=1))
    r = report_for(clips, music_available=False)
    c = codes(r)
    assert c["val.too_few"] == "error" and c["checks.music_unavailable"] == "info"
    fix = next(i for i in r.issues if i.code == "checks.music_unavailable").fix
    assert fix.action == "install_addon" and fix.params == {"resource": "separator"}
    narrow = {cl.id: np.full(500, 220.0 * 2 ** (i % 5 / 12), dtype=np.float32) for i, cl in enumerate(clips)}
    r = report_for(clips, f0=narrow)
    assert codes(r)["pitch.narrow"] == "info"
    assert r.pitch.low and r.pitch.high and 210 < r.pitch.median < 300
    assert sum(b.seconds for b in r.pitch.histogram) == 300 * 5.0


def test_duplicates_flag():
    c = make_clips(1)[0]
    assert compute_flags(c, duplicate=True) == ["duplicate"]


def test_levels():
    assert level_for(95, False) == "excellent" and level_for(80, False) == "good"
    assert level_for(60, False) == "fair" and level_for(40, False) == "poor" and level_for(99, True) == "blocked"


def test_clipping_detection_survives_rescaling():
    x = note(2.0, 220, amp=0.3)
    assert clipping_ratio(x) == 0.0
    hard = np.clip(x * 8, -1.0, 1.0)
    assert clipping_ratio(hard) > 0.05
    assert clipping_ratio(hard * 0.5) > 0.05  # flat tops remain after scaling down


def test_snr_estimate_needs_pauses():
    frames = np.full(200, -20.0)
    frames[::4] = -70.0
    voiced = frames > -50
    assert snr_estimate(frames, voiced) == 50.0
    assert snr_estimate(frames, np.ones(200, dtype=bool)) is None


def test_outlier_scores():
    rng = np.random.default_rng(0)
    features = rng.normal(0, 1, (40, 20))
    features[7] += 12
    scores = outlier_scores(features)
    assert int(np.argmax(scores)) == 7 and scores[7] > 3.5 and np.median(scores) < 1.5


def test_val_pick_is_diverse_and_never_flagged():
    clips = make_clips(100, val=0)
    for i, c in enumerate(clips):
        c.metrics.f0_median = 150.0 + 3 * i
    clips[50].flags = ["noisy"]
    picks = pick_val(clips, 10)
    assert len(picks) == 5  # ≤ 5 % of 100 clips
    assert "clp_50" not in picks
    medians = sorted(c.metrics.f0_median for c in clips if c.id in picks)
    assert medians[-1] - medians[0] > 150
    assert val_target(10, 10) == 2 and val_target(1, 1000) == 2 and val_target(50, 1000) == 20 and val_target(10, 2) == 1
