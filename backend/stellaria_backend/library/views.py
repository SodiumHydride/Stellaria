"""Computed wire shapes: ``Character`` and ``Model`` (stats, stage, health, steps, disk...)."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..engine import checkpoints, dataset
from ..schemas import (
    Character, CharacterHealthSummary, CharacterStats, DatasetInfo, DiskUsage, Model, ModelRecord, PitchRange,
)
from ..util.fsutil import dir_size

if TYPE_CHECKING:
    from ..context import AppContext

TRAINING_STATUSES = {"preparing", "preprocessing", "training", "stopping"}
_disk_cache: dict[str, tuple[float, DiskUsage]] = {}
DISK_TTL = 30.0


def character_view(ctx: "AppContext", cid: str, models: list[ModelRecord] | None = None) -> Character:
    chars = ctx.characters
    rec = chars.get(cid)
    raws = chars.raws(cid)
    clips = chars.clips(cid)
    active = [c for c in clips if c.status == "active"]
    report, revision = chars.health(cid)
    models = models if models is not None else ctx.models.list_records()
    mine = [m for m in models if any(mc.character_id == cid for mc in m.characters)]
    single = sorted((m for m in mine if m.kind == "single"), key=lambda m: m.updated_at, reverse=True)

    if any(m.status in TRAINING_STATUSES for m in mine):
        stage = "training"
    elif any(m.releases or checkpoints.has_trained(ctx.models.exp_dir(m.id)) for m in mine):
        stage = "trained"
    elif report is not None and not report.blocking and active:
        stage = "ready"
    elif raws or clips:
        stage = "material"
    else:
        stage = "empty"

    health = None
    pitch = None
    if report is not None:
        health = CharacterHealthSummary(score=report.score, level=report.level, checked_at=report.checked_at,
                                        stale=revision != rec.clips_revision)
        if report.pitch.low and report.pitch.high and report.pitch.median:
            pitch = PitchRange(low=report.pitch.low, high=report.pitch.high, median=report.pitch.median)
    return Character(
        **rec.model_dump(exclude={"clips_revision"}),
        stage=stage,
        stats=CharacterStats(
            raw_count=len(raws), clip_count=len(active), active_duration=round(sum(c.duration for c in active), 3),
            train_count=sum(1 for c in active if c.split == "train"),
            val_count=sum(1 for c in active if c.split == "val"),
            trashed_count=sum(1 for c in clips if c.status == "trashed"),
        ),
        health=health, model_ids=[m.id for m in mine], primary_model_id=single[0].id if single else None, pitch=pitch,
    )


def disk_usage(ctx: "AppContext", mid: str, fresh: bool = False) -> DiskUsage:
    cached = _disk_cache.get(mid)
    if cached and not fresh and time.monotonic() - cached[0] < DISK_TTL:
        return cached[1]
    ds = dir_size(ctx.models.dataset_dir(mid))
    ck = sum(p.stat().st_size for p in checkpoints.checkpoint_steps(ctx.models.exp_dir(mid)).values() if p.exists())
    rel = dir_size(ctx.models.releases_dir(mid))
    usage = DiskUsage(dataset=ds, checkpoints=ck, releases=rel, total=dir_size(ctx.models.dir(mid)))
    _disk_cache[mid] = (time.monotonic(), usage)
    return usage


def model_view(ctx: "AppContext", mid: str, with_disk: bool = True) -> Model:
    rec = ctx.models.get(mid)
    exp = ctx.models.exp_dir(mid)
    ckpts = checkpoints.list_checkpoints(rec, exp)
    runtime = ctx.training.runtime(mid)
    live = runtime is not None and rec.status in TRAINING_STATUSES
    step = max(runtime.step, ckpts[-1].step if ckpts else 0) if live else (ckpts[-1].step if ckpts else 0)
    ds_status, manifest = dataset.status(rec, ctx.characters, ctx.models.dataset_dir(mid))
    ds = DatasetInfo(status=ds_status)
    if manifest:
        ds.prepared_at = manifest.get("preparedAt")
        ds.clip_count = int(manifest.get("clipCount", 0))
        ds.duration = float(manifest.get("duration", 0.0))
    return Model(
        **rec.model_dump(exclude={"releases"}),
        releases=rec.releases, step=step, dataset=ds, checkpoints=ckpts,
        speed=runtime.speed if live else None, eta=runtime.eta if live else None,
        disk_usage=disk_usage(ctx, mid) if with_disk else None,
    )
