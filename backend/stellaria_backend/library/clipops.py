"""Clip editing: patch, split, rename/renumber, empty trash, validation pick, raw deletion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..audio.io import read_wav, write_wav
from ..errors import ApiError, not_found
from ..pipeline import rawcache
from ..pipeline.organize import check_pattern, format_name
from ..quality.valpick import apply_val, pick_val
from ..schemas import Clip
from ..util.fsutil import safe_filename
from ..util.ids import new_id
from ..util.timeutil import now_iso

if TYPE_CHECKING:
    from ..context import AppContext

PATCHABLE = {"status", "split", "reviewed", "name"}


def _find(clips: list[Clip], clip_id: str) -> Clip:
    for c in clips:
        if c.id == clip_id:
            return c
    raise not_found("clip", clip_id)


def _rename_file(ctx: "AppContext", cid: str, clip: Clip, new_name: str, taken: set[str]) -> None:
    name = new_name.strip()
    if not name or safe_filename(name) != name:
        raise ApiError("clip.bad_name", "This name cannot be used as a file name", status=422)
    if name == clip.name:
        return
    if name in taken:
        raise ApiError("clip.name_taken", f"A clip named '{name}' already exists", status=409)
    src = ctx.characters.clip_path(cid, clip)
    clip.file = f"clips/{name}.wav"
    src.rename(ctx.characters.clip_path(cid, clip))
    taken.discard(clip.name)
    taken.add(name)
    clip.name = name


def patch_clips(ctx: "AppContext", cid: str, ids: list[str], changes: dict[str, Any]) -> list[Clip]:
    unknown = set(changes) - PATCHABLE
    if unknown:
        raise ApiError("request.invalid", f"Cannot change {sorted(unknown)}", status=422)
    if "name" in changes and len(ids) != 1:
        raise ApiError("request.invalid", "Rename one clip at a time", status=422)
    with ctx.characters.lock(cid):
        clips = ctx.characters.clips(cid)
        taken = {c.name for c in clips}
        changed = []
        for clip_id in ids:
            clip = _find(clips, clip_id)
            if "name" in changes:
                _rename_file(ctx, cid, clip, str(changes["name"]), taken)
            for key in ("status", "split", "reviewed"):
                if key in changes:
                    setattr(clip, key, changes[key])
            changed.append(Clip.model_validate(clip.wire()))  # validates enum values
        ctx.characters.save_clips(cid, clips)
    ctx.emit_clips(cid)
    return changed


def split_clip(ctx: "AppContext", cid: str, clip_id: str, at: float) -> list[Clip]:
    with ctx.characters.lock(cid):
        clips = ctx.characters.clips(cid)
        clip = _find(clips, clip_id)
        x, sr = read_wav(ctx.characters.clip_path(cid, clip))
        cut = int(round(at * sr))
        if cut < int(0.1 * sr) or cut > len(x) - int(0.1 * sr):
            raise ApiError("clip.bad_split", "Split point too close to the clip edges", status=422)
        taken = {c.name for c in clips}
        parts = []
        for suffix, (s, e) in zip(("a", "b"), ((0, cut), (cut, len(x)))):
            name = f"{clip.name}_{suffix}"
            n = 2
            while name in taken:
                name = f"{clip.name}_{suffix}{n}"
                n += 1
            taken.add(name)
            write_wav(ctx.characters.clips_dir(cid) / f"{name}.wav", x[s:e], sr)
            offset = clip.start if clip.start is not None else None
            parts.append(Clip(
                id=new_id("clp"), name=name, file=f"clips/{name}.wav", raw_id=clip.raw_id, raw_name=clip.raw_name,
                start=None if offset is None else round(offset + s / sr, 3),
                end=None if offset is None else round(offset + e / sr, 3),
                duration=round((e - s) / sr, 3), split=clip.split, created_at=now_iso(),
            ))
        clip.status = "trashed"
        clips.extend(parts)
        ctx.characters.save_clips(cid, clips)
    ctx.emit_clips(cid)
    return parts


def _delete_clip_files(ctx: "AppContext", cid: str, clip: Clip) -> None:
    ctx.characters.clip_path(cid, clip).unlink(missing_ok=True)
    (ctx.characters.cache_dir(cid) / "analysis" / f"{clip.id}.npz").unlink(missing_ok=True)


def empty_trash(ctx: "AppContext", cid: str) -> int:
    with ctx.characters.lock(cid):
        clips = ctx.characters.clips(cid)
        trashed = [c for c in clips if c.status == "trashed"]
        for c in trashed:
            _delete_clip_files(ctx, cid, c)
        ctx.characters.save_clips(cid, [c for c in clips if c.status != "trashed"], bump=False)
    ctx.emit_clips(cid)
    return len(trashed)


def renumber(ctx: "AppContext", cid: str, pattern: str | None) -> list[Clip]:
    """Rename active clips 1..n (in import order), then trashed ones after them.
    Tokens: ``{name}``, ``{index}`` (``{index:03}``, ``{index:04}`` ...), ``{source}``."""
    pattern = pattern or "{name}_{index:04d}"
    check_pattern(pattern)
    with ctx.characters.lock(cid):
        rec = ctx.characters.get(cid)
        clips = ctx.characters.clips(cid)
        order = sorted(clips, key=lambda c: (c.status != "active", c.created_at, c.raw_id or "", c.start or 0.0))
        base = safe_filename(rec.name, "clip")
        targets = {c.id: format_name(pattern, base, i + 1, c.raw_name or "") for i, c in enumerate(order)}
        if len(set(targets.values())) != len(targets):
            raise ApiError("organize.bad_pattern", "The pattern does not give unique names", status=422)
        # two phases so that swapping names never overwrites a file
        for c in clips:
            tmp = f".renaming-{c.id}"
            ctx.characters.clip_path(cid, c).rename(ctx.characters.clips_dir(cid) / f"{tmp}.wav")
            c.file = f"clips/{tmp}.wav"
        for c in clips:
            src = ctx.characters.clip_path(cid, c)
            c.name = targets[c.id]
            c.file = f"clips/{c.name}.wav"
            src.rename(ctx.characters.clip_path(cid, c))
        ctx.characters.save_clips(cid, clips, bump=False)
    ctx.emit_clips(cid)
    return [c for c in clips if c.status == "active"]


def repick_val(ctx: "AppContext", cid: str, count: int | None) -> list[Clip]:
    with ctx.characters.lock(cid):
        clips = ctx.characters.clips(cid)
        changed = apply_val(clips, pick_val(clips, count if count is not None else 10))
        ctx.characters.save_clips(cid, clips, bump=bool(changed))
    ctx.emit_clips(cid)
    return [c for c in clips if c.status == "active"]


def delete_raw(ctx: "AppContext", cid: str, raw_id: str, with_clips: bool) -> None:
    with ctx.characters.lock(cid):
        raws = ctx.characters.raws(cid)
        raw = next((r for r in raws if r.id == raw_id), None)
        if raw is None:
            raise not_found("raw", raw_id)
        rawcache.raw_path(ctx, cid, raw).unlink(missing_ok=True)
        rawcache.forget(ctx, cid, raw.id)
        ctx.characters.save_raws(cid, [r for r in raws if r.id != raw_id])
        if with_clips:
            clips = ctx.characters.clips(cid)
            for c in clips:
                if c.raw_id == raw_id:
                    _delete_clip_files(ctx, cid, c)
            ctx.characters.save_clips(cid, [c for c in clips if c.raw_id != raw_id])
    ctx.emit_clips(cid)
