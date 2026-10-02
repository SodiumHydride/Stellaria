"""Import jobs: raw material for characters, media for projects."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..audio.io import AUDIO_EXTS, SR, decode, write_wav
from ..errors import ApiError
from ..jobs import JobContext
from ..schemas import Media, RawImport, msg
from ..util.ids import new_id
from ..util.timeutil import now_iso
from . import rawcache

if TYPE_CHECKING:
    from ..context import AppContext


def expand_paths(paths: list[Path]) -> list[Path]:
    """Files as given plus audio files found (recursively, sorted) in given folders."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            for root, _, files in os.walk(p):
                out.extend(Path(root) / f for f in sorted(files) if Path(f).suffix.lower() in AUDIO_EXTS)
        else:
            out.append(p)
    return out


def import_raw(ctx: "AppContext", jctx: JobContext, cid: str, paths: list[Path], names: dict[Path, str],
               cleanup: list[Path]) -> dict[str, Any]:
    """Copy originals into ``raw/`` (never modified afterwards) and validate them."""
    try:
        files = expand_paths(paths)
        if not files:
            raise ApiError("import.no_files", "No audio files found", status=422)
        raw_ids = []
        for i, src in enumerate(files):
            jctx.check_cancelled()
            name = names.get(src, src.name)
            jctx.progress(i / len(files), msg("stage.import", name=name, index=i + 1, total=len(files)))
            raw = RawImport(id=new_id("raw"), name=Path(name).stem, ext=Path(name).suffix.lower().lstrip("."),
                            size=0, imported_at=now_iso())
            try:
                if not src.is_file():
                    raise ApiError("import.not_found", f"'{src}' does not exist", status=422)
                dest = ctx.characters.raw_dir(cid) / f"{raw.id}.{raw.ext or 'bin'}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
                raw.size = dest.stat().st_size
                samples = rawcache.decoded(ctx, cid, raw)  # validates the whole file and warms the preview
                raw.duration = round(len(samples) / SR, 3) or None
            except ApiError as exc:
                raw.status, raw.error = "failed", msg(exc.code)
                jctx.log(f"{name}: {exc.message}")
            with ctx.characters.lock(cid):
                raws = ctx.characters.raws(cid)
                raws.append(raw)
                ctx.characters.save_raws(cid, raws)
            raw_ids.append(raw.id)
            ctx.emit_clips(cid)
        return {"rawIds": raw_ids}
    finally:
        for p in cleanup:
            shutil.rmtree(p, ignore_errors=True)


def import_media(ctx: "AppContext", jctx: JobContext, pid: str, paths: list[Path], names: dict[Path, str],
                 cleanup: list[Path], kind: str = "import") -> dict[str, Any]:
    """Convert files to 44.1 kHz wav (mono or stereo) under ``media/``."""
    try:
        files = expand_paths(paths)
        if not files:
            raise ApiError("import.no_files", "No audio files found", status=422)
        media_ids = []
        for i, src in enumerate(files):
            jctx.check_cancelled()
            name = names.get(src, src.name)
            jctx.progress(i / len(files), msg("stage.import", name=name, index=i + 1, total=len(files)))
            data = decode(src, SR, mono=False)
            media = add_media_audio(ctx, pid, data, Path(name).stem, kind)
            media_ids.append(media.id)
        return {"mediaIds": media_ids}
    finally:
        for p in cleanup:
            shutil.rmtree(p, ignore_errors=True)


def add_media_audio(ctx: "AppContext", pid: str, data, name: str, kind: str) -> Media:
    channels = 1 if data.ndim == 1 else data.shape[1]
    media = Media(id=new_id("med"), name=name, kind=kind, duration=round(len(data) / SR, 3), sample_rate=SR,
                  channels=channels, created_at=now_iso())
    write_wav(ctx.projects.media_path(pid, media.id), data, SR)
    return register_media(ctx, pid, media)


def register_media(ctx: "AppContext", pid: str, media: Media) -> Media:
    ctx.projects.add_media(pid, media)
    ctx.hub.publish("project.media", {"projectId": pid, "media": media.wire()})
    return media
