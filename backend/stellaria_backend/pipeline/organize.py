"""Organizing raw imports into training clips.

decode → (vocal separation) → (dereverb: the dry voice) → loudness normalize → slice (engine
Slicer) → split over-long slices at quiet points → drop short/silent/clipped → peak safety →
dedupe → name → write → health check (metrics, flags, validation pick, score).

The slicing preview (``preview.py``) runs the same ``normalize`` → ``plan_pieces`` →
``drop_reason`` steps, so what it shows is what organizing produces.

Clip name patterns use the tokens ``{name}`` (character name), ``{index}`` (with an optional
format such as ``{index:03}`` or ``{index:04d}``) and ``{source}`` (the raw file's stem).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..audio.dsp import active_rms_db
from ..audio.io import SR, decode, read_wav, write_wav
from ..errors import ApiError
from ..jobs import JobContext
from ..quality.analysis import clipping_ratio, pcm_hash, silence_stats
from ..quality.checker import run_health
from ..schemas import Clip, OrganizeOptions, RawImport, SliceOptions, msg
from ..util.fsutil import safe_filename
from ..util.ids import new_id
from ..util.timeutil import now_iso
from . import rawcache
from .slicing import split_long, voiced_segments

if TYPE_CHECKING:
    from ..context import AppContext

PEAK_LIMIT = 10 ** (-1 / 20)  # -1 dBFS
MAX_GAIN_DB = 30.0
CLIPPING_DROP = 0.001
SILENT_RATIO = 0.6
SILENT_RMS_DB = -45.0


# ---- names ----------------------------------------------------------------------------------
def format_name(pattern: str, name: str, index: int, source: str) -> str:
    return safe_filename(pattern.format(name=name, index=index, source=source or name))


def check_pattern(pattern: str) -> None:
    try:
        sample = pattern.format(name="x", index=1, source="y")
    except (KeyError, IndexError, ValueError) as exc:
        raise ApiError("organize.bad_pattern", f"Invalid name pattern: {exc}", status=422) from exc
    if not sample or safe_filename(sample) != sample:
        raise ApiError("organize.bad_pattern", "The name pattern makes invalid file names", status=422)
    if "{index" not in pattern:
        raise ApiError("organize.bad_pattern", "The name pattern needs an {index}", status=422)


class ClipNamer:
    def __init__(self, pattern: str, base: str, taken: set[str], start: int) -> None:
        self.pattern, self.base, self.taken, self.index = pattern, base, taken, start

    def next(self, source: str) -> str:
        while True:
            name = format_name(self.pattern, self.base, self.index, source)
            self.index += 1
            if name not in self.taken:
                self.taken.add(name)
                return name


# ---- shared with the preview ------------------------------------------------------------------
def normalize(x: np.ndarray, enabled: bool, target_db: float) -> np.ndarray:
    """Loudness normalization of a whole raw file (RMS over its sung parts)."""
    if not enabled or not len(x):
        return x
    gain_db = float(np.clip(target_db - active_rms_db(x, SR), -MAX_GAIN_DB, MAX_GAIN_DB))
    return x * np.float32(10 ** (gain_db / 20))


def plan_pieces(x: np.ndarray, slice_opts: SliceOptions, max_length: float, min_duration: float,
                engine_root: Path) -> list[tuple[int, int]]:
    """Sample ranges: engine Slicer, then over-long slices split at quiet points."""
    pieces: list[tuple[int, int]] = []
    for start, end in voiced_segments(x, SR, slice_opts, engine_root):
        pieces.extend(split_long(x, SR, start, end, max_length, min_duration))
    return pieces


def drop_reason(seg: np.ndarray, source_seg: np.ndarray, min_duration: float, drop_silent: bool,
                drop_clipping: bool) -> str | None:
    """Why organizing drops a piece (``too_short`` | ``clipping`` | ``silent``), or None.
    Clipping is judged on the audio before normalization."""
    if len(seg) / SR < min_duration:
        return "too_short"
    if drop_clipping and clipping_ratio(source_seg) >= CLIPPING_DROP:
        return "clipping"
    if drop_silent:
        silence_ratio, rms_db = silence_stats(seg, SR)
        if silence_ratio > SILENT_RATIO or rms_db < SILENT_RMS_DB:
            return "silent"
    return None


# ---- the organize job --------------------------------------------------------------------------
def organize(ctx: "AppContext", jctx: JobContext, cid: str, opts: OrganizeOptions) -> dict[str, Any]:
    chars = ctx.characters
    check_pattern(opts.name_pattern)
    if opts.separate_vocals:
        ctx.separation.require()
    if opts.dereverb:
        ctx.separation.require_cleanup()
    rec = chars.get(cid)
    raws = chars.raws(cid)
    if opts.raw_ids is None:
        targets = [r for r in raws if r.status in ("imported", "failed")]
    else:
        targets = [r for r in raws if r.id in set(opts.raw_ids)]
    clips = chars.clips(cid)
    namer = ClipNamer(opts.name_pattern, safe_filename(rec.name, "clip"), {c.name for c in clips}, len(clips) + 1)
    hashes = {_hash_file(chars.clip_path(cid, c)) for c in clips if c.status == "active" and opts.dedupe}
    dropped = {"tooShort": 0, "silent": 0, "clipping": 0, "duplicate": 0}
    new_ids: list[str] = []
    span = 0.7
    for i, raw in enumerate(targets):
        jctx.check_cancelled()
        jctx.progress(span * i / len(targets), msg("stage.organize", name=raw.name, index=i + 1, total=len(targets)))
        _set_raw(ctx, cid, raw.id, status="organizing", error=None)
        try:
            made = _organize_raw(ctx, jctx, cid, raw, opts, namer, hashes, dropped,
                                 (span * i / len(targets), span * (i + 1) / len(targets)))
        except ApiError as exc:
            _set_raw(ctx, cid, raw.id, status="failed", error=msg(exc.code, **exc.params))
            jctx.log(f"{raw.name}: {exc.message}")
            continue
        with chars.lock(cid):
            current = chars.clips(cid)
            current.extend(made)
            chars.save_clips(cid, current)
        new_ids.extend(c.id for c in made)
        raw_clip_count = sum(1 for c in chars.clips(cid) if c.raw_id == raw.id)
        _set_raw(ctx, cid, raw.id, status="organized", error=None, clip_count=raw_clip_count)
        jctx.log(f"{raw.name}: {len(made)} clips")
    jctx.stage("stage.health", span)
    report = run_health(ctx, jctx, cid, repick_val=True, val_count=opts.val_count, min_duration=opts.min_duration,
                        span=(span, 1.0))
    return {"clipIds": new_ids, "dropped": dropped, "rawIds": [r.id for r in targets],
            "health": {"score": report.score, "level": report.level}}


def _voice(ctx: "AppContext", jctx: JobContext, cid: str, raw: RawImport, opts: OrganizeOptions,
           span: tuple[float, float]) -> np.ndarray:
    """What gets sliced: the raw file, its separated vocals (cached for the preview), and with
    ``dereverb`` the dry version of either (the separation takes the first 60 % of ``span``)."""
    if not opts.separate_vocals and not opts.dereverb:
        return rawcache.decoded(ctx, cid, raw)
    lo, hi = span
    mid = (lo + (hi - lo) * 0.6 if opts.dereverb else hi) if opts.separate_vocals else lo
    device = ctx.separation.side_device()
    tmp = Path(tempfile.mkdtemp(prefix="sep-", dir=ctx.tmp_dir))
    try:
        voice = tmp / "input.wav"
        write_wav(voice, decode(rawcache.raw_path(ctx, cid, raw), SR, mono=False), SR)
        if opts.separate_vocals:
            ctx.separation.separate(jctx, voice, tmp / "vocals.wav", tmp / "inst.wav", (lo, mid), device=device)
            voice = tmp / "vocals.wav"
            vocals = decode(voice, SR)
            rawcache.save_vocals(ctx, cid, raw.id, vocals)
            if not opts.dereverb:
                return vocals
        ctx.separation.cleanup(jctx, voice, tmp / "dry.wav", span=(mid, hi), device=device)
        return decode(tmp / "dry.wav", SR)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _organize_raw(ctx: "AppContext", jctx: JobContext, cid: str, raw: RawImport, opts: OrganizeOptions,
                  namer: ClipNamer, hashes: set[str], dropped: dict[str, int], span: tuple[float, float]) -> list[Clip]:
    chars = ctx.characters
    if not rawcache.raw_path(ctx, cid, raw).is_file():
        raise ApiError("organize.raw_missing", f"The original of '{raw.name}' is missing", status=409)
    source = _voice(ctx, jctx, cid, raw, opts, span)
    x = normalize(source, opts.normalize, opts.target_db)
    made: list[Clip] = []
    counters = {"too_short": "tooShort", "silent": "silent", "clipping": "clipping"}
    for start, end in plan_pieces(x, opts.slice, opts.max_length, opts.min_duration, ctx.options.engine):
        seg = x[start:end]
        reason = drop_reason(seg, source[start:end], opts.min_duration, opts.drop_silent, opts.drop_clipping)
        if reason:
            dropped[counters[reason]] += 1
            continue
        peak = float(np.max(np.abs(seg))) if len(seg) else 0.0
        if peak > PEAK_LIMIT:
            seg = seg * np.float32(PEAK_LIMIT / peak)
        digest = pcm_hash(seg)
        if opts.dedupe and digest in hashes:
            dropped["duplicate"] += 1
            continue
        hashes.add(digest)
        name = namer.next(raw.name)
        write_wav(chars.clips_dir(cid) / f"{name}.wav", seg, SR)
        made.append(Clip(
            id=new_id("clp"), name=name, file=f"clips/{name}.wav", raw_id=raw.id, raw_name=raw.name,
            start=round(start / SR, 3), end=round(end / SR, 3), duration=round((end - start) / SR, 3),
            created_at=now_iso(),
        ))
    return made


def _hash_file(path: Path) -> str:
    try:
        x, _ = read_wav(path)
        return pcm_hash(x)
    except Exception:  # noqa: BLE001 - unreadable file cannot be a duplicate
        return ""


def _set_raw(ctx: "AppContext", cid: str, raw_id: str, **fields: Any) -> None:
    with ctx.characters.lock(cid):
        raws = ctx.characters.raws(cid)
        for r in raws:
            if r.id == raw_id:
                for k, v in fields.items():
                    setattr(r, k, v)
        ctx.characters.save_raws(cid, raws)
    ctx.emit_clips(cid)
