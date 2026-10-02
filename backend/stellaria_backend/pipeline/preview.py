"""Slicing preview for one raw import: same normalize → slice → split → drop rules as organize,
without writing anything. Synchronous; works from the decoded-audio cache."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from pydantic import Field

from ..audio.io import SR
from ..errors import ApiError, not_found
from ..schemas import OrganizeOptions, SliceOptions, Wire
from . import rawcache
from .organize import drop_reason, normalize, plan_pieces

if TYPE_CHECKING:
    from ..context import AppContext

_DEFAULTS = OrganizeOptions()


class PreviewRequest(Wire):
    raw_id: str
    slice: SliceOptions = Field(default_factory=SliceOptions)
    max_length: float = _DEFAULTS.max_length
    separate_vocals: bool = False
    n: int = 600
    # optional, so the preview matches a non-default organize run exactly
    normalize: bool = _DEFAULTS.normalize
    target_db: float = _DEFAULTS.target_db
    min_duration: float = _DEFAULTS.min_duration
    drop_silent: bool = _DEFAULTS.drop_silent
    drop_clipping: bool = _DEFAULTS.drop_clipping


class PreviewSegment(Wire):
    start: float
    end: float
    dropped: Literal["too_short", "silent", "clipping"] | None = None


def bucket_peaks(x: np.ndarray, n: int) -> list[float]:
    """Max |x| (0..1) in ``n`` equal buckets."""
    n = max(1, min(int(n), 20000))
    if not len(x):
        return [0.0] * n
    edges = np.linspace(0, len(x), n + 1).astype(np.int64)
    edges[-1] = len(x)
    ax = np.abs(x)
    starts = np.minimum(edges[:-1], len(x) - 1)
    values = np.maximum.reduceat(ax, starts)
    values[edges[1:] <= edges[:-1]] = 0.0  # empty buckets (n > samples)
    return [round(float(v), 4) for v in np.clip(values, 0, 1)]


def preview(ctx: "AppContext", cid: str, req: PreviewRequest) -> dict[str, Any]:
    raw = next((r for r in ctx.characters.raws(cid) if r.id == req.raw_id), None)
    if raw is None:
        if ctx.jobs.find_active("import", characterId=cid):
            raise ApiError("raw.not_ready", "The file is still being imported", status=409)
        raise not_found("raw", req.raw_id)
    if raw.status == "failed" and raw.error is not None:
        raise ApiError(raw.error.code, f"'{raw.name}' could not be imported", status=422)
    vocals = rawcache.cached_vocals(ctx, cid, raw.id) if req.separate_vocals else None
    source = vocals if vocals is not None else rawcache.decoded(ctx, cid, raw)
    x = normalize(source, req.normalize, req.target_db)
    segments = []
    for start, end in plan_pieces(x, req.slice, req.max_length, req.min_duration, ctx.options.engine):
        reason = drop_reason(x[start:end], source[start:end], req.min_duration, req.drop_silent, req.drop_clipping)
        segments.append(PreviewSegment(start=round(start / SR, 3), end=round(end / SR, 3), dropped=reason).wire())
    return {"duration": round(len(source) / SR, 3), "peaks": bucket_peaks(source, req.n), "segments": segments,
            "separated": vocals is not None}
