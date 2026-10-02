"""Decoded raw imports cached as ``<character>/cache/decoded/<raw_id>.npy`` (44.1 kHz mono float32).

Decoding an mp3 with ffmpeg takes about a second for a song; the slicing preview and organize
both start from the decoded samples, so they share this cache. Vocals separated by organize are
cached next to it (``<raw_id>.vocals.npy``) for the preview.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..audio.io import SR, decode
from ..errors import ApiError
from ..schemas import RawImport

if TYPE_CHECKING:
    from ..context import AppContext


def raw_path(ctx: "AppContext", cid: str, raw: RawImport) -> Path:
    return ctx.characters.raw_dir(cid) / f"{raw.id}.{raw.ext or 'bin'}"


def _cache_path(ctx: "AppContext", cid: str, raw_id: str, kind: str = "") -> Path:
    return ctx.characters.cache_dir(cid) / "decoded" / f"{raw_id}{'.' + kind if kind else ''}.npy"


def _save(path: Path, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, x.astype(np.float32, copy=False))
    tmp.replace(path)


def decoded(ctx: "AppContext", cid: str, raw: RawImport) -> np.ndarray:
    src = raw_path(ctx, cid, raw)
    if not src.is_file():
        raise ApiError("raw.not_ready", f"'{raw.name}' is not available yet", status=409)
    cache = _cache_path(ctx, cid, raw.id)
    if cache.is_file() and cache.stat().st_mtime >= src.stat().st_mtime:
        return np.load(cache)
    x = decode(src, SR)
    _save(cache, x)
    return x


def cached_vocals(ctx: "AppContext", cid: str, raw_id: str) -> np.ndarray | None:
    path = _cache_path(ctx, cid, raw_id, "vocals")
    return np.load(path) if path.is_file() else None


def save_vocals(ctx: "AppContext", cid: str, raw_id: str, x: np.ndarray) -> None:
    _save(_cache_path(ctx, cid, raw_id, "vocals"), x)


def forget(ctx: "AppContext", cid: str, raw_id: str) -> None:
    for kind in ("", "vocals"):
        _cache_path(ctx, cid, raw_id, kind).unlink(missing_ok=True)
