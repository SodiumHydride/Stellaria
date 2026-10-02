"""Song analysis for the canvas: ``/media/{id}/analysis`` (+ ``key``), ``/media/{id}/sections``,
and bpm/key filled into song nodes in the background after an import.

The heavy lifting lives in ``dsp.analysis`` (``sections(path, *, vocals_path=None)``,
``key(path)``) and ``mixing.analyze``; results are cached beside the media
(``<media>.analysis.json``, ``<media>.sections.json``). Until ``dsp.analysis`` is installed,
``key`` is null and sections are empty lists (bpm from the mixer's analysis).
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ApiError
from .mix import analyze_media

if TYPE_CHECKING:
    from ..context import AppContext

log = logging.getLogger(__name__)


def _fresh(cache: Path, *sources: Path) -> bool:
    if not cache.is_file():
        return False
    mtime = cache.stat().st_mtime
    return all(not s.is_file() or s.stat().st_mtime <= mtime for s in sources)


def _dsp_analysis():
    try:
        from ..dsp import analysis
    except ImportError:
        return None
    return analysis


def media_analysis(path: Path) -> dict[str, Any]:
    """``{bpm, lufs, peakDb, duration, key}`` (cached)."""
    cache = path.with_suffix(".analysis.json")
    data = dict(analyze_media(path))
    if "key" in data:
        return data
    dsp = _dsp_analysis()
    if dsp is None or not hasattr(dsp, "key"):
        return {**data, "key": None}
    try:
        data["key"] = dsp.key(str(path))
    except Exception:  # noqa: BLE001 - a key is a nicety
        log.exception("key detection failed for %s", path)
        data["key"] = None
    cache.write_text(json.dumps(data), encoding="utf-8")
    return data


def cached_analysis(path: Path) -> dict[str, Any] | None:
    cache = path.with_suffix(".analysis.json")
    if not _fresh(cache, path):
        return None
    data = json.loads(cache.read_text(encoding="utf-8"))
    return data if "key" in data else None


def song_vocals(ctx: "AppContext", pid: str, media_id: str) -> Path | None:
    """The vocals stem of a song node playing this media, if it has one."""
    for node in ctx.projects.get(pid).canvas.get("nodes", []):
        data = node.get("data") or {}
        if node.get("type") == "source" and data.get("mediaId") == media_id:
            vocals = (data.get("stems") or {}).get("vocals")
            if vocals:
                return ctx.projects.media_path(pid, vocals)
    return None


def sections(ctx: "AppContext", pid: str, media_id: str) -> dict[str, Any]:
    """``{bpm, beats, downbeats, phrases, sections}`` for split/comp/trim editors (cached)."""
    ctx.projects.get_media(pid, media_id)
    path = ctx.projects.media_path(pid, media_id)
    vocals = song_vocals(ctx, pid, media_id)
    cache = path.with_suffix(".sections.json")
    if _fresh(cache, path, *([vocals] if vocals else [])):
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if cached.get("vocals") == (vocals.name if vocals else None):
            return cached["data"]
    dsp = _dsp_analysis()
    if dsp is None or not hasattr(dsp, "sections"):
        return {"bpm": analyze_media(path).get("bpm"), "beats": [], "downbeats": [], "phrases": [], "sections": []}
    try:
        data = dsp.sections(str(path), vocals_path=str(vocals) if vocals else None)
    except ApiError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ApiError("analysis.failed", "Could not analyse this audio", status=500, detail=str(exc)) from exc
    cache.write_text(json.dumps({"vocals": vocals.name if vocals else None, "data": data}), encoding="utf-8")
    return data


class SongAnalysis:
    """Fills ``bpm``/``key`` of song nodes in the background (one analysis at a time)."""

    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="song-analysis")
        self._queued: set[tuple[str, str]] = set()
        self._again: set[tuple[str, str]] = set()  # asked again while queued: look for new song nodes once more
        self._lock = threading.Lock()

    def schedule(self, pid: str, media_id: str) -> None:
        with self._lock:
            if (pid, media_id) in self._queued:
                self._again.add((pid, media_id))
                return
            self._queued.add((pid, media_id))
        try:
            self._pool.submit(self._run, pid, media_id)
        except RuntimeError:  # shutting down
            pass

    def fill_new_sources(self, pid: str, canvas: dict[str, Any], stored: dict[str, Any]) -> None:
        """For song nodes a save adds: take bpm/key from the cache now, or analyse in the background."""
        known = {n.get("id") for n in stored.get("nodes", [])}
        for node in canvas.get("nodes", []):
            data = node.get("data") or {}
            if node.get("type") != "source" or node.get("id") in known or not data.get("mediaId"):
                continue
            if data.get("bpm") is not None and data.get("key") is not None:
                continue
            cached = cached_analysis(self.ctx.projects.media_path(pid, data["mediaId"]))
            if cached is not None:
                data["bpm"], data["key"] = _bpm(cached.get("bpm")), cached.get("key")
            else:
                self.schedule(pid, data["mediaId"])

    def _fill(self, pid: str, media_id: str, bpm: float | None, key: str | None) -> None:
        if not self.ctx.projects.dir(pid).is_dir():
            return
        for node in self.ctx.projects.get(pid).canvas.get("nodes", []):
            data = node.get("data") or {}
            if node.get("type") == "source" and data.get("mediaId") == media_id \
                    and (data.get("bpm"), data.get("key")) != (bpm, key):
                self.ctx.node_updates.update(self.ctx, pid, node["id"], bpm=bpm, key=key)

    def _run(self, pid: str, media_id: str) -> None:
        job = (pid, media_id)
        try:
            path = self.ctx.projects.media_path(pid, media_id)
            if not path.is_file():
                return
            result = media_analysis(path)
            bpm, key = _bpm(result.get("bpm")), result.get("key")
            while True:
                self._fill(pid, media_id, bpm, key)
                with self._lock:
                    if job not in self._again:
                        self._queued.discard(job)
                        return
                    self._again.discard(job)
        except Exception:  # noqa: BLE001 - background nicety
            log.exception("song analysis failed (%s, %s)", pid, media_id)
        finally:
            with self._lock:
                self._queued.discard(job)
                self._again.discard(job)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


def _bpm(value: Any) -> float | None:
    return round(float(value), 2) if isinstance(value, (int, float)) and value > 0 else None

