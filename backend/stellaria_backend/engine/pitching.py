"""The pitch editor's jobs: the pitch line of a render node's input (``pitch_edit.analysis``) and
quick previews of a phrase sung with the current edits.

The analysis is kept per node in ``<project>/pitch/<node>.json`` with what it was made from (input
media and pitch tracking settings), so a changed input is noticed. Previews are rendered into the
app's cache and never touch the node.
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..creation.canvas import Graph
from ..errors import ApiError, not_found
from ..jobs import Job, JobContext
from ..jobs.manager import PLACEMENT_CPU
from ..schemas import msg
from ..util.jsonio import read_json, write_json
from . import pitch_edit
from .rendering import RENDER_LOCK, _node, check_checkpoint, parse_render_params, run_render

if TYPE_CHECKING:
    from ..context import AppContext

PREVIEW_MAX_S = 60.0
PREVIEWS_KEPT = 6       # per project: each preview is a few MB and only the latest ones get played


def analysis_path(ctx: "AppContext", pid: str, node_id: str) -> Path:
    return ctx.projects.dir(pid) / "pitch" / f"{node_id}.json"


def preview_path(ctx: "AppContext", pid: str, job_id: str) -> Path:
    return ctx.options.data / "cache" / "pitch-preview" / pid / f"{job_id}.wav"


def prune_previews(ctx: "AppContext", pid: str, keep: int = PREVIEWS_KEPT) -> None:
    """Keeps only a project's newest ``keep`` preview renders."""
    folder = ctx.options.data / "cache" / "pitch-preview" / pid
    dated = []
    for f in folder.glob("*.wav"):
        try:
            dated.append((f.stat().st_mtime, f))
        except OSError:
            continue
    for _, old in sorted(dated, reverse=True)[keep:]:
        old.unlink(missing_ok=True)


def forget_project(ctx: "AppContext", pid: str) -> None:
    """A deleted project's pitch caches (previews, raw pitch lines)."""
    shutil.rmtree(ctx.options.data / "cache" / "pitch-preview" / pid, ignore_errors=True)
    for f in (ctx.options.data / "cache" / "pitch").glob(f"{pid}_*.npy"):
        f.unlink(missing_ok=True)


def _render_node(ctx: "AppContext", pid: str, node_id: str) -> tuple[dict[str, Any], str]:
    node = _node(ctx, pid, node_id)
    if node.get("type") != "render":
        raise ApiError("node.not_render", "Only render nodes have a pitch line", status=422)
    data = node.get("data", {})
    media_id = Graph(ctx.projects.get(pid).canvas).resolve(data.get("input"))
    if not media_id:
        raise ApiError("node.input_missing", "This node has no input yet", status=409,
                       params={"port": "in", "nodeId": node_id})
    return data, media_id


def _made_from(media_id: str, params: Any) -> dict[str, Any]:
    return {"mediaId": media_id, "f0Extractor": params.f0_extractor, "f0Min": params.f0_min, "f0Max": params.f0_max}


def get_analysis(ctx: "AppContext", pid: str, node_id: str) -> dict[str, Any]:
    """The stored analysis with ``current`` = still made from this input and these settings."""
    data, media_id = _render_node(ctx, pid, node_id)
    stored = read_json(analysis_path(ctx, pid, node_id), None)
    if not isinstance(stored, dict) or "midi" not in stored:
        raise ApiError("pitch.not_analyzed", "The pitch line hasn't been analysed yet", status=404)
    params = parse_render_params(data.get("params"))
    made = stored.get("madeFrom") or {}
    stored["current"] = made == _made_from(media_id, params)
    stored["offset"] = ctx.projects.get_media(pid, media_id).offset
    return stored


def submit_analysis(ctx: "AppContext", pid: str, node_id: str) -> Job:
    data, media_id = _render_node(ctx, pid, node_id)
    params = parse_render_params(data.get("params"))
    media = ctx.projects.get_media(pid, media_id)
    if params.f0_extractor == "rmvpe":
        ctx.resources.require(["rmvpe"])

    def run(jctx: JobContext) -> dict[str, Any]:
        raw = ctx.options.data / "cache" / "pitch" / f"{pid}_{node_id}.npy"
        args: dict[str, Any] = {
            "input": str(ctx.projects.media_path(pid, media_id)), "output": str(raw),
            "f0Extractor": params.f0_extractor, "f0CacheDir": str(ctx.options.data / "cache" / "f0"),
            **({"f0Min": params.f0_min} if params.f0_min is not None else {}),
            **({"f0Max": params.f0_max} if params.f0_max is not None else {}),
        }
        if jctx.job.placement == PLACEMENT_CPU:
            args["device"] = "cpu"
        jctx.stage("stage.pitch", 0.0)
        result = ctx.render.pitch(args, lambda v: jctx.progress(v, msg("stage.pitch")), cancel=jctx.job.cancel_event)
        f0 = np.load(raw, allow_pickle=False)
        raw.unlink(missing_ok=True)
        payload = pitch_edit.analysis(f0, float(result["frameS"]))
        payload["madeFrom"] = _made_from(media_id, params)
        write_json(analysis_path(ctx, pid, node_id), payload)
        return {"nodeId": node_id, "notes": len(payload["notes"])}

    return ctx.jobs.submit("pitch", msg("job.pitch", name=media.name), run,
                           target={"projectId": pid, "nodeId": node_id}, uses_gpu=True, cpu_fallback=True,
                           lock_key=RENDER_LOCK)


def submit_preview(ctx: "AppContext", pid: str, node_id: str, start: float, end: float, edit: Any) -> Job:
    """Sings ``start``–``end`` of the node's input (≤ 60 s) with its settings and ``edit``."""
    data, media_id = _render_node(ctx, pid, node_id)
    mid, cid = data.get("modelId"), data.get("characterId")
    if not mid or not cid:
        raise ApiError("node.incomplete", "Choose a character and a model first", status=422)
    media = ctx.projects.get_media(pid, media_id)
    start = max(0.0, float(start))
    end = min(float(end), media.duration)
    if not end - start > 0.05:
        raise ApiError("request.invalid", "Choose a part of the song to preview", status=422)
    end = min(end, start + PREVIEW_MAX_S)
    raw = dict(data.get("params") or {})
    raw["region"] = {"start": start, "end": end}
    raw["pitchEdit"] = edit
    params = parse_render_params(raw)
    release_id = data.get("releaseId")
    check_checkpoint(ctx, mid, release_id)
    ready = threading.Event()

    def run(jctx: JobContext) -> dict[str, Any]:
        ready.wait(10)
        out = preview_path(ctx, pid, jctx.job.id)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_render(ctx, jctx, mid, cid, release_id, params, ctx.projects.media_path(pid, media_id), out)
        prune_previews(ctx, pid)
        return {"nodeId": node_id, "start": start, "end": end,
                "url": f"/projects/{pid}/pitch-preview/{jctx.job.id}/audio"}

    job = ctx.jobs.submit("pitch", msg("job.pitch_preview", name=data.get("label") or media.name), run,
                          target={"projectId": pid, "nodeId": node_id}, uses_gpu=True, cpu_fallback=True,
                          lock_key=RENDER_LOCK)
    ready.set()
    return job


def preview_file(ctx: "AppContext", pid: str, job_id: str) -> Path:
    path = preview_path(ctx, pid, job_id)
    if not path.is_file():
        raise not_found("preview", job_id)
    return path

