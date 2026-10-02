"""Creation jobs: render nodes, quick renders, stem separation of source nodes.

Server-side node updates go through ``creation.updates.NodeUpdates`` (``project.nodes`` events,
ownership rules, stale propagation to dependents).

Renders and separations may run next to a training (see ``jobs.manager``): with placement
``cpu`` the worker renders on the CPU and the job stage is ``stage.render_cpu``. Rendering the
latest checkpoint of a model that is still training uses a stable copy of it (``torch.save``
writes in place); if that copy still cannot be read, the next older checkpoint is used.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..audio.io import SR, decode, probe, write_wav
from ..errors import ApiError, JobCancelled, not_found
from ..jobs import Job, JobContext
from ..jobs.manager import PLACEMENT_CPU
from ..pipeline.importing import register_media
from ..schemas import Media, Msg, RenderParams, msg
from ..util.ids import new_id
from ..util.timeutil import now_iso
from . import checkpoints
from ..creation.canvas import EXTRA_STEMS, Graph, empty_stems
from ..creation.updates import NodeUpdates
from .config import ENCODER_RESOURCE

if TYPE_CHECKING:
    from ..context import AppContext

RENDER_LOCK = "render-worker"  # one render at a time (the worker handles one request at once)
SEPARATE_LOCK = "separator"

def _find(canvas: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    return next((n for n in canvas.get("nodes", []) if n.get("id") == node_id), None)


def _node(ctx: "AppContext", pid: str, node_id: str) -> dict[str, Any]:
    node = _find(ctx.projects.get(pid).canvas, node_id)
    if node is None:
        raise not_found("node", node_id)
    return node


def check_checkpoint(ctx: "AppContext", mid: str, release_id: str | None) -> None:
    """Fail fast (at submit time) when there is nothing to render with."""
    if release_id:
        if not (checkpoints.release_dir(ctx.models.dir(mid), release_id) / "model.pt").is_file():
            raise not_found("release", release_id)
        return
    exp = ctx.models.exp_dir(mid)
    if checkpoints.latest_checkpoint(exp) is None or not (exp / "config.yaml").is_file():
        raise ApiError("render.no_checkpoint", "This model has no checkpoint yet", status=409)


def resolve_checkpoint(ctx: "AppContext", mid: str, release_id: str | None, skip: frozenset[int] = frozenset()) -> Path:
    """The checkpoint file to render with: a release, the latest checkpoint, or — while the
    model is training — a stable copy of the latest one."""
    check_checkpoint(ctx, mid, release_id)
    if release_id:
        return checkpoints.release_dir(ctx.models.dir(mid), release_id) / "model.pt"
    exp = ctx.models.exp_dir(mid)
    if ctx.training.active_job(mid) is None and not skip:
        return checkpoints.latest_checkpoint(exp)[1]
    snapshot = checkpoints.snapshot_latest(exp, ctx.models.cache_dir(mid) / "render", skip)
    if snapshot is None:
        raise ApiError("render.no_checkpoint", "No complete checkpoint is available yet", status=409)
    return snapshot


def render_args(ctx: "AppContext", mid: str, character_id: str, model_path: Path, params: RenderParams,
                input_path: Path, output_path: Path) -> dict[str, Any]:
    rec = ctx.models.get(mid)
    spk = {mc.character_id: mc.spk_id for mc in rec.characters}
    if character_id not in spk:
        raise ApiError("render.character_not_in_model", "This character is not part of the model", status=422)
    needed = ["nsf_hifigan", ENCODER_RESOURCE[rec.config.encoder]] + (["rmvpe"] if params.f0_extractor == "rmvpe" else [])
    ctx.resources.require(needed)
    mix = None
    if params.spk_mix and len(rec.characters) > 1:
        mix = {str(spk[c]): float(w) for c, w in params.spk_mix.items() if c in spk and w}
    return {
        "modelPath": str(model_path),
        "overrides": {"vocoderCkpt": str(ctx.resources.path("nsf_hifigan")),
                      "encoderCkpt": str(ctx.resources.path(ENCODER_RESOURCE[rec.config.encoder]))},
        "input": str(input_path), "output": str(output_path), "spkId": spk[character_id], "spkMix": mix or None,
        "key": params.key, "formant": params.formant, "register": params.vocal_register, "f0Extractor": params.f0_extractor,
        "steps": params.steps, "method": params.method, "tStart": params.t_start, "threshold": params.threshold,
        "region": [params.region.start, params.region.end] if params.region else None,
        "f0CacheDir": str(ctx.options.data / "cache" / "f0"),
        **({"pitchEdit": params.pitch_edit} if params.pitch_edit else {}),
        **({"f0Min": params.f0_min} if params.f0_min is not None else {}),
        **({"f0Max": params.f0_max} if params.f0_max is not None else {}),
    }


def parse_render_params(raw: dict[str, Any] | None) -> RenderParams:
    try:
        params = RenderParams.model_validate(raw or {})
    except Exception as exc:  # noqa: BLE001
        raise ApiError("request.invalid", "Invalid render params", status=422, detail=str(exc)) from exc
    if params.f0_min is not None and params.f0_max is not None and params.f0_min >= params.f0_max:
        raise ApiError("request.invalid", "f0Min must be below f0Max", status=422)
    return params


def run_render(ctx: "AppContext", jctx: JobContext, mid: str, character_id: str, release_id: str | None,
               params: RenderParams, input_path: Path, output_path: Path) -> dict[str, Any]:
    """Render through the worker on the job's placement; returns the worker result (with ``device``)."""
    on_cpu = jctx.job.placement == PLACEMENT_CPU
    stage = "stage.render_cpu" if on_cpu else "stage.render"
    skip: set[int] = set()
    while True:
        model_path = resolve_checkpoint(ctx, mid, release_id, frozenset(skip))
        args = render_args(ctx, mid, character_id, model_path, params, input_path, output_path)
        if on_cpu:
            args["device"] = "cpu"
        jctx.stage(stage, 0.0)
        try:
            return ctx.render.render(args, lambda v: jctx.progress(v, msg(stage)), cancel=jctx.job.cancel_event)
        except ApiError as exc:
            step = checkpoints.CKPT_RE.match(model_path.name)
            if exc.code != "render.bad_checkpoint" or release_id or step is None or len(skip) >= 2:
                raise
            jctx.log(f"checkpoint {model_path.name} is not readable yet, trying an older one")
            skip.add(int(step.group(1)))


# ---- render node -----------------------------------------------------------------------
def submit_node_render(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str) -> Job:
    node = _node(ctx, pid, node_id)
    if node.get("type") != "render":
        raise ApiError("node.not_render", "Only render nodes can be rendered", status=422)
    data = node.get("data", {})
    mid, cid = data.get("modelId"), data.get("characterId")
    if not mid or not cid:
        raise ApiError("node.incomplete", "Choose a character and a model first", status=422)
    model = ctx.models.get(mid)
    media_id = Graph(ctx.projects.get(pid).canvas).resolve(data.get("input"))
    source_media = ctx.projects.get_media(pid, media_id)
    params = parse_render_params(data.get("params"))
    release_id = data.get("releaseId")
    check_checkpoint(ctx, mid, release_id)
    char_name = ctx.characters.get(cid).name if ctx.characters.exists(cid) else model.name
    # a rendered region is written alone: it starts region.start into its input
    offset = source_media.offset + (max(0.0, params.region.start) if params.region else 0.0)

    queued = threading.Event()  # the "queued" node update must land before "running"

    def run(jctx: JobContext) -> dict[str, Any]:
        queued.wait(10)
        updates.update(ctx, pid, node_id, status="running", jobId=jctx.job.id, error=None)
        out_id = new_id("med")
        output = ctx.projects.media_path(pid, out_id)
        try:
            result = run_render(ctx, jctx, mid, cid, release_id, params, ctx.projects.media_path(pid, media_id), output)
            info = probe(output)
            media = register_media(ctx, pid, Media(
                id=out_id, name=f"{char_name} · {data.get('label') or model.name}", kind="render",
                duration=round(info.duration, 3), sample_rate=info.sample_rate, channels=info.channels,
                created_at=now_iso(), offset=round(offset, 6)))
        except JobCancelled:
            output.unlink(missing_ok=True)
            updates.update(ctx, pid, node_id, status="draft", jobId=None, error=None)
            raise
        except ApiError as exc:
            output.unlink(missing_ok=True)
            updates.update(ctx, pid, node_id, status="failed", error=Msg(code=exc.code, params=exc.params).wire())
            raise
        updates.finish(ctx, pid, node_id, status="done", mediaId=media.id, error=None)
        return {"mediaId": media.id, "nodeId": node_id, "device": result.get("device")}

    job = ctx.jobs.submit("render", msg("job.render", name=char_name),
                          run, target={"projectId": pid, "nodeId": node_id, "modelId": mid, "characterId": cid},
                          uses_gpu=True, cpu_fallback=True, lock_key=RENDER_LOCK)
    updates.update(ctx, pid, node_id, status="queued", jobId=job.id, error=None)
    queued.set()
    return job


# ---- quick render (model page "try it") -------------------------------------------------
def quick_render_path(ctx: "AppContext", mid: str, job_id: str) -> Path:
    return ctx.models.cache_dir(mid) / "quick" / f"{job_id}.wav"


def submit_quick_render(ctx: "AppContext", mid: str, source: Path, cleanup: list[Path], character_id: str,
                        release_id: str | None, params: RenderParams) -> Job:
    rec = ctx.models.get(mid)
    if not character_id:
        character_id = rec.characters[0].character_id
    check_checkpoint(ctx, mid, release_id)

    def run(jctx: JobContext) -> dict[str, Any]:
        try:
            output = quick_render_path(ctx, mid, jctx.job.id)
            output.parent.mkdir(parents=True, exist_ok=True)
            input_wav = output.with_name(f"{jctx.job.id}_in.wav")
            write_wav(input_wav, decode(source, SR), SR)
            result = run_render(ctx, jctx, mid, character_id, release_id, params, input_wav, output)
            input_wav.unlink(missing_ok=True)
            return {"url": f"/models/{mid}/quick-render/{jctx.job.id}/audio", "path": str(output),
                    "device": result.get("device")}
        finally:
            for p in cleanup:
                shutil.rmtree(p, ignore_errors=True)

    return ctx.jobs.submit("render", msg("job.quick_render", name=rec.name), run,
                           target={"modelId": mid, "characterId": character_id}, uses_gpu=True, cpu_fallback=True,
                           lock_key=RENDER_LOCK)


# ---- separation --------------------------------------------------------------------------
STEM_NAMES = {"vocals": "vocals", "inst": "instrumental", "drums": "drums", "bass": "bass", "other": "other"}


def separate_media(ctx: "AppContext", jctx: JobContext, pid: str, media: Media, ports: list[str], kind: str,
                   model: str | None = None) -> dict[str, str]:
    """Separate one media into ``ports`` (vocals/inst, + drums/bass/other) and register them as
    media of ``kind`` at the input's offset; returns {port: media id}."""
    four = any(p in EXTRA_STEMS for p in ports)
    wanted = ["vocals", "inst", *(EXTRA_STEMS if four else ())]
    tmp = Path(tempfile.mkdtemp(prefix="separate-", dir=ctx.projects.media_dir(pid)))
    paths = {p: tmp / f"{p}.wav" for p in wanted}
    on_cpu = jctx.job.placement == PLACEMENT_CPU
    try:
        jctx.stage("stage.separate", 0.0, **({"device": "cpu"} if on_cpu else {}))
        ctx.separation.separate(jctx, ctx.projects.media_path(pid, media.id), paths["vocals"], paths["inst"],
                                device="cpu" if on_cpu else None,
                                extra={p: paths[p] for p in EXTRA_STEMS} if four else None, model=model)
        ids: dict[str, str] = {}
        for port in ports:
            media_id = new_id("med")
            target = ctx.projects.media_path(pid, media_id)
            os.replace(paths[port], target)
            info = probe(target)
            register_media(ctx, pid, Media(
                id=media_id, name=f"{media.name} · {STEM_NAMES[port]}", kind=kind, duration=round(info.duration, 3),
                sample_rate=info.sample_rate, channels=info.channels, created_at=now_iso(), offset=media.offset))
            ids[port] = media_id
        return ids
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def submit_node_separate(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str, stems: int = 2,
                         model: str | None = None) -> Job:
    """Separate a source's song into vocals + accompaniment (``stems=4``: also drums, bass,
    other) with ``model`` (None = the default separation model); progress and the media ids land
    in the source node's ``stems`` (``project.nodes`` events). Stems the user provided are never
    overwritten."""
    node = _node(ctx, pid, node_id)
    if node.get("type") != "source":
        raise ApiError("node.not_source", "Only source nodes can be separated", status=422)
    if stems not in (2, 4):
        raise ApiError("request.invalid", "stems must be 2 or 4", status=422)
    data = node.get("data", {})
    media_id = data.get("mediaId")
    if not media_id:
        raise ApiError("node.input_missing", "This node has no audio", status=409, params={"port": "mix", "nodeId": node_id})
    source_media = ctx.projects.get_media(pid, media_id)
    provided = set((data.get("stems") or {}).get("provided") or [])
    ports = [p for p in ("vocals", "inst", *(EXTRA_STEMS if stems == 4 else ())) if p not in provided]
    if not ports:
        raise ApiError("node.not_runnable", "All stems were provided", status=409, params={"nodeId": node_id})
    model = ctx.separation.require(model)
    queued = threading.Event()

    def stems_update(finish: bool = False, **fields: Any) -> dict[str, Any]:
        current = (_node(ctx, pid, node_id).get("data") or {}).get("stems") or empty_stems()
        merged = {**empty_stems(), **current, **fields}
        if finish:
            updates.finish(ctx, pid, node_id, ports=list(ports), stems=merged)
        else:
            updates.update(ctx, pid, node_id, stems=merged)
        return merged

    def run(jctx: JobContext) -> dict[str, Any]:
        queued.wait(10)
        stems_update(status="running", jobId=jctx.job.id, error=None)
        try:
            ids = separate_media(ctx, jctx, pid, source_media, ports, "stem", model)
        except JobCancelled:
            stems_update(status="none", jobId=None, error=None)
            raise
        except ApiError as exc:
            stems_update(status="failed", jobId=None, error=Msg(code=exc.code, params=exc.params).wire())
            raise
        result = stems_update(finish=True, status="done", model=model, jobId=jctx.job.id, error=None, **ids)
        on_cpu = jctx.job.placement == PLACEMENT_CPU
        return {"nodeId": node_id, "stems": result, "device": "cpu" if on_cpu else ctx.separation.best_device}

    job = ctx.jobs.submit("separate", msg("job.separate", name=source_media.name), run,
                          target={"projectId": pid, "nodeId": node_id}, uses_gpu=True, cpu_fallback=True,
                          lock_key=SEPARATE_LOCK)
    stems_update(status="queued", jobId=job.id, error=None)
    queued.set()
    return job
