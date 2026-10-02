"""Projects: canvas/timeline, media (analysis, sections, export), pitch, running nodes
(``/run`` + the v2 aliases), song stems, mix processors, effects, key suggestion, mixdown."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response

import soundfile as sf
from pydantic import Field

from ..audio.io import SR, encode, peaks
from ..context import AppContext
from ..creation.analysis import media_analysis as analyze_with_key, sections as media_sections
from ..creation.canvas import migrate_canvas
from ..creation.mix import processors
from ..creation.run import import_stem, remove_stem, submit_node
from ..engine import mixdown as mixdown_engine
from ..engine import pitching
from ..engine.pitch import media_pitch, summarize, suggest_key
from ..errors import ApiError
from ..exports import destination, finish
from ..jobs import JobContext
from ..pipeline.importing import add_media_audio, import_media
from ..schemas import JobView, Media, PitchRange, Project, ProjectSummary, Wire, msg
from .deps import get_ctx, read_incoming, require_paths

router = APIRouter()


class ProjectCreate(Wire):
    name: str | None = None


class ProjectPut(Wire):
    name: str | None = None
    canvas: dict[str, Any] | None = None
    timeline: dict[str, Any] | None = None
    cover_color: str | None = None


class MediaExportBody(Wire):
    format: Literal["wav", "flac", "mp3"] = "wav"
    path: str | None = None
    normalize: bool = False


class RunBody(Wire):
    stems: Literal[2, 4] | None = None
    model: str | None = None  # songs: separation model id (None = the default)


class MixdownBody(Wire):
    format: Literal["wav", "flac", "mp3"] = "wav"
    # a file (a folder with ``stems``); none: the exports folder and a download URL (browser)
    path: str | None = None
    normalize: bool = True              # peak to −1 dBFS (``loudness`` wins when set)
    loudness: float | None = Field(None, ge=-40.0, le=-5.0)  # LUFS, true peak ≤ −1 dBTP
    sample_rate: Literal[44100, 48000] = 44100
    bit_depth: Literal[16, 24, 32] = 24  # 32 = float (FLAC writes 24)
    mp3_bitrate: Literal[192, 256, 320] = 320
    start: float | None = Field(None, ge=0)  # seconds; both set = only that range
    end: float | None = Field(None, ge=0)
    stems: bool = False                 # one file per audible track instead of the mix
    timeline: dict[str, Any] | None = None  # the arrangement as on screen (else the saved one)


def _summary(ctx: AppContext, project: Project) -> ProjectSummary:
    nodes = project.canvas.get("nodes", [])
    renders = [n for n in nodes if n.get("type") == "render"]
    character_ids = sorted({n.get("data", {}).get("characterId") for n in renders if n.get("data", {}).get("characterId")})
    return ProjectSummary(
        id=project.id, name=project.name, created_at=project.created_at, updated_at=project.updated_at,
        cover_color=project.cover_color, render_count=sum(1 for m in ctx.projects.media(project.id) if m.kind == "render"),
        character_ids=character_ids,
    )


@router.get("/projects", response_model=list[ProjectSummary])
def list_projects(ctx: AppContext = Depends(get_ctx)) -> list[ProjectSummary]:
    projects = sorted(ctx.projects.list(), key=lambda p: p.updated_at, reverse=True)
    return [_summary(ctx, p) for p in projects]


@router.post("/projects", response_model=Project)
def create_project(body: ProjectCreate | None = None, ctx: AppContext = Depends(get_ctx)) -> Project:
    return ctx.projects.create(body.name if body else None)


@router.get("/projects/{pid}", response_model=Project)
def get_project(pid: str, ctx: AppContext = Depends(get_ctx)) -> Project:
    return ctx.projects.get(pid)


@router.put("/projects/{pid}", response_model=Project)
def put_project(pid: str, body: ProjectPut, ctx: AppContext = Depends(get_ctx)) -> Project:
    with ctx.projects.lock(pid):
        project = ctx.projects.get(pid)
        if body.name is not None and body.name.strip():
            project.name = body.name.strip()
        if body.cover_color:
            project.cover_color = body.cover_color
        if body.canvas is not None:
            migrate_canvas(body.canvas)  # an older client may still send v1/v2 node data
            ctx.node_updates.merge_incoming(pid, body.canvas, project.canvas)
            ctx.analysis.fill_new_sources(pid, body.canvas, project.canvas)
            project.canvas = body.canvas
        if body.timeline is not None:
            project.timeline = body.timeline
        return ctx.projects.save(project)


@router.delete("/projects/{pid}")
def delete_project(pid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    if any(j.target.get("projectId") == pid for j in ctx.jobs.list(active_only=True)):
        raise ApiError("project.busy", "This project has running jobs", status=409)
    ctx.projects.delete(pid)
    pitching.forget_project(ctx, pid)
    return {"ok": True}


@router.post("/projects/{pid}/duplicate", response_model=Project)
def duplicate_project(pid: str, ctx: AppContext = Depends(get_ctx)) -> Project:
    return ctx.projects.duplicate(pid)


# ---- media --------------------------------------------------------------------------------
@router.get("/projects/{pid}/media", response_model=list[Media])
def list_media(pid: str, ctx: AppContext = Depends(get_ctx)) -> list[Media]:
    ctx.projects.get(pid)
    return ctx.projects.media(pid)


@router.post("/projects/{pid}/media", response_model=JobView)
async def add_media(pid: str, request: Request, ctx: AppContext = Depends(get_ctx)) -> JobView:
    project = ctx.projects.get(pid)
    incoming = await read_incoming(request, ctx)
    require_paths(incoming)
    def run(jctx: JobContext) -> dict[str, Any]:
        result = import_media(ctx, jctx, pid, incoming.paths, incoming.names, incoming.cleanup)
        for media_id in result.get("mediaIds", []):
            ctx.analysis.schedule(pid, media_id)  # bpm + key for the song nodes made from it
        return result

    job = ctx.jobs.submit("import", msg("job.import_media", name=project.name), run, target={"projectId": pid})
    return job.view()


def _media_file(ctx: AppContext, pid: str, mid: str) -> Path:
    ctx.projects.get_media(pid, mid)
    return ctx.projects.media_path(pid, mid)


@router.get("/projects/{pid}/media/{mid}/audio")
def media_audio(pid: str, mid: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    return FileResponse(_media_file(ctx, pid, mid), media_type="audio/wav")


@router.get("/projects/{pid}/media/{mid}/peaks")
def media_peaks(pid: str, mid: str, n: int = 800, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    values, duration = peaks(_media_file(ctx, pid, mid), n)
    return {"peaks": values, "duration": duration}


@router.get("/projects/{pid}/media/{mid}/analysis")
def media_analysis(pid: str, mid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return analyze_with_key(_media_file(ctx, pid, mid))


@router.get("/projects/{pid}/media/{mid}/sections")
def media_sections_route(pid: str, mid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return media_sections(ctx, pid, mid)


@router.post("/projects/{pid}/media/{mid}/export", response_model=JobView)
def media_export(pid: str, mid: str, body: MediaExportBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    source = _media_file(ctx, pid, mid)
    media = ctx.projects.get_media(pid, mid)

    def run(jctx: JobContext) -> dict[str, Any]:
        jctx.stage("stage.export", 0.1)
        data, sr = sf.read(str(source), dtype="float32", always_2d=True)
        if body.normalize and data.size and (peak := float(abs(data).max())) > 0:
            data = data * (10 ** (-1 / 20) / peak)
        dest, browser = destination(ctx, body.path, media.name, body.format)
        encode(dest, data if data.shape[1] > 1 else data[:, 0], sr, body.format)
        return finish(ctx, dest, browser)

    return ctx.jobs.submit("export", msg("job.export", name=media.name), run, target={"projectId": pid}).view()


@router.get("/projects/{pid}/media/{mid}/pitch")
def media_pitch_curve(pid: str, mid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return media_pitch(_media_file(ctx, pid, mid))


# ---- nodes --------------------------------------------------------------------------------
@router.post("/projects/{pid}/nodes/{node_id}/run", response_model=JobView)
def run_node(pid: str, node_id: str, body: RunBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return submit_node(ctx, pid, node_id, body.stems if body else None, model=body.model if body else None).view()


@router.post("/projects/{pid}/nodes/{node_id}/render", response_model=JobView)
def render_node(pid: str, node_id: str, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return submit_node(ctx, pid, node_id, expect="render").view()


# ---- the pitch editor (render nodes) -------------------------------------------------------
class PitchPreviewBody(Wire):
    start: float
    end: float
    pitch_edit: dict[str, Any] | None = None


@router.get("/projects/{pid}/nodes/{node_id}/pitch")
def node_pitch(pid: str, node_id: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """The pitch line of the node's input (``pitch.not_analyzed`` 404 before the first analysis)."""
    return pitching.get_analysis(ctx, pid, node_id)


@router.post("/projects/{pid}/nodes/{node_id}/pitch/analyze", response_model=JobView)
def analyze_node_pitch(pid: str, node_id: str, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return pitching.submit_analysis(ctx, pid, node_id).view()


@router.post("/projects/{pid}/nodes/{node_id}/pitch/preview", response_model=JobView)
def preview_node_pitch(pid: str, node_id: str, body: PitchPreviewBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return pitching.submit_preview(ctx, pid, node_id, body.start, body.end, body.pitch_edit).view()


@router.get("/projects/{pid}/pitch-preview/{job_id}/audio")
def pitch_preview_audio(pid: str, job_id: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    return FileResponse(pitching.preview_file(ctx, pid, job_id), media_type="audio/wav")


@router.post("/projects/{pid}/nodes/{node_id}/separate", response_model=JobView)
def separate_node(pid: str, node_id: str, body: RunBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return submit_node(ctx, pid, node_id, body.stems if body else None, expect="separate",
                       model=body.model if body else None).view()


@router.post("/projects/{pid}/nodes/{node_id}/mix", response_model=JobView)
def mix_node(pid: str, node_id: str, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return submit_node(ctx, pid, node_id, expect="mix").view()


@router.post("/projects/{pid}/nodes/{node_id}/stems")
async def add_stem(pid: str, node_id: str, request: Request, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """JSON ``{paths: [path], port}`` / ``{path, port}`` or multipart ``files`` + ``port`` → the song node."""
    ctx.projects.get(pid)
    incoming = await read_incoming(request, ctx)
    try:
        require_paths(incoming)
        if len(incoming.paths) != 1:
            raise ApiError("request.invalid", "Give exactly one file", status=422)
        path = incoming.paths[0]
        return await run_in_threadpool(import_stem, ctx, ctx.node_updates, pid, node_id, incoming.fields.get("port"),
                                       path, incoming.names.get(path))
    finally:
        for p in incoming.cleanup:
            shutil.rmtree(p, ignore_errors=True)


@router.delete("/projects/{pid}/nodes/{node_id}/stems/{port}")
def delete_stem(pid: str, node_id: str, port: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return remove_stem(ctx, ctx.node_updates, pid, node_id, port)


@router.get("/mix/processors")
def mix_processors(ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    return processors(ctx)


@router.get("/fx/effects")
def fx_effects() -> list[dict[str, Any]]:
    """``EffectSpec[]`` of the built-in effects (``fx.effect_specs()``; empty until ``fx`` is installed)."""
    try:
        from ..fx import effect_specs
    except ImportError:
        return []
    return effect_specs()


@router.get("/projects/{pid}/suggest-key")
def suggest(pid: str, mediaId: str, characterId: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:  # noqa: N803
    ctx.characters.get(characterId)
    source, seconds = summarize(media_pitch(_media_file(ctx, pid, mediaId))["f0"])
    report, _ = ctx.characters.health(characterId)
    character = None
    if report is not None and report.pitch.low and report.pitch.high and report.pitch.median:
        character = PitchRange(low=report.pitch.low, high=report.pitch.high, median=report.pitch.median)
    return suggest_key(source, seconds, character)


@router.post("/projects/{pid}/mixdown", response_model=JobView)
def mixdown(pid: str, body: MixdownBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    project = ctx.projects.get(pid)
    ranged = body.start is not None and body.end is not None
    span = {"start": body.start, "end": body.end} if ranged else {}
    float_out = body.format == "wav" and body.bit_depth == 32
    level = {"peak": body.normalize, "loudness": body.loudness}
    write = {"bits": body.bit_depth, "bitrate": body.mp3_bitrate}

    def run(jctx: JobContext) -> dict[str, Any]:
        jctx.stage("stage.mixdown", 0.1)
        current = ctx.projects.get(pid)
        if body.timeline is not None:
            current = current.model_copy(update={"timeline": body.timeline})
        known = {m.id for m in ctx.projects.media(pid)}

        def media_path(media_id: str) -> Path:
            if media_id not in known:
                raise ApiError("media_not_found", f"Media '{media_id}' not found", status=404)
            return ctx.projects.media_path(pid, media_id)

        if body.stems:
            return _mixdown_stems(ctx, jctx, current, media_path, body, span, level, write, float_out)
        mix = mixdown_engine.render_timeline(current.timeline, media_path, **span)
        if len(mix) == 0:
            raise ApiError("mixdown.empty", "The timeline is empty", status=409)
        jctx.progress(0.5)
        out = mixdown_engine.level(mixdown_engine.resample(mix, body.sample_rate), body.sample_rate,
                                   float_out=float_out, log=jctx.log, **level)
        jctx.progress(0.7)
        result: dict[str, Any] = {}
        if not ranged:
            # the whole song is also kept with the project's audio
            kept = out if body.sample_rate == SR else mixdown_engine.resample(out, SR, body.sample_rate)
            result["mediaId"] = add_media_audio(ctx, pid, kept, f"{current.name} · mixdown", "mixdown").id
        dest, browser = _mixdown_dest(ctx, body.path, current.name, body.format)
        mixdown_engine.write_audio(dest, out, body.sample_rate, body.format, title=current.name, log=jctx.log, **write)
        return {**result, **finish(ctx, dest, browser)}

    return ctx.jobs.submit("mixdown", msg("job.mixdown", name=project.name), run, target={"projectId": pid}).view()


def _mixdown_dest(ctx: AppContext, path: str | None, stem: str, ext: str) -> tuple[Path, bool]:
    """A chosen file is written as chosen; in a chosen folder a new name never overwrites."""
    from ..dsp.naming import claim, sanitize

    if path and Path(path).is_dir():
        return claim(Path(path), sanitize(stem, fallback="mix"), ext), False
    return destination(ctx, path, stem, ext)


def _mixdown_stems(ctx: AppContext, jctx: JobContext, project: Project, media_path: mixdown_engine.MediaPath,
                   body: MixdownBody, span: dict[str, Any], level: dict[str, Any], write: dict[str, Any],
                   float_out: bool) -> dict[str, Any]:
    """One file per audible track, all moved by the gain that levels their sum."""
    import zipfile

    from ..dsp.naming import claim, sanitize
    from ..util.ids import new_id

    stems = mixdown_engine.render_stems(project.timeline, media_path, **span)
    if not stems:
        raise ApiError("mixdown.empty", "The timeline is empty", status=409)
    stems = [(track, mixdown_engine.resample(x, body.sample_rate)) for track, x in stems]
    gain = mixdown_engine.level_gain(sum(x for _, x in stems), body.sample_rate, **level)
    browser = not body.path
    folder = Path(body.path) if body.path else \
        ctx.options.data / "exports" / f"{sanitize(project.name, fallback='mix')} stems-{new_id('x', 6)[2:]}"
    folder.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, (track, x) in enumerate(stems):
        jctx.progress(0.3 + 0.6 * i / len(stems))
        x = mixdown_engine.safe((x * gain).astype("float32"), body.sample_rate, float_out=float_out, log=jctx.log)
        name = sanitize(f"{project.name} - {i + 1:02d} {track.get('name') or ''}", fallback=f"track {i + 1}")
        dest = claim(folder, name, body.format)
        mixdown_engine.write_audio(dest, x, body.sample_rate, body.format, title=track.get("name") or None,
                                   log=jctx.log, **write)
        paths.append(str(dest))
    result: dict[str, Any] = {"paths": paths, "folder": str(folder)}
    if browser:
        # one download for the browser: the files in a zip
        archive = folder.parent / f"{folder.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as z:
            for p in paths:
                z.write(p, Path(p).name)
        result.update(finish(ctx, archive, True))
    return result
