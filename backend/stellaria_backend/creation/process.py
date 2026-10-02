"""Process nodes (``docs/canvas-v3.md`` §1.3, §2): trim, split, pitch, fx, separate, dereverb,
comp, concat, export.

Every op except ``separate`` and ``dereverb`` runs ``python -m stellaria_backend.dsp.run
<job.json>`` (JSON lines, see ``creation.cli``) in the ``dsp`` CPU pool::

    {"op", "inputs": [{"path", "offset"}], "params", "outputDir", "bpm",
     "vstPlugins": {id: VstPlugin}, "names": {project, song, voice, label, date, index}}
    → {"outputs": {port: {"path", "offset", "duration"}}, "result": {...}}

Outputs become media of kind ``process`` at the offsets the op computed; the node gets
``outputs`` (port → media id), ``result`` and ``status``, and its dependents turn stale.
``separate`` uses the separation worker (separator add-on) like a source's split; ``params.model``
is a separation model id (absent/null/'auto' = the default). ``dereverb`` uses the same worker
with a cleanup model (``params.model``: a cleanup model id, absent/null/'auto' = the default):
port ``dry`` is the voice without reverb (and echo), ``reverb`` what was removed (input − dry).
Both run on the GPU queue and fall back to the CPU next to a training. ``export`` writes
into ``params.folder`` (which must exist), else the job's ``defaultFolder`` =
``Preferences.exportFolder`` → ``<library>/exports`` (created when missing), and never
overwrites; ``result = {path, downloadUrl}``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import tempfile
import threading
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..audio.io import probe
from ..engine.rendering import SEPARATE_LOCK, separate_media
from ..errors import ApiError, JobCancelled, not_found
from ..jobs import Job, JobContext
from ..jobs.manager import PLACEMENT_CPU, POOL_DSP
from ..pipeline.importing import register_media
from ..schemas import Media, Msg, msg
from ..util.ids import new_id
from ..util.timeutil import now_iso
from .analysis import cached_analysis
from .canvas import PROCESS_OPS, Graph, find_node, inputs_of, is_bypassed, op_ports
from .cli import CliFailed, run_cli
from .updates import NodeUpdates

if TYPE_CHECKING:
    from ..context import AppContext

INPUT_COUNTS = {"comp": (2, 8), "concat": (2, 32)}  # others: exactly 1
FAIL_CODE = "process.failed"


def dsp_available() -> bool:
    try:
        return importlib.util.find_spec("stellaria_backend.dsp.run") is not None
    except ModuleNotFoundError:
        return False


def dsp_command(ctx: "AppContext", job_file: Path) -> list[str]:
    return [ctx.python, "-m", "stellaria_backend.dsp.run", str(job_file)]


def export_folder(ctx: "AppContext", folder: str | None = None) -> Path:
    """``folder`` → ``Preferences.exportFolder`` → ``<library>/exports``."""
    chosen = (folder or "").strip() or (ctx.prefs.get().export_folder or "").strip()
    path = Path(chosen).expanduser() if chosen else ctx.options.library / "exports"
    if not path.is_absolute():
        raise ApiError("request.relative_path", f"Paths must be absolute: {path}", status=422)
    return path


def stage_code(stage: Any, op: str) -> str:
    text = str(stage or "").strip()
    if not text:
        return f"stage.dsp.{op}"
    if text.startswith("stage."):
        return text
    return f"stage.{text}" if text.startswith("dsp.") else f"stage.dsp.{text}"


def _stage_params(line: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in line.items() if k not in ("progress", "stage") and isinstance(v, (str, int, float))
            and not isinstance(v, bool)}


# ---- names for export templates -----------------------------------------------------------------
def template_names(ctx: "AppContext", project, graph: Graph, node: dict[str, Any]) -> dict[str, Any]:
    data = node.get("data") or {}
    refs = inputs_of(node)
    song = graph.root_source(refs)
    song_name = ""
    if song is not None:
        sdata = song.get("data") or {}
        song_name = sdata.get("name") or ""
        if not song_name and sdata.get("mediaId"):
            try:
                song_name = ctx.projects.get_media(project.id, sdata["mediaId"]).name
            except ApiError:
                song_name = ""
    take = graph.nearest_take(refs)
    voice = ""
    if take is not None:
        cid = (take.get("data") or {}).get("characterId") or ""
        voice = ctx.characters.get(cid).name if cid and ctx.characters.exists(cid) else ""
    exports = [n["id"] for n in graph.nodes if n.get("type") == "process" and (n.get("data") or {}).get("op") == "export"]
    return {"project": project.name, "song": song_name or project.name, "voice": voice,
            "label": data.get("label") or song_name or project.name, "date": date.today().isoformat(),
            "index": exports.index(node["id"]) + 1 if node["id"] in exports else 1}


# ---- VST slots -------------------------------------------------------------------------------------
def _vst_slots(params: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in params.get("chain") or [] if isinstance(s, dict) and s.get("kind") == "vst" and not s.get("bypass")]


def vst_plugins(ctx: "AppContext", params: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``VstPlugin`` dicts of the chain's active VST slots (409 ``vst.host_missing``, 404
    ``vst.not_found`` {name}, 422 ``vst.instrument`` {name})."""
    slots = _vst_slots(params)
    if not slots:
        return {}
    registry = getattr(ctx, "vst", None)
    if registry is None:
        raise ApiError("vst.host_missing", "The VST host add-on (pedalboard) is not installed", status=409)
    registry.require_host()
    out = {}
    for slot in slots:
        plugin = registry.require(str(slot.get("pluginId") or ""))
        if plugin.get("category") == "instrument":
            raise ApiError("vst.instrument", f"'{plugin.get('name')}' is an instrument", status=422,
                           params={"name": plugin.get("name") or ""})
        out[plugin["id"]] = plugin
    return out


CRASH_NAME = re.compile(r"(?:vst|plugin)[^:=]*[:=]\s*['\"]?(.+?)['\"]?\s*$", re.IGNORECASE)


def crashed_plugin(exc: CliFailed, plugins: dict[str, dict[str, Any]]) -> str | None:
    """The plugin a crashed fx run was in, from its last stage/log line (else the chain's only VST)."""
    line = exc.run.last_progress
    for key in ("plugin", "pluginName", "name", "vst"):
        if isinstance(line.get(key), str) and line[key]:
            return line[key]
    for text in (exc.run.last_log, str(line.get("stage") or "")):
        m = CRASH_NAME.search(text or "")
        if m:
            return m.group(1).strip()
    names = sorted({p.get("name") or pid for pid, p in plugins.items()})
    return ", ".join(names) if names else None


# ---- the process job --------------------------------------------------------------------------------
def submit_node_process(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str) -> Job:
    project = ctx.projects.get(pid)
    node = find_node(project.canvas, node_id)
    if node is None:
        raise not_found("node", node_id)
    if node.get("type") != "process":
        raise ApiError("node.not_process", "Only process nodes can be processed", status=422)
    data = node.get("data") or {}
    op = data.get("op")
    if op not in PROCESS_OPS:
        raise ApiError("request.invalid", f"Unknown op '{op}'", status=422)
    if is_bypassed(node):
        raise ApiError("node.not_runnable", "A bypassed node passes its input through", status=409,
                       params={"nodeId": node_id})
    refs = inputs_of(node)
    lo, hi = INPUT_COUNTS.get(op, (1, 1))
    if not refs:
        raise ApiError("node.input_missing", "Connect an input first", status=409, params={"nodeId": node_id, "port": ""})
    if not lo <= len(refs) <= hi:
        raise ApiError("process.input_count", f"'{op}' takes {lo}–{hi} inputs", status=422, params={"min": lo, "max": hi})
    graph = Graph(project.canvas)
    media = {m.id: m for m in ctx.projects.media(pid)}
    sources: list[Media] = []
    for ref in refs:
        media_id = graph.resolve(ref)
        sources.append(media.get(media_id) or ctx.projects.get_media(pid, media_id))
    params = dict(data.get("params") or {})
    label = data.get("label") or ""
    title = msg(f"job.process.{op}", name=label or sources[0].name)
    target = {"projectId": pid, "nodeId": node_id}
    if op == "separate":
        return _submit_separate(ctx, updates, pid, node_id, sources[0], params, label, title, target)
    if op == "dereverb":
        return _submit_dereverb(ctx, updates, pid, node_id, sources[0], params, label, title, target)

    if not dsp_available():
        raise ApiError("dsp.unavailable", "Audio processing is not available in this version", status=409)
    plugins = vst_plugins(ctx, params) if op == "fx" else {}
    default_folder = str(export_folder(ctx)) if op == "export" else None
    song = graph.root_source(refs)
    sdata = (song or {}).get("data") or {}
    bpm = sdata.get("bpm")
    if bpm is None and sdata.get("mediaId"):  # analysis still running when the song node was saved
        bpm = (cached_analysis(ctx.projects.media_path(pid, sdata["mediaId"])) or {}).get("bpm")
    job_spec: dict[str, Any] = {
        "op": op, "inputs": [{"path": str(ctx.projects.media_path(pid, m.id)), "offset": float(m.offset)} for m in sources],
        "params": params, "outputDir": "", "bpm": float(bpm) if isinstance(bpm, (int, float)) else None,
        "vstPlugins": plugins, "names": template_names(ctx, project, graph, node),
        **({"defaultFolder": default_folder} if default_folder else {}),
    }
    queued = threading.Event()

    def run(jctx: JobContext) -> dict[str, Any]:
        queued.wait(10)
        updates.update(ctx, pid, node_id, status="running", jobId=jctx.job.id, error=None)
        workdir = Path(tempfile.mkdtemp(prefix=f"process-{op}-", dir=ctx.projects.media_dir(pid)))
        registered: list[str] = []
        try:
            spec = {**job_spec, "outputDir": str(workdir / "out")}
            (workdir / "out").mkdir()
            job_file = workdir / "job.json"
            job_file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            jctx.stage(stage_code(None, op), 0.0)

            def progress(value: float, line: dict[str, Any]) -> None:
                jctx.progress(value, msg(stage_code(line.get("stage"), op), **_stage_params(line)))

            try:
                result = run_cli(ctx, jctx, dsp_command(ctx, job_file), workdir, progress, FAIL_CODE, "Processing failed")
            except CliFailed as exc:
                if op == "fx" and exc.code == FAIL_CODE and plugins:
                    name = crashed_plugin(exc, plugins)
                    raise ApiError("vst.crashed", f"The plugin '{name}' crashed", status=500,
                                   detail=exc.detail, params={"name": name or ""}) from exc
                raise
            outputs: dict[str, str | None] = {}
            for port, info in (result.get("outputs") or {}).items():
                if not isinstance(info, dict) or not info.get("path"):
                    continue
                media_id = new_id("med")
                target_path = ctx.projects.media_path(pid, media_id)
                shutil.move(str(info["path"]), target_path)
                probed = probe(target_path)
                offset = info.get("offset")
                register_media(ctx, pid, Media(
                    id=media_id, name=f"{label or sources[0].name} · {port}" if port != "out" else (label or f"{sources[0].name} · {op}"),
                    kind="process", duration=round(probed.duration, 3), sample_rate=probed.sample_rate,
                    channels=probed.channels, created_at=now_iso(),
                    offset=round(float(offset if offset is not None else sources[0].offset), 6)))
                registered.append(media_id)
                outputs[port] = media_id
            facts = dict(result.get("result") or {})
            if op == "export" and facts.get("path"):
                facts["downloadUrl"] = f"/files/{ctx.files.register(Path(facts['path']))}"
        except JobCancelled:
            updates.update(ctx, pid, node_id, status="draft", jobId=None, error=None)
            raise
        except Exception as exc:
            error = Msg(code=exc.code, params=exc.params) if isinstance(exc, ApiError) else Msg(code=FAIL_CODE)
            updates.update(ctx, pid, node_id, status="failed", error=error.wire())
            raise
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
        updates.finish(ctx, pid, node_id, status="done", outputs=outputs, result=facts or None, error=None)
        return {"nodeId": node_id, "outputs": outputs, "result": facts or None}

    job = ctx.jobs.submit("process", title, run, target=target, lock_key=f"process:{pid}:{node_id}", pool=POOL_DSP)
    updates.update(ctx, pid, node_id, status="queued", jobId=job.id, error=None)
    queued.set()
    return job


def _submit_separate(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str, source: Media,
                     params: dict[str, Any], label: str, title, target: dict[str, str]) -> Job:
    model = ctx.separation.require(params.get("model"))
    stems = 4 if int(params.get("stems") or 2) == 4 else 2
    ports = op_ports({"op": "separate", "params": {"stems": stems}})
    queued = threading.Event()

    def run(jctx: JobContext) -> dict[str, Any]:
        queued.wait(10)
        updates.update(ctx, pid, node_id, status="running", jobId=jctx.job.id, error=None)
        named = source.model_copy(update={"name": label or source.name})
        try:
            ids = separate_media(ctx, jctx, pid, named, ports, "process", model)
        except JobCancelled:
            updates.update(ctx, pid, node_id, status="draft", jobId=None, error=None)
            raise
        except ApiError as exc:
            updates.update(ctx, pid, node_id, status="failed", error=Msg(code=exc.code, params=exc.params).wire())
            raise
        facts = {"model": model, "stems": stems}
        updates.finish(ctx, pid, node_id, status="done", outputs=ids, result=facts, error=None)
        return {"nodeId": node_id, "outputs": ids, "result": facts}

    job = ctx.jobs.submit("process", title, run, target=target, uses_gpu=True, cpu_fallback=True, lock_key=SEPARATE_LOCK)
    updates.update(ctx, pid, node_id, status="queued", jobId=job.id, error=None)
    queued.set()
    return job


def _submit_dereverb(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str, source: Media,
                     params: dict[str, Any], label: str, title, target: dict[str, str]) -> Job:
    model = ctx.separation.require_cleanup(params.get("model"))
    queued = threading.Event()

    def run(jctx: JobContext) -> dict[str, Any]:
        queued.wait(10)
        updates.update(ctx, pid, node_id, status="running", jobId=jctx.job.id, error=None)
        tmp = Path(tempfile.mkdtemp(prefix="dereverb-", dir=ctx.projects.media_dir(pid)))
        paths = {"dry": tmp / "dry.wav", "reverb": tmp / "reverb.wav"}
        on_cpu = jctx.job.placement == PLACEMENT_CPU
        try:
            jctx.stage("stage.dereverb", 0.0, **({"device": "cpu"} if on_cpu else {}))
            ctx.separation.cleanup(jctx, ctx.projects.media_path(pid, source.id), paths["dry"], paths["reverb"],
                                   device="cpu" if on_cpu else None, model=model)
            ids: dict[str, str | None] = {}
            for port, path in paths.items():
                media_id = new_id("med")
                target_path = ctx.projects.media_path(pid, media_id)
                shutil.move(str(path), target_path)
                info = probe(target_path)
                register_media(ctx, pid, Media(
                    id=media_id, name=f"{label or source.name} · {port}", kind="process",
                    duration=round(info.duration, 3), sample_rate=info.sample_rate, channels=info.channels,
                    created_at=now_iso(), offset=source.offset))
                ids[port] = media_id
        except JobCancelled:
            updates.update(ctx, pid, node_id, status="draft", jobId=None, error=None)
            raise
        except Exception as exc:
            error = Msg(code=exc.code, params=exc.params) if isinstance(exc, ApiError) else Msg(code="cleanup.failed")
            updates.update(ctx, pid, node_id, status="failed", error=error.wire())
            raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        facts = {"model": model}
        updates.finish(ctx, pid, node_id, status="done", outputs=ids, result=facts, error=None)
        return {"nodeId": node_id, "outputs": ids, "result": facts}

    job = ctx.jobs.submit("process", title, run, target=target, uses_gpu=True, cpu_fallback=True, lock_key=SEPARATE_LOCK)
    updates.update(ctx, pid, node_id, status="queued", jobId=job.id, error=None)
    queued.set()
    return job
