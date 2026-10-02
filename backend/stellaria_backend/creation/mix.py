"""Mix nodes (``docs/canvas-v3.md`` §1.3/§4): inputs (any node output, with a role, gain, pan,
mute, solo) → song variants.

``builtin`` (auto mix) and ``bus`` (plain sum) run as ``python -m stellaria_backend.mixing.run
<job.json>`` with job JSON v2 (inputs placed by their media offsets, plus the original song as
``reference`` when the lead vocal traces back to a source with a vocals stem). A plugin
processor runs through ``plugins.runner.run_plugin`` with vocals = lead + harmony inputs
(pre-summed when the plugin takes fewer) and backing = the sum of backing/other inputs, all
placed on one timeline. Without the ``mixing`` package a plain placed sum stands in.
Mix jobs run in the ``dsp`` CPU pool: they never wait for, or slow down, a GPU training.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import shutil
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import soundfile as sf

from ..audio.io import SR, probe, resample
from ..errors import ApiError, JobCancelled, not_found
from ..jobs import Job, JobContext
from ..jobs.manager import POOL_DSP
from ..pipeline.importing import register_media
from ..schemas import Media, Msg, msg
from ..util.fsutil import safe_filename
from ..util.ids import new_id
from ..util.timeutil import now_iso
from .canvas import MIX_ROLES, Graph, find_node
from .cli import run_cli
from .updates import NodeUpdates

if TYPE_CHECKING:
    from ..context import AppContext

MAX_VARIANTS = 4
BUILTIN = "builtin"
BUS = "bus"
VOCAL_ROLES = ("lead", "harmony")


# ---- the mixer package (owned by the mixer team) — guarded until it lands ------------------
def mixer_available() -> bool:
    try:
        return importlib.util.find_spec("stellaria_backend.mixing.run") is not None
    except ModuleNotFoundError:
        return False


def _label(code: str, zh: str, en: str) -> dict[str, Any]:
    return {"code": code, "params": {"zh-CN": zh, "en": en}}


def _fallback_processor() -> dict[str, Any]:
    """The built-in processor's description as §2 specifies it (used until ``mixing`` lands)."""
    def num(pid: str, zh: str, en: str, default: float, lo: float, hi: float, step: float, unit: str | None = None):
        spec = {"id": pid, "label": _label(f"mix.param.{pid}", zh, en), "type": "number", "default": default,
                "min": lo, "max": hi, "step": step}
        if unit:
            spec["unit"] = unit
        return spec

    return {
        "id": BUILTIN, "name": _label("mix.builtin", "一键混音", "One-click mix"), "kind": "builtin",
        "presets": [{"id": p, "name": _label(f"mix.preset.{p}", zh, en)} for p, zh, en in (
            ("natural", "自然", "Natural"), ("pop", "流行", "Pop"), ("airy", "空灵", "Airy"), ("dry", "干声", "Dry"))],
        "params": [
            num("vocalLevel", "人声音量", "Vocal level", 1.0, -6.0, 6.0, 0.5, "dB"),
            num("reverb", "混响", "Reverb", 0.35, 0.0, 1.0, 0.05),
            num("brightness", "明亮度", "Brightness", 0.0, -1.0, 1.0, 0.1),
            num("width", "混响宽度", "Width", 0.7, 0.0, 1.0, 0.05),
            num("targetLufs", "目标响度", "Target loudness", -14.0, -23.0, -9.0, 0.5, "LUFS"),
            num("backingGain", "伴奏音量", "Backing gain", 0.0, -12.0, 6.0, 0.5, "dB"),
        ],
        "maxVocals": 4, "available": True, "error": None,
    }


def _fallback_bus() -> dict[str, Any]:
    return {
        "id": BUS, "name": _label("mix.bus", "直接混合", "Bus (plain sum)"), "kind": "builtin", "presets": [],
        "params": [
            {"id": "normalize", "label": _label("mix.param.normalize", "响度标准化", "Normalize loudness"),
             "type": "boolean", "default": False},
            {"id": "targetLufs", "label": _label("mix.param.targetLufs", "目标响度", "Target loudness"), "type": "number",
             "default": -14.0, "min": -23.0, "max": -9.0, "step": 0.5, "unit": "LUFS"},
        ],
        "maxVocals": 16, "maxInputs": 16, "available": True, "error": None,
    }


def builtin_processor() -> dict[str, Any]:
    try:
        from ..mixing import builtin_processor as real
    except ImportError:
        return _fallback_processor()
    return real()


def bus_processor() -> dict[str, Any]:
    try:
        from ..mixing import bus_processor as real
    except ImportError:
        return _fallback_bus()
    return real()


def processors(ctx: "AppContext") -> list[dict[str, Any]]:
    """``GET /mix/processors``: the auto mix, the bus, then enabled ``mix`` plugins."""
    plugins = getattr(ctx, "plugins", None)
    return [builtin_processor(), bus_processor(), *(plugins.mix_processors() if plugins is not None else [])]


def analyze_media(path: Path) -> dict[str, Any]:
    """``{bpm, lufs, peakDb, duration}`` cached beside the media (``<media>.analysis.json``).
    Without the ``mixing`` package a rough, uncached estimate is returned (no BPM, ungated
    unweighted loudness)."""
    cache = path.with_suffix(".analysis.json")
    if cache.is_file() and cache.stat().st_mtime >= path.stat().st_mtime:
        return json.loads(cache.read_text(encoding="utf-8"))
    try:
        from ..mixing import analyze
    except ImportError:
        analyze = None
    if analyze is not None:
        result = analyze(path)
        cache.write_text(json.dumps(result), encoding="utf-8")
        return result
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mean_square = float(np.mean(data.astype(np.float64) ** 2)) if data.size else 0.0
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return {"bpm": None, "lufs": round(-0.691 + 10 * math.log10(max(mean_square, 1e-12)), 2),
            "peakDb": round(20 * math.log10(max(peak, 1e-9)), 2), "duration": round(len(data) / sr, 3)}


# ---- placing inputs on one timeline --------------------------------------------------------------
def _load(path: str) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return resample(data, sr, SR)


def place(item: dict[str, Any], base: float) -> np.ndarray:
    """One input as stereo (samples, 2) with gain and pan applied, padded to start at ``base``."""
    data = _load(item["path"])
    gain = 10 ** (float(item.get("gainDb", 0.0)) / 20)
    pan = max(-1.0, min(1.0, float(item.get("pan", 0.0))))
    if data.shape[1] == 1:
        angle = (pan + 1) * math.pi / 4  # constant power
        stereo = np.concatenate([data * math.cos(angle), data * math.sin(angle)], axis=1) * math.sqrt(2)
    else:
        stereo = data[:, :2].copy()
        if pan > 0:
            stereo[:, 0] *= 1 - pan
        elif pan < 0:
            stereo[:, 1] *= 1 + pan
    pad = int(round(max(0.0, float(item.get("offset", 0.0)) - base) * SR))
    return np.concatenate([np.zeros((pad, 2), dtype=np.float32), stereo * gain]).astype(np.float32)


def sum_tracks(tracks: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((max((len(t) for t in tracks), default=0), 2), dtype=np.float64)
    for t in tracks:
        out[: len(t)] += t
    return out.astype(np.float32)


def active(inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inputs that play: not muted and, when anything is soloed, soloed."""
    playing = [i for i in inputs if not i.get("mute")]
    soloed = [i for i in playing if i.get("solo")]
    return soloed or playing


def simple_mix(job: dict[str, Any]) -> dict[str, Any]:
    """Stand-in for the ``mixing`` package: the active inputs placed by offset and summed; the
    auto mix also peak-normalizes to -1 dBFS (bus: only when ``normalize``)."""
    inputs = active(job["inputs"])
    base = min((float(i.get("offset", 0.0)) for i in inputs), default=0.0)
    mix = sum_tracks([place(i, base) for i in inputs]).astype(np.float64)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 0 and (job.get("processor") != BUS or job.get("params", {}).get("normalize") or peak > 1.0):
        mix *= 10 ** (-1 / 20) / peak
    Path(job["output"]).parent.mkdir(parents=True, exist_ok=True)
    sf.write(job["output"], mix.astype(np.float32), SR, subtype="FLOAT")
    lufs = -0.691 + 10 * math.log10(max(float(np.mean(mix ** 2)) if mix.size else 0.0, 1e-12))
    return {"lufs": round(lufs, 2), "peakDb": round(20 * math.log10(max(float(np.max(np.abs(mix))) if mix.size else 0.0, 1e-9)), 2),
            "bpm": None, "offset": round(base, 6)}


# ---- running one variant -----------------------------------------------------------------------
def mixer_command(ctx: "AppContext", job_file: Path) -> list[str]:
    return [ctx.python, "-m", "stellaria_backend.mixing.run", str(job_file)]


def _run_mixer(ctx: "AppContext", jctx: JobContext, job: dict[str, Any], workdir: Path,
               span: tuple[float, float], stage_params: dict[str, Any]) -> dict[str, Any]:
    if not mixer_available():
        jctx.progress(span[0], msg("stage.mix.master", **stage_params))
        return simple_mix(job)
    job_file = workdir / f"job-{job['seed']}.json"
    job_file.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

    def progress(value: float, line: dict[str, Any]) -> None:
        jctx.progress(span[0] + (span[1] - span[0]) * value, msg(f"stage.{line.get('stage') or 'mix'}", **stage_params))

    return run_cli(ctx, jctx, mixer_command(ctx, job_file), workdir, progress, "mix.failed", "Mixing failed")


def _run_plugin(ctx: "AppContext", jctx: JobContext, processor: dict[str, Any], job: dict[str, Any], workdir: Path,
                names: dict[str, str], project_name: str, span: tuple[float, float]) -> dict[str, Any]:
    """vocals = lead + harmony inputs (one pre-summed file when the plugin takes fewer), backing
    = the backing/other inputs summed; everything placed on one timeline starting at the
    earliest input."""
    from ..plugins.runner import run_plugin

    inputs = active(job["inputs"])
    base = min(float(i.get("offset", 0.0)) for i in inputs)
    vocal_items = [i for i in inputs if i["role"] in VOCAL_ROLES]
    backing_items = [i for i in inputs if i["role"] not in VOCAL_ROLES]
    readable: dict[str | Path, str] = {}
    vocals: list[Path] = []
    if len(vocal_items) > int(processor.get("maxVocals") or 1):
        path = workdir / f"vocals-{job['seed']}.wav"
        sf.write(str(path), sum_tracks([place(i, base) for i in vocal_items]), SR, subtype="FLOAT")
        readable[path] = safe_filename(f"{project_name} - vocals")
        vocals.append(path)
    else:
        for n, item in enumerate(vocal_items):
            path = workdir / f"vocal-{job['seed']}-{n}.wav"
            sf.write(str(path), place(item, base), SR, subtype="FLOAT")
            readable[path] = safe_filename(f"{project_name} - {names.get(item['path']) or f'vocal {n + 1}'}")
            vocals.append(path)
    backing = None
    if backing_items:
        backing = workdir / f"backing-{job['seed']}.wav"
        sf.write(str(backing), sum_tracks([place(i, base) for i in backing_items]), SR, subtype="FLOAT")
        readable[backing] = safe_filename(f"{project_name} - backing")
    produced = run_plugin(ctx, jctx, processor["id"], vocals=vocals, backing=backing, input=None,
                          output=workdir / f"out-{job['seed']}.wav", params={**job["params"], "seed": job["seed"]},
                          names=readable, on_progress=lambda v: jctx.progress(span[0] + (span[1] - span[0]) * v))
    data = _load(str(produced))
    stereo = np.repeat(data, 2, axis=1) if data.shape[1] == 1 else data[:, :2]
    sf.write(job["output"], stereo, SR, subtype="FLOAT")
    analysis = analyze_media(Path(job["output"]))
    return {"lufs": analysis.get("lufs"), "peakDb": analysis.get("peakDb"), "bpm": analysis.get("bpm"),
            "offset": round(base, 6)}


# ---- the mix job ----------------------------------------------------------------------------------
def _processor(ctx: "AppContext", processor_id: str) -> dict[str, Any]:
    for p in processors(ctx):
        if p.get("id") == processor_id:
            if not p.get("available", True):
                raise ApiError("mix.processor_unavailable", f"'{processor_id}' cannot run right now", status=409,
                               params={"processor": processor_id})
            return p
    raise ApiError("mix.processor_unavailable", f"Unknown mix processor '{processor_id}'", status=409,
                   params={"processor": processor_id})


def _input_name(ctx: "AppContext", graph: Graph, ref: dict[str, Any], media: Media) -> str:
    take = graph.nearest_take([ref])
    if take is not None:
        tdata = take.get("data") or {}
        if tdata.get("label"):
            return str(tdata["label"])
        if ctx.characters.exists(tdata.get("characterId") or ""):
            return ctx.characters.get(tdata["characterId"]).name
    return media.name


def mix_job_inputs(ctx: "AppContext", pid: str, canvas: dict[str, Any], items: list[dict[str, Any]]):
    """Resolve mix inputs to v2 job inputs; also returns {path: display name} and the reference."""
    graph = Graph(canvas)
    media = {m.id: m for m in ctx.projects.media(pid)}
    inputs, names = [], {}
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = item.get("ref")
        media_id = graph.resolve(ref)
        m = media.get(media_id) or ctx.projects.get_media(pid, media_id)
        path = str(ctx.projects.media_path(pid, media_id))
        role = item.get("role") if item.get("role") in MIX_ROLES else "other"
        inputs.append({"path": path, "role": role, "gainDb": float(item.get("gainDb") or 0.0),
                       "pan": float(item.get("pan") or 0.0), "offset": float(m.offset), "mute": bool(item.get("mute")),
                       "solo": bool(item.get("solo")), "_ref": ref})
        names[path] = _input_name(ctx, graph, ref, m)
    reference = None
    for item in active(inputs):
        if item["role"] != "lead":
            continue
        song = graph.root_source([item["_ref"]])
        sdata = (song or {}).get("data") or {}
        vocals_id = (sdata.get("stems") or {}).get("vocals")
        if song is not None and vocals_id and sdata.get("mediaId") and vocals_id in media and sdata["mediaId"] in media:
            reference = {"vocals": str(ctx.projects.media_path(pid, vocals_id)),
                         "mix": str(ctx.projects.media_path(pid, sdata["mediaId"])),
                         "offset": float(media[sdata["mediaId"]].offset)}
            break
    for item in inputs:
        del item["_ref"]
    return inputs, names, reference


def submit_node_mix(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str) -> Job:
    project = ctx.projects.get(pid)
    node = find_node(project.canvas, node_id)
    if node is None:
        raise not_found("node", node_id)
    if node.get("type") != "mix":
        raise ApiError("node.not_mix", "Only mix nodes can be mixed", status=422)
    data = node.get("data") or {}
    if not data.get("inputs"):
        raise ApiError("mix.no_inputs", "Connect at least one input", status=409)
    processor_id = data.get("processor") or BUILTIN
    processor = _processor(ctx, processor_id)
    inputs, names, reference = mix_job_inputs(ctx, pid, project.canvas, data["inputs"])
    playing = active(inputs)
    vocal_count = sum(1 for i in playing if i["role"] in VOCAL_ROLES)
    max_inputs = int(processor.get("maxInputs") or 0)
    if max_inputs and len(inputs) > max_inputs:
        raise ApiError("mix.too_many_inputs", "This processor takes fewer inputs", status=422, params={"max": max_inputs})
    if processor_id != BUS and vocal_count == 0:
        raise ApiError("mix.no_vocals", "Connect at least one lead or harmony vocal", status=409)
    if processor_id == BUILTIN and vocal_count > int(processor.get("maxVocals") or 4):
        raise ApiError("mix.too_many_vocals", "This processor takes fewer vocals", status=422,
                       params={"max": int(processor.get("maxVocals") or 4)})
    variants = 1 if processor_id == BUS else max(1, min(MAX_VARIANTS, int(data.get("variants") or 1)))
    seeds = [0] + [random.randint(1, 2**31 - 1) for _ in range(variants - 1)]
    preset, params = data.get("preset") or ("natural" if processor_id == BUILTIN else None), dict(data.get("params") or {})
    label = data.get("label") or ("Bus" if processor_id == BUS else "Mix")
    queued = threading.Event()

    def run(jctx: JobContext) -> dict[str, Any]:
        queued.wait(10)
        updates.update(ctx, pid, node_id, status="running", jobId=jctx.job.id, error=None)
        workdir = Path(tempfile.mkdtemp(prefix="mix-", dir=ctx.tmp_dir))
        outputs: list[dict[str, Any]] = []
        output: Path | None = None  # the variant being written (removed if it fails)
        try:
            for i, seed in enumerate(seeds):
                jctx.check_cancelled()
                media_id = new_id("med")
                output = ctx.projects.media_path(pid, media_id)
                job = {"inputs": inputs, "reference": reference,
                       "processor": processor_id if processor_id in (BUILTIN, BUS) else BUILTIN,
                       "preset": preset, "params": params, "seed": seed, "output": str(output)}
                span = (i / len(seeds), (i + 1) / len(seeds))
                stage_params = {"variant": i + 1, "variants": len(seeds)}
                if processor_id in (BUILTIN, BUS):
                    result = _run_mixer(ctx, jctx, job, workdir, span, stage_params)
                else:
                    jctx.progress(span[0], msg("stage.mix.plugin", processor=processor_id, **stage_params))
                    result = _run_plugin(ctx, jctx, processor, job, workdir, names, project.name, span)
                info = probe(output)
                offset = result.get("offset")
                if offset is None:
                    offset = min(float(x.get("offset", 0.0)) for x in playing)
                register_media(ctx, pid, Media(
                    id=media_id, name=f"{label} · {i + 1}" if len(seeds) > 1 else label, kind="mix",
                    duration=round(info.duration, 3), sample_rate=info.sample_rate, channels=info.channels,
                    created_at=now_iso(), offset=round(float(offset), 6)))
                output = None
                lufs = result.get("lufs")
                outputs.append({"mediaId": media_id, "seed": seed, "lufs": None if lufs is None else round(float(lufs), 2)})
        except JobCancelled:
            updates.update(ctx, pid, node_id, status="draft", jobId=None, error=None)
            raise
        except Exception as exc:
            error = Msg(code=exc.code, params=exc.params) if isinstance(exc, ApiError) else Msg(code="mix.failed")
            updates.update(ctx, pid, node_id, status="failed", error=error.wire())
            raise
        finally:
            if output is not None:
                output.unlink(missing_ok=True)
            shutil.rmtree(workdir, ignore_errors=True)
        updates.finish(ctx, pid, node_id, status="done", outputs=outputs, selected=0, error=None)
        return {"nodeId": node_id, "outputs": outputs}

    job = ctx.jobs.submit("mix", msg("job.mix", name=label), run, target={"projectId": pid, "nodeId": node_id},
                          lock_key=f"mix:{pid}:{node_id}", pool=POOL_DSP)
    updates.update(ctx, pid, node_id, status="queued", jobId=job.id, error=None)
    queued.set()
    return job
