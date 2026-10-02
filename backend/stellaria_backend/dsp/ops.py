"""Job parsing and dispatch for ``python -m stellaria_backend.dsp.run`` (see ``run.py``)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .common import SR, bad_job, bad_params, seconds, write_wav

OPS = ("trim", "split", "pitch", "fx", "comp", "concat", "export")
INPUT_COUNTS = {"trim": (1, 1), "split": (1, 1), "pitch": (1, 1), "fx": (1, 1), "export": (1, 1),
                "comp": (2, 8), "concat": (2, 32)}


@dataclass
class Input:
    path: str
    offset: float = 0.0


@dataclass
class Job:
    op: str
    inputs: list[Input]
    params: dict
    output_dir: Path | None
    bpm: float | None = None
    vst_plugins: dict = field(default_factory=dict)
    names: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict)


class Context:
    """Progress (monotonic, 0..1) and log callbacks handed to the ops."""

    def __init__(self, progress: Callable[..., None] | None = None, log: Callable[[str], None] | None = None) -> None:
        self._progress = progress or (lambda *_a, **_k: None)
        self._log = log or (lambda _m: None)

    def progress(self, fraction: float, stage: str, **extra: Any) -> None:
        self._progress(fraction, stage, **extra)

    def log(self, message: str) -> None:
        self._log(message)


def parse_job(raw: Any) -> Job:
    if not isinstance(raw, dict):
        raise bad_job("The job must be a JSON object")
    op = raw.get("op")
    if op not in OPS:
        raise bad_job(f"Unknown op {op!r}; expected one of {', '.join(OPS)}")
    inputs_raw = raw.get("inputs")
    if not isinstance(inputs_raw, list):
        raise bad_job("'inputs' must be a list of {path, offset}")
    inputs: list[Input] = []
    for item in inputs_raw:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not item["path"]:
            raise bad_job("Every input needs a 'path'")
        offset = item.get("offset", 0.0)
        if offset is None:
            offset = 0.0
        if isinstance(offset, bool) or not isinstance(offset, (int, float)) or not math.isfinite(float(offset)):
            raise bad_job(f"Input offset {offset!r} is not a number")
        inputs.append(Input(item["path"], max(0.0, float(offset))))
    lo, hi = INPUT_COUNTS[op]
    if not lo <= len(inputs) <= hi:
        want = str(lo) if lo == hi else f"{lo}–{hi}"
        raise bad_params("inputs.count", f"'{op}' takes {want} input(s), got {len(inputs)}",
                         min=lo, max=hi, count=len(inputs))
    params = raw.get("params") if raw.get("params") is not None else {}
    if not isinstance(params, dict):
        raise bad_job("'params' must be an object")
    out_dir = raw.get("outputDir")
    if op != "export":
        if not isinstance(out_dir, str) or not out_dir:
            raise bad_job("'outputDir' is required")
    bpm = raw.get("bpm")
    if isinstance(bpm, bool) or not isinstance(bpm, (int, float)) or not math.isfinite(float(bpm)) or bpm <= 0:
        bpm = None
    plugins = raw.get("vstPlugins") if isinstance(raw.get("vstPlugins"), dict) else {}
    names = raw.get("names") if isinstance(raw.get("names"), dict) else {}
    return Job(op, inputs, params, Path(out_dir) if isinstance(out_dir, str) and out_dir else None,
               float(bpm) if bpm is not None else None, plugins, names, raw)


def _op(name: str):
    if name in ("trim", "split", "comp", "concat"):
        from . import edit

        return getattr(edit, name)
    if name == "pitch":
        from .pitch import pitch

        return pitch
    if name == "fx":
        from .fxop import fx

        return fx
    from .export import export

    return export


def run_job(raw: Any, progress: Callable[..., None] | None = None, log: Callable[[str], None] | None = None) -> dict:
    """Run one job → ``{"outputs": {port: {path, offset, duration}}, "result": {...}}``."""
    job = parse_job(raw)
    ctx = Context(progress, log)
    started = time.perf_counter()
    outputs, result = _op(job.op)(job, ctx)
    written: dict[str, dict] = {}
    if outputs:
        assert job.output_dir is not None
        for k, out in enumerate(outputs):
            ctx.progress(0.9 + 0.1 * k / len(outputs), "dsp.write")
            path = write_wav(job.output_dir / f"{out.port}.wav", out.audio, SR)
            written[out.port] = {"path": str(path), "offset": round(float(out.offset), 6),
                                 "duration": seconds(out.audio.shape[-1])}
    ctx.progress(1.0, "dsp.write" if outputs else f"dsp.{job.op}")
    ctx.log(f"{job.op}: done in {time.perf_counter() - started:.2f} s")
    return {"outputs": written, "result": result}
