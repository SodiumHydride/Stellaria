"""CLI: ``python -m stellaria_backend.dsp.run <job.json>`` — the process node's DSP ops.

job.json (docs/canvas-v3.md §6)::

    {"op": "trim|split|pitch|fx|comp|concat|export",
     "inputs": [{"path": str, "offset": float}], "params": {...}, "outputDir": str,
     "bpm": float | null, "vstPlugins": {id: VstPlugin}, "names": {...},
     "defaultFolder": str | null}            # export only: used (and created) when params.folder is null

stdout carries only JSON lines (one object each, flushed immediately), as ``mixing.run``::

    {"progress": 0..1, "stage": "dsp.<op>" | "dsp.read" | "dsp.write" | "dsp.fx.vst" | ..., ...}
        # fx lines also carry "slot" and "effect" (built-in) or "plugin" (VST name, emitted right
        # before the plugin runs: if it crashes the process, this is the last line)
    {"log": str}
    {"result": {"outputs": {port: {"path", "offset", "duration"}}, "result": {...}}}   # exit code 0
    {"error": {"code": str, "message": str, "params": {...}}}                          # exit code 1

Anything else written to stdout by libraries is redirected to stderr.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import IO, Any


def _protocol_stream() -> IO[str]:
    """A private handle on the real stdout; fd 1 is pointed at stderr for everyone else."""
    try:
        sys.stdout.flush()
        fd = os.dup(1)
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        return os.fdopen(fd, "w", encoding="utf-8", buffering=1, newline="\n")
    except (OSError, ValueError, AttributeError):
        stream = sys.stdout
        sys.stdout = sys.stderr
        return stream


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    out = _protocol_stream()

    def emit(obj: dict[str, Any]) -> None:
        out.write(json.dumps(obj, ensure_ascii=True, allow_nan=False) + "\n")
        out.flush()

    last = [-1.0]

    def progress(fraction: float, stage: str, **extra: Any) -> None:
        value = round(min(max(float(fraction), last[0]), 1.0), 3)   # never goes backwards
        last[0] = value
        emit({"progress": value, "stage": stage, **extra})

    def log(message: str) -> None:
        emit({"log": str(message)})

    try:
        from .common import DspError

        if len(argv) != 1:
            raise DspError("dsp.bad_job", "usage: python -m stellaria_backend.dsp.run <job.json>")
        try:
            with open(argv[0], encoding="utf-8") as f:
                job = json.load(f)
        except FileNotFoundError:
            raise DspError("dsp.bad_job", f"Job file not found: {argv[0]}") from None
        except (OSError, ValueError) as e:
            raise DspError("dsp.bad_job", f"Cannot read the job file: {e}") from None

        from .ops import run_job

        result = run_job(job, progress=progress, log=log)
        emit({"result": result})
        return 0
    except Exception as e:  # noqa: BLE001 - every failure becomes the final error line
        code = getattr(e, "code", None)
        message = getattr(e, "message", None) or str(e) or type(e).__name__
        params = getattr(e, "params", None)
        if not isinstance(code, str):
            code = "dsp.failed"
            message = f"{type(e).__name__}: {e}"
            params = {}
            traceback.print_exc(file=sys.stderr)
        try:
            emit({"error": {"code": code, "message": message, "params": _jsonable(params)}})
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        try:
            out.flush()
        except Exception:  # noqa: BLE001
            pass


def _jsonable(params: Any) -> dict:
    if not isinstance(params, dict):
        return {}
    clean: dict[str, Any] = {}
    for k, v in params.items():
        clean[str(k)] = v if isinstance(v, (str, int, float, bool)) or v is None else str(v)
    return clean


if __name__ == "__main__":
    sys.exit(main())
