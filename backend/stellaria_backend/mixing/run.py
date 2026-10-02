"""CLI: ``python -m stellaria_backend.mixing.run <job.json>``.

job.json v2 (``docs/canvas-v3.md`` §4)::

    {"inputs": [{"path": str, "role": "lead|harmony|backing|other", "gainDb": float, "pan": float,
                 "offset": float, "mute": bool?, "solo": bool?}],
     "reference": {"vocals": str | null, "mix": str | null, "offset": float} | null,
     "processor": "builtin" | "bus", "preset": str, "params": dict, "seed": int, "output": str}

v1 (``vocals[]`` + ``backing``, ``docs/creation-v2.md`` §6) is still accepted.

stdout carries only JSON lines (one object each, flushed immediately)::

    {"progress": 0..1, "stage": "mix.analyze" | "mix.vocal" | "mix.reverb" | "mix.master"}
    {"log": str}
    {"result": {"lufs": float, "truePeakDb": float, "peakDb": float, "bpm": float | null,
                "vocalToBackingDb": float | null, "offset": float}}       # last line, exit code 0
    {"error": {"code": str, "message": str}}                              # last line, exit code 1

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

    def progress(fraction: float, stage: str) -> None:
        value = round(min(max(float(fraction), last[0]), 1.0), 3)   # never goes backwards
        last[0] = value
        emit({"progress": value, "stage": stage})

    def log(message: str) -> None:
        emit({"log": str(message)})

    try:
        from .audioio import MixError

        if len(argv) != 1:
            raise MixError("mix.bad_job", "usage: python -m stellaria_backend.mixing.run <job.json>")
        try:
            with open(argv[0], encoding="utf-8") as f:
                job = json.load(f)
        except FileNotFoundError:
            raise MixError("mix.bad_job", f"Job file not found: {argv[0]}") from None
        except (OSError, ValueError) as e:
            raise MixError("mix.bad_job", f"Cannot read the job file: {e}") from None

        from .mixer import mix

        result = mix(job, progress=progress, log=log)
        emit({"result": result})
        return 0
    except Exception as e:  # noqa: BLE001 - every failure becomes the final error line
        code = getattr(e, "code", None)
        message = getattr(e, "message", None) or str(e) or type(e).__name__
        if not isinstance(code, str):
            code = "mix.failed"
            message = f"{type(e).__name__}: {e}"
            traceback.print_exc(file=sys.stderr)
        try:
            emit({"error": {"code": code, "message": message}})
        except Exception:  # noqa: BLE001
            pass
        return 1
    finally:
        try:
            out.flush()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    sys.exit(main())
