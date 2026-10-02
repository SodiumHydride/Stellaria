"""Running a JSON-lines CLI (``mixing.run``, ``dsp.run``) as a job's subprocess.

Protocol (stdout, one JSON object per line)::

    {"progress": 0..1, "stage": str, ...}     {"log": str}
    {"result": {...}}                          # last line, exit code 0
    {"error": {"code": str, "message": str, "params"?: {...}}}   # last line, exit code 1

Other stdout lines and stderr go to the job log. The process gets no console window, is killed
as a tree on cancel, and never inherits torch from the API process (it never had it).
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..engine.commands import BACKEND_ROOT
from ..errors import ApiError
from ..jobs import JobContext
from ..util import procs

if TYPE_CHECKING:
    from ..context import AppContext


@dataclass
class CliRun:
    """What a failed run left behind, for error mapping (e.g. a plugin that crashed the process)."""

    returncode: int | None = None
    last_progress: dict[str, Any] = field(default_factory=dict)
    last_log: str = ""
    tail: deque = field(default_factory=lambda: deque(maxlen=60))


class CliFailed(ApiError):
    def __init__(self, code: str, message: str, run: CliRun, params: dict[str, Any] | None = None) -> None:
        super().__init__(code, message, status=500, detail="\n".join(run.tail), params=params)
        self.run = run


def run_cli(ctx: "AppContext", jctx: JobContext, cmd: list[str], cwd: Path,
            on_progress: Callable[[float, dict[str, Any]], None], fail_code: str, fail_message: str) -> dict[str, Any]:
    """Run ``cmd`` and return its ``result``; raises ``CliFailed`` (code from the error line, else
    ``fail_code``) or ``JobCancelled``."""
    proc = procs.popen(cmd, cwd=cwd, env=procs.child_env(pythonpath=[BACKEND_ROOT]), stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
    jctx.on_cancel(lambda: procs.kill_tree(proc.pid))
    run = CliRun()

    def drain() -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            run.tail.append(line)
            jctx.log(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    assert proc.stdout is not None
    for raw in proc.stdout:
        text = raw.decode("utf-8", errors="replace").strip()
        try:
            line = json.loads(text)
        except ValueError:
            line = None
        if not isinstance(line, dict):
            if text:
                run.tail.append(text)
                jctx.log(text)
            continue
        if "progress" in line:
            run.last_progress = line
            try:
                on_progress(max(0.0, min(1.0, float(line["progress"]))), line)
            except (TypeError, ValueError):
                pass
        elif "log" in line:
            run.last_log = str(line["log"])
            run.tail.append(run.last_log)
            jctx.log(run.last_log)
        elif "result" in line:
            result = line["result"] if isinstance(line["result"], dict) else {}
        elif "error" in line:
            error = line["error"] if isinstance(line["error"], dict) else {}
    run.returncode = proc.wait()
    reader.join(timeout=5)
    jctx.on_cancel(None)
    jctx.check_cancelled()
    if error is not None or run.returncode != 0 or result is None:
        error = error or {}
        params = error.get("params") if isinstance(error.get("params"), dict) else None
        raise CliFailed(str(error.get("code") or fail_code), str(error.get("message") or fail_message), run, params)
    return result
