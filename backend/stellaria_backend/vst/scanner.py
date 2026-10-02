"""Plugin discovery and the worker subprocess.

* Default folders per OS (canvas-v3 §3.2) plus the user's folders.
* Plugin files: ``*.vst3`` (bundles or single files) everywhere, ``*.component`` on macOS; folders
  are searched recursively (vendor sub-folders) but never inside a bundle.
* A file's cache signature is its mtime (for bundles: the newest mtime inside it).
* Plugin ids are stable: ``sha1(path + pluginName)``.
* ``run_worker`` starts ``worker.py`` in its own process (no console window on Windows, own
  process group elsewhere), with a timeout and cancellation; it maps what happened to ``vst.*``
  errors.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..errors import ApiError
from ..util import procs

WORKER = Path(__file__).with_name("worker.py")
PREFIX = "STELLARIA_VST "
SCAN_TIMEOUT = 30.0
INSPECT_TIMEOUT = 30.0
EDITOR_TIMEOUT = 12 * 3600.0  # the window stays open as long as the user wants
MAX_DEPTH = 6
MAX_BUNDLE_ENTRIES = 4000
IS_MAC = sys.platform == "darwin"


def default_folders() -> list[str]:
    home = Path.home()
    if procs.IS_WINDOWS:
        common = os.environ.get("COMMONPROGRAMFILES") or r"C:\Program Files\Common Files"
        local = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return [str(Path(common) / "VST3"), str(Path(local) / "Programs" / "Common" / "VST3")]
    if IS_MAC:
        return ["/Library/Audio/Plug-Ins/VST3", str(home / "Library/Audio/Plug-Ins/VST3"),
                "/Library/Audio/Plug-Ins/Components", str(home / "Library/Audio/Plug-Ins/Components")]
    return [str(home / ".vst3"), "/usr/lib/vst3", "/usr/local/lib/vst3"]


def formats() -> list[str]:
    return ["vst3", "au"] if IS_MAC else ["vst3"]


def plugin_format(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix == ".vst3":
        return "vst3"
    if suffix == ".component" and IS_MAC:
        return "au"
    return None


def path_key(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(path))))


def plugin_id(path: str | os.PathLike[str], plugin_name: str | None) -> str:
    """Stable id: ``sha1(path + pluginName)`` (path normalized, case-folded on Windows)."""
    return hashlib.sha1((path_key(path) + (plugin_name or "")).encode("utf-8")).hexdigest()


def discover(folders: list[str]) -> list[Path]:
    """Plugin files/bundles in ``folders`` (existing ones), each once, in a stable order."""
    found: dict[str, Path] = {}

    def walk(folder: Path, depth: int) -> None:
        try:
            entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            return
        for entry in entries:
            if plugin_format(entry):
                found.setdefault(path_key(entry), entry)
            elif depth < MAX_DEPTH and entry.is_dir() and not entry.name.startswith("."):
                walk(entry, depth + 1)

    for folder in folders:
        path = Path(folder).expanduser()
        if path.is_dir():
            walk(path, 0)
        elif plugin_format(path) and path.exists():  # a plugin given directly
            found.setdefault(path_key(path), path)
    return list(found.values())


def signature(path: Path) -> float:
    """Newest mtime of a plugin file or bundle (bundles change inside, not at the top)."""
    try:
        newest = path.stat().st_mtime
    except OSError:
        return 0.0
    if path.is_dir():
        count = 0
        for root, dirs, files in os.walk(path):
            for name in files + dirs:
                count += 1
                if count > MAX_BUNDLE_ENTRIES:
                    return newest
                try:
                    newest = max(newest, os.stat(os.path.join(root, name)).st_mtime)
                except OSError:
                    pass
    return newest


# ---- worker ----------------------------------------------------------------------------
@dataclass
class WorkerRun:
    """How to start workers; tests swap ``python``/``script`` for fakes."""

    python: str
    script: Path = WORKER


def _spawn(argv: list[str], stdout: Any, stderr: Any) -> subprocess.Popen:
    kwargs: dict[str, Any] = {}
    if procs.IS_WINDOWS:
        kwargs["creationflags"] = procs.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    env = procs.child_env()
    env.pop("PYTHONPATH", None)  # the worker needs pedalboard + numpy from the runtime, nothing of ours
    return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, env=env,
                            cwd=tempfile.gettempdir(), **kwargs)


def _kill(proc: subprocess.Popen) -> None:
    from ..plugins.runner import kill_process_tree

    kill_process_tree(proc)


def run_worker(run: WorkerRun, args: list[str], *, name: str, timeout: float,
               cancelled: Callable[[], bool] | None = None,
               on_start: Callable[[Callable[[], None]], None] | None = None) -> dict[str, Any]:
    """Run ``worker.py <args>`` and return its ``ok`` payload.

    Raises ``vst.timeout`` {name, seconds}, ``vst.crashed`` {name} (died without an answer),
    ``vst.load_failed`` {name, reason}, ``vst.host_missing``; returns ``{"cancelled": True}`` when
    ``cancelled()`` turned true (the process is killed). ``on_start`` receives a kill function."""
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        try:
            proc = _spawn([run.python, str(run.script), *args], out, err)
        except OSError as exc:
            raise ApiError("vst.host_missing", f"Could not start the plugin host: {exc}", status=409,
                           detail=str(exc)) from exc
        kill_lock = threading.Lock()
        killed = False

        def kill() -> None:  # called by the cancel handler (API thread) and by this loop: once is enough
            nonlocal killed
            with kill_lock:
                if killed:
                    return
                killed = True
            _kill(proc)

        if on_start is not None:
            on_start(kill)
        deadline = time.monotonic() + timeout
        timed_out = False
        while proc.poll() is None:
            if cancelled is not None and cancelled():
                kill()
                break
            if time.monotonic() > deadline:
                timed_out = True
                kill()
                break
            try:
                proc.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
        proc.wait()
        out.seek(0)
        err.seek(0)
        text = out.read().decode("utf-8", errors="replace")
        tail = "\n".join(err.read().decode("utf-8", errors="replace").splitlines()[-40:])
    if cancelled is not None and cancelled():
        return {"cancelled": True}
    if timed_out:
        raise ApiError("vst.timeout", f"'{name}' did not answer within {int(timeout)} s", status=504,
                       detail=tail, params={"name": name, "seconds": int(timeout)})
    line = next((ln for ln in reversed(text.splitlines()) if ln.startswith(PREFIX)), None)
    answer: dict[str, Any] | None = None
    if line is not None:
        try:
            answer = json.loads(line[len(PREFIX):])
        except ValueError:
            answer = None
    if not isinstance(answer, dict):
        raise ApiError("vst.crashed", f"'{name}' crashed the plugin host (exit code {proc.returncode})", status=500,
                       detail=tail, params={"name": name})
    if "error" in answer:
        error = answer.get("error") or {}
        code = error.get("code") or "vst.load_failed"
        reason = f"{error.get('type') or 'Error'}: {error.get('message') or ''}".strip()
        if code == "vst.host_missing":
            raise ApiError("vst.host_missing", "The VST host add-on (pedalboard) is not installed", status=409,
                           detail=reason)
        raise ApiError("vst.load_failed", f"'{name}' could not be loaded: {reason}", status=422,
                       detail=f"{reason}\n{tail}".strip(), params={"name": name, "reason": reason[:300]})
    return answer.get("ok") or {}
