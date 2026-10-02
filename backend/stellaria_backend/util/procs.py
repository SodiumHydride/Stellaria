"""Subprocess helpers: no console windows on Windows, UTF-8 everywhere, process-tree control.

Never uses ``shell=True``; every argument list is passed as-is so paths with spaces or CJK
characters are safe.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import IO, Any, Mapping, Sequence

import psutil

IS_WINDOWS = sys.platform == "win32"
CREATE_NO_WINDOW = 0x08000000


def child_env(extra: Mapping[str, str] | None = None, pythonpath: Sequence[Path] = ()) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    if pythonpath:
        parts = [str(p) for p in pythonpath]
        if env.get("PYTHONPATH"):
            parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(parts)
    if extra:
        env.update(extra)
    return env


def popen(
    args: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stdin: int | IO[Any] | None = subprocess.DEVNULL,
    stdout: int | IO[Any] | None = subprocess.PIPE,
    stderr: int | IO[Any] | None = subprocess.STDOUT,
) -> subprocess.Popen:
    kwargs: dict[str, Any] = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = CREATE_NO_WINDOW
    return subprocess.Popen(
        [str(a) for a in args],
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else child_env(),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        **kwargs,
    )


def run(
    args: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess:
    kwargs: dict[str, Any] = {}
    if IS_WINDOWS:
        kwargs["creationflags"] = CREATE_NO_WINDOW
    return subprocess.run(
        [str(a) for a in args],
        cwd=str(cwd) if cwd else None,
        env=dict(env) if env is not None else child_env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=timeout,
        **kwargs,
    )


def kill_tree(pid: int, timeout: float = 5.0) -> None:
    """Terminate a process and all of its children (preprocess spawns worker processes).

    Tolerates processes that exit (or were already killed by someone else) while the tree is
    walked: ``NoSuchProcess`` (incl. zombies) and ``AccessDenied`` are ignored per process."""
    gone = (psutil.NoSuchProcess, psutil.AccessDenied)
    try:
        parent = psutil.Process(pid)
    except gone:
        return
    try:
        children = parent.children(recursive=True)
    except gone:  # the parent exited meanwhile; its children (if any) cannot be found from it any more
        children = []
    procs = children + [parent]
    for p in procs:
        try:
            p.terminate()
        except gone:
            pass
    try:
        _, alive = psutil.wait_procs(procs, timeout=timeout)
    except gone:
        alive = procs
    for p in alive:
        try:
            p.kill()
        except gone:
            pass


def process_alive(pid: int, create_time: float | None = None) -> bool:
    """True if ``pid`` runs and (when given) was created at ``create_time`` (guards pid reuse)."""
    try:
        proc = psutil.Process(pid)
        if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
            return False
        return create_time is None or abs(proc.create_time() - create_time) <= 1.0
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
