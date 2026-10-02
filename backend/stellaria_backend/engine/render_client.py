"""Client side of the persistent render worker (one request at a time).

The worker is spawned lazily, kept alive between renders (models stay loaded) and shut down
when a training starts while it is idle (to free VRAM), after ``gpuIdleReleaseMin`` minutes
without requests, on ``release()`` (the user's "free GPU memory") or when the backend exits.
Cancelling a render kills the worker; the next render starts a fresh one. Requests may carry
``"device": "cpu"`` to render next to a training that holds the GPU.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Callable

from ..errors import ApiError, JobCancelled
from ..util import procs
from .commands import engine_env

if TYPE_CHECKING:
    from ..context import AppContext

log = logging.getLogger(__name__)

STARTUP_TIMEOUT = 300.0
IDLE_CHECK_S = 20.0


class RenderClient:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self._proc: subprocess.Popen | None = None
        self._lines: queue.Queue = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=400)
        self._lock = threading.Lock()  # one request at a time
        self._proc_lock = threading.Lock()
        self._next_id = 0
        self.device: str | None = None
        self._last_used = time.monotonic()
        self._watcher: threading.Thread | None = None
        self._closing = threading.Event()

    @property
    def stderr_tail(self) -> list[str]:
        return list(self._stderr)

    # ---- process ---------------------------------------------------------------------
    def _spawn(self) -> None:
        args = [self.ctx.python, "-m", "stellaria_backend.engine.render_worker", "--engine", str(self.ctx.options.engine)]
        if self.ctx.options.fake_engine:
            args.append("--fake")
        env = engine_env(self.ctx, {"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
        proc = procs.popen(args, cwd=self.ctx.options.engine, env=env, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self._lines = queue.Queue()
        threading.Thread(target=self._read_stdout, args=(proc, self._lines), daemon=True, name="render-out").start()
        threading.Thread(target=self._read_stderr, args=(proc,), daemon=True, name="render-err").start()
        self._proc = proc
        msg = self._next_message(STARTUP_TIMEOUT, None)
        if msg.get("type") != "ready":
            raise ApiError("render.worker_failed", "The render worker did not start", status=500,
                           detail="\n".join(self.stderr_tail[-60:]))
        self.device = msg.get("device")
        log.info("render worker ready on %s", self.device)
        if self._watcher is None or not self._watcher.is_alive():
            self._watcher = threading.Thread(target=self._watch_idle, daemon=True, name="render-idle")
            self._watcher.start()

    def _read_stdout(self, proc: subprocess.Popen, lines: queue.Queue) -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                lines.put(json.loads(raw.decode("utf-8", errors="replace")))
            except ValueError:
                self._stderr.append(raw.decode("utf-8", errors="replace").rstrip())
        lines.put(None)  # EOF

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                self._stderr.append(line)
                log.debug("render worker: %s", line)

    def _next_message(self, timeout: float, cancel: threading.Event | None) -> dict[str, Any]:
        waited = 0.0
        while True:
            if cancel is not None and cancel.is_set():
                self.shutdown(kill=True)
                raise JobCancelled()
            try:
                msg = self._lines.get(timeout=0.2)
            except queue.Empty:
                waited += 0.2
                if waited >= timeout:
                    self.shutdown(kill=True)
                    raise ApiError("render.timeout", "The render worker stopped responding", status=500,
                                   detail="\n".join(self.stderr_tail[-60:])) from None
                continue
            if msg is None:
                self._proc = None
                raise ApiError("render.worker_crashed", "The render worker exited unexpectedly", status=500,
                               detail="\n".join(self.stderr_tail[-80:]))
            return msg

    def shutdown(self, kill: bool = False) -> None:
        with self._proc_lock:
            proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            if not kill and proc.stdin:
                proc.stdin.write(b'{"cmd": "shutdown"}\n')
                proc.stdin.flush()
                proc.wait(timeout=10)
                return
        except (OSError, subprocess.TimeoutExpired):
            pass
        procs.kill_tree(proc.pid)

    def shutdown_if_idle(self) -> bool:
        """Stop the worker (freeing its VRAM) unless a render is in progress."""
        if not self._lock.acquire(blocking=False):
            return False
        try:
            was_loaded = self.loaded
            self.shutdown()
        finally:
            self._lock.release()
        if was_loaded:
            self.ctx.hub.publish("render.status", self.status())
        return True

    def close(self) -> None:
        """Backend exit: stop the idle watcher and the worker."""
        self._closing.set()
        self.shutdown()

    # ---- memory ----------------------------------------------------------------------
    @property
    def loaded(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def status(self) -> dict[str, Any]:
        """``{loaded, busy, device, idleSeconds, releaseAfterMin}`` (device only while loaded)."""
        loaded = self.loaded
        return {"loaded": loaded, "busy": self._lock.locked(), "device": self.device if loaded else None,
                "idleSeconds": round(time.monotonic() - self._last_used) if loaded else None,
                "releaseAfterMin": self.ctx.prefs.get().gpu_idle_release_min}

    def release(self) -> bool:
        """Frees what the singing models hold (GPU memory included: the worker exits, the next render
        starts it again). False while a render or pitch analysis is running, or when nothing is loaded."""
        return self.loaded and self.shutdown_if_idle()

    def _watch_idle(self) -> None:
        while not self._closing.wait(IDLE_CHECK_S):
            minutes = self.ctx.prefs.get().gpu_idle_release_min
            if minutes <= 0 or not self.loaded or self._lock.locked():
                continue
            if time.monotonic() - self._last_used >= minutes * 60 and self.release():
                log.info("render worker released after %d idle minute(s)", minutes)

    # ---- requests --------------------------------------------------------------------
    def render(self, args: dict[str, Any], on_progress: Callable[[float], None],
               cancel: threading.Event | None = None, timeout: float = 3600.0) -> dict[str, Any]:
        return self._request("render", args, on_progress, cancel, timeout)

    def pitch(self, args: dict[str, Any], on_progress: Callable[[float], None],
              cancel: threading.Event | None = None, timeout: float = 1800.0) -> dict[str, Any]:
        """The raw pitch line of ``args["input"]`` saved to ``args["output"]`` (see the worker)."""
        return self._request("pitch", args, on_progress, cancel, timeout)

    def _request(self, cmd: str, args: dict[str, Any], on_progress: Callable[[float], None],
                 cancel: threading.Event | None, timeout: float) -> dict[str, Any]:
        try:
            with self._lock:
                try:
                    return self._send(cmd, args, on_progress, cancel, timeout)
                finally:
                    self._last_used = time.monotonic()
        finally:
            self.ctx.hub.publish("render.status", self.status())  # after the lock: no longer busy

    def _send(self, cmd: str, args: dict[str, Any], on_progress: Callable[[float], None],
              cancel: threading.Event | None, timeout: float) -> dict[str, Any]:
        if self._proc is None or self._proc.poll() is not None:
            self._spawn()
            self.ctx.hub.publish("render.status", self.status())
        self._next_id += 1
        rid = self._next_id
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write((json.dumps({"id": rid, "cmd": cmd, "args": args}) + "\n").encode("utf-8"))
            self._proc.stdin.flush()
        except OSError as exc:
            self._proc = None
            raise ApiError("render.worker_crashed", "The render worker is not running", status=500,
                           detail="\n".join(self.stderr_tail[-60:])) from exc
        while True:
            msg = self._next_message(timeout, cancel)
            if msg.get("id") != rid:
                continue
            if msg["type"] == "progress":
                on_progress(float(msg.get("value", 0.0)))
            elif msg["type"] == "result":
                return msg.get("data") or {}
            elif msg["type"] == "error":
                raise ApiError(msg.get("code", "render.failed"), msg.get("message", "Render failed"),
                               status=500, detail=msg.get("detail", ""))
