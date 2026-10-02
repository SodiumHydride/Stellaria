"""``VstRegistry`` (``ctx.vst``): folders, scanning with a cache, the editor and inspection.

Never imports pedalboard. Everything that loads a plugin runs ``worker.py`` in a short-lived
subprocess (see ``scanner.run_worker``), so a plugin that crashes or hangs only costs that
process. Files:

* ``<data>/vst/settings.json`` — ``{"folders": [...]}`` (the user's folders)
* ``<data>/vst/cache.json`` — ``{"version", "lastScan", "files": {key: {path, signature, plugins}}}``
  keyed by the normalized path; a file is re-loaded only when its signature (mtime) changes.
"""

from __future__ import annotations

import base64
import binascii
import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..errors import ApiError, JobCancelled
from ..schemas import Msg, msg
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso
from . import scanner
from .models import VstParam, VstPlugin, VstStatus

if TYPE_CHECKING:
    from ..context import AppContext
    from ..jobs import Job, JobContext

log = logging.getLogger(__name__)

CACHE_VERSION = 1
HOST_NEGATIVE_TTL = 5.0


def host_missing() -> ApiError:
    return ApiError("vst.host_missing", "The VST host add-on (pedalboard) is not installed", status=409)


def not_found(pid: str) -> ApiError:
    return ApiError("vst.not_found", f"VST plugin '{pid}' not found", status=404, params={"name": pid})


def _check_state(state: str | None) -> str | None:
    if state is None or state == "":
        return None
    try:
        base64.b64decode(state, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ApiError("request.invalid", "state must be base64", status=422, detail=str(exc)) from exc
    return state


class VstRegistry:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.root = ctx.options.data / "vst"
        self.root.mkdir(parents=True, exist_ok=True)
        self._settings_path = self.root / "settings.json"
        self._cache_path = self.root / "cache.json"
        self._lock = threading.RLock()
        # the interpreter the vsthost add-on is installed into (the add-on installer uses sys.executable)
        self.worker = scanner.WorkerRun(python=os.environ.get("STELLARIA_VST_PYTHON") or sys.executable)
        self.scan_workers = 3
        self.editor_override: bool | None = None  # tests / development
        self._scanning = False
        self._host: tuple[float, dict[str, Any]] | None = None

    # ---- host --------------------------------------------------------------------------
    def host_info(self, refresh: bool = False) -> dict[str, Any]:
        """``{installed, version, formats}`` without importing pedalboard here."""
        if self.worker.python == sys.executable and self.worker.script == scanner.WORKER:
            importlib.invalidate_caches()
            installed = importlib.util.find_spec("pedalboard") is not None
            version = None
            if installed:
                try:
                    version = metadata.version("pedalboard")
                except metadata.PackageNotFoundError:
                    version = None
            return {"installed": installed, "version": version, "formats": scanner.formats()}
        cached = self._host
        if cached and not refresh and (cached[1]["installed"] or time.monotonic() - cached[0] < HOST_NEGATIVE_TTL):
            return cached[1]
        try:  # another interpreter: ask it (cached; a positive answer stays valid)
            ok = scanner.run_worker(self.worker, ["probe"], name="pedalboard", timeout=60.0)
            info = {"installed": True, "version": ok.get("version"), "formats": ok.get("formats") or scanner.formats()}
        except ApiError:
            info = {"installed": False, "version": None, "formats": scanner.formats()}
        self._host = (time.monotonic(), info)
        return info

    def host_available(self) -> bool:
        return bool(self.host_info()["installed"])

    def require_host(self) -> None:
        if not self.host_available():
            raise host_missing()

    def editor_supported(self) -> bool:
        if self.editor_override is not None:
            return self.editor_override
        return sys.platform in ("win32", "darwin")

    # ---- settings ----------------------------------------------------------------------
    def folders(self) -> list[str]:
        data = read_json(self._settings_path, {}) or {}
        return [str(f) for f in data.get("folders", []) if isinstance(f, str)]

    def set_folders(self, folders: list[str]) -> dict[str, Any]:
        clean: list[str] = []
        for folder in folders:
            text = (folder or "").strip()
            if not text:
                continue
            if not Path(text).expanduser().is_absolute():
                raise ApiError("request.relative_path", f"Paths must be absolute: {text}", status=422)
            if scanner.path_key(text) not in {scanner.path_key(c) for c in clean}:
                clean.append(text)
        with self._lock:
            write_json(self._settings_path, {"folders": clean})
        status = self.status()
        self.ctx.hub.publish("vst.status", status)
        return status

    def scan_folders(self) -> list[str]:
        return scanner.default_folders() + self.folders()

    # ---- cache & views -----------------------------------------------------------------
    def _load_cache(self) -> dict[str, Any]:
        data = read_json(self._cache_path, None)
        if not isinstance(data, dict) or data.get("version") != CACHE_VERSION:
            return {"version": CACHE_VERSION, "lastScan": None, "files": {}}
        data.setdefault("files", {})
        return data

    def _save_cache(self, cache: dict[str, Any]) -> None:
        with self._lock:
            write_json(self._cache_path, cache)

    def status(self) -> dict[str, Any]:
        host = self.host_info()
        return VstStatus(
            host_installed=host["installed"], host_version=host["version"], formats=host["formats"],
            folders=self.folders(), default_folders=scanner.default_folders(),
            editor_supported=self.editor_supported(), last_scan=self._load_cache().get("lastScan"),
            scanning=self._scanning,
        ).wire()

    def plugins(self) -> list[dict[str, Any]]:
        """``VstPlugin`` wire dicts from the cache (instruments included, marked by ``category``)."""
        files = self._load_cache()["files"]
        out = [p for entry in files.values() for p in entry.get("plugins", [])]
        return sorted(out, key=lambda p: ((p.get("name") or "").lower(), p.get("path") or ""))

    def get(self, plugin_id: str) -> dict[str, Any] | None:
        return next((p for p in self.plugins() if p.get("id") == plugin_id), None)

    def require(self, plugin_id: str) -> dict[str, Any]:
        plugin = self.get(plugin_id)
        if plugin is None:
            raise not_found(plugin_id)
        return plugin

    # ---- scanning ----------------------------------------------------------------------
    @staticmethod
    def _failed(path: Path, fmt: str, plugin_name: str | None, exc: ApiError, when: str) -> dict[str, Any]:
        return VstPlugin(
            id=scanner.plugin_id(path, plugin_name), name=plugin_name or path.stem, format=fmt,  # type: ignore[arg-type]
            path=str(path), plugin_name=plugin_name, status="failed", error=Msg(code=exc.code, params=exc.params),
            scanned_at=when,
        ).wire()

    @staticmethod
    def _plugin(path: Path, fmt: str, info: dict[str, Any], when: str) -> dict[str, Any]:
        params = []
        for raw in info.get("params") or []:
            try:
                params.append(VstParam.model_validate(raw))
            except Exception:  # noqa: BLE001 - one odd parameter does not hide the plugin
                log.warning("skipping a parameter of %s: %r", path, raw)
        name = info.get("pluginName")
        return VstPlugin(
            id=scanner.plugin_id(path, name), name=str(info.get("name") or name or path.stem),
            vendor=info.get("vendor"), version=info.get("version"), format=fmt,  # type: ignore[arg-type]
            path=str(path), plugin_name=name, category=info.get("category") or "effect", params=params,
            status="ok", scanned_at=when,
        ).wire()

    def _scan_file(self, path: Path, cancelled: Callable[[], bool],
                   track: Callable[[Callable[[], None]], None]) -> list[dict[str, Any]] | None:
        fmt = scanner.plugin_format(path) or "vst3"
        when = now_iso()

        def ask(args: list[str], name: str) -> dict[str, Any]:
            return scanner.run_worker(self.worker, args, name=name, timeout=scanner.SCAN_TIMEOUT,
                                      cancelled=cancelled, on_start=track)

        try:
            answer = ask(["scan", str(path)], path.stem)
        except ApiError as exc:
            if exc.code == "vst.host_missing":
                raise
            return [self._failed(path, fmt, None, exc, when)]
        if answer.get("cancelled"):
            return None
        if "names" in answer:  # a shell bundle: one process per plugin inside
            out: list[dict[str, Any]] = []
            for name in answer["names"]:
                try:
                    sub = ask(["scan", str(path), name], name)
                except ApiError as exc:
                    if exc.code == "vst.host_missing":
                        raise
                    out.append(self._failed(path, fmt, name, exc, when))
                    continue
                if sub.get("cancelled"):
                    return None
                out += [self._plugin(path, fmt, {**p, "pluginName": name}, when) for p in sub.get("plugins", [])]
            return out
        return [self._plugin(path, fmt, p, when) for p in answer.get("plugins", [])]

    def submit_scan(self, full: bool = False) -> "Job":
        self.require_host()
        existing = self.ctx.jobs.find_active("vst_scan")
        if existing is not None:
            return existing

        def run(jctx: "JobContext") -> dict[str, Any]:
            self._scanning = True
            self.ctx.hub.publish("vst.status", self.status())
            kills: set[Callable[[], None]] = set()
            kills_lock = threading.Lock()

            def track(kill: Callable[[], None]) -> None:
                with kills_lock:
                    kills.add(kill)

            def cancel_all() -> None:
                with kills_lock:
                    pending = list(kills)
                for kill in pending:
                    kill()

            jctx.on_cancel(cancel_all)
            try:
                return self._run_scan(jctx, full, track)
            finally:
                jctx.on_cancel(None)
                self._scanning = False
                self.ctx.hub.publish("vst.status", self.status())

        return self.ctx.jobs.submit("vst_scan", msg("job.vst_scan"), run, lock_key="vst:scan")

    def _run_scan(self, jctx: "JobContext", full: bool, track: Callable[[Callable[[], None]], None]) -> dict[str, Any]:
        files = scanner.discover(self.scan_folders())
        cache = self._load_cache()
        entries: dict[str, Any] = cache["files"]
        present = {scanner.path_key(f): f for f in files}
        for key in [k for k in entries if k not in present]:
            del entries[key]  # uninstalled
        todo = []
        for key, path in present.items():
            sig = scanner.signature(path)
            if full or entries.get(key, {}).get("signature") != sig:
                todo.append((key, path, sig))
        jctx.log(f"{len(files)} plugin file(s) in {len(self.scan_folders())} folder(s); loading {len(todo)}")
        total = len(todo)
        jctx.progress(0.0 if total else 1.0, msg("stage.vst_scan", name="", done=0, total=total))
        done = 0
        if todo:
            with ThreadPoolExecutor(max_workers=max(1, self.scan_workers), thread_name_prefix="vst-scan") as pool:
                futures = {pool.submit(self._scan_file, path, lambda: jctx.cancelled, track): (key, path, sig)
                           for key, path, sig in todo}
                try:
                    for future in as_completed(futures):
                        key, path, sig = futures[future]
                        plugins = future.result()
                        done += 1
                        if plugins is not None:
                            entries[key] = {"path": str(path), "signature": sig, "plugins": plugins}
                            self._save_cache(cache)
                            for p in plugins:
                                state = "failed: " + p["error"]["code"] if p["status"] == "failed" else "ok"
                                jctx.log(f"{path.name}: {p['name']} ({state})")
                        jctx.progress(done / total, msg("stage.vst_scan", name=path.stem, done=done, total=total))
                except ApiError:
                    for f in futures:
                        f.cancel()
                    raise
        self._save_cache(cache)
        jctx.check_cancelled()
        cache["lastScan"] = now_iso()
        self._save_cache(cache)
        plugins = self.plugins()
        self.ctx.hub.publish("vst.plugins", plugins)
        failed = sum(1 for p in plugins if p.get("status") == "failed")
        return {"found": len(plugins) - failed, "failed": failed, "scanned": total}

    # ---- editor & inspect ------------------------------------------------------------
    def _job_file(self, plugin: dict[str, Any], state: str | None) -> Path:
        folder = self.ctx.tmp_dir / "vst"
        folder.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(prefix="vst-", suffix=".json", dir=folder)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"path": plugin["path"], "pluginName": plugin.get("pluginName"), "state": state}, f)
        return Path(name)

    def inspect(self, plugin_id: str, state: str | None) -> dict[str, Any]:
        """``{params}``: current values for a state (a short subprocess)."""
        plugin = self.require(plugin_id)
        state = _check_state(state)
        self.require_host()
        job_file = self._job_file(plugin, state)
        try:
            ok = scanner.run_worker(self.worker, ["inspect", str(job_file)], name=plugin["name"],
                                    timeout=scanner.INSPECT_TIMEOUT)
        finally:
            job_file.unlink(missing_ok=True)
        return {"params": ok.get("params") or {}}

    def submit_editor(self, plugin_id: str, state: str | None) -> "Job":
        plugin = self.require(plugin_id)
        state = _check_state(state)
        if not self.editor_supported():
            raise ApiError("vst.editor_unsupported", "Plugin windows are only supported on Windows and macOS",
                           status=409)
        self.require_host()

        def run(jctx: "JobContext") -> dict[str, Any]:
            job_file = self._job_file(plugin, state)
            try:
                ok = scanner.run_worker(self.worker, ["editor", str(job_file)], name=plugin["name"],
                                        timeout=scanner.EDITOR_TIMEOUT, cancelled=lambda: jctx.cancelled,
                                        on_start=jctx.on_cancel)
            finally:
                jctx.on_cancel(None)
                job_file.unlink(missing_ok=True)
            if ok.get("cancelled"):
                raise JobCancelled()
            return {"state": ok.get("state"), "params": ok.get("params") or {}}

        return self.ctx.jobs.submit("vst_editor", msg("job.vst_editor", name=plugin["name"]), run,
                                    target={"vstPluginId": plugin_id})
