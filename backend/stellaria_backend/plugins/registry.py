"""``PluginRegistry`` (``ctx.plugins``): the plugins the user added, persisted in
``<data>/plugins/registry.json``, their status, interpreter checks, install and test jobs.

The registry stores the user's choices (enabled, interpreter, python path, setting values, a
rename) and the last interpreter check. Manifests are re-read from disk on every view, so edits
to a folder's ``stellaria-plugin.json`` take effect at once; the last valid manifest is kept to
show something sensible while the current one is broken.
"""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from ..errors import ApiError
from ..schemas import Msg, msg
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso
from . import envs
from .manifest import (
    ID_RE, RESERVED_IDS, bad_manifest, guess_requirements, load_folder_manifest, load_wizard_manifest,
    script_candidates, setting_tokens, slugify, tokens, uses_python, wizard_manifest,
)
from .models import Manifest, Plugin, PluginPatch, PythonCheck, RegistryEntry, WizardSpec
from .pycheck import run_check

if TYPE_CHECKING:
    from ..context import AppContext
    from ..jobs import Job, JobContext

log = logging.getLogger(__name__)

TESTS_KEPT = 5
TORCH_DISTS = {"torch", "torchaudio", "torchvision"}


@dataclass
class RunTarget:
    """Everything ``run_plugin`` needs, resolved and checked."""

    id: str
    name: str
    manifest: Manifest
    folder: Path
    python: Path | None
    env: dict[str, str]
    settings: dict[str, Any] = field(default_factory=dict)
    custom: bool = False


def not_found(pid: str) -> ApiError:
    return ApiError("plugin.not_found", f"Plugin '{pid}' not found", status=404, params={"id": pid})


def _absolute_folder(folder: str) -> Path:
    if not folder or not folder.strip():
        raise ApiError("request.invalid", "folder is required", status=422)
    path = Path(folder.strip()).expanduser()
    if not path.is_absolute():
        raise ApiError("request.relative_path", f"Paths must be absolute: {folder}", status=422)
    if not path.is_dir():
        raise ApiError("plugin.folder_missing", f"Folder not found: {folder}", status=404, params={"folder": folder})
    return path.resolve()


class PluginRegistry:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.root = ctx.options.data / "plugins"
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "registry.json"
        self._lock = threading.RLock()
        self._check_locks: dict[str, threading.Lock] = {}
        self._entries: dict[str, RegistryEntry] = self._load()

    # ---- persistence -----------------------------------------------------------------
    def _load(self) -> dict[str, RegistryEntry]:
        raw = read_json(self._path, {}) or {}
        out: dict[str, RegistryEntry] = {}
        for item in raw.get("plugins", []) if isinstance(raw, dict) else []:
            try:
                entry = RegistryEntry.model_validate(item)
            except ValidationError:
                log.warning("ignoring a broken plugin registry entry: %r", item)
                continue
            out[entry.id] = entry
        return out

    def _save(self) -> None:
        with self._lock:
            write_json(self._path, {"version": 1, "plugins": [e.model_dump(by_alias=True, mode="json")
                                                             for e in self._entries.values()]})

    def _entry(self, pid: str) -> RegistryEntry:
        with self._lock:
            entry = self._entries.get(pid)
        if entry is None:
            raise not_found(pid)
        return entry

    def _wizard_path(self, pid: str) -> Path:
        return self.root / f"{pid}.json"

    def _publish(self, pid: str) -> None:
        try:
            self.ctx.hub.publish("plugin", self.get(pid).wire())
        except ApiError:
            pass

    # ---- manifests & views -----------------------------------------------------------
    def manifest(self, pid: str) -> Manifest:
        return self._manifest(self._entry(pid))

    def _manifest(self, entry: RegistryEntry) -> Manifest:
        folder = Path(entry.folder)
        if not folder.is_dir():
            raise ApiError("plugin.folder_missing", f"The plugin folder is gone: {entry.folder}", status=409,
                           params={"folder": entry.folder})
        if entry.manifest_source == "wizard":
            try:
                manifest, _ = load_wizard_manifest(self._wizard_path(entry.id))
            except FileNotFoundError as exc:
                raise bad_manifest("the wizard manifest file is missing") from exc
        else:
            try:
                manifest = load_folder_manifest(folder)
            except FileNotFoundError as exc:
                raise bad_manifest("stellaria-plugin.json was removed from the plugin folder") from exc
            if manifest.id != entry.id:
                raise bad_manifest(f"the manifest id changed from '{entry.id}' to '{manifest.id}'")
        data = manifest.wire()
        if entry.last_manifest != data:
            with self._lock:
                entry.last_manifest = data
                self._save()
        return manifest

    @staticmethod
    def _settings(entry: RegistryEntry, manifest: Manifest) -> dict[str, Any]:
        return {s.id: entry.setting_values.get(s.id, s.default) for s in manifest.settings}

    @staticmethod
    def _check_key(entry: RegistryEntry, manifest: Manifest) -> str:
        return "|".join([entry.interpreter, entry.python_path or "", manifest.runtime, *manifest.requirements])

    def _check_current(self, entry: RegistryEntry, manifest: Manifest) -> bool:
        if entry.check is None or entry.checked_for != self._check_key(entry, manifest):
            return False
        if entry.interpreter == "builtin" and entry.check.error is None:
            return envs.env_ready(self.ctx.options.data, entry.id)  # env removed / app runtime changed
        return True

    def _status(self, entry: RegistryEntry, manifest: Manifest) -> tuple[str, Msg | None]:
        if uses_python(manifest):
            check = entry.check
            if not self._check_current(entry, manifest) or check is None:
                return "needs_setup", Msg(code="plugin.not_checked")
            if check.error is not None:
                return "error", check.error
            if check.missing:
                return "needs_setup", msg("plugin.packages_missing", packages=", ".join(check.missing))
        else:
            from .runner import resolve_program

            first = manifest.command[0] if manifest.command else ""
            if set(tokens(first)) <= {"folder"}:
                program = first.replace("{folder}", str(entry.folder))
                if resolve_program(program, Path(entry.folder), None) is None:
                    return "error", msg("plugin.program_missing", program=program)
        values = self._settings(entry, manifest)
        for sid in sorted(setting_tokens(manifest)):
            if values.get(sid) in (None, ""):
                return "needs_setup", msg("plugin.setting_missing", setting=sid)
        return "ready", None

    def _view(self, entry: RegistryEntry) -> Plugin:
        status: str
        error: Msg | None
        try:
            manifest: Manifest | None = self._manifest(entry)
            status, error = self._status(entry, manifest)  # type: ignore[arg-type]
        except ApiError as exc:
            status, error = "error", Msg(code=exc.code, params=exc.params)
            manifest = None
            if entry.last_manifest:
                try:
                    manifest = Manifest.model_validate(entry.last_manifest)
                except ValidationError:
                    manifest = None
        if manifest is None:
            manifest = Manifest(id=entry.id, name=entry.name or entry.id, command=["{python}"])
        return Plugin(
            id=entry.id, name=entry.name or manifest.name, version=manifest.version, kind=manifest.kind,
            runtime=manifest.runtime, folder=entry.folder, manifest_source=entry.manifest_source,
            interpreter=entry.interpreter, python_path=entry.python_path, status=status,  # type: ignore[arg-type]
            check=entry.check, params=manifest.params, settings=manifest.settings,
            setting_values=self._settings(entry, manifest), enabled=entry.enabled, added_at=entry.added_at,
            error=error, requirements=list(manifest.requirements), max_vocals=manifest.vocals_limit,
            checked_at=entry.checked_at,
        )

    def list(self, refresh: bool = True) -> list[Plugin]:
        """All plugins in the order they were added. ``refresh`` re-checks plugins whose last check
        no longer matches (interpreter/requirements changed, builtin env gone) — once, then the
        stored check is current again (also when it failed)."""
        with self._lock:
            entries = sorted(self._entries.values(), key=lambda e: e.added_at)
        if refresh:
            for entry in entries:
                if entry.enabled and self._needs_refresh(entry):
                    try:
                        self.check(entry.id)
                    except ApiError:
                        pass
            with self._lock:
                entries = sorted(self._entries.values(), key=lambda e: e.added_at)
        return [self._view(e) for e in entries]

    def _needs_refresh(self, entry: RegistryEntry) -> bool:
        try:
            manifest = self._manifest(entry)
        except ApiError:
            return False
        return uses_python(manifest) and not self._check_current(entry, manifest)

    def get(self, pid: str) -> Plugin:
        return self._view(self._entry(pid))

    def mix_processors(self) -> list[dict[str, Any]]:
        """``MixProcessor`` wire dicts for enabled plugins of kind 'mix' (status from the last check)."""
        out = []
        for plugin in self.list():
            if not plugin.enabled or plugin.kind != "mix":
                continue
            out.append({
                "id": plugin.id, "name": plugin.name, "kind": "plugin", "presets": [],
                "params": [p.wire() for p in plugin.params], "maxVocals": plugin.max_vocals,
                "available": plugin.status == "ready", "error": plugin.error.wire() if plugin.error else None,
            })
        return out

    # ---- add / wizard / update / delete ----------------------------------------------
    def add_folder(self, folder: str) -> Plugin:
        path = _absolute_folder(folder)
        try:
            manifest = load_folder_manifest(path)
        except FileNotFoundError:
            raise ApiError("plugin.no_manifest", f"No stellaria-plugin.json in {path}", status=409,
                           detail={"candidates": script_candidates(path), "requirements": guess_requirements(path),
                                   "folder": str(path)},  # type: ignore[arg-type]
                           params={"folder": str(path)}) from None
        with self._lock:
            existing = self._entries.get(manifest.id)
            if existing is not None and not (existing.manifest_source == "folder" and Path(existing.folder) == path):
                raise ApiError("plugin.exists", f"A plugin with id '{manifest.id}' is already added", status=409,
                               params={"id": manifest.id, "folder": existing.folder})
            if existing is None:  # adding the same folder again just re-checks it
                self._entries[manifest.id] = RegistryEntry(id=manifest.id, folder=str(path), manifest_source="folder",
                                                           added_at=now_iso(), last_manifest=manifest.wire())
                self._save()
        return self.check(manifest.id)

    def _new_id(self, name: str, folder: Path) -> str:
        base = slugify(name) or slugify(folder.name) or "plugin"
        if base in RESERVED_IDS or not ID_RE.match(base):
            base = "plugin"
        candidate, i = base, 2
        while candidate in self._entries or self._wizard_path(candidate).exists():
            candidate = f"{base[:56]}-{i}"
            i += 1
        return candidate

    def add_wizard(self, spec: WizardSpec) -> Plugin:
        path = _absolute_folder(spec.folder)
        if not spec.name.strip():
            raise bad_manifest("name is empty")
        with self._lock:
            pid = self._new_id(spec.name, path)
            manifest = wizard_manifest(spec.model_copy(update={"folder": str(path)}), pid)
            data = {**manifest.wire(), "folder": str(path)}
            write_json(self._wizard_path(pid), data)
            self._entries[pid] = RegistryEntry(id=pid, folder=str(path), manifest_source="wizard",
                                               added_at=now_iso(), last_manifest=manifest.wire())
            self._save()
        return self.check(pid)

    def update(self, pid: str, patch: PluginPatch) -> Plugin:
        fields = patch.model_fields_set
        with self._lock:
            entry = self._entry(pid).model_copy(deep=True)
            before = (entry.interpreter, entry.python_path)
            if "enabled" in fields and patch.enabled is not None:
                entry.enabled = patch.enabled
            if "name" in fields:
                entry.name = (patch.name or "").strip() or None
            if "python_path" in fields:
                value = (patch.python_path or "").strip() or None
                if value and not Path(value).expanduser().is_absolute():
                    raise ApiError("request.relative_path", f"Paths must be absolute: {value}", status=422)
                entry.python_path = value
            if "interpreter" in fields and patch.interpreter is not None:
                entry.interpreter = patch.interpreter
            if entry.interpreter == "custom" and not entry.python_path:
                raise ApiError("plugin.python_missing", "Choose a Python interpreter for this plugin", status=422,
                               params={"path": ""})
            if "setting_values" in fields and patch.setting_values is not None:
                values = dict(entry.setting_values)
                for key, value in patch.setting_values.items():
                    if value is None:
                        values.pop(key, None)
                    else:
                        values[key] = value
                entry.setting_values = values
            self._entries[pid] = entry
            self._save()
        if (entry.interpreter, entry.python_path) != before:
            return self.check(pid)
        self._publish(pid)
        return self.get(pid)

    def remove(self, pid: str) -> None:
        entry = self._entry(pid)
        for job in self.ctx.jobs.list(active_only=True):
            if job.target.get("pluginId") == pid:
                raise ApiError("plugin.busy", "The plugin is installing or running", status=409, params={"id": pid})
        with self._lock:
            self._entries.pop(pid, None)
            self._save()
        if entry.manifest_source == "wizard":
            self._wizard_path(pid).unlink(missing_ok=True)
        envs.remove_env(self.ctx.options.data, pid)
        shutil.rmtree(self.root / "tests" / pid, ignore_errors=True)
        self.ctx.hub.publish("plugin.deleted", {"id": pid})

    # ---- checks ----------------------------------------------------------------------
    def check(self, pid: str) -> Plugin:
        """Re-check the plugin's interpreter (creates a builtin env when missing)."""
        with self._lock:
            lock = self._check_locks.setdefault(pid, threading.Lock())
        with lock:
            entry = self._entry(pid)
            try:
                manifest = self._manifest(entry)
            except ApiError:
                self._publish(pid)
                return self.get(pid)
            result = self._run_check(entry, manifest)
            with self._lock:
                current = self._entries.get(pid)
                if current is None:
                    raise not_found(pid)
                current.check = result
                current.checked_at = now_iso()
                current.checked_for = self._check_key(current, manifest)
                self._save()
        self._publish(pid)
        return self.get(pid)

    def _run_check(self, entry: RegistryEntry, manifest: Manifest) -> PythonCheck | None:
        if not uses_python(manifest):
            return None
        min_version = (3, 8) if manifest.runtime == "python" else None
        wants_torch = any(envs.normalize_name(envs.requirement_name(r)) in TORCH_DISTS for r in manifest.requirements)
        if entry.interpreter == "builtin":
            venv = envs.env_dir(self.ctx.options.data, entry.id)
            try:
                py = envs.ensure_env(self.ctx, entry.id)
            except ApiError as exc:
                return PythonCheck(ok=False, executable=str(envs.venv_python(venv)),
                                   error=msg(exc.code, reason=str(exc.detail)[-300:]))
            env = envs.interpreter_env(py, custom=False, venv=venv)
        else:
            if not entry.python_path:
                return PythonCheck(ok=False, executable="", error=msg("python.not_found", path=""))
            py = envs.resolve_python_path(entry.python_path)
            env = envs.interpreter_env(py, custom=True)
        return run_check(py, manifest.requirements, torch=wants_torch, min_version=min_version, env=env)

    def run_target(self, pid: str, require_enabled: bool = True) -> RunTarget:
        """Resolve and verify what a run needs; raises ``plugin.*`` errors."""
        entry = self._entry(pid)
        if require_enabled and not entry.enabled:
            raise ApiError("plugin.disabled", f"Plugin '{pid}' is disabled", status=409, params={"id": pid})
        manifest = self._manifest(entry)
        name = entry.name or manifest.name
        settings = self._settings(entry, manifest)
        for sid in sorted(setting_tokens(manifest)):
            if settings.get(sid) in (None, ""):
                raise ApiError("plugin.setting_missing", f"Set '{sid}' in the plugin's settings first", status=409,
                               params={"setting": sid})
        folder = Path(entry.folder)
        custom = entry.interpreter == "custom"
        if not uses_python(manifest):
            return RunTarget(pid, name, manifest, folder, None, envs.interpreter_env(None, custom=True), settings, True)
        if not self._check_current(entry, manifest) or entry.check is None or not entry.check.ok:
            self.check(pid)
            entry = self._entry(pid)
        check = entry.check
        if check is None or check.error is not None:
            code = check.error.code if check is not None and check.error is not None else "python.not_found"
            path = (entry.python_path or "") if custom else str(envs.venv_python(envs.env_dir(self.ctx.options.data, pid)))
            raise ApiError("plugin.python_missing", f"The plugin's Python interpreter is not usable ({code})",
                           status=409, detail=path, params={"path": path, "reason": code})
        if check.missing:
            raise ApiError("plugin.packages_missing", "The plugin needs packages that are not installed: "
                           + ", ".join(check.missing), status=409, params={"packages": ", ".join(check.missing)})
        if custom:
            python = envs.resolve_python_path(entry.python_path or "")
            env = envs.interpreter_env(python, custom=True)
        else:
            venv = envs.env_dir(self.ctx.options.data, pid)
            python = envs.ensure_env(self.ctx, pid)
            env = envs.interpreter_env(python, custom=False, venv=venv)
        return RunTarget(pid, name, manifest, folder, python, env, settings, custom)

    # ---- jobs ------------------------------------------------------------------------
    def submit_install(self, pid: str, target: str) -> "Job":
        entry = self._entry(pid)
        manifest = self._manifest(entry)
        if target not in ("builtin", "custom"):
            raise ApiError("request.invalid", "target must be 'builtin' or 'custom'", status=422)
        if target == "custom" and not entry.python_path:
            raise ApiError("plugin.python_missing", "Choose a Python interpreter for this plugin first", status=409,
                           params={"path": ""})
        existing = self.ctx.jobs.find_active("install_plugin", pluginId=pid)
        if existing is not None:
            return existing
        name = entry.name or manifest.name

        def missing_in(python: Path, env: dict[str, str]) -> list[str]:
            found = run_check(python, manifest.requirements, torch=False, env=env)
            if found.error is not None:
                raise ApiError("plugin.python_missing", f"The interpreter is not usable ({found.error.code})",
                               status=409, params={"path": str(python), "reason": found.error.code})
            return found.missing

        def run(jctx: "JobContext") -> dict[str, Any]:
            data = self.ctx.options.data
            installed: list[str] = []
            if target == "builtin":
                jctx.stage("stage.plugin_env")
                venv = envs.env_dir(data, pid)
                python = envs.ensure_env(self.ctx, pid, log=jctx.log)
                missing = missing_in(python, envs.interpreter_env(python, custom=False, venv=venv))
                if missing:
                    jctx.log(f"missing: {', '.join(missing)}")
                    jctx.stage("stage.install")
                    installed = envs.install_into_env(self.ctx, jctx, python, venv, list(manifest.requirements))
                else:
                    jctx.log("all requirements are already available")
            else:
                python = envs.resolve_python_path(entry.python_path or "")
                if not python.is_file():
                    raise ApiError("plugin.python_missing", f"Python not found: {python}", status=409,
                                   params={"path": str(python), "reason": "python.not_found"})
                missing = missing_in(python, envs.interpreter_env(python, custom=True))
                if missing:
                    jctx.log(f"missing: {', '.join(missing)}")
                    jctx.stage("stage.install")
                    installed = envs.install_into_custom(self.ctx, jctx, python, missing)
                else:
                    jctx.log("all requirements are already available")
            jctx.stage("stage.check")
            plugin = self.check(pid)
            if self._entry(pid).interpreter == target and plugin.check is not None and plugin.check.missing:
                raise ApiError("plugin.packages_missing", "Still missing after the install: "
                               + ", ".join(plugin.check.missing), status=409,
                               params={"packages": ", ".join(plugin.check.missing)})
            return {"pluginId": pid, "target": target, "installed": installed, "status": plugin.status}

        return self.ctx.jobs.submit("install_plugin", msg("job.install_plugin", plugin=name), run,
                                    target={"pluginId": pid}, lock_key=f"plugin:{pid}")

    def test_output(self, pid: str, job_id: str) -> Path:
        return self.root / "tests" / pid / f"{job_id}.wav"

    def submit_test(self, pid: str, params: dict[str, Any] | None = None) -> "Job":
        from .runner import run_plugin
        from .testdata import ensure_testdata

        entry = self._entry(pid)
        manifest = self._manifest(entry)
        name = entry.name or manifest.name

        def run(jctx: "JobContext") -> dict[str, Any]:
            import soundfile as sf

            backing, vocal = ensure_testdata(self.ctx.options.data)
            output = self.test_output(pid, jctx.job.id)
            if manifest.kind == "mix":
                run_plugin(self.ctx, jctx, pid, vocals=[vocal], backing=backing, input=None, output=output,
                           params=params or {}, require_enabled=False)
            else:
                run_plugin(self.ctx, jctx, pid, vocals=[], backing=None, input=vocal, output=output,
                           params=params or {}, require_enabled=False)
            info = sf.info(str(output))
            self._trim_tests(pid, keep=output)
            return {"mediaUrl": f"/plugins/{pid}/test/{jctx.job.id}/audio", "pluginId": pid,
                    "duration": info.frames / info.samplerate if info.samplerate else 0.0}

        return self.ctx.jobs.submit("plugin_test", msg("job.plugin_test", plugin=name), run,
                                    target={"pluginId": pid}, lock_key=f"plugin:{pid}")

    def _trim_tests(self, pid: str, keep: Path) -> None:
        folder = self.root / "tests" / pid
        files = sorted(folder.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for old in files[TESTS_KEPT:]:
            if old != keep:
                old.unlink(missing_ok=True)

    # ---- helpers for the API ---------------------------------------------------------
    def wizard_hints(self, folder: str) -> dict[str, Any]:
        path = _absolute_folder(folder)
        return {"candidates": script_candidates(path), "requirements": guess_requirements(path)}
