"""Separation models (vocals / accompaniment) and cleanup models (the dry voice without reverb,
echo or noise): which exist, which are ready, which is the default.

* Demucs ``htdemucs`` / ``htdemucs_ft``: part of the ``separator`` add-on (Demucs fetches its
  own weights). Separation only.
* Roformer checkpoints offered for download: manifest resources with a ``separation`` block
  ``{arch, config, sdr?, aliases?}`` or a ``cleanup`` block ``{arch, config, kind, target, sdr?,
  aliases?}``; ``config`` names a file in ``roformer/configs``.
* Roformer checkpoints imported from a folder someone already has (Music-Source-Separation-
  Training, its GUIs, UVR …): ``<data>/separation-models/<id>/{model.json, config.yaml, <ckpt>}``;
  ``model.json`` says the role (``separation`` when absent) and, for cleanup, the kind.

Every entry has a role: ``separation`` or ``cleanup``; most methods take the role and default
to separation. Roformer models need the add-on too (its packages run them; Demucs adds
drums/bass/other for 4 stems). The default separation model is ``Preferences.separationModel``
while that one is ready, else the best ready one: Roformer resources by vocal SDR, then imported
Roformers, then ``htdemucs``. The default cleanup model is ``Preferences.cleanupModel`` while
ready, else the first ready one that removes reverb (resources in manifest order, then imported
ones), else none. This module never imports torch: configs are read with ``roformer.config``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ApiError, not_found
from ..jobs import JobContext
from ..resources.manager import walk_folders
from ..schemas import CleanupModel, CleanupModels, Msg, SeparationModel, SeparationModels, msg
from ..util.ids import new_id
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso
from .roformer.config import CONFIG_DIR, ConfigProblem, RoformerSpec, cleanup_kind, describe, load_config

if TYPE_CHECKING:
    from ..context import AppContext
    from ..resources.manifest import ResourceDef

ADDON = "separator"
DEFAULT_DEMUCS = "htdemucs"
DEMUCS_NAMES: dict[str, dict[str, str]] = {
    "htdemucs": {"zh-CN": "Demucs（htdemucs）", "en": "Demucs (htdemucs)"},
    "htdemucs_ft": {"zh-CN": "Demucs 精调版（htdemucs_ft，约慢 4 倍）",
                    "en": "Demucs fine-tuned (htdemucs_ft, ~4× slower)"},
}
AUTO = ("", "auto")
RUNTIME_MODULES = ("einops", "rotary_embedding_torch")
CHECKPOINT_SUFFIXES = (".ckpt", ".chpt", ".pth", ".pt")
CONFIG_SUFFIXES = (".yaml", ".yml")
MIN_CHECKPOINT = 1 << 20  # smaller files are not separation models
SEPARATION, CLEANUP = "separation", "cleanup"
ID_PREFIX = {SEPARATION: "sep", CLEANUP: "cln"}
AUTO_CLEANUP_KINDS = ("dereverb", "dereverb_echo")  # what 'automatic' may pick: no denoise-only models


def _block(rdef: "ResourceDef") -> dict[str, Any] | None:
    return rdef.separation or rdef.cleanup


def _role(rdef: "ResourceDef") -> str:
    return SEPARATION if rdef.separation else CLEANUP


@dataclass
class SeparationEntry:
    id: str
    engine: str  # 'demucs' | 'roformer'
    arch: str  # 'htdemucs' | 'bs_roformer' | 'mel_band_roformer'
    source: str  # 'addon' | 'resource' | 'local'
    name: dict[str, str]
    resource_id: str | None = None
    sdr: float | None = None
    checkpoint: Path | None = None
    config: Path | None = None
    size: int | None = None
    created_at: str = ""
    role: str = SEPARATION
    kind: str | None = None  # cleanup: 'dereverb' | 'dereverb_echo' | 'denoise'


@dataclass
class Candidate:
    """A checkpoint found in a folder, with the config that describes it."""

    checkpoint: Path
    config: Path | None = None  # None for manifest checkpoints (their config ships with the app)
    spec: RoformerSpec | None = None
    resource_id: str | None = None


@dataclass
class _Scan:
    checkpoints: list[Path] = field(default_factory=list)
    configs: list[Path] = field(default_factory=list)
    gui_configs: dict[str, list[Path]] = field(default_factory=dict)  # checkpoint name → config paths
    gui_types: dict[str, str] = field(default_factory=dict)  # checkpoint name → model type


class SeparationCatalog:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.root = ctx.options.data / "separation-models"
        self._importing: set[tuple[str, int]] = set()
        self._lock = threading.Lock()

    # ---- entries ---------------------------------------------------------------------------
    def _resource_defs(self, role: str | None = None) -> list["ResourceDef"]:
        """Checkpoint resources of ``role`` (None: both)."""
        return [r for r in self.ctx.resources.manifest.resources
                if _block(r) is not None and (role is None or _role(r) == role)]

    def _resource_entries(self, role: str = SEPARATION) -> list[SeparationEntry]:
        """Separation resources best first (by vocal SDR); cleanup resources in manifest order."""
        models_dir = self.ctx.resources.models_dir
        entries = []
        for r in self._resource_defs(role):
            block = _block(r) or {}
            entries.append(SeparationEntry(
                id=r.id, engine="roformer", arch=block["arch"], source="resource", name=r.name, resource_id=r.id,
                sdr=block.get("sdr"), checkpoint=r.primary_file(models_dir), config=CONFIG_DIR / block["config"],
                size=r.size, role=role, kind=block.get("kind") if role == CLEANUP else None))
        return sorted(entries, key=lambda e: -(e.sdr or 0.0)) if role == SEPARATION else entries

    def _local_record(self, folder: Path) -> dict[str, Any] | None:
        try:
            rec = read_json(folder / "model.json")
        except (OSError, ValueError):
            return None
        return rec if isinstance(rec, dict) and rec.get("id") == folder.name and rec.get("file") else None

    def _local_entries(self, role: str = SEPARATION) -> list[SeparationEntry]:
        entries = []
        if self.root.is_dir():
            for folder in self.root.iterdir():
                rec = self._local_record(folder) if folder.is_dir() else None
                if rec is None or str(rec.get("role") or SEPARATION) != role:
                    continue
                name = str(rec.get("name") or rec["id"])
                entries.append(SeparationEntry(
                    id=rec["id"], engine="roformer", arch=str(rec.get("arch") or ""), source="local",
                    name={"zh-CN": name, "en": name}, checkpoint=folder / rec["file"], config=folder / "config.yaml",
                    size=rec.get("size"), created_at=str(rec.get("createdAt") or ""), role=role,
                    kind=str(rec.get("kind") or "dereverb") if role == CLEANUP else None))
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def entries(self, role: str = SEPARATION) -> list[SeparationEntry]:
        demucs = [SeparationEntry(id=mid, engine="demucs", arch="htdemucs", source="addon", name=names,
                                  resource_id=ADDON) for mid, names in DEMUCS_NAMES.items()]
        return [*(demucs if role == SEPARATION else []), *self._resource_entries(role), *self._local_entries(role)]

    def get(self, model_id: str, role: str = SEPARATION) -> SeparationEntry | None:
        return next((e for e in self.entries(role) if e.id == model_id), None)

    # ---- readiness -------------------------------------------------------------------------
    def addon_ready(self) -> bool:
        return self.ctx.resources.is_ready(ADDON)

    def runtime_ready(self) -> bool:
        """The add-on's Roformer packages are importable (installs from before lack them)."""
        if self.ctx.options.fake_engine:
            return True
        importlib.invalidate_caches()
        return all(importlib.util.find_spec(m) is not None for m in RUNTIME_MODULES)

    def _readiness(self) -> tuple[bool, bool]:
        """(add-on ready, its Roformer packages importable)."""
        addon = self.addon_ready()
        return addon, addon and self.runtime_ready()

    def missing(self, entry: SeparationEntry, readiness: tuple[bool, bool] | None = None) -> str | None:
        """What keeps ``entry`` from running: 'addon', 'addon_update', 'model' or None."""
        addon, runtime = readiness or self._readiness()
        if not addon:
            return "addon"
        if entry.engine == "demucs":
            return None
        if not runtime:
            return "addon_update"
        if entry.source == "resource":
            return None if self.ctx.resources.is_ready(entry.resource_id or "") else "model"
        files = (entry.checkpoint, entry.config)
        return None if all(f is not None and f.is_file() for f in files) else "model"

    def best(self, readiness: tuple[bool, bool] | None = None, role: str = SEPARATION) -> str | None:
        """The best ready model: ``htdemucs`` when no Roformer separation model is ready; None
        when no cleanup model that removes reverb is ready."""
        readiness = readiness or self._readiness()
        candidates = [*self._resource_entries(role), *self._local_entries(role)]
        if role == CLEANUP:
            candidates = [e for e in candidates if e.kind in AUTO_CLEANUP_KINDS]
        for entry in candidates:
            if self.missing(entry, readiness) is None:
                return entry.id
        return DEFAULT_DEMUCS if role == SEPARATION else None

    def _chosen(self, readiness: tuple[bool, bool], role: str) -> str | None:
        """The preferred model of ``role`` while it is ready, else the best ready one."""
        prefs = self.ctx.prefs.get()
        chosen = prefs.separation_model if role == SEPARATION else prefs.cleanup_model
        if chosen and chosen not in AUTO:
            entry = self.get(chosen, role)
            if entry is not None and self.missing(entry, readiness) is None:
                return entry.id
        return self.best(readiness, role)

    def default(self, readiness: tuple[bool, bool] | None = None) -> str:
        """``Preferences.separationModel`` while it is ready, else the best ready model."""
        return self._chosen(readiness or self._readiness(), SEPARATION) or DEFAULT_DEMUCS

    def cleanup_default(self, readiness: tuple[bool, bool] | None = None) -> str | None:
        """``Preferences.cleanupModel`` while it is ready, else the first ready dereverb model."""
        return self._chosen(readiness or self._readiness(), CLEANUP)

    def known(self, model: Any, role: str = SEPARATION) -> bool:
        key = str(model or "").strip()
        return key in AUTO or self.get(key, role) is not None

    def resolve(self, model: Any, role: str = SEPARATION) -> SeparationEntry:
        """The model to separate (or clean up) with; None / '' / 'auto' = the default, and for
        cleanup while none is ready, the first downloadable one (so the error names what to
        get). 422 ``{role}.unknown_model`` {model}; 409 ``addon.missing`` {resource},
        ``addon.outdated`` {resource}, ``{role}.model_missing`` {model, resource?},
        ``cleanup.unavailable`` (no cleanup model exists at all)."""
        key = str(model or "").strip()
        readiness = self._readiness()
        if key not in AUTO:
            wanted: str | None = key
        elif role == SEPARATION:
            wanted = self.default(readiness)
        else:
            fallback = next((e.id for e in self._resource_entries(CLEANUP)), None)
            wanted = self.cleanup_default(readiness) or fallback
            if wanted is None:
                raise ApiError("cleanup.unavailable", "No vocal cleanup model is available", status=409)
        entry = self.get(wanted, role) if wanted else None
        if entry is None:
            raise ApiError(f"{role}.unknown_model", f"Unknown {role} model '{key}'", status=422, params={"model": key})
        reason = self.missing(entry, readiness)
        if reason == "addon":
            raise ApiError("addon.missing", "Vocal separation needs the 'separator' add-on", status=409,
                           detail=f"resource={ADDON}", params={"resource": ADDON})
        if reason == "addon_update":
            raise ApiError("addon.outdated", "Reinstall the vocal separation add-on to use Roformer models", status=409,
                           params={"resource": ADDON})
        if reason == "model":
            params = {"model": entry.id, **({"resource": entry.resource_id} if entry.source == "resource" else {})}
            raise ApiError(f"{role}.model_missing", f"The {role} model '{entry.id}' is not installed",
                           status=409, params=params)
        return entry

    # ---- wire view -------------------------------------------------------------------------
    def _state(self, entry: SeparationEntry, readiness: tuple[bool, bool]) -> dict[str, Any]:
        """What both views say about an entry: readiness, what is missing, size, checkpoint path."""
        reason = self.missing(entry, readiness)
        size = entry.size
        if size is None and entry.checkpoint is not None and entry.checkpoint.is_file():
            size = entry.checkpoint.stat().st_size
        path = str(entry.checkpoint) if entry.checkpoint is not None and reason is None and entry.source != "addon" \
            else None
        return {"ready": reason is None, "missing": reason, "size": size, "path": path}

    def _view(self, entry: SeparationEntry, readiness: tuple[bool, bool]) -> SeparationModel:
        return SeparationModel(
            id=entry.id, engine=entry.engine, arch=entry.arch, source=entry.source, resource_id=entry.resource_id,
            name=Msg(code=f"separation.{entry.id}.name", params=dict(entry.name)),
            quality="high" if entry.engine == "roformer" else "standard", sdr=entry.sdr,
            **self._state(entry, readiness))

    def view(self) -> SeparationModels:
        readiness = self._readiness()
        return SeparationModels(models=[self._view(e, readiness) for e in self.entries()],
                                default_model=self.default(readiness))

    def cleanup_view(self) -> CleanupModels:
        readiness = self._readiness()
        models = [
            CleanupModel(id=entry.id, kind=entry.kind or "dereverb", arch=entry.arch, source=entry.source,
                         resource_id=entry.resource_id, sdr=entry.sdr,
                         name=Msg(code=f"cleanup.{entry.id}.name", params=dict(entry.name)),
                         **self._state(entry, readiness))
            for entry in self.entries(CLEANUP)
        ]
        return CleanupModels(models=models, default_model=self.cleanup_default(readiness))

    def publish(self, role: str | None = None) -> None:
        """``separation`` / ``cleanup`` events with the current views (None: both)."""
        if role in (None, SEPARATION):
            self.ctx.hub.publish("separation", self.view().wire(), throttle_key="separation", min_interval=0.25)
        if role in (None, CLEANUP):
            self.ctx.hub.publish("cleanup", self.cleanup_view().wire(), throttle_key="cleanup", min_interval=0.25)

    def resource_changed(self, rdef: "ResourceDef") -> None:
        """A resource was installed or removed: refresh the lists it affects."""
        if rdef.id == ADDON:
            self.publish()
        elif _block(rdef) is not None:
            self.publish(_role(rdef))

    # ---- importing from a folder -----------------------------------------------------------
    def _manifest_match(self, ckpt: Path, size: int) -> "ResourceDef | None":
        name = ckpt.name.lower()
        for rdef in self._resource_defs():
            names = {Path(rdef.install.get("dest", "")).name.lower(),
                     *(str(a).lower() for a in (_block(rdef) or {}).get("aliases") or [])}
            if name in names and (rdef.size is None or rdef.size == size):
                return rdef
        return None

    def _local_duplicate(self, ckpt: Path, size: int) -> str | None:
        for entry in [*self._local_entries(SEPARATION), *self._local_entries(CLEANUP)]:
            if entry.checkpoint is not None and entry.checkpoint.name == ckpt.name and entry.size == size:
                return entry.id
        return None

    @staticmethod
    def _walk(root: Path) -> _Scan:
        found = _Scan()
        for folder, entries in walk_folders(root):
            for entry in entries.values():
                if not entry.is_file():
                    continue
                path, lower = Path(entry.path), entry.name.lower()
                if lower.endswith(CHECKPOINT_SUFFIXES):
                    found.checkpoints.append(path)
                elif lower.endswith(CONFIG_SUFFIXES):
                    found.configs.append(path)
                elif lower.startswith("model_config") and lower.endswith(".json"):
                    _read_gui_config(path, found)
        return found

    def _config_for(self, ckpt: Path, scan: _Scan) -> Path | None:
        for path in scan.gui_configs.get(ckpt.name, []):
            if path.is_file():
                return path
        stem, name = ckpt.stem.lower(), ckpt.name.lower()
        wanted = {stem, name, f"config_{stem}", f"{stem}_config"}
        matches = [c for c in scan.configs if c.stem.lower() in wanted]
        if matches:  # nearest first: same folder, then by path length
            return sorted(matches, key=lambda c: (c.parent != ckpt.parent, len(c.parts), str(c)))[0]
        beside = [c for c in scan.configs if c.parent == ckpt.parent]
        if len(beside) == 1 and len([k for k in scan.checkpoints if k.parent == ckpt.parent]) == 1:
            return beside[0]
        return None

    def scan(self, root: Path, config: Path | None = None) -> tuple[list[Candidate], list[dict[str, Any]]]:
        """Separation and cleanup checkpoints under ``root`` (a folder, or one checkpoint with an
        optional ``config``) → (candidates, skipped [{path, reason: Msg}]). Checkpoints without a config
        are not reported when they are probably another kind of model (``.pt`` files, folders
        holding a ``config.json``)."""
        if root.is_file():
            if not root.name.lower().endswith(CHECKPOINT_SUFFIXES):
                raise ApiError("request.invalid", "Choose the checkpoint (.ckpt, .pth, .pt) itself", status=422)
            scan = _Scan(checkpoints=[root])
            scan.configs = [config] if config is not None else [
                p for p in root.parent.iterdir() if p.is_file() and p.suffix.lower() in CONFIG_SUFFIXES]
            single = True
        else:
            scan, single = self._walk(root), False
        candidates: list[Candidate] = []
        skipped: list[dict[str, Any]] = []
        seen: set[str] = set()

        def skip(path: Path, code: str, **params: Any) -> None:
            skipped.append({"path": str(path), "reason": msg(code, **params).wire()})

        for ckpt in sorted(scan.checkpoints):
            real = os.path.realpath(ckpt)
            if real in seen:
                continue
            seen.add(real)
            try:
                size = ckpt.stat().st_size
            except OSError:
                continue
            if size < MIN_CHECKPOINT and not single:
                continue
            rdef = self._manifest_match(ckpt, size)
            if rdef is not None:
                if self.ctx.resources.view(rdef.id).status in ("ready", "downloading", "installing"):
                    skip(ckpt, "separation.skip.duplicate", model=rdef.id)
                else:
                    candidates.append(Candidate(checkpoint=ckpt, resource_id=rdef.id))
                continue
            cfg = config if single and config is not None else self._config_for(ckpt, scan)
            if cfg is None:
                if single or (ckpt.suffix.lower() != ".pt" and not (ckpt.parent / "config.json").is_file()):
                    skip(ckpt, "separation.skip.no_config")
                continue
            try:
                spec = describe(load_config(cfg), scan.gui_types.get(ckpt.name))
            except ConfigProblem as exc:
                skip(ckpt, exc.code, **exc.params)
                continue
            duplicate = self._local_duplicate(ckpt, size)
            with self._lock:
                importing = (ckpt.name, size) in self._importing
            if duplicate or importing:
                skip(ckpt, "separation.skip.duplicate", model=duplicate or ckpt.stem)
                continue
            candidates.append(Candidate(checkpoint=ckpt, config=cfg, spec=spec))
        return candidates, skipped

    def import_from(self, path: Path, config: Path | None = None) -> dict[str, Any]:
        """Import every usable separation or cleanup checkpoint in ``path`` (a folder, or one
        checkpoint): manifest checkpoints as those resources, others as imported models (one job
        each). Returns ``{found: [{id, name, arch, role, kind, path, config, jobId, resourceId}],
        skipped: [{path, reason}]}`` (``kind``: the cleanup kind, None for separation)."""
        if not path.exists():
            raise ApiError("import.not_found", f"'{path}' does not exist", status=422, params={"path": str(path)})
        if config is not None and not config.is_file():
            raise ApiError("import.not_found", f"'{config}' does not exist", status=422, params={"path": str(config)})
        candidates, skipped = self.scan(path, config)
        found = []
        for cand in candidates:
            if cand.resource_id is not None:
                rdef = self.ctx.resources.definition(cand.resource_id)
                block = _block(rdef) or {}
                job = self.ctx.resources.import_local(cand.resource_id, cand.checkpoint)
                found.append({"id": cand.resource_id, "name": rdef.name.get("en") or cand.resource_id,
                              "arch": block.get("arch"), "role": _role(rdef), "kind": block.get("kind"),
                              "path": str(cand.checkpoint), "config": None, "jobId": job.id,
                              "resourceId": cand.resource_id})
                continue
            mid, job_id = self._submit_import(cand)
            role = cand.spec.role if cand.spec else SEPARATION
            found.append({"id": mid, "name": cand.checkpoint.stem, "arch": cand.spec.arch if cand.spec else None,
                          "role": role, "kind": cleanup_kind(cand.checkpoint.name) if role == CLEANUP else None,
                          "path": str(cand.checkpoint), "config": str(cand.config), "jobId": job_id,
                          "resourceId": None})
        return {"found": found, "skipped": skipped}

    def _submit_import(self, cand: Candidate) -> tuple[str, str]:
        assert cand.config is not None and cand.spec is not None
        ckpt, spec = cand.checkpoint, cand.spec
        size = ckpt.stat().st_size
        role = spec.role
        mid = new_id(ID_PREFIX[role])
        folder = self.root / mid
        key = (ckpt.name, size)
        with self._lock:
            self._importing.add(key)

        def run(jctx: JobContext) -> dict[str, Any]:
            try:
                jctx.stage("stage.import_local", 0.0)
                folder.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(cand.config, folder / "config.yaml")  # type: ignore[arg-type]
                self.ctx.resources.copy_verified(jctx, ckpt, folder / ckpt.name, None)
                write_json(folder / "model.json", {
                    "id": mid, "name": ckpt.stem, "arch": spec.arch, "role": role,
                    **({"kind": cleanup_kind(ckpt.name)} if role == CLEANUP else {}), "file": ckpt.name,
                    "size": size, "instruments": spec.instruments, "outputs": spec.outputs, "sourcePath": str(ckpt),
                    "configSourcePath": str(cand.config), "createdAt": now_iso(),
                })
            except BaseException:
                shutil.rmtree(folder, ignore_errors=True)
                raise
            finally:
                with self._lock:
                    self._importing.discard(key)
                self.publish(role)
            return {"separationModelId": mid, "role": role, "path": str(folder / ckpt.name)}

        title = msg("job.import_separation" if role == SEPARATION else "job.import_cleanup", name=ckpt.stem)
        job = self.ctx.jobs.submit("download", title, run, target={"separationModelId": mid}, lock_key=f"sep:{mid}")
        return mid, job.id

    def delete(self, model_id: str, role: str = SEPARATION) -> None:
        """Remove an imported model of ``role`` (manifest ones are removed under Resources: 409
        ``separation.read_only``)."""
        entry = self.get(model_id, role)
        if entry is None:
            raise not_found(f"{role}_model", model_id)
        if entry.source != "local":
            raise ApiError("separation.read_only", "Built-in and downloadable models are managed under Resources",
                           status=409, params={"model": model_id})
        shutil.rmtree(self.root / model_id, ignore_errors=True)
        self.publish(role)


def _read_gui_config(path: Path, scan: _Scan) -> None:
    """MSST-GUI's ``model_config_*.json``: ``config_paths`` {checkpoint: [config, config-fast]}
    relative to its folder, ``model_types`` {checkpoint: type}."""
    try:
        data = read_json(path)
    except (OSError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for name, paths in (data.get("config_paths") or {}).items():
        items = paths if isinstance(paths, list) else [paths]
        scan.gui_configs.setdefault(str(name), []).extend(path.parent / str(p) for p in items if p)
    for name, kind in (data.get("model_types") or {}).items():
        if isinstance(kind, str) and kind:
            scan.gui_types.setdefault(str(name), kind)
