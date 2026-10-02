"""Resource status, model downloads and add-on installation.

* Model files are ready when every destination file exists (``install.dest`` / ``install.files``).
* Add-ons are ready when their package imports *and* the install finished (``.ready`` marker,
  written after the add-on's model weights were fetched too).
* ``failed`` states are persisted in ``<data>/models/resources.json``; live progress is in memory.
* With ``--fake-engine`` installs are simulated and leave a marker in ``models/.fake/`` that only
  counts in fake mode, so a dev data folder never mixes fake and real files.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from ..errors import ApiError, not_found
from ..jobs import Job, JobContext
from ..schemas import Msg, Resource, msg
from ..util import procs
from ..util.jsonio import read_json, write_json
from .downloader import LatencyProbe, Progress, download, order_sources, sha256_of
from .manifest import ResourceDef, Source, load_manifest

if TYPE_CHECKING:
    from ..context import AppContext

log = logging.getLogger(__name__)

HF_SOURCES = [
    Source(id="huggingface", region="global", url="https://huggingface.co/"),
    Source(id="hf-mirror", region="cn", url="https://hf-mirror.com/"),
]


class ResourceManager:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.manifest = load_manifest(ctx.options.resources)
        self.models_dir = ctx.options.data / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.models_dir / "resources.json"
        self._live: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.probe = LatencyProbe()

    # ---- status ----------------------------------------------------------------------
    def definition(self, rid: str) -> ResourceDef:
        rdef = self.manifest.get(rid)
        if rdef is None:
            raise not_found("resource", rid)
        return rdef

    def _fake_marker(self, rid: str) -> Path:
        return self.models_dir / ".fake" / rid

    def _addon_dir(self, rid: str) -> Path:
        return self.models_dir / rid

    def addon_env(self, rid: str) -> dict[str, str]:
        """Environment for add-on subprocesses (where their own model weights are cached)."""
        base = self._addon_dir(rid)
        env = {"HF_HOME": str(base / "hf"), "TORCH_HOME": str(base / "torch")}
        if self._hf_endpoint():
            env["HF_ENDPOINT"] = self._hf_endpoint()
        return env

    def is_ready(self, rid: str) -> bool:
        rdef = self.definition(rid)
        if self.ctx.options.fake_engine and self._fake_marker(rid).exists():
            return True
        if rdef.kind == "addon":
            module = rdef.module_name
            importlib.invalidate_caches()
            return bool(module and importlib.util.find_spec(module)) and (self._addon_dir(rid) / ".ready").exists()
        files = rdef.required_files(self.models_dir)
        return bool(files) and all(f.is_file() for f in files)

    def path(self, rid: str) -> Path | None:
        """Primary file of a model resource (what the engine loads), or None if missing."""
        rdef = self.definition(rid)
        if rdef.kind == "addon":
            return self._addon_dir(rid)
        return rdef.primary_file(self.models_dir)

    def require(self, rids: list[str]) -> None:
        """Raise ``engine.missing_model`` naming the first missing resource."""
        if self.ctx.options.fake_engine:
            return
        for rid in rids:
            if not self.is_ready(rid):
                raise ApiError(
                    "engine.missing_model", f"Pretrained model '{rid}' is not installed", status=409,
                    detail=f"resource={rid}", params={"resource": rid},
                )

    def view(self, rid: str) -> Resource:
        rdef = self.definition(rid)
        live = self._live.get(rid)
        persisted = (read_json(self._state_path, {}) or {}).get(rid, {})
        ready = self.is_ready(rid)
        if live:
            status = live["status"]
        elif ready:
            status = "ready"
        elif persisted.get("status") == "failed":
            status = "failed"
        else:
            status = "missing"
        path = self.path(rid)
        error = persisted.get("error") if status == "failed" else None
        return Resource(
            id=rid, kind=rdef.kind, required=rdef.required,
            name=Msg(code=f"resource.{rid}.name", params=rdef.name),
            description=Msg(code=f"resource.{rid}.description", params=rdef.description),
            size=rdef.size or 0, status=status,
            progress=live.get("progress") if live else (1.0 if ready else None),
            bytes_done=live.get("bytesDone") if live else None,
            speed=live.get("speed") if live else None,
            source=(live or {}).get("source") or persisted.get("source"),
            error=Msg.model_validate(error) if error else None,
            path=str(path) if path and ready else None,
            license=rdef.license, homepage=rdef.homepage,
        )

    def list(self) -> list[Resource]:
        return [self.view(r.id) for r in self.manifest.resources]

    def _publish(self, rid: str) -> None:
        self.ctx.hub.publish("resource", self.view(rid).wire(), throttle_key=f"res:{rid}", min_interval=0.25)

    def _persist(self, rid: str, entry: dict[str, Any] | None) -> None:
        with self._lock:
            state = read_json(self._state_path, {}) or {}
            if entry is None:
                state.pop(rid, None)
            else:
                state[rid] = entry
            write_json(self._state_path, state)

    def _set_live(self, rid: str, **values: Any) -> None:
        with self._lock:
            self._live.setdefault(rid, {}).update(values)
        self._publish(rid)

    # ---- sources ---------------------------------------------------------------------
    def _region(self) -> str:
        return self.ctx.prefs.get().mirror_region

    def ordered(self, sources: list[Source], preferred: str | None = None) -> list[Source]:
        region = self._region()
        latency = self.probe.measure(sources) if region == "auto" else None
        ordered = order_sources(sources, region, latency)
        if preferred:
            ordered.sort(key=lambda s: 0 if s.id == preferred else 1)
        return ordered

    def _hf_endpoint(self) -> str | None:
        region = self._region()
        if region == "cn":
            return "https://hf-mirror.com"
        if region == "auto":
            measured = self.probe.last
            hf, mirror = measured.get("huggingface"), measured.get("hf-mirror")
            if hf is None and mirror is not None:
                return "https://hf-mirror.com"
        return None

    def sources_info(self) -> dict[str, Any]:
        everything = [s for r in self.manifest.resources for s in r.sources] + self.manifest.pypi
        measured = self.probe.measure(everything)
        return {
            "region": self._region(),
            "measured": [{"sourceId": sid, "latencyMs": ms} for sid, ms in sorted(measured.items())],
        }

    # ---- install / delete ------------------------------------------------------------
    def install(self, rid: str, source: str | None = None) -> Job:
        rdef = self.definition(rid)
        existing = self.ctx.jobs.find_active("download", resourceId=rid) or self.ctx.jobs.find_active(
            "install_addon", resourceId=rid)
        if existing:
            return existing
        job_type = "install_addon" if rdef.kind == "addon" else "download"
        self._set_live(rid, status="downloading" if job_type == "download" else "installing", progress=0.0)

        def run(jctx: JobContext) -> dict[str, Any]:
            try:
                if self.ctx.options.fake_engine:
                    self._fake_install(jctx, rdef)
                elif rdef.kind == "addon":
                    self._install_addon(jctx, rdef)
                else:
                    self._download_model(jctx, rdef, source)
                self._persist(rid, None)
                return {"resourceId": rid, "path": str(self.path(rid))}
            except Exception as exc:
                if not jctx.cancelled:
                    err = exc if isinstance(exc, ApiError) else ApiError("download.failed", str(exc))
                    self._persist(rid, {"status": "failed", "error": msg(err.code, **err.params).wire(),
                                        "source": (self._live.get(rid) or {}).get("source")})
                raise
            finally:
                with self._lock:
                    self._live.pop(rid, None)
                self.ctx.hub.publish("resource", self.view(rid).wire(), throttle_key=f"res:{rid}")
                self._separation_changed(rdef)

        return self.ctx.jobs.submit(job_type, msg(f"job.{job_type}", resource=rid), run,
                                    target={"resourceId": rid}, lock_key=f"res:{rid}")

    def import_local(self, rid: str, path: Path) -> Job:
        """Install a model resource from a local file: the file itself, the official zip, or a
        folder (e.g. DDSP-SVC's ``pretrain/<name>/``) holding the files the zip would contain."""
        rdef = self.definition(rid)
        if rdef.kind == "addon":
            raise ApiError("resource.not_importable", "Add-ons cannot be imported from a file", status=422)
        if not path.exists():
            raise ApiError("import.not_found", f"'{path}' does not exist", status=422)
        self._set_live(rid, status="installing", progress=0.0, source="local")

        def run(jctx: JobContext) -> dict[str, Any]:
            try:
                jctx.stage("stage.import_local", 0.0)
                self._import_local(jctx, rdef, path)
                if not self.is_ready(rid):
                    missing = [p.name for p in rdef.required_files(self.models_dir) if not p.is_file()]
                    raise ApiError("resource.incomplete", "Some files of this model were not found", status=422,
                                   detail=", ".join(missing), params={"files": ", ".join(missing)})
                self._persist(rid, None)
                return {"resourceId": rid, "path": str(self.path(rid))}
            except Exception as exc:
                if not jctx.cancelled:
                    err = exc if isinstance(exc, ApiError) else ApiError("resource.import_failed", str(exc))
                    self._persist(rid, {"status": "failed", "error": msg(err.code, **err.params).wire(), "source": "local"})
                raise
            finally:
                with self._lock:
                    self._live.pop(rid, None)
                self.ctx.hub.publish("resource", self.view(rid).wire(), throttle_key=f"res:{rid}")
                self._separation_changed(rdef)

        return self.ctx.jobs.submit("download", msg("job.import_local", resource=rid), run,
                                    target={"resourceId": rid}, lock_key=f"res:{rid}")

    def import_folder(self, root: Path) -> dict[str, Any]:
        """Find every model that isn't installed yet in a folder someone already has (a DDSP-SVC
        checkout or bundle, its ``pretrain/``, a single model's folder) and import each one found;
        separation and cleanup checkpoints (an MSST folder) too, see ``SeparationCatalog.import_from``.
        Returns ``{found: [{id, path, jobId}], missing: [id], separation: {found, skipped}}``."""
        if not root.is_dir():
            raise ApiError("import.not_found", f"'{root}' is not a folder", status=422, params={"path": str(root)})
        wanted = [r for r in self.manifest.resources
                  if r.kind == "model" and not r.separation and not r.cleanup and not self.is_ready(r.id)]
        located = find_models(root, [r.id for r in wanted])
        found, missing = [], []
        for rdef in wanted:
            path = located.get(rdef.id)
            if path is None:
                missing.append(rdef.id)
                continue
            if rdef.install_type == "file":  # the checkpoint itself
                path = path / Path(rdef.install["dest"]).name
            job = self.import_local(rdef.id, path)
            found.append({"id": rdef.id, "path": str(path), "jobId": job.id})
        return {"found": found, "missing": missing, "separation": self.ctx.separation.catalog.import_from(root)}

    def _import_local(self, jctx: JobContext, rdef: ResourceDef, path: Path) -> None:
        if rdef.install_type == "file":
            if not path.is_file():
                raise ApiError("request.invalid", "Choose the model file itself", status=422)
            dest = self.models_dir / rdef.install["dest"]
            self.copy_verified(jctx, path, dest, rdef.sha256)
            return
        dest_dir = self.models_dir / rdef.install["dest"]
        mapping: dict[str, str] = rdef.install.get("files", {})
        if path.is_file() and zipfile.is_zipfile(path):
            if rdef.sha256:
                jctx.stage("stage.verify", 0.1)
                self._check_sha(path, rdef.sha256)
            jctx.stage("stage.extract", 0.5)
            extract_files(path, dest_dir, mapping)
            return
        # a folder (or a file inside it) laid out like the archive's contents
        folder = path if path.is_dir() else path.parent
        found = 0
        for member, target in mapping.items():
            names = (Path(target).name, Path(member).name, Path(target).stem)  # `model` for `model.ckpt`
            for candidate in (folder / name for name in dict.fromkeys(names)):
                if candidate.is_file():
                    self.copy_verified(jctx, candidate, dest_dir / target, None)
                    found += 1
                    break
        if not found:
            raise ApiError("resource.incomplete", "No files of this model were found there", status=422,
                           params={"files": ", ".join(Path(t).name for t in mapping.values())})

    def copy_verified(self, jctx: JobContext, src: Path, dest: Path, sha256: str | None) -> None:
        """Copy ``src`` to ``dest`` (verified when a checksum is known). On the same drive a hard
        link is made instead: no second copy on disk, and the file stays if the original goes."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        part.unlink(missing_ok=True)
        try:
            os.link(src, part)
        except OSError:
            total = max(1, src.stat().st_size)
            done = 0
            with open(src, "rb") as fin, open(part, "wb") as fout:
                for block in iter(lambda: fin.read(1 << 20), b""):
                    jctx.check_cancelled()
                    fout.write(block)
                    done += len(block)
                    jctx.progress(done / total, msg("stage.import_local"))
        if sha256:
            jctx.stage("stage.verify", 1.0)
            try:
                self._check_sha(part, sha256)
            except ApiError:
                part.unlink(missing_ok=True)
                raise
        part.replace(dest)

    @staticmethod
    def _check_sha(path: Path, expected: str) -> None:
        digest = sha256_of(path)
        if digest.lower() != expected.lower():
            raise ApiError("resource.checksum", "The file does not match the expected checksum", status=422,
                           detail=f"sha256 {digest} != expected {expected}")

    def install_required(self) -> list[Job]:
        return [self.install(r.id) for r in self.manifest.resources if r.required and not self.is_ready(r.id)]

    def delete(self, rid: str) -> Resource:
        rdef = self.definition(rid)
        if self._live.get(rid):
            raise ApiError("resource.busy", "Resource is being installed", status=409)
        self._fake_marker(rid).unlink(missing_ok=True)
        if rdef.kind == "addon":
            shutil.rmtree(self._addon_dir(rid), ignore_errors=True)
        elif rdef.install_type == "zip":
            shutil.rmtree(self.models_dir / rdef.install["dest"], ignore_errors=True)
        else:
            for f in rdef.dest_files(self.models_dir):
                f.unlink(missing_ok=True)
                f.with_name(f.name + ".part").unlink(missing_ok=True)
        self._persist(rid, None)
        self._publish(rid)
        self._separation_changed(rdef)
        return self.view(rid)

    def _separation_changed(self, rdef: ResourceDef) -> None:
        self.ctx.separation.catalog.resource_changed(rdef)

    # ---- implementations -------------------------------------------------------------
    def _progress_cb(self, jctx: JobContext, rid: str):
        def cb(p: Progress) -> None:
            frac = p.bytes_done / p.total if p.total else None
            self._set_live(rid, progress=frac, bytesDone=p.bytes_done, speed=p.speed, source=p.source)
            eta = (p.total - p.bytes_done) / p.speed if p.total and p.speed else None
            jctx.progress(frac, msg("stage.download", source=p.source), eta)
        return cb

    def _download_model(self, jctx: JobContext, rdef: ResourceDef, preferred: str | None) -> None:
        sources = self.ordered(rdef.sources, preferred)
        jctx.log(f"sources: {', '.join(s.id for s in sources)}")
        if rdef.install_type == "file":
            dest = self.models_dir / rdef.install["dest"]
            used = download(sources, dest, size=rdef.size, sha256=rdef.sha256, cancel=jctx.job.cancel_event,
                            on_progress=self._progress_cb(jctx, rdef.id), log=jctx.log)
            jctx.log(f"downloaded from {used.id}")
            return
        archive = self.models_dir / ".downloads" / f"{rdef.id}.zip"
        used = download(sources, archive, size=rdef.size, sha256=rdef.sha256, cancel=jctx.job.cancel_event,
                        on_progress=self._progress_cb(jctx, rdef.id), log=jctx.log)
        jctx.log(f"downloaded from {used.id}")
        jctx.stage("stage.extract")
        self._set_live(rdef.id, status="installing")
        extract_files(archive, self.models_dir / rdef.install["dest"], rdef.install.get("files", {}))
        archive.unlink(missing_ok=True)

    def _install_addon(self, jctx: JobContext, rdef: ResourceDef) -> None:
        jctx.stage("stage.install")
        from ..plugins import envs

        if envs.layer_marker() is None and os.environ.get("STELLARIA_CUSTOM_RUNTIME") == "1":
            # the user's own Python without the shell's layer: installing would change their
            # environment, which never happens
            raise ApiError("addon.needs_layer", "Add-ons go into Stellaria's own folder next to your Python, "
                           "which is missing", status=409)
        if envs.layer_marker() is not None:
            # The user's own Python through the shell's layer: only what it lacks, into the layer
            # (uv would not see their packages there and pull torch again).
            try:
                envs.install_into_env(self.ctx, jctx, Path(sys.executable), Path(sys.prefix), list(rdef.packages))
            except ApiError as exc:
                raise ApiError("addon.install_failed", "Installing the add-on failed", status=502,
                               detail=exc.detail) from exc
            self._finish_addon(jctx, rdef)
            return
        constraints = Path(tempfile.mkstemp(prefix="stellaria-constraints-", suffix=".txt")[1])
        try:
            constraints.write_text("\n".join(_pinned_runtime()), encoding="utf-8")
            errors = []
            for mirror in self.ordered(self.manifest.pypi):
                cmd = self._pip_command(rdef.packages, mirror.url, constraints)
                jctx.log(f"$ {' '.join(cmd)}")
                # uv splits --constraint values on spaces (UV_CONSTRAINT can list several files): pass
                # the file by its bare name and run in its folder, so a temp path with a space works
                proc = procs.popen(cmd, env=procs.child_env(), cwd=constraints.parent)
                jctx.on_cancel(lambda p=proc: procs.kill_tree(p.pid))
                assert proc.stdout is not None
                for raw in proc.stdout:
                    jctx.log(raw.decode("utf-8", errors="replace").rstrip())
                proc.wait()
                jctx.on_cancel(None)
                jctx.check_cancelled()
                if proc.returncode == 0:
                    break
                errors.append(f"{mirror.id}: exit code {proc.returncode}")
            else:
                raise ApiError("addon.install_failed", "Installing the add-on failed", status=502,
                               detail="\n".join(errors + list(jctx.job.log)[-40:]))
        finally:
            constraints.unlink(missing_ok=True)
        self._finish_addon(jctx, rdef)

    def _finish_addon(self, jctx: JobContext, rdef: ResourceDef) -> None:
        if rdef.id == "separator":
            jctx.stage("stage.download")
            self.probe.measure(HF_SOURCES)
            from ..engine.separation import fetch_separator_weights

            fetch_separator_weights(self.ctx, jctx)
        marker = self._addon_dir(rdef.id) / ".ready"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"), encoding="utf-8")

    def _pip_command(self, packages: list[str], index_url: str | None, constraints: Path) -> list[str]:
        uv = self.ctx.options.uv or (Path(found) if (found := shutil.which("uv")) else None)
        index = ["--index-url", index_url] if index_url else []
        if uv:
            return [str(uv), "pip", "install", "--python", sys.executable, *packages, *index,
                    "--constraint", constraints.name]
        return [sys.executable, "-m", "pip", "install", *packages, *index, "--constraint", constraints.name]

    def _fake_install(self, jctx: JobContext, rdef: ResourceDef) -> None:
        total = rdef.size or 50_000_000
        steps = 20
        for i in range(1, steps + 1):
            jctx.check_cancelled()
            time.sleep(0.1)
            done = total * i // steps
            self._progress_cb(jctx, rdef.id)(Progress(done, total, total / 2.0, "fake"))
        marker = self._fake_marker(rdef.id)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("fake", encoding="utf-8")


def _pinned_runtime() -> list[str]:
    """Constraints so an add-on install never swaps torch or numpy under the engine."""
    pins = []
    for dist in ("torch", "torchaudio", "numpy"):
        try:
            pins.append(f"{dist}=={metadata.version(dist)}")
        except metadata.PackageNotFoundError:
            pass
    return pins


def extract_files(archive: Path, dest_dir: Path, mapping: dict[str, str]) -> None:
    """Extract ``mapping`` (member → relative destination) from ``archive`` into ``dest_dir``."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for member, target in mapping.items():
            if member not in names:
                raise ApiError("download.bad_archive", f"'{member}' missing in archive", status=502)
            out = (dest_dir / target).resolve()
            if dest_dir.resolve() not in out.parents:
                raise ApiError("download.bad_archive", f"unsafe path '{target}'", status=502)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.name + ".tmp")
            with zf.open(member) as src, open(tmp, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            tmp.replace(out)


# ---- finding models in a folder someone already has ------------------------------------------------
# Per model: words a folder's name contains, and the file sets (any one) that make it that model.
# `pytorch_model.bin` and `model.pt` are common names, so they only count inside a matching folder.
MODEL_FOLDERS: dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = {
    "contentvec": (("contentvec", "content-vec", "content_vec"), (("pytorch_model.bin",),)),
    "rmvpe": (("rmvpe",), (("model.pt",),)),
    "nsf_hifigan": (("hifigan",), (("model.ckpt", "config.json"), ("model", "config.json"))),
    "hubertsoft": (("hubert",), (("hubert-soft-0d54a1f4.pt",),)),
}
_SKIP_DIRS = {".git", "node_modules", "__pycache__", "site-packages", "dist-packages", "lib", "Lib", "include",
              "Include", "Scripts", "share", "exp", "logs", "dataset", "data", "raw", "samples", "outputs"}
_MAX_DEPTH = 4
_MAX_DIRS = 5000


def walk_folders(root: Path) -> Iterator[tuple[Path, dict[str, os.DirEntry]]]:
    """``(folder, {name: entry})`` for ``root`` and its sub-folders, breadth first, up to
    ``_MAX_DEPTH`` levels and ``_MAX_DIRS`` folders. Python environments and datasets are skipped."""
    queue: list[tuple[Path, int]] = [(root, 0)]
    seen = 0
    while queue and seen < _MAX_DIRS:
        folder, depth = queue.pop(0)
        seen += 1
        try:
            entries = {e.name: e for e in os.scandir(folder)}
        except OSError:
            continue
        yield folder, entries
        if depth >= _MAX_DEPTH:
            continue
        for entry in sorted(entries.values(), key=lambda e: e.name.lower()):
            if entry.is_dir(follow_symlinks=False) and entry.name not in _SKIP_DIRS and not entry.name.startswith(".") \
                    and not (Path(entry.path) / "pyvenv.cfg").exists() and not (Path(entry.path) / "conda-meta").exists():
                queue.append((Path(entry.path), depth + 1))


def find_models(root: Path, ids: list[str]) -> dict[str, Path]:
    """The folder holding each model in ``ids`` under ``root``, shallowest first. Python
    environments and datasets are not searched."""
    found: dict[str, Path] = {}
    wanted = [i for i in ids if i in MODEL_FOLDERS]
    if not wanted:
        return found
    for folder, entries in walk_folders(root):
        name = folder.name.lower()
        for rid in wanted:
            if rid in found:
                continue
            words, sets = MODEL_FOLDERS[rid]
            if not any(w in name for w in words) and not (rid == "hubertsoft" and "hubert-soft-0d54a1f4.pt" in entries):
                continue
            for files in sets:
                if all(f in entries and entries[f].is_file() for f in files):
                    found[rid] = folder
                    break
        if len(found) == len(wanted):
            break
    return found
