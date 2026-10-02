"""``POST /system/diagnostics``: a zip to attach to a bug report — backend and shell logs, the
system report, preferences, resources, recent jobs and a library summary (counts and sizes).
No audio, no models, no tokens."""

from __future__ import annotations

import json
import platform
import sys
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import __version__
from .jobs import JobContext
from .util.fsutil import dir_size

if TYPE_CHECKING:
    from .context import AppContext

LOG_LIMIT = 20 * 1024 * 1024  # per file: the tail of very large logs


def _library(ctx: "AppContext") -> dict[str, Any]:
    def summary(base: Path) -> dict[str, Any]:
        items = [d for d in sorted(base.iterdir()) if d.is_dir()] if base.is_dir() else []
        return {"count": len(items), "bytes": sum(dir_size(d) for d in items)}

    projects = []
    for p in ctx.projects.list():
        nodes = p.canvas.get("nodes", [])
        kinds: dict[str, int] = {}
        for n in nodes:
            key = n.get("type") if n.get("type") != "process" else f"process.{(n.get('data') or {}).get('op')}"
            kinds[str(key)] = kinds.get(str(key), 0) + 1
        projects.append({"id": p.id, "nodes": kinds, "media": len(ctx.projects.media(p.id))})
    return {
        "characters": summary(ctx.lib.characters_dir), "models": summary(ctx.lib.models_dir),
        "projects": {**summary(ctx.lib.projects_dir), "items": projects},
        "modelStatuses": [{"id": m.id, "status": m.status, "steps": m.target_steps, "kind": m.kind}
                          for m in ctx.models.list_records()],
    }


def _add_log(zf: zipfile.ZipFile, path: Path, arcname: str) -> None:
    size = path.stat().st_size
    if size <= LOG_LIMIT:
        zf.write(path, arcname)
        return
    with open(path, "rb") as f:
        f.seek(size - LOG_LIMIT)
        zf.writestr(arcname, f.read())


def write_diagnostics(ctx: "AppContext", jctx: JobContext) -> dict[str, Any]:
    folder = ctx.options.data / "exports" / "diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / f"stellaria-diagnostics-{time.strftime('%Y%m%d-%H%M%S')}.zip"
    jctx.stage("stage.diagnostics", 0.0)

    def dump(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)

    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        about = {"version": __version__, "python": sys.version, "platform": platform.platform(),
                 "fakeEngine": ctx.options.fake_engine, "createdAt": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        zf.writestr("about.json", dump(about))
        zf.writestr("system.json", dump(ctx.system.info().wire()))
        jctx.progress(0.2)
        zf.writestr("preferences.json", dump(ctx.prefs.get().wire()))
        zf.writestr("resources.json", dump([r.wire() for r in ctx.resources.list()]))
        jobs = [{"id": j.id, "type": j.type, "status": j.status, "createdAt": j.created_at,
                 "error": j.error.wire() if j.error else None} for j in ctx.jobs.list()]
        zf.writestr("jobs.json", dump(jobs))
        jctx.progress(0.4)
        zf.writestr("library.json", dump(_library(ctx)))
        jctx.progress(0.7)
        logs = ctx.options.data / "logs"
        if logs.is_dir():
            for path in sorted(logs.rglob("*")):
                if path.is_file():
                    _add_log(zf, path, f"logs/{path.relative_to(logs).as_posix()}")
    jctx.progress(1.0)
    return {"path": str(dest), "downloadUrl": f"/files/{ctx.files.register(dest)}"}
