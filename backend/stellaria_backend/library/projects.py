"""Projects (canvas + timeline) and their media.

Canvases are brought to v2 on load (``creation.canvas.migrate_canvas``; saved once when it
changed something) and always stored with ``edges: []`` — edges are derived from node data."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from ..creation.canvas import migrate_canvas
from ..errors import not_found
from ..schemas import Media, Project
from ..util.fsutil import remove_tree
from ..util.ids import new_id
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso
from .store import Library


# muted cover colors (the frontend maps unknown values too)
COVER_PALETTE = ["#9aaabd", "#c9a6a0", "#a7b59f", "#cdbb9f", "#b0a3bb", "#95b0ad", "#d0adb9", "#bca48e", "#b3b494", "#d5b39a"]


def empty_canvas() -> dict:
    return {"nodes": [], "edges": [], "viewport": {"x": 0, "y": 0, "zoom": 1}}


def empty_timeline() -> dict:
    return {"tracks": [], "length": 0}


class ProjectStore:
    def __init__(self, lib: Library) -> None:
        self.lib = lib

    def dir(self, pid: str) -> Path:
        return self.lib.projects_dir / pid

    def media_dir(self, pid: str) -> Path:
        return self.dir(pid) / "media"

    def media_path(self, pid: str, mid: str) -> Path:
        return self.media_dir(pid) / f"{mid}.wav"

    def lock(self, pid: str):
        return self.lib.lock(f"prj:{pid}")

    def _load(self, folder: Path) -> Project | None:
        data = read_json(folder / "project.json")
        if data is None:
            return None
        project = Project.model_validate(data)
        if migrate_canvas(project.canvas):
            with self.lock(project.id):
                self.save(project, touch=False)
        return project

    def reset_interrupted(self) -> int:
        """At startup no job is alive, so nodes still marked queued/running were cut off by the
        last shutdown or a crash: show them as interrupted (keeping any earlier result) instead of
        spinning forever. Returns the number of nodes reset."""
        reset = 0
        for folder in sorted(self.lib.projects_dir.iterdir()):
            if not folder.is_dir():
                continue
            with self.lock(folder.name):
                project = self._load(folder)
                if project is None:
                    continue
                changed = 0
                for node in project.canvas.get("nodes") or []:
                    data = node.get("data")
                    if not isinstance(data, dict):
                        continue
                    state = data.get("stems") if node.get("type") == "source" else data
                    if not isinstance(state, dict) or state.get("status") not in ("queued", "running"):
                        continue
                    if node.get("type") == "source":
                        has_result = bool(state.get("vocals") and state.get("inst"))
                    else:
                        has_result = bool(data.get("mediaId") or data.get("outputs"))
                    state["jobId"] = None
                    if has_result:
                        state["status"] = "done"
                        state["error"] = None
                        if node.get("type") != "source":
                            data["stale"] = True
                    else:
                        state["status"] = "failed"
                        state["error"] = {"code": "job.interrupted", "params": {}}
                    changed += 1
                if changed:
                    self.save(project, touch=False)
                    reset += changed
        return reset

    def get(self, pid: str) -> Project:
        project = self._load(self.dir(pid))
        if project is None:
            raise not_found("project", pid)
        return project

    def list(self) -> list[Project]:
        projects = (self._load(d) for d in sorted(self.lib.projects_dir.iterdir()) if d.is_dir())
        return [p for p in projects if p is not None]

    def save(self, project: Project, touch: bool = True) -> Project:
        if touch:
            project.updated_at = now_iso()
        project.canvas["edges"] = []
        write_json(self.dir(project.id) / "project.json", project.wire())
        return project

    def create(self, name: str | None) -> Project:
        now = now_iso()
        project = Project(
            id=new_id("prj"), name=(name or "").strip() or "Untitled", created_at=now, updated_at=now,
            cover_color=random.choice(COVER_PALETTE), canvas=empty_canvas(), timeline=empty_timeline(),
        )
        self.media_dir(project.id).mkdir(parents=True, exist_ok=True)
        write_json(self.dir(project.id) / "media.json", [])
        return self.save(project, touch=False)

    def delete(self, pid: str) -> None:
        self.get(pid)
        remove_tree(self.dir(pid))

    def duplicate(self, pid: str) -> Project:
        src = self.get(pid)
        now = now_iso()
        new_pid = new_id("prj")
        shutil.copytree(self.dir(pid), self.dir(new_pid))
        copy = src.model_copy(update={"id": new_pid, "name": f"{src.name} (copy)", "created_at": now, "updated_at": now})
        return self.save(copy, touch=False)

    # ---- media -----------------------------------------------------------------------
    def media(self, pid: str) -> list[Media]:
        return [Media.model_validate(m) for m in read_json(self.dir(pid) / "media.json", [])]

    def get_media(self, pid: str, mid: str) -> Media:
        for m in self.media(pid):
            if m.id == mid:
                return m
        raise not_found("media", mid)

    def add_media(self, pid: str, media: Media) -> None:
        with self.lock(pid):
            items = self.media(pid)
            items.append(media)
            write_json(self.dir(pid) / "media.json", [m.wire() for m in items])
