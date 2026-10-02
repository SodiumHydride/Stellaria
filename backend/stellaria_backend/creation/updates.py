"""Server-side node updates and how they merge with the frontend's saves.

Jobs write node fields (status, outputs, …) into ``project.json`` and send them as
``project.nodes`` events ``{projectId, nodes}``. Ownership (``docs/canvas-v3.md`` §1.4):

* Server-owned, for every node type: ``status``, ``jobId``, ``error``, ``outputs``, ``result``,
  ``mediaId``, ``stems``, ``bpm``, ``key``. A ``PUT`` keeps the stored values of these for nodes
  that already exist (a new node takes what the client sent).
* ``stale``: incoming OR stored — the client can set it, only a successful run clears it.
* When a run succeeds with new outputs (``finish``), the node's direct dependents (also those
  reading through bypassed fx/pitch nodes) get ``stale: true`` in the same event.
* Other fields a job sets (a mix's ``selected``) are re-applied to incoming saves until the
  client echoes them, so a debounced save already in flight cannot undo them.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Iterable

from .canvas import STALE_TYPES, Graph, find_node

if TYPE_CHECKING:
    from ..context import AppContext

SERVER_FIELDS = ("status", "jobId", "error", "outputs", "result", "mediaId", "stems", "bpm", "key")


class NodeUpdates:
    def __init__(self) -> None:
        self._pending: dict[str, dict[str, dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _remember(self, pid: str, node_id: str, fields: dict[str, Any]) -> None:
        client = {k: v for k, v in fields.items() if k not in SERVER_FIELDS and k != "stale"}
        if client:
            with self._lock:
                self._pending.setdefault(pid, {}).setdefault(node_id, {}).update(client)

    def update(self, ctx: "AppContext", pid: str, node_id: str, **fields: Any) -> dict[str, Any] | None:
        """Set fields on a node, save, and send it as a ``project.nodes`` event."""
        return self.apply(ctx, pid, node_id, fields)

    def finish(self, ctx: "AppContext", pid: str, node_id: str, ports: Iterable[str] | None = None,
               **fields: Any) -> dict[str, Any] | None:
        """A run of ``node_id`` produced new outputs: set its fields (``stale: false``) and mark the
        nodes reading from it (optionally only through ``ports``) stale, in one event."""
        return self.apply(ctx, pid, node_id, {**fields, "stale": False}, stale_ports=ports, propagate=True)

    def apply(self, ctx: "AppContext", pid: str, node_id: str, fields: dict[str, Any],
              stale_ports: Iterable[str] | None = None, propagate: bool = False) -> dict[str, Any] | None:
        with ctx.projects.lock(pid):
            project = ctx.projects.get(pid)
            node = find_node(project.canvas, node_id)
            if node is None:
                return None
            data = node.setdefault("data", {})
            if node.get("type") not in STALE_TYPES:
                fields = {k: v for k, v in fields.items() if k != "stale"}
            data.update(fields)
            changed = [node]
            if propagate:
                for dep in Graph(project.canvas).dependents(node_id, stale_ports):
                    ddata = dep.setdefault("data", {})
                    if dep.get("type") in STALE_TYPES and not ddata.get("stale"):
                        ddata["stale"] = True
                        changed.append(dep)
            ctx.projects.save(project)
        self._remember(pid, node_id, fields)
        ctx.hub.publish("project.nodes", {"projectId": pid, "nodes": changed})
        return node

    def merge_incoming(self, pid: str, canvas: dict[str, Any], stored: dict[str, Any] | None = None) -> None:
        """Apply the ownership rules to a canvas the client is saving (in place)."""
        stored_by_id = {n.get("id"): n for n in (stored or {}).get("nodes", [])}
        for node in canvas.get("nodes", []):
            old = stored_by_id.get(node.get("id"))
            if old is None or old.get("type") != node.get("type"):
                continue
            data, old_data = node.setdefault("data", {}), old.get("data") or {}
            for key in SERVER_FIELDS:
                if key in old_data:
                    data[key] = old_data[key]
            if "stale" in data or "stale" in old_data:
                data["stale"] = bool(data.get("stale")) or bool(old_data.get("stale"))
        with self._lock:
            pending = self._pending.get(pid)
            if not pending:
                return
            present = set()
            for node in canvas.get("nodes", []):
                fields = pending.get(node.get("id"))
                if not fields:
                    continue
                present.add(node["id"])
                data = node.setdefault("data", {})
                if all(data.get(k) == v for k, v in fields.items()):
                    del pending[node["id"]]  # echoed back: the client is up to date
                else:
                    data.update(fields)
            for node_id in [n for n in pending if n not in present]:
                del pending[node_id]  # node deleted by the user
