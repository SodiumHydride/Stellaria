"""The canvas graph (``docs/canvas-v3.md`` §1): every edge is audio, edges are derived from node
data, and every output port resolves to a media id.

* ``migrate_canvas`` brings v1 (stem nodes, ``inputNodeId``) and v2 (``port: 'mix'`` on
  non-source nodes, mix ``vocals``/``backing``) canvases to v3 and fills defaults. Idempotent;
  the project store runs it on load and saves once when it changed something.
* ``Graph`` answers questions about a canvas: a node's inputs, what a port carries (following
  bypassed fx/pitch nodes), who reads from a node, the song or take a port derives from.
"""

from __future__ import annotations

import copy
from collections import deque
from typing import Any, Iterable

from ..errors import ApiError

STEM_PORTS = ("vocals", "inst", "drums", "bass", "other")
EXTRA_STEMS = ("drums", "bass", "other")
SOURCE_PORTS = ("mix", *STEM_PORTS)
LEGACY_STEM = {"vocals": "vocals", "instrumental": "inst", "inst": "inst"}
PROCESS_OPS = ("trim", "split", "pitch", "fx", "separate", "dereverb", "comp", "concat", "export")
DEREVERB_PORTS = ("dry", "reverb")
BYPASSABLE = ("pitch", "fx")
RUNNABLE_TYPES = ("source", "render", "mix", "process")
STALE_TYPES = ("render", "mix", "process")
MIX_ROLES = ("lead", "harmony", "backing", "other")
MAX_DEPTH = 64


def empty_stems() -> dict[str, Any]:
    return {"status": "none", "vocals": None, "inst": None, "drums": None, "bass": None, "other": None,
            "provided": [], "model": None, "jobId": None, "error": None}


SOURCE_DEFAULTS: dict[str, Any] = {"bpm": None, "key": None}
AUDIO_DEFAULTS: dict[str, Any] = {"mediaId": None, "name": "", "role": "backing"}
RENDER_DEFAULTS: dict[str, Any] = {"stale": False, "batchId": None}
MIX_DEFAULTS: dict[str, Any] = {
    "inputs": [], "processor": "builtin", "preset": None, "params": {}, "variants": 1,
    "outputs": [], "selected": 0, "status": "draft", "jobId": None, "error": None, "stale": False, "label": "",
}
MIX_INPUT_DEFAULTS: dict[str, Any] = {"role": "lead", "gainDb": 0.0, "pan": 0.0, "mute": False, "solo": False}
PROCESS_DEFAULTS: dict[str, Any] = {
    "label": "", "inputs": [], "params": {}, "outputs": {}, "status": "draft", "jobId": None, "error": None,
    "stale": False, "bypass": False, "result": None,
}


def _fill(data: dict[str, Any], defaults: dict[str, Any]) -> bool:
    changed = False
    for key, value in defaults.items():
        if key not in data:
            data[key] = copy.deepcopy(value)
            changed = True
    return changed


def _fix_ref(ref: Any, types: dict[str, str]) -> bool:
    """v2: ``port: 'mix'`` (or no port) on a non-source node meant its own output → ``out``."""
    if not isinstance(ref, dict) or not ref.get("nodeId"):
        return False
    kind = types.get(ref["nodeId"])
    if kind is not None and kind != "source" and ref.get("port") in (None, "", "mix"):
        ref["port"] = "out"
        return True
    if kind == "source" and not ref.get("port"):
        ref["port"] = "mix"
        return True
    return False


def _migrate_mix(data: dict[str, Any], types: dict[str, str]) -> bool:
    changed = False
    if "inputs" not in data:
        inputs = []
        for v in data.get("vocals") or []:
            if isinstance(v, dict) and v.get("nodeId"):
                inputs.append({"id": str(v["nodeId"]), "ref": {"nodeId": v["nodeId"], "port": "out"}, "role": "lead",
                               "gainDb": float(v.get("gainDb") or 0.0), "pan": float(v.get("pan") or 0.0),
                               "mute": False, "solo": False})
        backing = data.get("backing")
        if isinstance(backing, dict) and backing.get("nodeId"):
            port = backing.get("port") or "mix"
            inputs.append({"id": f"{backing['nodeId']}:{port}", "ref": {"nodeId": backing["nodeId"], "port": port},
                           "role": "backing", "gainDb": float(backing.get("gainDb") or 0.0), "pan": 0.0,
                           "mute": False, "solo": False})
        data["inputs"] = inputs
        changed = True
    for key in ("vocals", "backing"):
        if key in data:
            del data[key]
            changed = True
    for i, item in enumerate(data["inputs"]):
        if not isinstance(item, dict):
            continue
        changed |= _fill(item, MIX_INPUT_DEFAULTS)
        ref = item.get("ref")
        if "id" not in item:
            item["id"] = f"{(ref or {}).get('nodeId', 'input')}:{(ref or {}).get('port', i)}"
            changed = True
        changed |= _fix_ref(ref, types)
    return changed


def migrate_canvas(canvas: dict[str, Any]) -> bool:
    """Bring a canvas to v3 in place; returns True when anything changed."""
    changed = False
    nodes: list[dict[str, Any]] = canvas.setdefault("nodes", [])
    by_id = {n.get("id"): n for n in nodes}

    # 1) v1: fold stem nodes into their sources
    stem_targets: dict[str, tuple[str, str]] = {}  # stem node id → (source id, port)
    for node in [n for n in nodes if n.get("type") == "stem"]:
        data = node.get("data") or {}
        source = by_id.get(data.get("sourceNodeId"))
        port = LEGACY_STEM.get(str(data.get("stem")))
        if source is not None and port is not None:
            stems = source.setdefault("data", {}).setdefault("stems", empty_stems())
            if data.get("mediaId"):
                stems[port] = data["mediaId"]
            stem_targets[node["id"]] = (source["id"], port)
    if any(n.get("type") == "stem" for n in nodes):
        nodes[:] = [n for n in nodes if n.get("type") != "stem"]
        changed = True
        for source_id in {s for s, _ in stem_targets.values()}:
            stems = by_id[source_id]["data"]["stems"]
            stems["status"] = "done" if stems.get("vocals") and stems.get("inst") else stems.get("status", "none")
    types = {n.get("id"): n.get("type") for n in nodes}

    # 2) node data: defaults, inputs as port references
    for node in nodes:
        data = node.setdefault("data", {})
        kind = node.get("type")
        if kind == "source":
            if not isinstance(data.get("stems"), dict):
                data["stems"] = empty_stems()
                changed = True
            else:
                changed |= _fill(data["stems"], empty_stems())
            changed |= _fill(data, SOURCE_DEFAULTS)
        elif kind == "audio":
            changed |= _fill(data, AUDIO_DEFAULTS)
        elif kind == "render":
            if "input" not in data:
                old = data.get("inputNodeId")
                node_id, port = stem_targets.get(old, (old, "mix"))
                data["input"] = {"nodeId": node_id, "port": port} if node_id else None
                changed = True
            if "inputNodeId" in data:
                del data["inputNodeId"]
                changed = True
            changed |= _fix_ref(data.get("input"), types)
            changed |= _fill(data, RENDER_DEFAULTS)
        elif kind == "mix":
            changed |= _migrate_mix(data, types)
            changed |= _fill(data, MIX_DEFAULTS)
        elif kind == "process":
            changed |= _fill(data, PROCESS_DEFAULTS)
            for ref in data.get("inputs") or []:
                changed |= _fix_ref(ref, types)
        if node.get("parentId") in stem_targets:  # a stem used to be someone's parent group: drop the link
            del node["parentId"]
            changed = True

    # 3) edges are derived from node data
    if canvas.get("edges"):
        changed = True
    canvas["edges"] = []
    return changed


# ---- the graph ---------------------------------------------------------------------------------
def input_missing(node_id: str | None, port: str | None) -> ApiError:
    return ApiError("node.input_missing", f"The input '{port}' of '{node_id}' has no audio yet", status=409,
                    params={"nodeId": node_id or "", "port": port or ""})


def find_node(canvas: dict[str, Any], node_id: str | None) -> dict[str, Any] | None:
    return next((n for n in canvas.get("nodes", []) if n.get("id") == node_id), None) if node_id else None


def is_bypassed(node: dict[str, Any] | None) -> bool:
    if node is None or node.get("type") != "process":
        return False
    data = node.get("data") or {}
    return bool(data.get("bypass")) and data.get("op") in BYPASSABLE


def inputs_of(node: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The port references a node reads from, in order."""
    if node is None:
        return []
    data = node.get("data") or {}
    kind = node.get("type")
    if kind == "render":
        ref = data.get("input")
        return [ref] if isinstance(ref, dict) and ref.get("nodeId") else []
    if kind == "mix":
        return [i["ref"] for i in data.get("inputs") or []
                if isinstance(i, dict) and isinstance(i.get("ref"), dict) and i["ref"].get("nodeId")]
    if kind == "process":
        return [r for r in data.get("inputs") or [] if isinstance(r, dict) and r.get("nodeId")]
    return []


def op_ports(data: dict[str, Any]) -> list[str]:
    params = data.get("params") or {}
    op = data.get("op")
    if op == "split":
        return [f"part{i + 1}" for i in range(len(params.get("cuts") or []) + 1)]
    if op == "separate":
        return list(STEM_PORTS) if int(params.get("stems") or 2) == 4 else ["vocals", "inst"]
    if op == "dereverb":
        return list(DEREVERB_PORTS)
    if op == "export":
        return []
    return ["out"]


def output_ports(node: dict[str, Any]) -> list[str]:
    kind = node.get("type")
    if kind == "source":
        return list(SOURCE_PORTS)
    if kind in ("audio", "render", "mix"):
        return ["out"]
    if kind == "process":
        return op_ports(node.get("data") or {})
    return []


class Graph:
    def __init__(self, canvas: dict[str, Any]) -> None:
        self.canvas = canvas
        self.nodes: list[dict[str, Any]] = canvas.get("nodes", [])
        self.by_id = {n.get("id"): n for n in self.nodes}

    def node(self, node_id: str | None) -> dict[str, Any] | None:
        return self.by_id.get(node_id) if node_id else None

    def chain(self, ref: dict[str, Any]) -> list[dict[str, Any]]:
        """The references a port's audio passes through: ``ref`` itself, then the input of each
        bypassed fx/pitch node it leads to."""
        out = [ref]
        seen = {ref.get("nodeId")}
        node = self.node(ref.get("nodeId"))
        while is_bypassed(node) and len(out) < MAX_DEPTH:
            inputs = inputs_of(node)
            if not inputs or inputs[0].get("nodeId") in seen:
                break
            out.append(inputs[0])
            seen.add(inputs[0].get("nodeId"))
            node = self.node(inputs[0].get("nodeId"))
        return out

    def media_of(self, ref: dict[str, Any] | None) -> str | None:
        """The media a port carries now, or None."""
        if not ref:
            return None
        last = self.chain(ref)[-1]
        node = self.node(last.get("nodeId"))
        if node is None or is_bypassed(node):
            return None
        data = node.get("data") or {}
        port = last.get("port") or ("mix" if node.get("type") == "source" else "out")
        kind = node.get("type")
        if kind == "source":
            return data.get("mediaId") if port == "mix" else (data.get("stems") or {}).get(port)
        if kind in ("audio", "render"):
            return data.get("mediaId") if port in ("out", "mix") else None
        if kind == "mix":
            if port not in ("out", "mix"):
                return None
            outputs = [o for o in data.get("outputs") or [] if isinstance(o, dict)]
            selected = int(data.get("selected") or 0)
            chosen = outputs[selected] if 0 <= selected < len(outputs) else (outputs[0] if outputs else None)
            return (chosen or {}).get("mediaId")
        if kind == "process":
            return (data.get("outputs") or {}).get(port)
        return None

    def resolve(self, ref: dict[str, Any] | None) -> str:
        """Media id behind a port; 409 ``node.input_missing`` ``{nodeId, port}`` when there is none
        (naming the node that lacks the audio, past bypassed nodes)."""
        if not ref or not ref.get("nodeId"):
            raise input_missing((ref or {}).get("nodeId"), (ref or {}).get("port"))
        media_id = self.media_of(ref)
        if not media_id:
            last = self.chain(ref)[-1]
            raise input_missing(last.get("nodeId"), last.get("port"))
        return media_id

    def dependents(self, node_id: str, ports: Iterable[str] | None = None) -> list[dict[str, Any]]:
        """Nodes that read from ``node_id`` (through bypassed nodes too), optionally only via ``ports``."""
        wanted = set(ports) if ports is not None else None
        out = []
        for n in self.nodes:
            if n.get("id") == node_id:
                continue
            for ref in inputs_of(n):
                if any(r.get("nodeId") == node_id and (wanted is None or r.get("port") in wanted) for r in self.chain(ref)):
                    out.append(n)
                    break
        return out

    def upstream(self, start: Iterable[dict[str, Any]]):
        """Nodes reachable upstream from the given references, nearest first (breadth-first)."""
        queue = deque(r.get("nodeId") for r in start)
        seen: set[str] = set()
        while queue:
            node_id = queue.popleft()
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            node = self.node(node_id)
            if node is None:
                continue
            yield node
            queue.extend(r.get("nodeId") for r in inputs_of(node))

    def root_source(self, refs: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
        """The song the given inputs derive from (first source walking upstream)."""
        return next((n for n in self.upstream(refs) if n.get("type") == "source"), None)

    def nearest_take(self, refs: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
        return next((n for n in self.upstream(refs) if n.get("type") == "render"), None)


def resolve_port(canvas: dict[str, Any], node_id: str | None, port: str | None) -> str:
    """v2 helper: media id behind ``(node_id, port)``."""
    return Graph(canvas).resolve({"nodeId": node_id, "port": port})
