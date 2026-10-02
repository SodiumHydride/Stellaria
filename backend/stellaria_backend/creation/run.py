"""``POST /projects/{id}/nodes/{nodeId}/run``: one entry point for every runnable node, and the
song-side helpers around it (importing/removing user-provided stems)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ..audio.io import SR, decode
from ..engine.rendering import submit_node_render, submit_node_separate
from ..errors import ApiError, not_found
from ..jobs import Job
from ..pipeline.importing import add_media_audio
from .canvas import STEM_PORTS, empty_stems, find_node, is_bypassed
from .mix import submit_node_mix
from .process import submit_node_process
from .updates import NodeUpdates

if TYPE_CHECKING:
    from ..context import AppContext

STEM_LABELS = {"vocals": "vocals", "inst": "instrumental", "drums": "drums", "bass": "bass", "other": "other"}


def _busy(ctx: "AppContext", node: dict[str, Any]) -> None:
    data = node.get("data") or {}
    state = (data.get("stems") or {}) if node.get("type") == "source" else data
    if state.get("status") in ("queued", "running") and state.get("jobId"):
        try:
            job = ctx.jobs.get(state["jobId"])
        except ApiError:
            return
        if job.active:
            raise ApiError("node.busy", "This node is already running", status=409,
                           params={"nodeId": node.get("id") or "", "jobId": job.id})


def submit_node(ctx: "AppContext", pid: str, node_id: str, stems: int | None = None,
                expect: str | None = None, model: str | None = None) -> Job:
    """Start the job of any runnable node: source → separate (2 stems, 4 on request; ``model``
    = separation model id, None = the default), render, mix, process (a ``separate`` /
    ``dereverb`` process node names its model in its params). ``expect`` (the v2 aliases) keeps
    their old type errors."""
    node = find_node(ctx.projects.get(pid).canvas, node_id)
    if node is None:
        raise not_found("node", node_id)
    kind = node.get("type")
    submit: dict[str, Callable[[], Job]] = {
        "source": lambda: submit_node_separate(ctx, ctx.node_updates, pid, node_id, stems or 2, model),
        "render": lambda: submit_node_render(ctx, ctx.node_updates, pid, node_id),
        "mix": lambda: submit_node_mix(ctx, ctx.node_updates, pid, node_id),
        "process": lambda: submit_node_process(ctx, ctx.node_updates, pid, node_id),
    }
    if expect is None and (kind not in submit or is_bypassed(node)):
        raise ApiError("node.not_runnable", f"A '{kind}' node has nothing to run", status=409, params={"nodeId": node_id})
    _busy(ctx, node)
    if expect is not None:
        return {"separate": submit["source"], "render": submit["render"], "mix": submit["mix"]}[expect]()
    return submit[kind]()


# ---- user-provided stems ------------------------------------------------------------------------
def _source(ctx: "AppContext", pid: str, node_id: str) -> dict[str, Any]:
    node = find_node(ctx.projects.get(pid).canvas, node_id)
    if node is None:
        raise not_found("node", node_id)
    if node.get("type") != "source":
        raise ApiError("node.not_source", "Stems belong to song nodes", status=422)
    return node


def _check_port(port: Any) -> str:
    if port not in STEM_PORTS:
        raise ApiError("request.invalid", f"port must be one of {', '.join(STEM_PORTS)}", status=422)
    return str(port)


def import_stem(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str, port: Any, path: Path,
                display_name: str | None = None) -> dict[str, Any]:
    """Use a file as one of a song's stems (e.g. the official instrumental): media ``import`` at
    offset 0, listed in ``stems.provided`` so a split never overwrites it."""
    port = _check_port(port)
    node = _source(ctx, pid, node_id)
    if not path.is_file():
        raise ApiError("import.not_found", f"'{path}' does not exist", status=422)
    _busy(ctx, node)
    song = (node.get("data") or {}).get("name") or Path(display_name or path.name).stem
    data = decode(path, SR, mono=False)
    media = add_media_audio(ctx, pid, data, f"{song} · {STEM_LABELS[port]}", "import")
    current = {**empty_stems(), **((_source(ctx, pid, node_id).get("data") or {}).get("stems") or {})}
    provided = [p for p in current.get("provided") or [] if p != port] + [port]
    stems = {**current, port: media.id, "provided": provided}
    if stems.get("vocals") and stems.get("inst") and stems.get("status") in ("none", "failed"):
        stems.update(status="done", error=None)
    return updates.finish(ctx, pid, node_id, ports=[port], stems=stems) or node


def remove_stem(ctx: "AppContext", updates: NodeUpdates, pid: str, node_id: str, port: Any) -> dict[str, Any]:
    port = _check_port(port)
    node = _source(ctx, pid, node_id)
    current = {**empty_stems(), **((node.get("data") or {}).get("stems") or {})}
    if port not in (current.get("provided") or []):
        raise ApiError("stems.not_provided", "Only stems you imported can be removed", status=409, params={"port": port})
    stems = {**current, port: None, "provided": [p for p in current["provided"] if p != port]}
    if not any(stems.get(p) for p in STEM_PORTS):
        stems.update(status="none", error=None)
    return updates.finish(ctx, pid, node_id, ports=[port], stems=stems) or node
