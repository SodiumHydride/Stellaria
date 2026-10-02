"""Freeing disk space of a model: dataset features and/or checkpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..errors import ApiError
from ..jobs import Job, JobContext
from ..schemas import msg
from ..util.fsutil import dir_size, remove_tree
from . import checkpoints

if TYPE_CHECKING:
    from ..context import AppContext

CHECKPOINT_MODES = ("none", "keep_latest", "releases_only")


def slim(ctx: "AppContext", jctx: JobContext, mid: str, dataset_too: bool, checkpoints_mode: str) -> dict[str, Any]:
    if ctx.training.active_job(mid) is not None:
        raise ApiError("model.busy", "Pause the training before slimming the model", status=409)
    freed = 0
    if dataset_too:
        jctx.stage("stage.slim_dataset")
        for path in (ctx.models.dataset_dir(mid), ctx.models.cache_dir(mid)):
            freed += dir_size(path)
            remove_tree(path)
    if checkpoints_mode != "none":
        jctx.stage("stage.slim_checkpoints")
        steps = checkpoints.checkpoint_steps(ctx.models.exp_dir(mid))
        keep = {max(steps)} if steps and checkpoints_mode == "keep_latest" else set()
        for step, path in steps.items():
            if step not in keep:
                freed += path.stat().st_size
                path.unlink(missing_ok=True)
    ctx.emit_model(mid)
    return {"freedBytes": freed}


def submit_slim(ctx: "AppContext", mid: str, dataset_too: bool, checkpoints_mode: str) -> Job:
    if checkpoints_mode not in CHECKPOINT_MODES:
        raise ApiError("request.invalid", f"checkpoints must be one of {CHECKPOINT_MODES}", status=422)
    rec = ctx.models.get(mid)
    return ctx.jobs.submit("slim", msg("job.slim", name=rec.name),
                           lambda jctx: slim(ctx, jctx, mid, dataset_too, checkpoints_mode),
                           target={"modelId": mid}, lock_key=f"mdl:{mid}")
