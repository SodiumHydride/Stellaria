"""Models, training, metrics, validation, releases, checkpoints, slim, export, quick render."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, Response

from ..context import AppContext
from ..engine import checkpoints
from ..engine.basemodels import STRUCTURE
from ..engine.config import default_config, merge_config, recommended_preset
from ..engine.external import ExternalImport, submit_import_external
from ..engine.logs import tail_lines
from ..audio.io import peaks
from ..engine.rendering import parse_render_params, quick_render_path, submit_quick_render
from ..engine.slim import submit_slim
from ..engine.validation import audio_file, mel_payload, snapshot
from ..errors import ApiError, not_found
from ..library.views import TRAINING_STATUSES, model_view
from ..packs import export_model
from ..schemas import BaseModelInfo, JobView, Model, Release, ScalarSeries, TrainConfig, ValidationSnapshot, Wire, msg
from ..util.fsutil import remove_tree
from ..util.jsonio import write_json
from .deps import get_ctx, read_incoming, require_paths

router = APIRouter()


class ModelCreate(Wire):
    name: str | None = None
    character_ids: list[str]
    config: dict[str, Any] | None = None


class ModelPatch(Wire):
    name: str | None = None
    config: dict[str, Any] | None = None
    target_steps: int | None = None


class TrainBody(Wire):
    fresh: bool = False


class ReleaseCreate(Wire):
    step: int
    name: str | None = None
    notes: str = ""


class ReleasePatch(Wire):
    name: str | None = None
    notes: str | None = None


class SlimBody(Wire):
    dataset: bool = False
    checkpoints: Literal["none", "keep_latest", "releases_only"] = "none"


class ModelExportBody(Wire):
    release_id: str
    path: str | None = None


class BaseImportBody(Wire):
    ckpt_path: str
    config_path: str | None = None
    name: str = ""


class BaseFromReleaseBody(Wire):
    model_id: str
    release_id: str
    name: str = ""


class BasePatch(Wire):
    name: str | None = None
    notes: str | None = None


def _busy(ctx: AppContext, mid: str) -> bool:
    return ctx.models.get(mid).status in TRAINING_STATUSES or ctx.training.active_job(mid) is not None


def _with_base(ctx: AppContext, cfg: TrainConfig) -> tuple[TrainConfig, str | None]:
    """With ``baseModelId`` the structural fields come from the base model (request values are
    ignored); also returns the base's ``glu_type``."""
    if not cfg.base_model_id:
        return cfg, None
    structure = ctx.base_models.structure(cfg.base_model_id)
    glu = structure.pop("glu_type")
    return TrainConfig.model_validate({**cfg.model_dump(), **structure}), glu


@router.get("/models", response_model=list[Model])
def list_models(characterId: str | None = None, ctx: AppContext = Depends(get_ctx)) -> list[Model]:  # noqa: N803
    recs = ctx.models.list_records()
    if characterId:
        recs = [r for r in recs if any(mc.character_id == characterId for mc in r.characters)]
    return [model_view(ctx, r.id) for r in recs]


@router.post("/models", response_model=Model)
def create_model(body: ModelCreate, ctx: AppContext = Depends(get_ctx)) -> Model:
    if not body.character_ids or len(set(body.character_ids)) != len(body.character_ids):
        raise ApiError("model.bad_characters", "Give one or more distinct characters", status=422)
    names = [ctx.characters.get(c).name for c in body.character_ids]
    config = merge_config(default_config(recommended_preset(ctx.system.best_vram_gb())), body.config)
    config, glu = _with_base(ctx, config)
    name = (body.name or "").strip() or " & ".join(names)
    rec = ctx.models.create(name, body.character_ids, config)
    if glu is not None:
        with ctx.models.lock(rec.id):
            rec = ctx.models.get(rec.id)
            rec.glu_type = glu
            ctx.models.save(rec, touch=False)
    ctx.emit_model(rec.id)
    return model_view(ctx, rec.id)


@router.post("/models/import-external", response_model=JobView)
def import_external(body: ExternalImport, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return submit_import_external(ctx, body).view()


@router.get("/models/{mid}", response_model=Model)
def get_model(mid: str, ctx: AppContext = Depends(get_ctx)) -> Model:
    return model_view(ctx, mid)


@router.patch("/models/{mid}", response_model=Model)
def patch_model(mid: str, body: ModelPatch, ctx: AppContext = Depends(get_ctx)) -> Model:
    with ctx.models.lock(mid):
        rec = ctx.models.get(mid)
        if body.config and _busy(ctx, mid):
            raise ApiError("model.busy", "Pause the training before changing its settings", status=409)
        if body.name is not None and body.name.strip():
            rec.name = body.name.strip()
        if body.config:
            _apply_config(ctx, rec, merge_config(rec.config, body.config))
        if body.target_steps is not None:
            if body.target_steps < 1:
                raise ApiError("request.invalid", "targetSteps must be positive", status=422)
            rec.target_steps = body.target_steps
            rec.config.target_steps = body.target_steps
        latest = checkpoints.latest_checkpoint(ctx.models.exp_dir(mid))
        if rec.status in ("done", "paused") and latest:
            rec.status = "done" if latest[0] >= rec.target_steps else "paused"  # more steps: it can continue
        ctx.models.save(rec)
    # a running training picks up a new target by itself (see TrainingMonitor / _train_job_inner)
    ctx.emit_model(mid)
    return model_view(ctx, mid)


def _apply_config(ctx: AppContext, rec, new: TrainConfig) -> None:
    """Non-structural fields always apply; structure and ``baseModelId`` only before training
    saved its first checkpoint (409 ``train.config_mismatch`` after)."""
    old = rec.config
    new, glu = _with_base(ctx, new)
    changed = [f for f in (*STRUCTURE, "base_model_id") if getattr(old, f) != getattr(new, f)]
    if changed:
        exp = ctx.models.exp_dir(rec.id)
        if checkpoints.has_trained(exp):
            raise ApiError("train.config_mismatch", "The model structure and base model are fixed once training "
                           "saved a checkpoint; start a new model to change them", status=409,
                           detail=", ".join(changed))
        # a base-model seed that was never trained on would no longer match: seed again
        (exp / "model_0.pt").unlink(missing_ok=True)
        (exp / "config.yaml").unlink(missing_ok=True)
        if new.base_model_id:
            rec.glu_type = glu
        elif old.base_model_id:
            rec.glu_type = None
    rec.config = new
    rec.target_steps = new.target_steps


@router.delete("/models/{mid}")
def delete_model(mid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    rec = ctx.models.get(mid)
    if _busy(ctx, mid):
        raise ApiError("model.busy", "Pause the training before deleting the model", status=409)
    ctx.models.delete(mid)
    ctx.metrics.forget(mid)
    ctx.hub.publish("model.deleted", {"id": mid})
    for mc in rec.characters:
        ctx.emit_character(mc.character_id)
    return {"ok": True}


# ---- training ---------------------------------------------------------------------------
@router.post("/models/{mid}/train", response_model=JobView)
def train(mid: str, body: TrainBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.training.start(mid, fresh=bool(body and body.fresh)).view()


@router.post("/models/{mid}/pause", response_model=Model)
def pause(mid: str, ctx: AppContext = Depends(get_ctx)) -> Model:
    ctx.training.pause(mid)
    return model_view(ctx, mid)


@router.post("/models/{mid}/stop", response_model=Model)
def stop(mid: str, ctx: AppContext = Depends(get_ctx)) -> Model:
    ctx.training.pause(mid)
    return model_view(ctx, mid)


@router.get("/models/{mid}/metrics", response_model=list[ScalarSeries])
def metrics(mid: str, tags: str | None = None, since: int | None = None, maxPoints: int = 1500,  # noqa: N803
            ctx: AppContext = Depends(get_ctx)) -> list[ScalarSeries]:
    ctx.models.get(mid)
    wanted = [t for t in tags.split(",") if t] if tags else None
    return ctx.metrics.polled(mid).scalars(wanted, since, max(10, min(maxPoints, 20000)))


@router.get("/models/{mid}/validation", response_model=ValidationSnapshot | None)
def validation(mid: str, step: int | None = None, ctx: AppContext = Depends(get_ctx)) -> ValidationSnapshot | None:
    ctx.models.get(mid)
    return snapshot(mid, ctx.metrics.polled(mid), step)


@router.get("/models/{mid}/validation/steps", response_model=list[int])
def validation_steps(mid: str, ctx: AppContext = Depends(get_ctx)) -> list[int]:
    ctx.models.get(mid)
    return ctx.metrics.polled(mid).validation_steps()


@router.get("/models/{mid}/validation/{step}/{name:path}/gt.wav")
def validation_gt(mid: str, step: int, name: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    ctx.models.get(mid)
    path = audio_file(ctx.models.cache_dir(mid), ctx.metrics.polled(mid), step, name, "gt")
    return FileResponse(path, media_type="audio/wav")


@router.get("/models/{mid}/validation/{step}/{name:path}/pred.wav")
def validation_pred(mid: str, step: int, name: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    ctx.models.get(mid)
    path = audio_file(ctx.models.cache_dir(mid), ctx.metrics.polled(mid), step, name, "pred")
    return FileResponse(path, media_type="audio/wav")


@router.get("/models/{mid}/validation/{step}/{name:path}/mel")
def validation_mel(mid: str, step: int, name: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    ctx.models.get(mid)
    return mel_payload(ctx.models.cache_dir(mid), ctx.metrics.polled(mid), step, name)


@router.get("/models/{mid}/log")
def model_log(mid: str, tail: int = 500, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    ctx.models.get(mid)
    lines = tail_lines(ctx.models.log_path(mid), max(1, min(tail, 20000)))
    if not lines:
        lines = tail_lines(ctx.models.exp_dir(mid) / "log_info.txt", tail)
    return {"lines": lines}


# ---- releases & checkpoints -------------------------------------------------------------
@router.post("/models/{mid}/releases", response_model=Release)
def create_release(mid: str, body: ReleaseCreate, ctx: AppContext = Depends(get_ctx)) -> Release:
    with ctx.models.lock(mid):
        rec = ctx.models.get(mid)
        release = checkpoints.create_release(rec, ctx.models.dir(mid), body.step, body.name, body.notes)
        rec.releases.append(release)
        ctx.models.save(rec)
    ctx.emit_model(mid)
    return release


def _release(rec, rel_id: str) -> Release:
    for r in rec.releases:
        if r.id == rel_id:
            return r
    raise not_found("release", rel_id)


@router.patch("/models/{mid}/releases/{rel_id}", response_model=Release)
def patch_release(mid: str, rel_id: str, body: ReleasePatch, ctx: AppContext = Depends(get_ctx)) -> Release:
    with ctx.models.lock(mid):
        rec = ctx.models.get(mid)
        release = _release(rec, rel_id)
        if body.name is not None and body.name.strip():
            release.name = body.name.strip()
        if body.notes is not None:
            release.notes = body.notes
        ctx.models.save(rec)
        write_json(checkpoints.release_dir(ctx.models.dir(mid), rel_id) / "release.json", release.wire())
    ctx.emit_model(mid)
    return release


@router.delete("/models/{mid}/releases/{rel_id}")
def delete_release(mid: str, rel_id: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    with ctx.models.lock(mid):
        rec = ctx.models.get(mid)
        _release(rec, rel_id)
        rec.releases = [r for r in rec.releases if r.id != rel_id]
        remove_tree(checkpoints.release_dir(ctx.models.dir(mid), rel_id))
        ctx.models.save(rec)
    ctx.emit_model(mid)
    return {"ok": True}


@router.delete("/models/{mid}/checkpoints/{step}")
def delete_checkpoint(mid: str, step: int, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    if _busy(ctx, mid):
        raise ApiError("model.busy", "Pause the training before deleting checkpoints", status=409)
    path = checkpoints.checkpoint_steps(ctx.models.exp_dir(mid)).get(step)
    if path is None:
        raise not_found("checkpoint", str(step))
    path.unlink()
    ctx.emit_model(mid)
    return {"ok": True}


@router.post("/models/{mid}/slim", response_model=JobView)
def slim(mid: str, body: SlimBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return submit_slim(ctx, mid, body.dataset, body.checkpoints).view()


@router.post("/models/{mid}/export", response_model=JobView)
def export(mid: str, body: ModelExportBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    rec = ctx.models.get(mid)
    _release(rec, body.release_id)
    job = ctx.jobs.submit("export", msg("job.export", name=rec.name),
                          lambda jctx: export_model(ctx, jctx, mid, body.release_id, body.path), target={"modelId": mid})
    return job.view()


@router.post("/models/{mid}/quick-render", response_model=JobView)
async def quick_render(mid: str, request: Request, ctx: AppContext = Depends(get_ctx)) -> JobView:
    ctx.models.get(mid)
    incoming = await read_incoming(request, ctx)
    require_paths(incoming)
    fields = incoming.fields
    params = parse_render_params(fields.get("params"))
    job = submit_quick_render(ctx, mid, incoming.paths[0], incoming.cleanup, fields.get("characterId") or "",
                              fields.get("releaseId"), params)
    return job.view()


def _quick_output(ctx: AppContext, mid: str, job_id: str):
    ctx.models.get(mid)
    path = quick_render_path(ctx, mid, job_id)
    if not job_id.startswith("job_") or not path.is_file():
        raise not_found("render", job_id)
    return path


@router.get("/models/{mid}/quick-render/{job_id}/audio")
def quick_render_audio(mid: str, job_id: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    return FileResponse(_quick_output(ctx, mid, job_id), media_type="audio/wav")


@router.get("/models/{mid}/quick-render/{job_id}/peaks")
def quick_render_peaks(mid: str, job_id: str, n: int = 800, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    values, duration = peaks(_quick_output(ctx, mid, job_id), n)
    return {"peaks": values, "duration": duration}


# ---- base models (底模) ------------------------------------------------------------------
@router.get("/base-models", response_model=list[BaseModelInfo])
def list_base_models(ctx: AppContext = Depends(get_ctx)) -> list[BaseModelInfo]:
    return ctx.base_models.list()


@router.post("/base-models/import", response_model=JobView)
def import_base_model(body: BaseImportBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.base_models.submit_import(Path(body.ckpt_path), Path(body.config_path) if body.config_path else None,
                                         body.name.strip()).view()


@router.post("/base-models/from-release", response_model=BaseModelInfo)
def base_model_from_release(body: BaseFromReleaseBody, ctx: AppContext = Depends(get_ctx)) -> BaseModelInfo:
    return ctx.base_models.from_release(body.model_id, body.release_id, body.name.strip())


@router.get("/base-models/{bid}", response_model=BaseModelInfo)
def get_base_model(bid: str, ctx: AppContext = Depends(get_ctx)) -> BaseModelInfo:
    return ctx.base_models.get(bid)


@router.patch("/base-models/{bid}", response_model=BaseModelInfo)
def patch_base_model(bid: str, body: BasePatch, ctx: AppContext = Depends(get_ctx)) -> BaseModelInfo:
    return ctx.base_models.patch(bid, body.name, body.notes)


@router.delete("/base-models/{bid}")
def delete_base_model(bid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    ctx.base_models.delete(bid)
    return {"ok": True}
