"""Characters, raw imports, clips, organize, health, character export, pack import."""

from __future__ import annotations

import io
from typing import Any, Literal

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import FileResponse, Response

from ..audio.io import peaks
from ..context import AppContext
from ..errors import ApiError, not_found
from ..library import clipops
from ..library.views import TRAINING_STATUSES, character_view
from ..packs import export_character, import_pack
from ..pipeline.importing import import_raw
from ..pipeline.organize import organize
from ..pipeline.preview import PreviewRequest, preview
from ..quality.checker import run_health
from ..schemas import Character, Clip, Consent, HealthReport, JobView, OrganizeOptions, RawImport, Wire, msg
from ..util.timeutil import now_iso
from .deps import get_ctx, read_incoming, require_paths

router = APIRouter()


class CharacterCreate(Wire):
    name: str
    color: str | None = None
    tags: list[str] | None = None


class CharacterPatch(Wire):
    name: str | None = None
    color: str | None = None
    tags: list[str] | None = None
    notes: str | None = None
    consent: Consent | None = None
    pinned: bool | None = None
    archived: bool | None = None


class ClipChanges(Wire):
    status: Literal["active", "trashed"] | None = None
    split: Literal["train", "val"] | None = None
    reviewed: bool | None = None
    name: str | None = None


class ClipsPatch(Wire):
    ids: list[str]
    set: ClipChanges


class SplitBody(Wire):
    at: float


class RenameBody(Wire):
    pattern: str | None = None


class PickValBody(Wire):
    count: int | None = None


class HealthBody(Wire):
    advanced: bool = False


class ExportBody(Wire):
    include_material: bool = False
    path: str | None = None


def _view(ctx: AppContext, cid: str) -> Character:
    return character_view(ctx, cid)


# ---- characters ---------------------------------------------------------------------------
@router.get("/characters", response_model=list[Character])
def list_characters(archived: int | None = None, ctx: AppContext = Depends(get_ctx)) -> list[Character]:
    models = ctx.models.list_records()
    out = [character_view(ctx, r.id, models) for r in ctx.characters.list_records()]
    if archived is not None:
        out = [c for c in out if c.archived == bool(archived)]
    return out


@router.post("/characters", response_model=Character)
def create_character(body: CharacterCreate, ctx: AppContext = Depends(get_ctx)) -> Character:
    rec = ctx.characters.create(body.name, body.color, body.tags)
    ctx.emit_character(rec.id)
    return _view(ctx, rec.id)


@router.get("/characters/{cid}", response_model=Character)
def get_character(cid: str, ctx: AppContext = Depends(get_ctx)) -> Character:
    return _view(ctx, cid)


@router.patch("/characters/{cid}", response_model=Character)
def patch_character(cid: str, body: CharacterPatch, ctx: AppContext = Depends(get_ctx)) -> Character:
    with ctx.characters.lock(cid):
        rec = ctx.characters.get(cid)
        for key, value in body.model_dump(exclude_unset=True).items():
            if value is not None:
                setattr(rec, key, Consent.model_validate(value) if key == "consent" else value)
        ctx.characters.save(rec)
    ctx.emit_character(cid)
    return _view(ctx, cid)


@router.delete("/characters/{cid}")
def delete_character(cid: str, deleteModels: int = 0, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:  # noqa: N803
    ctx.characters.get(cid)
    models = [m for m in ctx.models.list_records() if any(mc.character_id == cid for mc in m.characters)]
    if any(m.status in TRAINING_STATUSES for m in models):
        raise ApiError("character.busy", "A model of this character is training", status=409)
    if deleteModels:
        for m in models:
            ctx.models.delete(m.id)
            ctx.metrics.forget(m.id)
            ctx.hub.publish("model.deleted", {"id": m.id})
    ctx.characters.delete(cid)
    ctx.hub.publish("character.deleted", {"id": cid})
    return {"ok": True}


@router.put("/characters/{cid}/avatar", response_model=Character)
async def put_avatar(cid: str, file: UploadFile = File(...), ctx: AppContext = Depends(get_ctx)) -> Character:
    from PIL import Image, ImageOps

    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGBA"), (256, 256), Image.Resampling.LANCZOS)
    except Exception as exc:  # noqa: BLE001
        raise ApiError("avatar.invalid", "This image cannot be read", status=422, detail=str(exc)) from exc
    with ctx.characters.lock(cid):
        rec = ctx.characters.get(cid)
        image.save(ctx.characters.avatar_path(cid), format="PNG")
        rec.has_avatar = True
        rec.avatar_version += 1
        ctx.characters.save(rec)
    ctx.emit_character(cid)
    return _view(ctx, cid)


@router.get("/characters/{cid}/avatar")
def get_avatar(cid: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    path = ctx.characters.avatar_path(cid)
    if not path.is_file():
        raise not_found("avatar", cid)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "max-age=31536000"})


@router.post("/characters/{cid}/opened", response_model=Character)
def opened(cid: str, ctx: AppContext = Depends(get_ctx)) -> Character:
    with ctx.characters.lock(cid):
        rec = ctx.characters.get(cid)
        rec.last_opened_at = now_iso()
        ctx.characters.save(rec, touch=False)
    ctx.emit_character(cid)
    return _view(ctx, cid)


# ---- raw imports ----------------------------------------------------------------------------
@router.get("/characters/{cid}/raw", response_model=list[RawImport])
def list_raw(cid: str, ctx: AppContext = Depends(get_ctx)) -> list[RawImport]:
    ctx.characters.get(cid)
    return ctx.characters.raws(cid)


@router.post("/characters/{cid}/raw", response_model=JobView)
async def add_raw(cid: str, request: Request, ctx: AppContext = Depends(get_ctx)) -> JobView:
    rec = ctx.characters.get(cid)
    incoming = await read_incoming(request, ctx)
    require_paths(incoming)
    job = ctx.jobs.submit(
        "import", msg("job.import", name=rec.name),
        lambda jctx: import_raw(ctx, jctx, cid, incoming.paths, incoming.names, incoming.cleanup),
        target={"characterId": cid}, lock_key=f"chr:{cid}",
    )
    return job.view()


@router.delete("/characters/{cid}/raw/{raw_id}")
def delete_raw(cid: str, raw_id: str, withClips: int = 0, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:  # noqa: N803
    clipops.delete_raw(ctx, cid, raw_id, bool(withClips))
    return {"ok": True}


# ---- clips ----------------------------------------------------------------------------------
@router.get("/characters/{cid}/clips", response_model=list[Clip])
def list_clips(cid: str, status: Literal["active", "trashed", "all"] = "active",
               ctx: AppContext = Depends(get_ctx)) -> list[Clip]:
    ctx.characters.get(cid)
    clips = ctx.characters.clips(cid)
    return clips if status == "all" else [c for c in clips if c.status == status]


@router.patch("/characters/{cid}/clips", response_model=list[Clip])
def patch_clips(cid: str, body: ClipsPatch, ctx: AppContext = Depends(get_ctx)) -> list[Clip]:
    ctx.characters.get(cid)
    return clipops.patch_clips(ctx, cid, body.ids, body.set.model_dump(exclude_none=True))


@router.post("/characters/{cid}/clips/empty-trash")
def empty_trash(cid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    ctx.characters.get(cid)
    return {"deleted": clipops.empty_trash(ctx, cid)}


@router.post("/characters/{cid}/clips/rename", response_model=list[Clip])
def rename_clips(cid: str, body: RenameBody | None = None, ctx: AppContext = Depends(get_ctx)) -> list[Clip]:
    ctx.characters.get(cid)
    return clipops.renumber(ctx, cid, body.pattern if body else None)


@router.post("/characters/{cid}/clips/pick-val", response_model=list[Clip])
def pick_val(cid: str, body: PickValBody | None = None, ctx: AppContext = Depends(get_ctx)) -> list[Clip]:
    ctx.characters.get(cid)
    return clipops.repick_val(ctx, cid, body.count if body else None)


@router.post("/characters/{cid}/clips/{clip_id}/split", response_model=list[Clip])
def split_clip(cid: str, clip_id: str, body: SplitBody, ctx: AppContext = Depends(get_ctx)) -> list[Clip]:
    ctx.characters.get(cid)
    return clipops.split_clip(ctx, cid, clip_id, body.at)


def _clip(ctx: AppContext, cid: str, clip_id: str) -> Clip:
    for c in ctx.characters.clips(cid):
        if c.id == clip_id:
            return c
    raise not_found("clip", clip_id)


@router.get("/characters/{cid}/clips/{clip_id}/audio")
def clip_audio(cid: str, clip_id: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    return FileResponse(ctx.characters.clip_path(cid, _clip(ctx, cid, clip_id)), media_type="audio/wav")


@router.get("/characters/{cid}/clips/{clip_id}/peaks")
def clip_peaks(cid: str, clip_id: str, n: int = 800, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    values, duration = peaks(ctx.characters.clip_path(cid, _clip(ctx, cid, clip_id)), n)
    return {"peaks": values, "duration": duration}


# ---- organize & health ----------------------------------------------------------------------
@router.get("/characters/{cid}/organize/defaults", response_model=OrganizeOptions)
def organize_defaults(cid: str, ctx: AppContext = Depends(get_ctx)) -> OrganizeOptions:
    ctx.characters.get(cid)
    return OrganizeOptions()


@router.post("/characters/{cid}/organize/preview")
def organize_preview(cid: str, body: PreviewRequest, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Synchronous: where the slicer would cut one raw file with these options."""
    ctx.characters.get(cid)
    return preview(ctx, cid, body)


@router.post("/characters/{cid}/organize", response_model=JobView)
def start_organize(cid: str, body: dict[str, Any] | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    rec = ctx.characters.get(cid)
    defaults = OrganizeOptions().wire()
    patch = body or {}
    if isinstance(patch.get("slice"), dict):
        patch = {**patch, "slice": {**defaults["slice"], **patch["slice"]}}
    try:
        opts = OrganizeOptions.model_validate({**defaults, **patch})
    except Exception as exc:  # noqa: BLE001
        raise ApiError("request.invalid", "Invalid organize options", status=422, detail=str(exc)) from exc
    job = ctx.jobs.submit("organize", msg("job.organize", name=rec.name), lambda jctx: organize(ctx, jctx, cid, opts),
                          target={"characterId": cid}, lock_key=f"chr:{cid}")
    return job.view()


@router.post("/characters/{cid}/health", response_model=JobView)
def start_health(cid: str, body: HealthBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    rec = ctx.characters.get(cid)
    advanced = bool(body and body.advanced)

    def run(jctx) -> dict[str, Any]:
        report = run_health(ctx, jctx, cid, advanced=advanced)
        return {"score": report.score, "level": report.level, "blocking": report.blocking}

    job = ctx.jobs.submit("health", msg("job.health", name=rec.name), run, target={"characterId": cid},
                          lock_key=f"chr:{cid}")
    return job.view()


@router.get("/characters/{cid}/health", response_model=HealthReport | None)
def get_health(cid: str, ctx: AppContext = Depends(get_ctx)) -> HealthReport | None:
    ctx.characters.get(cid)
    return ctx.characters.health(cid)[0]


# ---- export / packs -------------------------------------------------------------------------
@router.post("/characters/{cid}/export", response_model=JobView)
def export(cid: str, body: ExportBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    rec = ctx.characters.get(cid)
    job = ctx.jobs.submit("export", msg("job.export", name=rec.name),
                          lambda jctx: export_character(ctx, jctx, cid, body.include_material, body.path),
                          target={"characterId": cid})
    return job.view()


@router.post("/packs/import", response_model=JobView)
async def pack_import(request: Request, ctx: AppContext = Depends(get_ctx)) -> JobView:
    incoming = await read_incoming(request, ctx)
    require_paths(incoming)
    path = incoming.paths[0]
    job = ctx.jobs.submit("import_pack", msg("job.import_pack", name=incoming.names.get(path, path.name)),
                          lambda jctx: import_pack(ctx, jctx, path, incoming.cleanup))
    return job.view()
