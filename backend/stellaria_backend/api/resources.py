"""Pretrained models and add-ons; mirror latency."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from ..context import AppContext
from ..engine.separation_models import CLEANUP
from ..schemas import CleanupModels, JobView, Resource, SeparationModels, Wire
from .deps import get_ctx

router = APIRouter()


class InstallBody(Wire):
    source: str | None = None


class ImportBody(Wire):
    path: str


class SeparationImportBody(Wire):
    path: str  # a folder (e.g. Music-Source-Separation-Training or its GUI), or one checkpoint (separation or cleanup)
    config: str | None = None  # the checkpoint's YAML config, when ``path`` is a checkpoint


@router.get("/resources", response_model=list[Resource])
def list_resources(ctx: AppContext = Depends(get_ctx)) -> list[Resource]:
    return ctx.resources.list()


@router.get("/resources/sources")
def sources(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.resources.sources_info()


# separation and cleanup routes come before /resources/{rid}/… so neither is taken for an id
@router.get("/resources/separation", response_model=SeparationModels)
def separation_models(ctx: AppContext = Depends(get_ctx)) -> SeparationModels:
    """Every vocal separation model, whether it is ready, and the default one."""
    return ctx.separation.catalog.view()


@router.post("/resources/separation/import")
def import_separation(body: SeparationImportBody, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Import the vocal separation and cleanup checkpoints found in a folder (or one checkpoint)."""
    return ctx.separation.catalog.import_from(Path(body.path), Path(body.config) if body.config else None)


@router.delete("/resources/separation/{model_id}", response_model=SeparationModels)
def delete_separation(model_id: str, ctx: AppContext = Depends(get_ctx)) -> SeparationModels:
    ctx.separation.catalog.delete(model_id)
    return ctx.separation.catalog.view()


@router.get("/resources/cleanup", response_model=CleanupModels)
def cleanup_models(ctx: AppContext = Depends(get_ctx)) -> CleanupModels:
    """Every vocal cleanup (dereverb) model, whether it is ready, and the default one."""
    return ctx.separation.catalog.cleanup_view()


@router.delete("/resources/cleanup/{model_id}", response_model=CleanupModels)
def delete_cleanup(model_id: str, ctx: AppContext = Depends(get_ctx)) -> CleanupModels:
    ctx.separation.catalog.delete(model_id, CLEANUP)
    return ctx.separation.catalog.cleanup_view()


@router.post("/resources/install-required", response_model=list[JobView])
def install_required(ctx: AppContext = Depends(get_ctx)) -> list[JobView]:
    return [j.view() for j in ctx.resources.install_required()]


@router.post("/resources/import-folder")
def import_folder(body: ImportBody, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Import every missing model found in a folder (e.g. an existing DDSP-SVC checkout)."""
    return ctx.resources.import_folder(Path(body.path))


@router.post("/resources/{rid}/install", response_model=JobView)
def install(rid: str, body: InstallBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.resources.install(rid, body.source if body else None).view()


@router.post("/resources/{rid}/import", response_model=JobView)
def import_local(rid: str, body: ImportBody, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.resources.import_local(rid, Path(body.path)).view()


@router.delete("/resources/{rid}", response_model=Resource)
def delete(rid: str, ctx: AppContext = Depends(get_ctx)) -> Resource:
    return ctx.resources.delete(rid)
