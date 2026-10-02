"""System, preferences, disk, shutdown, train presets."""

from __future__ import annotations

import asyncio
import shutil
from typing import Any

from fastapi import APIRouter, Depends, Request

from .. import __version__
from ..context import AppContext
from ..diagnostics import write_diagnostics
from ..errors import ApiError
from ..engine.config import presets
from ..schemas import GpuInfo, JobView, Preferences, SystemInfo, TrainPreset, msg
from ..util.fsutil import dir_size
from .deps import get_ctx

router = APIRouter()


@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "version": __version__}


@router.get("/system", response_model=SystemInfo)
def system(ctx: AppContext = Depends(get_ctx)) -> SystemInfo:
    return ctx.system.info()


@router.get("/system/gpu", response_model=list[GpuInfo])
def system_gpu(ctx: AppContext = Depends(get_ctx)) -> list[GpuInfo]:
    return ctx.system.gpus()


@router.post("/system/diagnostics", response_model=JobView)
def diagnostics(ctx: AppContext = Depends(get_ctx)) -> JobView:
    """A zip of logs, the system report, preferences and a library summary (no audio, no models)."""
    return ctx.jobs.submit("diagnostics", msg("job.diagnostics"), lambda jctx: write_diagnostics(ctx, jctx)).view()


@router.get("/system/render")
def render_status(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Whether the singing models are loaded (holding GPU / memory)."""
    return ctx.render.status()


@router.post("/system/render/release")
def release_render(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Free the memory the singing models hold now; ``released`` is false while one is singing."""
    return {"released": ctx.render.release(), **ctx.render.status()}


@router.get("/preferences", response_model=Preferences)
def get_preferences(ctx: AppContext = Depends(get_ctx)) -> Preferences:
    return ctx.prefs.get()


@router.put("/preferences", response_model=Preferences)
def put_preferences(body: dict[str, Any], ctx: AppContext = Depends(get_ctx)) -> Preferences:
    for key, role in (("separationModel", "separation"), ("cleanupModel", "cleanup")):
        model = body.get(key)
        if isinstance(model, str) and model.strip() in ("", "auto"):
            body = {**body, key: None}
        elif model is not None and not ctx.separation.catalog.known(model, role):
            raise ApiError(f"{role}.unknown_model", f"Unknown {role} model '{model}'", status=422,
                           params={"model": str(model)})
    return ctx.prefs.update(body)


@router.get("/disk")
def disk(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    usage = shutil.disk_usage(ctx.options.library)

    def sizes(base) -> list[dict[str, Any]]:
        return [{"id": d.name, "bytes": dir_size(d)} for d in sorted(base.iterdir()) if d.is_dir()]

    return {
        "libraryFree": usage.free, "libraryTotal": usage.total,
        "characters": sizes(ctx.lib.characters_dir), "models": sizes(ctx.lib.models_dir),
        "projects": sizes(ctx.lib.projects_dir),
    }


@router.post("/shutdown")
async def shutdown(request: Request, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    """Respond first, then stop gracefully (trainings get STOP) and exit."""
    ctx.shutdown_requested.set()
    request_exit = getattr(request.app.state, "request_exit", None)
    if request_exit is not None:
        asyncio.get_running_loop().call_later(0.2, request_exit)
    return {"ok": True}


@router.get("/train/presets", response_model=list[TrainPreset])
def train_presets(ctx: AppContext = Depends(get_ctx)) -> list[TrainPreset]:
    return presets(ctx.system.best_vram_gb())
