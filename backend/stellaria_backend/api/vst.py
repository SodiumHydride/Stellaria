"""VST3 / Audio Unit host (canvas-v3 §3.2): status, folders, scanning, plugins, editor, inspect."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from ..context import AppContext
from ..schemas import JobView, Wire
from .deps import get_ctx

router = APIRouter()


class SettingsBody(Wire):
    folders: list[str]


class ScanBody(Wire):
    full: bool = False


class StateBody(Wire):
    state: str | None = None


@router.get("/vst/status")
def vst_status(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.vst.status()


@router.patch("/vst/settings")
def vst_settings(body: SettingsBody, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.vst.set_folders(body.folders)


@router.post("/vst/scan", response_model=JobView)
def vst_scan(body: ScanBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.vst.submit_scan(full=bool(body and body.full)).view()


@router.get("/vst/plugins")
def vst_plugins(ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    return ctx.vst.plugins()


@router.get("/vst/plugins/{plugin_id}")
def vst_plugin(plugin_id: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.vst.require(plugin_id)


@router.post("/vst/plugins/{plugin_id}/editor", response_model=JobView)
def vst_editor(plugin_id: str, body: StateBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.vst.submit_editor(plugin_id, body.state if body else None).view()


@router.post("/vst/plugins/{plugin_id}/inspect")
def vst_inspect(plugin_id: str, body: StateBody | None = None, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.vst.inspect(plugin_id, body.state if body else None)
