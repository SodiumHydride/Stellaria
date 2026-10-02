"""Plugins (Settings → 插件): add, wizard, settings, interpreter checks, installs, tests."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response

from ..audio.io import peaks
from ..context import AppContext
from ..errors import ApiError
from ..plugins.envs import interpreter_env, resolve_python_path
from ..plugins.models import PluginPatch, WizardSpec
from ..plugins.pycheck import run_check
from ..schemas import JobView, Wire
from .deps import get_ctx

router = APIRouter()


class AddBody(Wire):
    folder: str


class InstallBody(Wire):
    target: Literal["builtin", "custom"] = "builtin"


class TestBody(Wire):
    params: dict[str, Any] | None = None


class PythonCheckBody(Wire):
    path: str
    packages: list[str] | None = None


@router.get("/plugins")
def list_plugins(ctx: AppContext = Depends(get_ctx)) -> list[dict[str, Any]]:
    return [p.wire() for p in ctx.plugins.list()]


@router.post("/plugins")
def add_plugin(body: AddBody, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.plugins.add_folder(body.folder).wire()


@router.post("/plugins/wizard")
def add_wizard(body: WizardSpec, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.plugins.add_wizard(body).wire()


@router.get("/plugins/{pid}")
def get_plugin(pid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.plugins.get(pid).wire()


@router.patch("/plugins/{pid}")
def update_plugin(pid: str, body: PluginPatch, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.plugins.update(pid, body).wire()


@router.delete("/plugins/{pid}")
def delete_plugin(pid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    ctx.plugins.remove(pid)
    return {"ok": True}


@router.post("/plugins/{pid}/check")
def check_plugin(pid: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.plugins.check(pid).wire()


@router.post("/plugins/{pid}/install-requirements", response_model=JobView)
def install_requirements(pid: str, body: InstallBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.plugins.submit_install(pid, body.target if body else "builtin").view()


@router.post("/plugins/{pid}/test", response_model=JobView)
def test_plugin(pid: str, body: TestBody | None = None, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.plugins.submit_test(pid, body.params if body else None).view()


def _test_output(ctx: AppContext, pid: str, job_id: str):
    ctx.plugins.get(pid)
    path = ctx.plugins.test_output(pid, job_id)
    if not job_id.startswith("job_") or "/" in job_id or "\\" in job_id or not path.is_file():
        raise ApiError("plugin.test_not_found", f"No test output for job '{job_id}'", status=404)
    return path


@router.get("/plugins/{pid}/test/{job_id}/audio")
def test_audio(pid: str, job_id: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    return FileResponse(_test_output(ctx, pid, job_id), media_type="audio/wav")


@router.get("/plugins/{pid}/test/{job_id}/peaks")
def test_peaks(pid: str, job_id: str, n: int = 800, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    values, duration = peaks(_test_output(ctx, pid, job_id), n)
    return {"peaks": values, "duration": duration}


@router.post("/python/check")
def python_check(body: PythonCheckBody) -> dict[str, Any]:
    exe = resolve_python_path(body.path) if body.path.strip() else None
    env = interpreter_env(exe, custom=True) if exe is not None else None
    return run_check(body.path, body.packages, torch=True, env=env).wire()
