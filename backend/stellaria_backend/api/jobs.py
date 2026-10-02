"""Jobs: list, inspect, logs, cancel, clear."""

from __future__ import annotations

from collections import deque
from typing import Any

from fastapi import APIRouter, Depends

from ..context import AppContext
from ..schemas import JobView
from .deps import get_ctx

router = APIRouter()


@router.get("/jobs", response_model=list[JobView])
def list_jobs(active: int = 0, ctx: AppContext = Depends(get_ctx)) -> list[JobView]:
    return [j.view() for j in ctx.jobs.list(active_only=bool(active))]


@router.get("/jobs/{job_id}", response_model=JobView)
def get_job(job_id: str, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.jobs.get(job_id).view()


@router.get("/jobs/{job_id}/log")
def job_log(job_id: str, tail: int = 500, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    job = ctx.jobs.get(job_id)
    return {"lines": list(deque(job.log, maxlen=max(1, tail)))}


@router.post("/jobs/{job_id}/cancel", response_model=JobView)
def cancel_job(job_id: str, ctx: AppContext = Depends(get_ctx)) -> JobView:
    return ctx.jobs.cancel(job_id).view()


@router.post("/jobs/clear")
def clear_jobs(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    ctx.jobs.clear()
    return {"ok": True}
