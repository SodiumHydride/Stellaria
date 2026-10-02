"""Job queue and scheduler.

* Every job runs its function on its own thread and reports through a ``JobContext``.
* GPU jobs (train, render, separate) run one at a time. Jobs submitted with
  ``cpu_fallback`` (render, separate) get a *placement* when they start while a training holds
  the GPU, following the ``gpuPolicy`` preference:
  ``smart`` → next to the training on the GPU when ≥ 2.5 GB VRAM is free, else on the CPU at once;
  ``cpu`` → always on the CPU; ``queue`` → wait (``waiting_gpu``) until the training ends/pauses.
* ``lock_key`` serializes jobs touching the same object (e.g. two organize runs on one
  character) while unrelated CPU jobs run in parallel (up to ``max_cpu_jobs``).
* Jobs with a ``pool`` (``dsp``: process and mix jobs) have their own limit
  (``max(1, cpu_count // 4)``) instead; ``vst_editor`` jobs (an open plugin window, mostly
  idle) take no slot at all.
* Progress events are throttled to one per 250 ms per job; status changes go out at once.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import traceback
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Callable

from ..errors import ApiError, JobCancelled, not_found
from ..events import EventHub
from ..schemas import JobError, JobView, Msg
from ..util.ids import new_id
from ..util.timeutil import now_iso

log = logging.getLogger(__name__)

ACTIVE = {"queued", "waiting_gpu", "running", "stopping"}
FINISHED = {"succeeded", "failed", "cancelled"}
MAX_JOBS = 200
LOG_LINES = 5000
SMART_MIN_FREE_VRAM = int(2.5 * 1024**3)
PLACEMENT_BEST = "best"  # the best device the job's worker has (CUDA, else MPS, else CPU)
PLACEMENT_CPU = "cpu"  # forced CPU, next to a training that holds the GPU
POOL_DSP = "dsp"  # CPU-heavy creation jobs (process, mix)
UNCAPPED_TYPES = {"vst_editor"}  # an open plugin window waits on the user, not the CPU


def default_pools() -> dict[str, int]:
    return {POOL_DSP: max(1, (os.cpu_count() or 4) // 4)}

JobFn = Callable[["JobContext"], "dict[str, Any] | None"]


@dataclass(eq=False)
class Job:
    id: str
    type: str
    title: Msg
    target: dict[str, str]
    uses_gpu: bool
    fn: JobFn
    lock_key: str | None = None
    cpu_fallback: bool = False  # may run on the CPU next to a GPU training (render, separate)
    pool: str | None = None  # CPU jobs with their own concurrency limit (POOL_DSP)
    placement: str | None = None  # PLACEMENT_* once a GPU job started
    status: str = "queued"
    progress: float | None = None
    stage: Msg | None = None
    eta: float | None = None
    created_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    error: JobError | None = None
    result: dict[str, Any] | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    cancel_handler: Callable[[], None] | None = None
    log: deque = field(default_factory=lambda: deque(maxlen=LOG_LINES))
    unsent_log: list[str] = field(default_factory=list)
    last_log_flush: float = 0.0

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def view(self) -> JobView:
        return JobView(
            id=self.id, type=self.type, title=self.title, target=self.target, status=self.status,
            progress=self.progress, stage=self.stage, eta=self.eta, uses_gpu=self.uses_gpu,
            created_at=self.created_at, started_at=self.started_at, finished_at=self.finished_at,
            error=self.error, result=self.result,
        )


class JobContext:
    """Handed to job functions: progress/stage/log reporting and cancellation."""

    def __init__(self, manager: "JobManager", job: Job) -> None:
        self._manager = manager
        self.job = job

    @property
    def cancelled(self) -> bool:
        return self.job.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled()

    def on_cancel(self, handler: Callable[[], None] | None) -> None:
        """Register what cancelling means while this section runs (kill a process, write STOP)."""
        self.job.cancel_handler = handler
        if handler is not None and self.cancelled:
            handler()

    def progress(self, value: float | None, stage: Msg | None = None, eta: float | None = None) -> None:
        job = self.job
        job.progress = None if value is None else round(max(0.0, min(1.0, value)), 4)
        if stage is not None:
            job.stage = stage
        job.eta = eta
        self._manager.publish(job, throttled=True)

    def stage(self, code: str, progress: float | None = None, **params: Any) -> None:
        self.job.stage = Msg(code=code, params=params)
        self.job.progress = progress
        self.job.eta = None
        self._manager.publish(self.job, throttled=False)

    def log(self, line: str) -> None:
        self.log_lines([line])

    def log_lines(self, lines: list[str]) -> None:
        if not lines:
            return
        job = self.job
        job.log.extend(lines)
        job.unsent_log.extend(lines)
        self._manager.flush_log(job)


class JobManager:
    def __init__(
        self,
        hub: EventHub,
        *,
        gpu_policy: Callable[[], str] = lambda: "queue",
        free_vram: Callable[[], int | None] = lambda: None,
        max_cpu_jobs: int = 3,
        pools: dict[str, int] | None = None,
    ) -> None:
        self.hub = hub
        self._gpu_policy = gpu_policy
        self._free_vram = free_vram
        self._max_cpu_jobs = max_cpu_jobs
        self.pools = pools if pools is not None else default_pools()
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._threads: dict[str, threading.Thread] = {}
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._stopping = False
        self._dispatcher = threading.Thread(target=self._dispatch_loop, name="jobs-dispatcher", daemon=True)
        self._dispatcher.start()

    # ---- public API ------------------------------------------------------------------
    def submit(
        self,
        type_: str,
        title: Msg,
        fn: JobFn,
        *,
        target: dict[str, str] | None = None,
        uses_gpu: bool = False,
        lock_key: str | None = None,
        cpu_fallback: bool = False,
        pool: str | None = None,
    ) -> Job:
        job = Job(id=new_id("job"), type=type_, title=title, target=target or {}, uses_gpu=uses_gpu,
                  fn=fn, lock_key=lock_key, cpu_fallback=cpu_fallback, pool=pool)
        with self._lock:
            if self._stopping:
                raise ApiError("app.shutting_down", "The backend is shutting down", status=503)
            self._jobs[job.id] = job
            self._trim()
            self._wake.notify_all()
        self.publish(job)
        return job

    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise not_found("job", job_id)
        return job

    def list(self, active_only: bool = False) -> list[Job]:
        with self._lock:
            jobs = list(self._jobs.values())
        if active_only:
            jobs = [j for j in jobs if j.active]
        return list(reversed(jobs))

    def find_active(self, type_: str, **target: str) -> Job | None:
        for job in self.list(active_only=True):
            if job.type == type_ and all(job.target.get(k) == v for k, v in target.items()):
                return job
        return None

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        handler = None
        with self._lock:
            if job.status in ("queued", "waiting_gpu"):
                job.cancel_event.set()
                self._finish(job, "cancelled")
            elif job.status == "running":
                job.status = "stopping"
                job.cancel_event.set()
                handler = job.cancel_handler
            elif job.status == "stopping":
                handler = job.cancel_handler
        self.publish(job)
        if handler is not None:
            try:
                handler()
            except Exception:  # noqa: BLE001
                log.exception("cancel handler of %s failed", job.id)
        return job

    def clear(self) -> None:
        with self._lock:
            for jid in [j.id for j in self._jobs.values() if not j.active]:
                del self._jobs[jid]

    def shutdown(self, timeout: float = 20.0) -> None:
        """Cancel everything and wait (bounded) for running jobs to wind down."""
        with self._lock:
            self._stopping = True
            ids = [j.id for j in self._jobs.values() if j.active]
            self._wake.notify_all()
        for jid in ids:
            self.cancel(jid)
        deadline = time.monotonic() + timeout
        for thread in list(self._threads.values()):
            thread.join(max(0.0, deadline - time.monotonic()))

    def publish(self, job: Job, throttled: bool = False) -> None:
        self.hub.publish("job", job.view().wire(), throttle_key=f"job:{job.id}",
                         min_interval=0.25 if throttled else 0.0)

    def flush_log(self, job: Job, force: bool = False) -> None:
        now = time.monotonic()
        if not job.unsent_log or (not force and now - job.last_log_flush < 0.25):
            return
        lines, job.unsent_log = job.unsent_log, []
        job.last_log_flush = now
        topic = f"job.log:{job.id}"
        if self.hub.has_subscribers(topic):
            self.hub.publish("job.log", {"jobId": job.id, "lines": lines}, topic=topic)

    # ---- scheduling ------------------------------------------------------------------
    def _dispatch_loop(self) -> None:
        while True:
            with self._lock:
                if self._stopping and not any(j.status in ("queued", "waiting_gpu") for j in self._jobs.values()):
                    return
                self._schedule()
                waiting = any(j.status == "waiting_gpu" for j in self._jobs.values())
                self._wake.wait(timeout=0.25 if waiting else 0.5)
            for job in self.list(active_only=True):
                self.flush_log(job)

    def _schedule(self) -> None:
        running = [j for j in self._jobs.values() if j.status in ("running", "stopping")]
        held_keys = {j.lock_key for j in running if j.lock_key}
        gpu_holders = [j for j in running if j.uses_gpu and j.placement != PLACEMENT_CPU]
        cpu_slots = self._max_cpu_jobs - sum(1 for j in running if not j.uses_gpu and not j.pool
                                             and j.type not in UNCAPPED_TYPES)
        pool_used = Counter(j.pool for j in running if j.pool and not j.uses_gpu)
        gpu_blocked = False
        for job in [j for j in self._jobs.values() if j.status in ("queued", "waiting_gpu")]:
            if self._stopping or (job.lock_key and job.lock_key in held_keys):
                continue
            if job.uses_gpu:
                placement = self._placement(job, gpu_holders, gpu_blocked)
                if placement is None:
                    gpu_blocked = True
                    if job.status != "waiting_gpu":
                        job.status = "waiting_gpu"
                        job.stage = Msg(code="stage.waiting_gpu")
                        self.publish(job)
                    continue
                job.placement = placement
                if placement != PLACEMENT_CPU:
                    gpu_holders.append(job)
            elif job.type in UNCAPPED_TYPES:
                pass
            elif job.pool:
                if pool_used[job.pool] >= self.pools.get(job.pool, 1):
                    continue
                pool_used[job.pool] += 1
            else:
                if cpu_slots <= 0:
                    continue
                cpu_slots -= 1
            if job.lock_key:
                held_keys.add(job.lock_key)
            self._start(job)

    def _placement(self, job: Job, gpu_holders: list[Job], gpu_blocked: bool) -> str | None:
        """Where a GPU job may start now (``None`` = it has to wait for the GPU)."""
        if not gpu_holders and not gpu_blocked:
            return PLACEMENT_BEST
        if not job.cpu_fallback or not gpu_holders or any(j.type != "train" for j in gpu_holders):
            return None
        policy = self._gpu_policy()
        if policy == "cpu":
            return PLACEMENT_CPU
        if policy == "smart":
            free = self._free_vram()
            return PLACEMENT_BEST if free is not None and free >= SMART_MIN_FREE_VRAM else PLACEMENT_CPU
        return None  # 'queue'

    def training_placement(self) -> str:
        """Placement for GPU work done inside non-GPU jobs (e.g. separation while organizing)."""
        holders = [j for j in self.list(active_only=True) if j.status in ("running", "stopping") and j.type == "train"]
        if not holders:
            return PLACEMENT_BEST
        policy = self._gpu_policy()
        if policy == "smart":
            free = self._free_vram()
            if free is not None and free >= SMART_MIN_FREE_VRAM:
                return PLACEMENT_BEST
        return PLACEMENT_CPU

    def _start(self, job: Job) -> None:
        job.status = "running"
        job.started_at = now_iso()
        if job.stage is not None and job.stage.code == "stage.waiting_gpu":
            job.stage = None
        thread = threading.Thread(target=self._run, args=(job,), name=f"job-{job.type}-{job.id}", daemon=True)
        self._threads[job.id] = thread
        self.publish(job)
        thread.start()

    def _run(self, job: Job) -> None:
        ctx = JobContext(self, job)
        status, result, error = "succeeded", None, None
        try:
            result = job.fn(ctx)
            if job.cancel_event.is_set():
                status = "cancelled"
        except JobCancelled:
            status = "cancelled"
        except ApiError as exc:
            status, error = "failed", JobError(code=exc.code, message=exc.message, detail=exc.detail, params=exc.params)
        except Exception as exc:  # noqa: BLE001 - every failure becomes {code, message, detail}
            log.exception("job %s (%s) crashed", job.id, job.type)
            status = "failed"
            error = JobError(code="internal", message=str(exc) or type(exc).__name__, detail=traceback.format_exc())
        with self._lock:
            job.result = result if isinstance(result, dict) else job.result
            job.error = error
            self._finish(job, status)
            self._threads.pop(job.id, None)
            self._wake.notify_all()
        self.flush_log(job, force=True)
        self.publish(job)

    def _finish(self, job: Job, status: str) -> None:
        job.status = status
        job.finished_at = now_iso()
        job.cancel_handler = None
        if status == "succeeded":
            job.progress = 1.0
        job.eta = None

    def _trim(self) -> None:
        while len(self._jobs) > MAX_JOBS:
            victim = next((j for j in self._jobs.values() if not j.active), None)
            if victim is None:
                return
            del self._jobs[victim.id]
