"""Job queue: cancellation, GPU serialization, smart policy, per-object locks, throttled events."""

from __future__ import annotations

import threading
import time

from stellaria_backend.events import EventHub
from stellaria_backend.jobs import JobManager
from stellaria_backend.schemas import msg
from conftest import wait_for


def manager(policy="queue", free=None):
    return JobManager(EventHub(), gpu_policy=lambda: policy, free_vram=lambda: free)


def sleeper(seconds: float, log: list | None = None, name: str = ""):
    def fn(jctx):
        if log is not None:
            log.append(("start", name, time.monotonic()))
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            jctx.check_cancelled()
            time.sleep(0.01)
        if log is not None:
            log.append(("end", name, time.monotonic()))
        return {"name": name}
    return fn


def test_gpu_jobs_run_one_at_a_time():
    jm = manager()
    log: list = []
    jobs = [jm.submit("render", msg("t"), sleeper(0.2, log, f"g{i}"), uses_gpu=True) for i in range(3)]
    wait_for(lambda: jm.get(jobs[1].id).status == "waiting_gpu", 5)
    wait_for(lambda: all(jm.get(j.id).status == "succeeded" for j in jobs), 10)
    starts = [t for kind, _, t in log if kind == "start"]
    ends = [t for kind, _, t in log if kind == "end"]
    assert [n for kind, n, _ in log if kind == "start"] == ["g0", "g1", "g2"]
    assert starts[1] >= ends[0] and starts[2] >= ends[1]
    assert jm.get(jobs[2].id).result == {"name": "g2"}
    jm.shutdown(1)


def test_cpu_jobs_run_in_parallel_but_lock_keys_serialize():
    jm = manager()
    log: list = []
    a = jm.submit("health", msg("t"), sleeper(0.3, log, "a"), lock_key="chr:1")
    b = jm.submit("health", msg("t"), sleeper(0.3, log, "b"), lock_key="chr:2")
    c = jm.submit("organize", msg("t"), sleeper(0.1, log, "c"), lock_key="chr:1")
    wait_for(lambda: all(jm.get(j.id).status == "succeeded" for j in (a, b, c)), 10)
    t = {(k, n): v for k, n, v in log}
    assert t[("start", "b")] < t[("end", "a")]  # different keys overlap
    assert t[("start", "c")] >= t[("end", "a")]  # same key waits
    jm.shutdown(1)


def test_cancel_queued_and_running():
    jm = manager()
    stopped = threading.Event()
    first = jm.submit("train", msg("t"), sleeper(5), uses_gpu=True)
    queued = jm.submit("render", msg("t"), sleeper(5), uses_gpu=True)
    wait_for(lambda: jm.get(first.id).status == "running", 5)
    assert jm.cancel(queued.id).status == "cancelled"

    def with_handler(jctx):
        jctx.on_cancel(stopped.set)
        while not stopped.is_set():
            time.sleep(0.01)
        return {"paused": True}

    jm.cancel(first.id)
    wait_for(lambda: jm.get(first.id).status == "cancelled", 5)
    job = jm.submit("train", msg("t"), with_handler, uses_gpu=True)
    wait_for(lambda: jm.get(job.id).status == "running", 5)
    assert jm.cancel(job.id).status == "stopping"
    wait_for(lambda: jm.get(job.id).status == "cancelled", 5)
    assert stopped.is_set()
    jm.shutdown(1)


def run_next_to_training(policy: str, free: int | None, job_type: str = "render"):
    """Start a training, then a job of ``job_type``; returns (manager, train job, other job)."""
    jm = manager(policy, free)
    train = jm.submit("train", msg("t"), sleeper(1.0), uses_gpu=True)
    wait_for(lambda: jm.get(train.id).status == "running", 5)
    other = jm.submit(job_type, msg("t"), sleeper(0.05), uses_gpu=True, cpu_fallback=job_type in ("render", "separate"),
                      lock_key="render-worker")
    return jm, train, other


def test_smart_policy_renders_next_to_training_or_on_cpu():
    for free, placement in ((8 << 30, "best"), (int(2.6 * (1 << 30)), "best"), (1 << 30, "cpu"), (None, "cpu")):
        jm, train, render = run_next_to_training("smart", free)
        wait_for(lambda: jm.get(render.id).status == "succeeded", 0.6)  # well before the training ends
        assert render.placement == placement and jm.get(train.id).status == "running"
        jm.shutdown(1)


def test_cpu_policy_always_renders_on_cpu_during_training():
    jm, train, separate = run_next_to_training("cpu", 8 << 30, "separate")
    wait_for(lambda: jm.get(separate.id).status == "succeeded", 0.6)
    assert separate.placement == "cpu"
    jm.shutdown(1)


def test_queue_policy_waits_for_the_training():
    jm, train, render = run_next_to_training("queue", 8 << 30)
    time.sleep(0.3)
    assert jm.get(render.id).status == "waiting_gpu"
    wait_for(lambda: jm.get(render.id).status == "succeeded", 5)
    assert render.placement == "best" and jm.get(train.id).status == "succeeded"
    jm.shutdown(1)


def test_without_training_gpu_jobs_still_take_turns():
    jm = manager("smart", 8 << 30)
    log: list = []
    a = jm.submit("render", msg("t"), sleeper(0.3, log, "a"), uses_gpu=True, cpu_fallback=True, lock_key="render-worker")
    b = jm.submit("separate", msg("t"), sleeper(0.1, log, "b"), uses_gpu=True, cpu_fallback=True)
    c = jm.submit("render", msg("t"), sleeper(0.1, log, "c"), uses_gpu=True, cpu_fallback=True, lock_key="render-worker")
    wait_for(lambda: all(jm.get(j.id).status == "succeeded" for j in (a, b, c)), 5)
    t = {(k, n): v for k, n, v in log}
    assert t[("start", "b")] >= t[("end", "a")] and t[("start", "c")] >= t[("end", "b")]
    assert {a.placement, b.placement, c.placement} == {"best"}
    jm.shutdown(1)


def test_training_never_starts_next_to_another_gpu_job():
    jm = manager("smart", 8 << 30)
    render = jm.submit("render", msg("t"), sleeper(0.3), uses_gpu=True, cpu_fallback=True)
    wait_for(lambda: jm.get(render.id).status == "running", 5)
    train = jm.submit("train", msg("t"), sleeper(0.05), uses_gpu=True)
    time.sleep(0.15)
    assert jm.get(train.id).status == "waiting_gpu"
    wait_for(lambda: jm.get(train.id).status == "succeeded", 5)
    jm.shutdown(1)


def test_failures_become_error_objects():
    from stellaria_backend.errors import ApiError

    jm = manager()

    def boom(jctx):
        raise ApiError("engine.out_of_memory", "oom", detail="CUDA out of memory")

    def crash(jctx):
        raise ValueError("bad")

    a = jm.submit("train", msg("t"), boom)
    b = jm.submit("train", msg("t"), crash)
    wait_for(lambda: jm.get(a.id).status == "failed" and jm.get(b.id).status == "failed", 5)
    assert jm.get(a.id).error.code == "engine.out_of_memory" and jm.get(a.id).error.detail == "CUDA out of memory"
    assert jm.get(b.id).error.code == "internal" and "ValueError" in jm.get(b.id).error.detail
    jm.clear()
    assert jm.list() == []
    jm.shutdown(1)


def test_progress_events_are_throttled():
    hub = EventHub()
    sent = []
    hub._deliver = lambda t, d, topic: sent.append((t, d))  # type: ignore[method-assign]
    jm = JobManager(hub)

    def chatty(jctx):
        for i in range(200):
            jctx.progress(i / 200)
            time.sleep(0.002)
        return {}

    job = jm.submit("health", msg("t"), chatty)
    wait_for(lambda: jm.get(job.id).status == "succeeded", 5)
    time.sleep(0.3)
    job_events = [d for t, d in sent if t == "job" and d["id"] == job.id]
    assert 3 <= len(job_events) <= 12  # ~0.4 s of progress at ≤ 4/s plus status changes
    assert job_events[-1]["status"] == "succeeded" and job_events[-1]["progress"] == 1.0
    jm.shutdown(1)
    hub.close()
