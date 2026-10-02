"""Training lifecycle: prepare → preprocess → train, pause (STOP), re-attach, monitoring.

Nothing is injected into the training loop. Progress comes from the tfevents files the
engine writes; pausing writes ``exp/STOP`` (the engine saves a checkpoint, then exits).
The training process writes its output straight into ``<model>/model.log`` (not a pipe),
so it survives a backend restart; ``exp/train.pid`` lets the next backend re-attach.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil

from ..errors import ApiError, JobCancelled
from ..jobs import Job, JobContext
from ..schemas import Msg, msg
from ..util import procs
from ..util.timeutil import now_iso
from . import checkpoints, config as engine_config, dataset
from .commands import engine_env, engine_script
from .logs import LogTail, classify_failure, tail_lines
from .preprocess import check_after_skips, run_preprocess, skipped_files

if TYPE_CHECKING:
    from ..context import AppContext

log = logging.getLogger(__name__)

ACTIVE_STATUSES = {"preparing", "preprocessing", "training", "stopping"}


@dataclass
class Runtime:
    step: int = 0
    speed: float | None = None
    eta: float | None = None


class _Process:
    """A training process we started (Popen) or found again after a restart (psutil)."""

    def __init__(self, pid: int, popen: Any = None, log_offset: int | None = None) -> None:
        self.pid = pid
        self.popen = popen
        self.log_offset = log_offset  # where this run's output starts in model.log
        self._ps = None if popen else psutil.Process(pid)

    def running(self) -> bool:
        if self.popen is not None:
            return self.popen.poll() is None
        try:
            return self._ps.is_running() and self._ps.status() != psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return False

    @property
    def returncode(self) -> int | None:
        return self.popen.returncode if self.popen is not None else None


class TrainingService:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self._runtime: dict[str, Runtime] = {}
        self._stop_requested: set[str] = set()
        self._lock = threading.Lock()

    # ---- queries ---------------------------------------------------------------------
    def runtime(self, mid: str) -> Runtime | None:
        return self._runtime.get(mid)

    def active_job(self, mid: str) -> Job | None:
        return self.ctx.jobs.find_active("train", modelId=mid)

    def pid_path(self, mid: str) -> Path:
        return self.ctx.models.exp_dir(mid) / "train.pid"

    # ---- status ----------------------------------------------------------------------
    def set_status(self, mid: str, status: str, error: Msg | None = None, **fields: Any) -> None:
        with self.ctx.models.lock(mid):
            rec = self.ctx.models.get(mid)
            rec.status = status
            rec.error = error
            for k, v in fields.items():
                setattr(rec, k, v)
            self.ctx.models.save(rec)
        self.ctx.emit_model(mid)

    def _resting_status(self, mid: str) -> str:
        rec = self.ctx.models.get(mid)
        latest = checkpoints.latest_checkpoint(self.ctx.models.exp_dir(mid))
        if latest is None:
            return "idle"
        return "done" if latest[0] >= rec.target_steps else "paused"

    # ---- start / pause ---------------------------------------------------------------
    def start(self, mid: str, fresh: bool = False) -> Job:
        rec = self.ctx.models.get(mid)
        if self.active_job(mid):
            raise ApiError("model.busy", "This model is already training", status=409)
        blockers = self.ctx.system.train_blockers()
        if blockers:
            raise ApiError(blockers[0].code, "Training needs an NVIDIA GPU with CUDA", status=409)
        self._check_health(rec)
        self._stop_requested.discard(mid)
        previous = rec.status
        self.set_status(mid, "preparing")
        try:
            return self.ctx.jobs.submit(
                "train", msg("job.train", name=rec.name), lambda jctx: self._train_job(jctx, mid, fresh),
                target={"modelId": mid}, uses_gpu=True, lock_key=f"mdl:{mid}",
            )
        except ApiError:
            self.set_status(mid, previous)
            raise

    def _check_health(self, rec) -> None:
        for mc in rec.characters:
            if not self.ctx.characters.exists(mc.character_id):
                continue
            report, revision = self.ctx.characters.health(mc.character_id)
            current = self.ctx.characters.get(mc.character_id).clips_revision
            if report is not None and report.blocking and revision == current:
                name = self.ctx.characters.get(mc.character_id).name
                raise ApiError("train.health_blocking", f"The material of '{name}' has blocking issues",
                               status=409, detail=mc.character_id, params={"characterId": mc.character_id})

    def pause(self, mid: str) -> None:
        job = self.active_job(mid)
        if job is None:
            rec = self.ctx.models.get(mid)
            if rec.status in ACTIVE_STATUSES:  # stale status without a job: just settle it
                self.set_status(mid, self._resting_status(mid))
                return
            raise ApiError("model.not_training", "This model is not training", status=409)
        if job.status in ("queued", "waiting_gpu"):
            self.ctx.jobs.cancel(job.id)
            self.set_status(mid, self._resting_status(mid))
            return
        self._stop_requested.add(mid)
        if job.cancel_handler is not None:
            job.cancel_handler()
        if self.ctx.models.get(mid).status == "training":
            self.set_status(mid, "stopping")

    def _write_stop(self, mid: str) -> None:
        stop = self.ctx.models.exp_dir(mid) / "STOP"
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.write_text("stop", encoding="utf-8")

    def stop_all(self, timeout: float = 20.0) -> None:
        """Ask every running training to stop (checkpoint first) and wait up to ``timeout``."""
        jobs = [j for j in self.ctx.jobs.list(active_only=True) if j.type == "train"]
        for job in jobs:
            mid = job.target.get("modelId")
            if mid:
                try:
                    self.pause(mid)
                except ApiError:
                    pass
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and any(j.active for j in jobs):
            time.sleep(0.2)

    # ---- the train job ---------------------------------------------------------------
    def _train_job(self, jctx: JobContext, mid: str, fresh: bool) -> dict[str, Any]:
        try:
            return self._train_job_inner(jctx, mid, fresh)
        except JobCancelled:
            self.set_status(mid, self._resting_status(mid))
            if mid in self._stop_requested and not jctx.cancelled:
                return {"paused": True}
            raise
        except ApiError as exc:
            self.set_status(mid, "failed", Msg(code=exc.code, params=exc.params))
            raise
        except Exception as exc:
            self.set_status(mid, "failed", Msg(code="internal", params={"message": str(exc)}))
            raise
        finally:
            self._stop_requested.discard(mid)
            self._runtime.pop(mid, None)

    def _train_job_inner(self, jctx: JobContext, mid: str, fresh: bool) -> dict[str, Any]:
        ctx = self.ctx
        stop = lambda: mid in self._stop_requested  # noqa: E731
        rec = ctx.models.get(mid)

        # 1) prepare: assemble the dataset (skipped when the fingerprint is unchanged)
        jctx.stage("stage.prepare")
        self.set_status(mid, "preparing")
        entries = dataset.plan(rec, ctx.characters)
        dataset.validate(entries)
        ds_dir = ctx.models.dataset_dir(mid)
        manifest = dataset.read_manifest(ds_dir)
        fp = dataset.fingerprint(entries)
        if not manifest or manifest.get("fingerprint") != fp or not (ds_dir / "train" / "audio").is_dir():
            jctx.log(f"assembling dataset: {len(entries)} clips")
            manifest = dataset.assemble(ds_dir, entries)
        else:
            jctx.log("dataset unchanged")
        if stop() or jctx.cancelled:
            raise JobCancelled()

        self._require_resources(rec)
        self.write_engine_config(mid, entries)

        # 2) preprocess (skipped when dataset and feature settings are unchanged)
        pfp = dataset.preprocess_fingerprint(fp, rec)
        if (manifest.get("preprocess") or {}).get("fingerprint") != pfp:
            if skipped_files(ds_dir):  # give clips a previous extractor failed on another chance
                manifest = dataset.assemble(ds_dir, entries)
            self.set_status(mid, "preprocessing")
            jctx.stage("stage.preprocess", 0.0, done=0, total=len(entries))
            run_preprocess(ctx, jctx, mid, len(entries), stop)
            self._flag_skipped(mid, skipped_files(ds_dir))
            check_after_skips(ds_dir)
            dataset.mark_preprocessed(ds_dir, pfp)
        else:
            jctx.log("features up to date, preprocessing skipped")

        # 3) train
        exp = ctx.models.exp_dir(mid)
        if fresh:
            self._wipe_experiment(mid)
        if ctx.models.get(mid).config.base_model_id and checkpoints.latest_checkpoint(exp) is None:
            jctx.stage("stage.seed_base")
            ctx.base_models.seed(jctx, mid)
        self._check_arch(mid)
        while True:
            rec = ctx.models.get(mid)
            latest = checkpoints.latest_checkpoint(exp)
            start_step = latest[0] if latest else 0
            if start_step >= rec.target_steps:
                self.set_status(mid, "done")
                self._after_done(mid)
                return {"done": True, "step": start_step}
            if stop() or jctx.cancelled:
                raise JobCancelled()
            self.write_engine_config(mid, entries)  # targetSteps may have changed
            proc = self._launch(mid)
            result = self._supervise(jctx, mid, proc, start_step)
            if result is not None:
                return result
            latest = checkpoints.latest_checkpoint(exp)
            if (latest[0] if latest else 0) <= start_step:
                raise classify_failure(tail_lines(ctx.models.log_path(mid), 200), "engine.train_failed",
                                       "Training ended without making progress")

    def _require_resources(self, rec) -> None:
        needed = ["nsf_hifigan", engine_config.ENCODER_RESOURCE[rec.config.encoder]]
        if rec.config.f0_extractor == "rmvpe":
            needed.append("rmvpe")
        self.ctx.resources.require(needed)
        if rec.config.base_model_id and checkpoints.latest_checkpoint(self.ctx.models.exp_dir(rec.id)) is None:
            self.ctx.base_models.files(rec.config.base_model_id)  # a missing base fails before preprocessing

    def write_engine_config(self, mid: str, entries: list[dataset.DatasetEntry]) -> None:
        ctx = self.ctx
        rec = ctx.models.get(mid)
        torch = ctx.system.torch() if not ctx.options.fake_engine else None
        device = "cpu" if ctx.options.fake_engine else ctx.system.train_device()
        shortest = min((e.duration for e in entries), default=2.0)
        conf = engine_config.build_engine_config(
            engine_config.load_template(ctx.options.engine), model_dir=ctx.models.dir(mid), cfg=rec.config,
            target_steps=rec.target_steps, n_spk=len(rec.characters),
            encoder_ckpt=ctx.resources.path(engine_config.ENCODER_RESOURCE[rec.config.encoder]),
            vocoder_ckpt=ctx.resources.path("nsf_hifigan"), device=device,
            fused_available=bool(torch and torch.device == "cuda" and torch.triton),
            duration=engine_config.training_duration(shortest), glu_type=rec.glu_type,
        )
        engine_config.write_yaml(ctx.models.config_path(mid), conf)

    def _flag_skipped(self, mid: str, skipped: list[str]) -> None:
        if not skipped:
            return
        manifest = dataset.read_manifest(self.ctx.models.dataset_dir(mid)) or {}
        by_path = {e["path"]: e for e in manifest.get("entries", [])}
        per_char: dict[str, set[str]] = {}
        for rel in skipped:
            entry = by_path.get(rel)
            if entry:
                per_char.setdefault(entry["characterId"], set()).add(entry["clipId"])
        for cid, ids in per_char.items():
            with self.ctx.characters.lock(cid):
                clips = self.ctx.characters.clips(cid)
                for c in clips:
                    if c.id in ids and "f0_failed" not in c.flags:
                        c.flags.append("f0_failed")
                self.ctx.characters.save_clips(cid, clips, bump=False)
            self.ctx.emit_clips(cid)

    def _wipe_experiment(self, mid: str) -> None:
        exp = self.ctx.models.exp_dir(mid)
        for path in checkpoints.checkpoint_steps(exp).values():
            path.unlink(missing_ok=True)
        shutil.rmtree(exp / "logs", ignore_errors=True)
        (exp / "config.yaml").unlink(missing_ok=True)
        shutil.rmtree(self.ctx.models.cache_dir(mid) / "validation", ignore_errors=True)
        self.ctx.metrics.reader(mid).reset()
        self.set_status(mid, "preparing", training_time=0.0)

    def _check_arch(self, mid: str) -> None:
        exp = self.ctx.models.exp_dir(mid)
        if checkpoints.latest_checkpoint(exp) is None:
            return
        used = engine_config.read_yaml(exp / "config.yaml")
        if used is None:
            return
        rec = self.ctx.models.get(mid)
        wanted = {f: getattr(rec.config, f) for f in engine_config.ARCH_FIELDS}
        if engine_config.arch_of_engine_config(used) != wanted:
            raise ApiError("train.config_mismatch",
                           "The model structure changed since the last checkpoint; start fresh to apply it",
                           status=409, detail=json.dumps({"checkpoint": engine_config.arch_of_engine_config(used),
                                                          "config": wanted}))

    def _launch(self, mid: str) -> _Process:
        ctx = self.ctx
        exp = ctx.models.exp_dir(mid)
        exp.mkdir(parents=True, exist_ok=True)
        (exp / "STOP").unlink(missing_ok=True)  # a stale STOP would end the run at once
        ctx.render.shutdown_if_idle()  # free VRAM held by the render worker
        cmd, cwd = engine_script(ctx, "train_reflow.py", ["-c", str(ctx.models.config_path(mid))])
        log_path = ctx.models.log_path(mid)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n=== training {now_iso()} ===\n")
        offset = log_path.stat().st_size
        log_file = open(log_path, "ab")
        try:
            popen = procs.popen(cmd, cwd=cwd, env=engine_env(ctx), stdout=log_file, stderr=log_file)
        finally:
            log_file.close()  # the child keeps its own handle
        info = {"pid": popen.pid, "createTime": psutil.Process(popen.pid).create_time(), "startedAt": now_iso(),
                "fake": ctx.options.fake_engine}
        self.pid_path(mid).write_text(json.dumps(info), encoding="utf-8")
        self.set_status(mid, "training")
        return _Process(popen.pid, popen, offset)

    def _supervise(self, jctx: JobContext, mid: str, proc: _Process, start_step: int) -> dict[str, Any] | None:
        """Wait for the training process; returns the job result, or None to launch again."""
        ctx = self.ctx
        jctx.stage("stage.train", None, step=start_step, target=ctx.models.get(mid).target_steps)
        jctx.on_cancel(lambda: self._write_stop(mid))
        monitor = TrainingMonitor(self, mid, jctx, LogTail(ctx.models.log_path(mid), proc.log_offset))
        monitor.start()
        try:
            while proc.running():
                time.sleep(0.5)
        finally:
            monitor.stop()
            jctx.on_cancel(None)
            self.pid_path(mid).unlink(missing_ok=True)
        training_time = ctx.metrics.polled(mid).training_time()
        with ctx.models.lock(mid):
            rec = ctx.models.get(mid)
            rec.training_time = training_time
            ctx.models.save(rec)
        checkpoints.prune_pause_checkpoints(rec, ctx.models.exp_dir(mid))
        latest = checkpoints.latest_checkpoint(ctx.models.exp_dir(mid))
        step = latest[0] if latest else 0
        stopped = mid in self._stop_requested or jctx.cancelled
        if step >= rec.target_steps:
            self.set_status(mid, "done")
            self._after_done(mid)
            return {"done": True, "step": step}
        if stopped:
            self.set_status(mid, "paused" if latest else "idle")
            if jctx.cancelled:
                raise JobCancelled()
            return {"paused": True, "step": step}
        lines = tail_lines(ctx.models.log_path(mid), 200)
        if proc.returncode == 0 or (proc.returncode is None and any("training stopped" in line for line in lines[-5:])):
            return None  # ended by max_steps below a raised target: continue
        raise classify_failure(lines, "engine.train_failed", "Training stopped with an error")

    def _after_done(self, mid: str) -> None:
        if self.ctx.prefs.get().auto_slim_after_training:
            from .slim import submit_slim

            submit_slim(self.ctx, mid, dataset_too=True, checkpoints_mode="keep_latest")

    # ---- re-attach after a backend restart --------------------------------------------
    def reattach_all(self) -> None:
        for rec in self.ctx.models.list_records():
            info = self._read_pid(rec.id)
            if info and procs.process_alive(info["pid"], info.get("createTime")):
                log.info("re-attaching to training of %s (pid %s)", rec.id, info["pid"])
                self._runtime[rec.id] = Runtime()
                self.ctx.jobs.submit(
                    "train", msg("job.train", name=rec.name), lambda jctx, m=rec.id, i=info: self._reattached(jctx, m, i),
                    target={"modelId": rec.id}, uses_gpu=True, lock_key=f"mdl:{rec.id}",
                )
                if rec.status != "training":
                    self.set_status(rec.id, "training")
            else:
                self.pid_path(rec.id).unlink(missing_ok=True)
                if rec.status in ACTIVE_STATUSES:
                    self.set_status(rec.id, self._resting_status(rec.id))

    def _read_pid(self, mid: str) -> dict[str, Any] | None:
        try:
            return json.loads(self.pid_path(mid).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None

    def _reattached(self, jctx: JobContext, mid: str, info: dict[str, Any]) -> dict[str, Any]:
        try:
            proc = _Process(int(info["pid"]))
        except psutil.NoSuchProcess:
            self.set_status(mid, self._resting_status(mid))
            return {"reattached": True}
        latest = checkpoints.latest_checkpoint(self.ctx.models.exp_dir(mid))
        try:
            result = self._supervise(jctx, mid, proc, latest[0] if latest else 0)
        except JobCancelled:
            raise
        except ApiError as exc:
            self.set_status(mid, "failed", Msg(code=exc.code, params=exc.params))
            raise
        finally:
            self._stop_requested.discard(mid)
            self._runtime.pop(mid, None)
        if result is None:  # cannot relaunch a process we did not start with this job's state
            self.set_status(mid, self._resting_status(mid))
            return {"reattached": True}
        return {**result, "reattached": True}


class TrainingMonitor:
    """Polls tfevents (1/s) and pushes metrics/validation/model events and job progress."""

    def __init__(self, service: TrainingService, mid: str, jctx: JobContext, tail: LogTail) -> None:
        self.service = service
        self.ctx = service.ctx
        self.mid = mid
        self.jctx = jctx
        self.tail = tail
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"monitor-{mid}", daemon=True)
        self._last_ckpts: set[int] = set()
        self._last_model_emit = 0.0

    def start(self) -> None:
        self.service._runtime.setdefault(self.mid, Runtime())
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        try:
            self.tick()  # final read so the last points and log lines are not lost
        except Exception:  # noqa: BLE001
            log.exception("final monitor tick failed for %s", self.mid)

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - monitoring must never kill the job
                log.exception("training monitor tick failed for %s", self.mid)

    def tick(self) -> None:
        ctx, mid = self.ctx, self.mid
        lines = self.tail.read()
        if lines:
            self.jctx.log_lines(lines)
        reader = ctx.metrics.reader(mid)
        result = reader.poll()
        rec = ctx.models.get(mid)
        runtime = self.service._runtime.setdefault(mid, Runtime())
        if result.scalars:
            step = max(s for s, _ in result.scalars.values())
            ctx.hub.publish("metrics", {"modelId": mid, "step": step,
                                        "scalars": {tag: v for tag, (_, v) in result.scalars.items()}})
        for step in result.new_validation_steps:
            ctx.hub.publish("validation", {"modelId": mid, "step": step})
        if result.last_step is not None:
            runtime.step = result.last_step
            stop_file = ctx.models.exp_dir(mid) / "STOP"
            if runtime.step >= rec.target_steps and not stop_file.exists():
                self.service._write_stop(mid)  # target lowered below the run's max_steps
        runtime.speed = reader.speed()
        runtime.eta = (rec.target_steps - runtime.step) / runtime.speed if runtime.speed else None
        self.jctx.progress(runtime.step / max(1, rec.target_steps),
                           msg("stage.train", step=runtime.step, target=rec.target_steps), runtime.eta)
        ckpts = set(checkpoints.checkpoint_steps(ctx.models.exp_dir(mid)))
        changed = ckpts != self._last_ckpts
        if changed:
            self._last_ckpts = ckpts
            checkpoints.prune_pause_checkpoints(rec, ctx.models.exp_dir(mid))
        now = time.monotonic()
        if changed or now - self._last_model_emit >= 2.0:
            self._last_model_emit = now
            ctx.emit_model(mid, throttled=not changed)
