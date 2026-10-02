"""Runs the engine's ``preprocess.py`` and turns its tqdm output into job progress."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..errors import ApiError, JobCancelled
from ..jobs import JobContext
from ..schemas import msg
from ..util import procs
from ..util.timeutil import now_iso
from .commands import engine_env, engine_script
from .logs import classify_failure, split_lines

if TYPE_CHECKING:
    from ..context import AppContext

TQDM = re.compile(r"(\d+)%\|.*?\|\s*(\d+)/(\d+)")


class TqdmPhases:
    """Progress over consecutive tqdm bars (train set, then validation set)."""

    def __init__(self, grand_total: int) -> None:
        self.grand_total = max(1, grand_total)
        self.completed = 0
        self.done = 0
        self.total: int | None = None

    def feed(self, line: str) -> float | None:
        m = TQDM.search(line)
        if not m:
            return None
        done, total = int(m.group(2)), int(m.group(3))
        if self.total is not None and (total != self.total or done < self.done):
            self.completed += self.total
        self.done, self.total = done, total
        return min(1.0, (self.completed + done) / self.grand_total)


def preprocess_workers(device: str, vram_gb: float | None) -> int:
    if device == "cpu":
        return 1
    return 2 if (vram_gb or 0) >= 10 else 1


def run_preprocess(ctx: "AppContext", jctx: JobContext, mid: str, n_files: int,
                   should_stop: Callable[[], bool]) -> None:
    device = "cpu" if ctx.options.fake_engine else ctx.system.train_device()
    workers = preprocess_workers(device, ctx.system.best_vram_gb())
    config = ctx.models.config_path(mid)
    cmd, cwd = engine_script(ctx, "preprocess.py", ["-c", str(config), "-d", device, "-j", str(workers)])
    log_path = ctx.models.log_path(mid)
    phases = TqdmPhases(n_files)
    captured: list[str] = []
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"\n=== preprocess {now_iso()} ({device}, {workers} worker(s)) ===\n")
        proc = procs.popen(cmd, cwd=cwd, env=engine_env(ctx), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        jctx.on_cancel(lambda: procs.kill_tree(proc.pid))
        assert proc.stdout is not None
        rest = ""
        last_progress_line = None
        while True:
            chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            lines, rest = split_lines(rest + chunk.decode("utf-8", errors="replace"))
            for line in lines:
                value = phases.feed(line)
                if value is not None:
                    last_progress_line = line
                    jctx.progress(value, msg("stage.preprocess", done=phases.completed + phases.done, total=n_files))
                    continue
                captured.append(line)
                log_file.write(line + "\n")
                jctx.log(line)
        if last_progress_line:
            log_file.write(last_progress_line + "\n")
        rc = proc.wait()
        jctx.on_cancel(None)
    if should_stop() or jctx.cancelled:
        raise JobCancelled()
    if rc != 0:
        raise classify_failure(captured, "engine.preprocess_failed", "Preprocessing failed")
    dataset_dir = ctx.models.dataset_dir(mid)
    for split in ("train", "val"):
        if not (dataset_dir / split / "pitch_aug_dict.npy").exists():
            raise classify_failure(captured, "engine.preprocess_failed", f"Preprocessing produced no {split} features")


def skipped_files(dataset_dir: Path) -> list[str]:
    """Dataset-relative paths ('train/audio/x.wav') that preprocess moved to ``skip/`` (F0 failed)."""
    out = []
    for split in ("train", "val"):
        skip = dataset_dir / split / "skip"
        if skip.is_dir():
            for p in skip.rglob("*.wav"):
                out.append(f"{split}/audio/{p.relative_to(skip).as_posix()}")
    return out


def remaining_audio(dataset_dir: Path, split: str) -> int:
    audio = dataset_dir / split / "audio"
    return sum(1 for _ in audio.rglob("*.wav")) if audio.is_dir() else 0


def check_after_skips(dataset_dir: Path) -> None:
    for split in ("train", "val"):
        if remaining_audio(dataset_dir, split) == 0:
            raise ApiError("dataset.f0_failed", f"Pitch extraction failed for every {split} clip", status=409)
