"""Checkpoints in ``exp/`` (``model_<step>.pt``) and named releases in ``releases/``.

Kinds: ``permanent`` = multiple of intervalForceSave, ``regular`` = multiple of intervalVal,
``pause`` = anything else (saved because of STOP or max_steps). Pause checkpoints are removed
once a newer regular checkpoint exists.
"""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from ..errors import ApiError, not_found
from ..schemas import Checkpoint, ModelRecord, Release
from ..util.ids import new_id
from ..util.jsonio import write_json
from ..util.timeutil import iso_from_timestamp, now_iso

CKPT_RE = re.compile(r"^model_(\d+)\.pt$")


def checkpoint_steps(exp_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if exp_dir.is_dir():
        for p in exp_dir.iterdir():
            m = CKPT_RE.match(p.name)
            if m and p.is_file():
                out[int(m.group(1))] = p
    return out


def kind_of(step: int, interval_val: int, interval_force_save: int) -> str:
    if interval_force_save > 0 and step % interval_force_save == 0:
        return "permanent"
    if interval_val > 0 and step % interval_val == 0:
        return "regular"
    return "pause"


def list_checkpoints(rec: ModelRecord, exp_dir: Path) -> list[Checkpoint]:
    by_step = {r.step: r.id for r in rec.releases}
    out = []
    for step, path in sorted(checkpoint_steps(exp_dir).items()):
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        out.append(Checkpoint(
            step=step, file=path.name, size=st.st_size, created_at=iso_from_timestamp(st.st_mtime),
            kind=kind_of(step, rec.config.interval_val, rec.config.interval_force_save),
            release_id=by_step.get(step),
        ))
    return out


def latest_checkpoint(exp_dir: Path) -> tuple[int, Path] | None:
    steps = checkpoint_steps(exp_dir)
    if not steps:
        return None
    step = max(steps)
    return step, steps[step]


def has_trained(exp_dir: Path) -> bool:
    """Whether training saved a checkpoint (a step-0 base-model seed does not count)."""
    return any(step > 0 for step in checkpoint_steps(exp_dir))


def prune_pause_checkpoints(rec: ModelRecord, exp_dir: Path) -> list[int]:
    """Delete pause checkpoints older than the newest regular/permanent one, and the base-model
    seed (``model_0.pt``) once training saved anything newer."""
    ckpts = list_checkpoints(rec, exp_dir)
    regular = [c.step for c in ckpts if c.kind != "pause" and c.step > 0]
    if not regular:
        return []
    newest = max(regular)
    removed = []
    for c in ckpts:
        if (c.kind == "pause" or c.step == 0) and c.step < newest:
            (exp_dir / c.file).unlink(missing_ok=True)
            removed.append(c.step)
    return removed


def create_release(rec: ModelRecord, model_dir: Path, step: int, name: str | None, notes: str) -> Release:
    src = checkpoint_steps(model_dir / "exp").get(step)
    if src is None:
        raise not_found("checkpoint", str(step))
    config = model_dir / "exp" / "config.yaml"
    if not config.exists():
        config = model_dir / "config.yaml"
    if not config.exists():
        raise ApiError("model.no_config", "The model has no engine config yet", status=409)
    release = Release(id=new_id("rel"), name=(name or "").strip() or f"Step {step}", step=step,
                      created_at=now_iso(), notes=notes or "", size=src.stat().st_size)
    dest = model_dir / "releases" / release.id
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest / "model.pt")
    shutil.copy2(config, dest / "config.yaml")
    write_json(dest / "release.json", release.wire())
    return release


def release_dir(model_dir: Path, release_id: str) -> Path:
    return model_dir / "releases" / release_id


def wait_stable(path: Path, settle: float = 1.0, timeout: float = 20.0) -> bool:
    """True once ``path`` has not changed for ``settle`` seconds. ``torch.save`` writes the
    file in place, so a checkpoint of a running training may still be growing."""
    deadline = time.monotonic() + timeout
    last, since = None, time.monotonic()
    while time.monotonic() < deadline:
        st = path.stat()  # FileNotFoundError: deleted by the engine meanwhile
        signature = (st.st_size, st.st_mtime_ns)
        if signature != last:
            last, since = signature, time.monotonic()
        if time.time() - st.st_mtime >= settle and time.monotonic() - since >= min(settle, 0.2):
            return True
        time.sleep(0.1)
    return False


def snapshot_latest(exp_dir: Path, dest_dir: Path, skip: set[int] | frozenset[int] = frozenset(),
                    settle: float = 1.0, timeout: float = 20.0) -> Path | None:
    """Copy the newest complete checkpoint (+ ``config.yaml``) of a training that is still
    running into ``dest_dir`` and return the copy, so it can neither change nor be deleted by
    the engine while a render loads it. Older snapshots are removed."""
    config = exp_dir / "config.yaml"
    for step, path in sorted(checkpoint_steps(exp_dir).items(), reverse=True):
        if step in skip:
            continue
        dest = dest_dir / f"model_{step}.pt"
        try:
            if not wait_stable(path, settle, timeout):
                continue
            size = path.stat().st_size
            if not (dest.is_file() and dest.stat().st_size == size):
                dest_dir.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".tmp")
                shutil.copy2(path, tmp)
                if tmp.stat().st_size != size or path.stat().st_size != size:
                    tmp.unlink(missing_ok=True)
                    continue
                tmp.replace(dest)
            shutil.copy2(config, dest_dir / "config.yaml")
        except FileNotFoundError:
            continue
        for old in dest_dir.glob("model_*.pt"):
            if old != dest:
                old.unlink(missing_ok=True)
        return dest
    return None
