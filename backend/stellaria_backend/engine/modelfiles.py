"""Model files from elsewhere (external models, base models): validation in a subprocess,
step detection, copying with progress, and engine configs pointed at this machine."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ApiError
from ..jobs import JobContext
from ..schemas import TrainConfig
from ..util import procs
from .commands import engine_env

if TYPE_CHECKING:
    from ..context import AppContext

F0_EXTRACTORS = {"rmvpe", "fcpe", "parselmouth", "harvest", "dio", "crepe"}


def run_inspector(ctx: "AppContext", jctx: JobContext | None, args: list[str]) -> dict[str, Any]:
    """Run ``engine.inspect_worker`` (torch lives there) and return its result."""
    cmd = [ctx.python, "-m", "stellaria_backend.engine.inspect_worker", *args, "--engine", str(ctx.options.engine)]
    if ctx.options.fake_engine:
        cmd.append("--fake")
    proc = procs.popen(cmd, cwd=ctx.options.engine, env=engine_env(ctx), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if jctx is not None:
        jctx.on_cancel(lambda: procs.kill_tree(proc.pid))
    tail: deque[str] = deque(maxlen=60)

    def drain() -> None:
        assert proc.stderr is not None
        for raw in proc.stderr:
            line = raw.decode("utf-8", errors="replace").rstrip()
            tail.append(line)
            if jctx is not None:
                jctx.log(line)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    out, _ = proc.communicate()
    reader.join(timeout=5)
    if jctx is not None:
        jctx.on_cancel(None)
        jctx.check_cancelled()
    lines = [line for line in out.decode("utf-8", errors="replace").splitlines() if line.strip()]
    message = json.loads(lines[-1]) if lines else {}
    if "result" not in message:
        error = message.get("error") or {}
        raise ApiError(error.get("code", "model.inspect_failed"), error.get("message", "Could not read the model"),
                       status=500, detail=error.get("detail") or "\n".join(tail))
    return message["result"]


def inspect_files(ctx: "AppContext", jctx: JobContext | None, config: Path, ckpts: list[Path]) -> dict[str, Any]:
    """Validate a reflow DDSP-SVC config and strictly load every checkpoint against it.
    Raises ``model.import_incompatible`` ({reason}) or ``model.import_bad_checkpoint`` ({file})."""
    args = ["inspect", "--config", str(config)]
    for c in ckpts:
        args += ["--ckpt", str(c)]
    result = run_inspector(ctx, jctx, args)
    if result.get("incompatible"):
        reason = result["incompatible"]
        raise ApiError("model.import_incompatible", f"This model cannot be used here ({reason})", status=422,
                       params={"reason": reason}, detail=json.dumps(result.get("config"), ensure_ascii=False))
    for entry in result["checkpoints"]:
        if not entry["ok"]:
            raise ApiError("model.import_bad_checkpoint", f"'{Path(entry['path']).name}' cannot be loaded", status=422,
                           params={"file": Path(entry["path"]).name}, detail=entry.get("error") or "")
    return result


def assign_steps(entries: list[dict[str, Any]]) -> list[int]:
    """Training step of each checkpoint (``global_step`` or ``model_<step>.pt``), made unique."""
    used: set[int] = set()
    steps = []
    for entry in entries:
        step = entry.get("step")
        step = int(step) if step is not None else max(used, default=0) + 1
        while step in used:
            step += 1
        used.add(step)
        steps.append(step)
    return steps


def train_config_from(summary: dict[str, Any], base: TrainConfig) -> TrainConfig:
    """A TrainConfig carrying an imported model's structure."""
    f0 = summary.get("f0Extractor")
    return base.model_copy(update={
        "encoder": summary["encoder"], "n_layers": int(summary["nLayers"]), "n_chans": int(summary["nChans"]),
        "n_aux_layers": int(summary.get("nAuxLayers") or base.n_aux_layers),
        "n_aux_chans": int(summary.get("nAuxChans") or base.n_aux_chans),
        "use_pitch_aug": bool(summary.get("usePitchAug")), "f0_extractor": f0 if f0 in F0_EXTRACTORS else base.f0_extractor,
    })


def glu_type_of(summary: dict[str, Any]) -> str:
    """Configs written before ``glu_type`` existed belong to atanglu-trained checkpoints (engine rule)."""
    return summary.get("gluType") or "atanglu"


def copy_file(jctx: JobContext | None, src: Path, dest: Path, span: tuple[float, float] = (0.0, 1.0)) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    total, done = max(1, src.stat().st_size), 0
    with open(src, "rb") as fin, open(tmp, "wb") as fout:
        for block in iter(lambda: fin.read(1 << 22), b""):
            if jctx is not None:
                jctx.check_cancelled()
            fout.write(block)
            done += len(block)
            if jctx is not None:
                jctx.progress(span[0] + (span[1] - span[0]) * done / total)
    shutil.copystat(src, tmp)
    tmp.replace(dest)
