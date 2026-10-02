"""Runs the separation worker as a subprocess (vocals/instrumental, the dry voice, music
residue, weights).

``model`` arguments are separation model ids (``separation_models``); None = the default
(``Preferences.separationModel`` while ready, else the best installed model). ``cleanup`` takes
cleanup model ids (None = ``Preferences.cleanupModel`` while ready, else the best installed one).
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ApiError
from ..jobs import JobContext
from ..jobs.manager import PLACEMENT_CPU
from ..util import procs
from .commands import engine_env
from .separation_models import ADDON, CLEANUP, DEFAULT_DEMUCS, SEPARATION, SeparationCatalog

if TYPE_CHECKING:
    from ..context import AppContext


class SeparationClient:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.catalog = SeparationCatalog(ctx)

    @property
    def available(self) -> bool:
        return self.ctx.resources.is_ready(ADDON)

    def require(self, model: Any = None, role: str = SEPARATION) -> str:
        """The id of the model to separate (``role`` cleanup: to clean up) with (raises
        ``addon.missing`` and the errors of ``SeparationCatalog.resolve``)."""
        if not self.available:
            raise ApiError("addon.missing", "Vocal separation needs the 'separator' add-on", status=409,
                           detail=f"resource={ADDON}", params={"resource": ADDON})
        return self.catalog.resolve(model, role).id

    def require_cleanup(self, model: Any = None) -> str:
        return self.require(model, CLEANUP)

    @property
    def best_device(self) -> str:
        return self.ctx.system.torch().device

    @property
    def demucs_model(self) -> str:
        """The add-on's own Demucs model (fetched at install, splits drums/bass/other)."""
        return self.ctx.resources.definition(ADDON).install.get("model", DEFAULT_DEMUCS)

    @property
    def default_model(self) -> str:
        return self.catalog.default()

    def _model_args(self, model: str | None, role: str = SEPARATION) -> list[str]:
        """Worker arguments for a model id (already checked by ``require``; the add-on's install
        fetches Demucs weights before the add-on counts as ready)."""
        entry = self.catalog.get(model, role) if model else None
        if entry is None:
            entry = self.catalog.resolve(model, role)
        if role == CLEANUP:
            return ["--cleanup", str(entry.checkpoint), "--cleanup-config", str(entry.config)]
        if entry.engine == "demucs":
            return ["--model", entry.id]
        return ["--model", self.demucs_model, "--roformer", str(entry.checkpoint),
                "--roformer-config", str(entry.config)]

    def run(self, jctx: JobContext | None, args: list[str], span: tuple[float, float] = (0.0, 1.0),
            device: str | None = None, model: str | None = None, role: str = SEPARATION) -> dict[str, Any]:
        cmd = [self.ctx.python, "-m", "stellaria_backend.engine.separate_worker", *self._model_args(model, role), *args]
        if device:
            cmd += ["--device", device]
        if self.ctx.options.fake_engine:
            cmd.append("--fake")
        env = engine_env(self.ctx, self.ctx.resources.addon_env(ADDON))
        proc = procs.popen(cmd, cwd=self.ctx.options.engine, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if jctx is not None:
            jctx.on_cancel(lambda: procs.kill_tree(proc.pid))
        stderr: deque[str] = deque(maxlen=200)

        def drain() -> None:
            assert proc.stderr is not None
            for raw in proc.stderr:
                stderr.append(raw.decode("utf-8", errors="replace").rstrip())

        threading.Thread(target=drain, daemon=True).start()
        result: dict[str, Any] | None = None
        error: dict[str, Any] | None = None
        assert proc.stdout is not None
        for raw in proc.stdout:
            try:
                msg = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if msg.get("type") == "progress" and jctx is not None:
                jctx.progress(span[0] + (span[1] - span[0]) * float(msg["value"]))
            elif msg.get("type") == "result":
                result = msg.get("data") or {}
            elif msg.get("type") == "error":
                error = msg
        proc.wait()
        if jctx is not None:
            jctx.on_cancel(None)
            jctx.check_cancelled()
        if error is not None or result is None:
            error = error or {}
            code, text = ("separate.failed", "Separation failed") if role == SEPARATION else \
                ("cleanup.failed", "Cleanup failed")
            raise ApiError(error.get("code", code), error.get("message", text), status=500,
                           detail=error.get("detail") or "\n".join(stderr), params=error.get("params") or {})
        return result

    def side_device(self) -> str | None:
        """Device for separations inside CPU jobs (organize, health): follows ``gpuPolicy``
        while a training holds the GPU, else the best device."""
        return "cpu" if self.ctx.jobs.training_placement() == PLACEMENT_CPU else None

    def separate(self, jctx: JobContext | None, src: Path, vocals: Path, instrumental: Path,
                 span: tuple[float, float] = (0.0, 1.0), device: str | None = None,
                 extra: dict[str, Path] | None = None, model: str | None = None) -> None:
        """Vocals + instrumental; with ``extra`` = {drums, bass, other} paths also those stems."""
        model = self.require(model)
        vocals.parent.mkdir(parents=True, exist_ok=True)
        args = ["--input", str(src), "--vocals", str(vocals), "--instrumental", str(instrumental)]
        for name, path in (extra or {}).items():
            args += [f"--{name}", str(path)]
        self.run(jctx, args, span, device, model)

    def cleanup(self, jctx: JobContext | None, src: Path, dry: Path, wet: Path | None = None,
                span: tuple[float, float] = (0.0, 1.0), device: str | None = None, model: str | None = None) -> str:
        """The dry voice of ``src`` (without reverb / echo), and with ``wet`` what was removed
        (``src`` − dry); returns the cleanup model id."""
        model = self.require_cleanup(model)
        dry.parent.mkdir(parents=True, exist_ok=True)
        args = ["--input", str(src), "--dry", str(dry), *(["--wet", str(wet)] if wet is not None else [])]
        self.run(jctx, args, span, device, model, CLEANUP)
        return model

    def measure(self, jctx: JobContext | None, files: list[Path], span: tuple[float, float],
                model: str | None = None) -> dict[str, float]:
        model = self.require(model)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8",
                                         dir=self.ctx.tmp_dir) as f:
            json.dump([str(p) for p in files], f)
            list_path = Path(f.name)
        try:
            return self.run(jctx, ["--measure", str(list_path)], span, self.side_device(), model).get("ratios", {})
        finally:
            list_path.unlink(missing_ok=True)


def fetch_separator_weights(ctx: "AppContext", jctx: JobContext) -> None:
    """Let Demucs download the weights of the add-on's own model into the add-on folder (part of
    installing the add-on). Other Demucs models download when first used; Roformer checkpoints
    are resources of their own."""
    client = SeparationClient(ctx)
    client.run(jctx, ["--download"], model=client.demucs_model)
