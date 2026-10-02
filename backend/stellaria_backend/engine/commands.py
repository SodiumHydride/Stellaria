"""Command lines and environment for engine subprocesses (real scripts or fake stand-ins)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..util import procs

if TYPE_CHECKING:
    from ..context import AppContext

BACKEND_ROOT = Path(__file__).resolve().parents[2]  # folder that contains `stellaria_backend`

FAKE_MODULES = {"preprocess.py": "stellaria_backend.fake.preprocess", "train_reflow.py": "stellaria_backend.fake.trainer"}


def engine_env(ctx: "AppContext", extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {"STELLARIA_FAKE_SPEED": str(ctx.options.fake_speed)}
    rmvpe = ctx.resources.path("rmvpe")
    if rmvpe is not None:
        env["DDSP_SVC_RMVPE_PATH"] = str(rmvpe)
    if extra:
        env.update(extra)
    return procs.child_env(env, pythonpath=[BACKEND_ROOT])


def engine_script(ctx: "AppContext", script: str, args: list[str]) -> tuple[list[str], Path]:
    """``python <engine>/<script> args`` (cwd = engine root), or the fake module in fake mode."""
    if ctx.options.fake_engine:
        return [ctx.python, "-m", FAKE_MODULES[script], *args], ctx.options.engine
    return [ctx.python, str(ctx.options.engine / script), *args], ctx.options.engine
