"""Plugin interpreters: per-plugin environments derived from the app runtime, the environment
variables plugin processes get, and requirement installs.

``builtin`` environments live in ``<data>/plugins/envs/<id>``. They are venvs of the app's own
interpreter (``sys.executable``) with ``--system-site-packages``. Because the app runtime is
itself a venv, a venv made from it does not see the runtime's packages (venvs always point at
the *base* interpreter), so a ``_stellaria_runtime.pth`` file adds the runtime's site-packages
*after* the plugin's own: numpy/scipy/librosa/torch come from the runtime, anything the plugin
installs wins.

Installing into a builtin env resolves the plugin's requirements with ``uv pip compile`` against
the runtime's installed versions as constraints (falling back to pinning only torch/numpy when
that conflicts), then installs only what the runtime does not already provide (``--no-deps``),
so nothing large is downloaded twice. Mirrors and their order are the add-on installer's.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from collections import deque
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from ..errors import ApiError
from ..util import procs
from ..util.fsutil import remove_tree
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso

if TYPE_CHECKING:
    from ..context import AppContext
    from ..jobs import JobContext

IS_WINDOWS = procs.IS_WINDOWS
ENV_MARKER = "stellaria-env.json"
BRIDGE_PTH = "_stellaria_runtime.pth"
# Written by the shell in its layer over the user's own Python (desktop/src-tauri/src/layer.rs)
LAYER_MARKER = "stellaria-layer.json"
CORE_PINS = ("torch", "torchaudio", "numpy")

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(req: str) -> str:
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", req)
    return m.group(1) if m else req.strip()


# ---- interpreters ----------------------------------------------------------------------
def envs_root(data: Path) -> Path:
    return data / "plugins" / "envs"


def env_dir(data: Path, pid: str) -> Path:
    return envs_root(data) / pid


def venv_python(env: Path) -> Path:
    return env / "Scripts" / "python.exe" if IS_WINDOWS else env / "bin" / "python"


def resolve_python_path(path: str | os.PathLike[str]) -> Path:
    """The python executable for what the user picked: the executable itself, or an
    environment folder (venv, conda env, Python install folder)."""
    p = Path(path).expanduser()
    if p.is_dir():
        names = ("python.exe", "Scripts/python.exe", "bin/python.exe") if IS_WINDOWS else \
            ("bin/python3", "bin/python", "python3", "python")
        for name in names:
            if (p / name).is_file():
                return p / name
    return p


def uv_binary(ctx: "AppContext") -> Path | None:
    if ctx.options.uv:
        return Path(ctx.options.uv)
    found = shutil.which("uv")
    return Path(found) if found else None


def interpreter_env(python: Path | None, *, custom: bool, venv: Path | None = None,
                    extra: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for a plugin process (and for checking its interpreter): UTF-8, unbuffered,
    the interpreter's folders first on PATH, none of the backend's own Python variables."""
    env = procs.child_env(extra)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
        env.pop(key, None)
    dirs: list[Path] = []
    if python is not None:
        bindir = python.parent
        dirs.append(bindir)
        if IS_WINDOWS:  # conda environments need their DLL folders on PATH (what `conda activate` does)
            dirs += [bindir / s for s in ("Library/mingw-w64/bin", "Library/usr/bin", "Library/bin", "Scripts")
                     if (bindir / s).is_dir()]
    if custom:
        # the user's own interpreter should behave as it does in their terminal
        env.pop("PYTHONNOUSERSITE", None)
        env.pop("VIRTUAL_ENV", None)
        if python is not None:
            root = python.parent.parent if python.parent.name in ("bin", "Scripts") else python.parent
            if (root / "pyvenv.cfg").is_file():
                env["VIRTUAL_ENV"] = str(root)
            if (root / "conda-meta").is_dir():
                env["CONDA_PREFIX"] = str(root)
    else:
        env["PYTHONNOUSERSITE"] = "1"
        if venv is not None:
            env["VIRTUAL_ENV"] = str(venv)
    if dirs:
        current = env.get("PATH", "")
        env["PATH"] = os.pathsep.join([str(d) for d in dirs] + ([current] if current else []))
    return env


# ---- the shell's layer over the user's Python ------------------------------------------
def layer_marker() -> dict | None:
    """The shell's marker when the backend runs in its layer over the user's own Python: a venv
    holding only what their environment lacks, which reaches their packages through a ``.pth``
    file. Everything installed from here belongs in the layer, never in their environment."""
    marker = read_json(Path(sys.prefix) / LAYER_MARKER, None)
    return marker if isinstance(marker, dict) else None


def in_shadow_layer() -> bool:
    """The layer over an embeddable Python (a copy of its executable with our own ``._pth``):
    venvs made from it would ignore their own folders, so plugins cannot get environments."""
    return (layer_marker() or {}).get("kind") == "shadow"


def layer_sites() -> list[str]:
    """The user's site-packages behind the layer (none outside one)."""
    marker = layer_marker() or {}
    return [s for s in marker.get("sites") or [] if isinstance(s, str) and os.path.isdir(s)]


# ---- builtin environments --------------------------------------------------------------
def runtime_site_dirs() -> list[str]:
    """site-packages of the app runtime (the interpreter running the backend) — in the shell's
    layer, its own and then the user's behind it."""
    out: list[str] = []
    paths = sysconfig.get_paths()
    for key in ("purelib", "platlib"):
        p = paths.get(key)
        if p and os.path.isdir(p) and p not in out:
            out.append(p)
    for p in layer_sites():
        if p not in out:
            out.append(p)
    return out


def _env_site_dirs(env: Path) -> list[Path]:
    if IS_WINDOWS:
        return [env / "Lib" / "site-packages"]
    return sorted(env.glob("lib/python*/site-packages"))


def _write_bridge(env: Path) -> None:
    # ascii() keeps the file ASCII-only: .pth files are read with the locale encoding
    text = "".join(f"import site; site.addsitedir({ascii(d)})\n" for d in runtime_site_dirs())
    for site_dir in _env_site_dirs(env):
        target = site_dir / BRIDGE_PTH
        try:
            if not target.is_file() or target.read_text(encoding="ascii", errors="replace") != text:
                target.write_text(text, encoding="ascii")
        except OSError:
            pass


def env_ready(data: Path, pid: str) -> bool:
    env = env_dir(data, pid)
    marker = read_json(env / ENV_MARKER, None) if env.is_dir() else None
    return venv_python(env).exists() and isinstance(marker, dict) and _marker_matches(marker)


def _wanted_marker() -> dict[str, str | None]:
    # in the shell's layer, sys.executable stays the same when the user picks another Python
    layer = (layer_marker() or {}).get("base")
    return {"base": sys.executable, "version": "%d.%d" % sys.version_info[:2], "layer": layer}


def _marker_matches(marker: dict) -> bool:
    return all(marker.get(k) == v for k, v in _wanted_marker().items())


def ensure_env(ctx: "AppContext", pid: str, log: Callable[[str], None] | None = None) -> Path:
    """Create (or re-create, after an app runtime change) the plugin's environment; returns
    its python. Raises ``plugin.env_failed``."""
    data = ctx.options.data
    env = env_dir(data, pid)
    py = venv_python(env)
    if in_shadow_layer():
        raise ApiError("plugin.env_unsupported", "Plugins with their own environment need Stellaria's runtime "
                       "(your Python is an embeddable one)", status=409)
    with _lock(f"env:{pid}"):
        if env_ready(data, pid):
            _write_bridge(env)
            return py
        if env.exists():
            try:
                remove_tree(env)
            except OSError as exc:
                raise ApiError("plugin.env_failed", "Could not replace the plugin's Python environment "
                               "(is the plugin still running?)", status=409, detail=str(exc)) from exc
        env.parent.mkdir(parents=True, exist_ok=True)
        uv = uv_binary(ctx)
        if uv is not None:
            cmd = [str(uv), "venv", "--python", sys.executable, "--system-site-packages", str(env)]
        else:
            cmd = [sys.executable, "-m", "venv", "--system-site-packages", "--without-pip", str(env)]
        if log:
            log("$ " + subprocess.list2cmdline(cmd))
        try:
            result = procs.run(cmd, env=procs.child_env(), timeout=300)
            output = (result.stdout + result.stderr).decode("utf-8", errors="replace")
            code = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            output, code = str(exc), -1
        if log and output.strip():
            for line in output.strip().splitlines():
                log(line)
        if code != 0 or not py.exists():
            shutil.rmtree(env, ignore_errors=True)
            raise ApiError("plugin.env_failed", "Could not create the plugin's Python environment", status=500,
                           detail=output[-4000:])
        _write_bridge(env)
        write_json(env / ENV_MARKER, {**_wanted_marker(), "createdAt": now_iso(), "tool": "uv" if uv else "venv",
                                      "runtimeSite": runtime_site_dirs()})
        return py


def remove_env(data: Path, pid: str) -> None:
    with _lock(f"env:{pid}"):
        shutil.rmtree(env_dir(data, pid), ignore_errors=True)


# ---- installs --------------------------------------------------------------------------
def _public(version: str) -> str:
    return version.split("+", 1)[0]


def runtime_versions() -> dict[str, str]:
    """normalized distribution name → version, for the interpreter running the backend."""
    out: dict[str, str] = {}
    for dist in metadata.distributions():
        try:
            name = dist.metadata["Name"]
        except Exception:  # noqa: BLE001 - broken metadata
            continue
        if name:
            out.setdefault(normalize_name(name), dist.version)
    return out


def _pins(versions: dict[str, str], full: bool) -> list[str]:
    names = sorted(versions) if full else [n for n in CORE_PINS if n in versions]
    return [f"{n}=={_public(versions[n])}" for n in names if versions[n]]


def _requirement_lines(text: str) -> list[str]:
    lines, current = [], ""
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip() if not raw.lstrip().startswith("#") else ""
        if line.endswith("\\"):
            current += line[:-1] + " "
            continue
        line = (current + line).strip()
        current = ""
        if line and not line.startswith(("#", "-")):
            lines.append(line)
    return lines


def provided_by_runtime(line: str, versions: dict[str, str]) -> bool:
    """True when a resolved ``name==version`` line is already installed in the app runtime."""
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*([^\s;]+)", line)
    if not m:
        return False
    have = versions.get(normalize_name(m.group(1)))
    if have is None:
        return False
    try:
        from packaging.version import Version

        return Version(_public(have)) == Version(_public(m.group(2)))
    except Exception:  # noqa: BLE001 - packaging missing or odd version strings
        return _public(have) == _public(m.group(2))


def _run_logged(jctx: "JobContext", cmd: list[str], env: dict[str, str], tail: deque, cwd: Path | None = None) -> int:
    jctx.log("$ " + subprocess.list2cmdline(cmd))
    try:
        proc = procs.popen(cmd, env=env, cwd=cwd)
    except OSError as exc:
        jctx.log(f"could not start: {exc}")
        tail.append(str(exc))
        return -1
    jctx.on_cancel(lambda p=proc: procs.kill_tree(p.pid))
    try:
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip()
            if line:
                jctx.log(line)
                tail.append(line)
        proc.wait()
    finally:
        jctx.on_cancel(None)
    jctx.check_cancelled()
    return proc.returncode


def _mirrors(ctx: "AppContext") -> list[str | None]:
    """Index URLs in the add-on installer's order (region preference / measured latency)."""
    try:
        urls: list[str | None] = [s.url for s in ctx.resources.ordered(ctx.resources.manifest.pypi)]
    except Exception:  # noqa: BLE001 - no manifest mirrors → the installer's default index
        urls = []
    return urls or [None]


def _install_failed(errors: list[str], tail: deque) -> ApiError:
    return ApiError("plugin.install_failed", "Installing the plugin's requirements failed", status=502,
                    detail="\n".join(errors + list(tail)[-60:]))


def install_into_env(ctx: "AppContext", jctx: "JobContext", python: Path, venv: Path,
                     requirements: list[str]) -> list[str]:
    """Install ``requirements`` into a builtin plugin env; returns the installed lines."""
    if not requirements:
        return []
    uv = uv_binary(ctx)
    env = interpreter_env(python, custom=False, venv=venv)
    work = Path(tempfile.mkdtemp(prefix="plugin-install-", dir=ctx.tmp_dir))
    tail: deque = deque(maxlen=200)
    errors: list[str] = []
    try:
        versions = runtime_versions()
        req_in = work / "requirements.in"
        req_in.write_text("\n".join(requirements) + "\n", encoding="utf-8")
        full, core = work / "runtime-full.txt", work / "runtime-core.txt"
        full.write_text("\n".join(_pins(versions, True)) + "\n", encoding="utf-8")
        core.write_text("\n".join(_pins(versions, False)) + "\n", encoding="utf-8")
        if uv is None:
            return _pip_install(ctx, jctx, python, env, requirements, ["--constraint", core.name], tail, cwd=work)
        # uv splits --constraint values on spaces: run in the work folder with relative file names
        for url in _mirrors(ctx):
            index = ["--index-url", url] if url else []
            resolved = work / "resolved.txt"
            resolved.unlink(missing_ok=True)
            for pins in (full, core):
                code = _run_logged(jctx, [str(uv), "pip", "compile", req_in.name, "--python", str(python),
                                          "--constraint", pins.name, "--output-file", resolved.name,
                                          "--no-header", "--no-annotate", *index], env, tail, cwd=work)
                if code == 0:
                    break
                if pins is full:
                    jctx.log("resolving against the app runtime's exact versions failed; "
                             "retrying with only torch/numpy pinned")
            else:
                errors.append(f"{url or 'default index'}: could not resolve the requirements")
                continue
            lines = _requirement_lines(resolved.read_text(encoding="utf-8"))
            todo = [line for line in lines if not provided_by_runtime(line, versions)]
            if not todo:
                jctx.log("everything is already provided by the app runtime")
                return []
            todo_file = work / "install.txt"
            todo_file.write_text("\n".join(todo) + "\n", encoding="utf-8")
            code = _run_logged(jctx, [str(uv), "pip", "install", "--python", str(python), "--no-deps",
                                      "-r", todo_file.name, *index], env, tail, cwd=work)
            if code == 0:
                return todo
            errors.append(f"{url or 'default index'}: exit code {code}")
        raise _install_failed(errors, tail)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def install_into_custom(ctx: "AppContext", jctx: "JobContext", python: Path, requirements: list[str]) -> list[str]:
    """Install into the user's own interpreter (only on explicit request)."""
    if not requirements:
        return []
    env = interpreter_env(python, custom=True)
    tail: deque = deque(maxlen=200)
    uv = uv_binary(ctx)
    if uv is None:
        return _pip_install(ctx, jctx, python, env, requirements, [], tail)
    errors: list[str] = []
    for url in _mirrors(ctx):
        index = ["--index-url", url] if url else []
        code = _run_logged(jctx, [str(uv), "pip", "install", "--python", str(python), *requirements, *index], env, tail)
        if code == 0:
            return list(requirements)
        errors.append(f"{url or 'default index'}: exit code {code}")
    raise _install_failed(errors, tail)


def _pip_install(ctx: "AppContext", jctx: "JobContext", python: Path, env: dict[str, str], requirements: list[str],
                 extra: list[str], tail: deque, cwd: Path | None = None) -> list[str]:
    """Fallback without uv: the interpreter's own pip (bootstrapped with ensurepip if needed)."""
    if _run_logged(jctx, [str(python), "-m", "pip", "--version"], env, tail) != 0:
        _run_logged(jctx, [str(python), "-m", "ensurepip", "--default-pip"], env, tail)
    errors: list[str] = []
    for url in _mirrors(ctx):
        index = ["--index-url", url] if url else []
        code = _run_logged(jctx, [str(python), "-m", "pip", "install", "--disable-pip-version-check",
                                  *requirements, *extra, *index], env, tail, cwd=cwd)
        if code == 0:
            return list(requirements)
        errors.append(f"{url or 'default index'}: exit code {code}")
    raise _install_failed(errors, tail)
