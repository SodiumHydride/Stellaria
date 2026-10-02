"""Check a Python interpreter: version, missing distributions, torch (if importable).

Runs the interpreter with a small embedded script (no console window, timeout, process-tree
kill on hang). The script only uses the standard library and works on old interpreters too, so
a wrong pick (Python 2, a non-Python program, a hanging wrapper) gives a clear error instead of a
crash. Requirements are compared by normalized distribution name via ``importlib.metadata``;
markers and version specifiers are honoured when ``packaging`` is importable there.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Mapping

from ..schemas import Msg
from ..util import procs
from .envs import interpreter_env, resolve_python_path
from .models import PythonCheck, TorchCheck

MARKER = "STELLARIA_CHECK "
TIMEOUT = 30.0
TORCH_TIMEOUT = 180.0  # a cold `import torch` on Windows can take a while

CHECK_SCRIPT = r'''
import json, re, sys
def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()
args = json.loads(sys.argv[1])
out = {"version": "%d.%d.%d" % tuple(sys.version_info[:3]), "executable": sys.executable, "missing": []}
Requirement = None
for modname in ("packaging.requirements", "pip._vendor.packaging.requirements"):
    try:
        Requirement = __import__(modname, fromlist=["Requirement"]).Requirement
        break
    except Exception:
        pass
versions = {}
try:
    try:
        from importlib import metadata as md
    except ImportError:
        import importlib_metadata as md
    for d in md.distributions():
        try:
            name = d.metadata["Name"]
        except Exception:
            name = None
        if name and norm(name) not in versions:
            versions[norm(name)] = d.version
except Exception:
    try:
        import pkg_resources
        for d in pkg_resources.working_set:
            versions.setdefault(norm(d.project_name), d.version)
    except Exception:
        pass
for raw in args.get("packages") or []:
    name, spec, applies = None, None, True
    if Requirement is not None:
        try:
            r = Requirement(raw)
            name, spec = r.name, r.specifier
            if r.marker is not None:
                applies = bool(r.marker.evaluate())
        except Exception:
            name = None
    if name is None:
        m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", raw)
        name = m.group(1) if m else raw.strip()
    if not applies:
        continue
    have = versions.get(norm(name))
    shown = raw.split(";")[0].strip()
    if have is None:
        out["missing"].append(shown)
    elif spec is not None and str(spec):
        try:
            if not spec.contains(have, prereleases=True):
                out["missing"].append(shown)
        except Exception:
            pass
if args.get("torch"):
    try:
        import importlib.util
        found = importlib.util.find_spec("torch") is not None
    except Exception:
        found = True
    if found:
        try:
            import torch
            cuda_ok = False
            try:
                cuda_ok = bool(torch.cuda.is_available())
            except Exception:
                pass
            out["torch"] = {"version": str(torch.__version__), "cuda": getattr(torch.version, "cuda", None),
                            "cudaAvailable": cuda_ok}
        except Exception as e:
            out["torchError"] = "%s: %s" % (type(e).__name__, e)
sys.stdout.write("STELLARIA_CHECK " + json.dumps(out) + "\n")
sys.stdout.flush()
'''


def _failed(executable: str, code: str, version: str | None = None, **params: object) -> PythonCheck:
    return PythonCheck(ok=False, executable=executable, version=version, error=Msg(code=code, params=params))


def _version_tuple(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in text.split(".")[:3])
    except ValueError:
        return (0,)


def run_check(path: str | Path, packages: list[str] | None = None, *, torch: bool = True,
              min_version: tuple[int, int] | None = None, env: Mapping[str, str] | None = None,
              timeout: float | None = None) -> PythonCheck:
    """``PythonCheck`` for ``path`` (an executable or an environment folder).

    Error codes (``check.error.code``, shared with the shell's ``check_python`` where they mean the
    same): ``python.not_found`` {path}, ``python.failed`` {path, reason} (could not run it / not a
    Python), ``python.check_failed`` {reason} (unreadable answer), ``python.timeout`` {seconds},
    ``python.too_old`` {version, min} (plugins with the python runtime need 3.8+)."""
    given = str(path)
    if not given.strip():
        return _failed(given, "python.not_found", path=given)
    exe = resolve_python_path(given)
    if not exe.is_file():
        return _failed(str(exe), "python.not_found", path=str(exe))
    args = json.dumps({"packages": list(packages or []), "torch": torch})
    limit = timeout if timeout is not None else (TORCH_TIMEOUT if torch else TIMEOUT)
    run_env = dict(env) if env is not None else interpreter_env(exe, custom=True)
    cwd = Path(tempfile.gettempdir())  # `-c` puts the cwd on sys.path: keep it neutral
    try:
        proc = procs.popen([str(exe), "-c", CHECK_SCRIPT, args], cwd=cwd, env=run_env, stderr=subprocess.PIPE)
    except OSError as exc:
        return _failed(str(exe), "python.failed", path=str(exe), reason=str(exc)[:300])
    started = time.monotonic()
    try:
        out, err = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        procs.kill_tree(proc.pid, timeout=3)
        try:
            proc.communicate(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        return _failed(str(exe), "python.timeout", seconds=round(time.monotonic() - started))
    text = out.decode("utf-8", errors="replace")
    line = next((ln for ln in reversed(text.splitlines()) if ln.startswith(MARKER)), None)
    if line is None:
        reason = err.decode("utf-8", errors="replace").strip().splitlines()[-1:] or [f"exit code {proc.returncode}"]
        return _failed(str(exe), "python.failed", path=str(exe), reason=reason[0][:300])
    try:
        data = json.loads(line[len(MARKER):])
    except ValueError:
        return _failed(str(exe), "python.check_failed", reason="unreadable answer: " + line[:200])
    version = str(data.get("version") or "")
    executable = str(data.get("executable") or exe)
    torch_info = None
    if isinstance(data.get("torch"), dict):
        t = data["torch"]
        torch_info = TorchCheck(version=str(t.get("version")), cuda=t.get("cuda"),
                                cuda_available=bool(t.get("cudaAvailable")))
    missing = [str(m) for m in data.get("missing") or []]
    error = None
    if min_version is not None and _version_tuple(version)[:2] < min_version:
        error = Msg(code="python.too_old", params={"version": version, "min": "%d.%d" % min_version})
    return PythonCheck(ok=error is None and not missing, executable=executable, version=version or None,
                       missing=missing, error=error, torch=torch_info)


def check_python(path: Path | str, packages: list[str] | None = None, **kwargs: object) -> dict:
    """``PythonCheck`` wire dict (the internal interface from creation-v2 §6)."""
    return run_check(path, packages, **kwargs).wire()  # type: ignore[arg-type]
