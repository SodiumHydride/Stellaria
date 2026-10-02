"""Plugin manifests: loading, validation (with readable ``plugin.bad_manifest`` reasons), command
tokens, wizard manifests for scripts without one, and hints for the wizard.

Manifests live either in the plugin folder (``stellaria-plugin.json``) or, when created by the
wizard, in ``<data>/plugins/<id>.json`` (the manifest plus ``"folder"``).
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from pydantic import ValidationError

from ..errors import ApiError
from .models import Manifest, ParamSpec, WizardSpec

MANIFEST_FILE = "stellaria-plugin.json"
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
PARAM_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")
RESERVED_IDS = {"builtin", "wizard", "envs", "testdata", "tests", "registry"}
TOKEN_RE = re.compile(r"\{\{|\}\}|\{([^{}]*)\}")

SIMPLE_TOKENS = {"python", "folder", "backing", "vocal", "vocals", "input", "output", "outdir", "params_json"}
MIX_ONLY = {"backing", "vocal", "vocals"}
EFFECT_ONLY = {"input"}
OUTPUT_TOKENS = {"output", "outdir"}
PATH_TOKENS = {"python", "folder", "backing", "vocal", "input", "output", "outdir", "params_json"}

SCRIPT_EXTS = (".py", ".pyw", ".exe", ".bat", ".cmd", ".sh")
DEFAULT_OUTPUT_GLOBS = ["output/*.flac", "output/*.wav", "output/*.mp3"]
MAX_TIMEOUT = 24 * 3600.0

# import name → distribution name, where they differ (for the wizard's requirement guess)
IMPORT_TO_DIST = {
    "cv2": "opencv-python", "PIL": "pillow", "sklearn": "scikit-learn", "yaml": "PyYAML",
    "skimage": "scikit-image", "bs4": "beautifulsoup4", "Crypto": "pycryptodome",
    "dateutil": "python-dateutil", "parselmouth": "praat-parselmouth", "win32api": "pywin32",
    "win32con": "pywin32", "win32gui": "pywin32", "pythoncom": "pywin32", "pywintypes": "pywin32",
    "attr": "attrs", "pkg_resources": "setuptools", "fitz": "PyMuPDF", "serial": "pyserial",
    "dotenv": "python-dotenv", "jwt": "PyJWT", "zmq": "pyzmq", "Levenshtein": "python-Levenshtein",
    "pyloudnorm": "pyloudnorm", "noisereduce": "noisereduce", "audiomentations": "audiomentations",
}


def bad_manifest(reason: str, detail: str = "") -> ApiError:
    return ApiError("plugin.bad_manifest", f"Invalid plugin manifest: {reason}", status=422,
                    detail=detail or reason, params={"reason": reason})


# ---- tokens ----------------------------------------------------------------------------
def tokens(arg: str) -> list[str]:
    """Token names used in one command argument (``{{``/``}}`` are literal braces)."""
    return [m.group(1) for m in TOKEN_RE.finditer(arg) if m.group(1) is not None]


def command_tokens(manifest: Manifest) -> set[str]:
    return {t for arg in manifest.command or [] for t in tokens(arg)}


def uses_python(manifest: Manifest) -> bool:
    return manifest.runtime == "python" or "python" in command_tokens(manifest)


def setting_tokens(manifest: Manifest) -> set[str]:
    return {t.split(":", 1)[1] for t in command_tokens(manifest) if t.startswith("setting:")}


def _check_token(token: str, arg: str, index: int, manifest: Manifest) -> None:
    where = f"command[{index}]"
    if token.startswith("setting:"):
        sid = token.split(":", 1)[1]
        if sid not in {s.id for s in manifest.settings}:
            raise bad_manifest(f"{where} uses {{setting:{sid}}} but there is no setting '{sid}'")
        return
    if token not in SIMPLE_TOKENS:
        raise bad_manifest(f"unknown token {{{token}}} in {where}")
    if token == "vocals" and arg != "{vocals}":
        raise bad_manifest(f"{{vocals}} must be a whole argument ({where})")
    if manifest.kind == "mix" and token in EFFECT_ONLY:
        raise bad_manifest(f"{{{token}}} is only for 'effect' plugins ({where})")
    if manifest.kind == "effect" and token in MIX_ONLY:
        raise bad_manifest(f"{{{token}}} is only for 'mix' plugins ({where})")


def expand_arg(arg: str, values: dict[str, str]) -> str:
    def repl(m: re.Match[str]) -> str:
        if m.group(0) == "{{":
            return "{"
        if m.group(0) == "}}":
            return "}"
        return values[m.group(1)]

    return TOKEN_RE.sub(repl, arg)


# ---- validation ------------------------------------------------------------------------
def _relative_ok(text: str) -> bool:
    if not text or text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", text):
        return False
    return ".." not in PurePosixPath(text.replace("\\", "/")).parts


def _check_specs(specs: list[ParamSpec], what: str) -> None:
    seen: set[str] = set()
    for spec in specs:
        if not PARAM_ID_RE.match(spec.id):
            raise bad_manifest(f"{what} id '{spec.id}' is not a valid identifier")
        if spec.id in seen:
            raise bad_manifest(f"duplicate {what} id '{spec.id}'")
        seen.add(spec.id)
        d = spec.default
        if spec.type == "number" and d is not None and (isinstance(d, bool) or not isinstance(d, (int, float))):
            raise bad_manifest(f"{what} '{spec.id}': default must be a number")
        if spec.type == "boolean" and d is not None and not isinstance(d, bool):
            raise bad_manifest(f"{what} '{spec.id}': default must be true or false")
        if spec.type == "choice":
            if not spec.options:
                raise bad_manifest(f"{what} '{spec.id}': a choice needs options")
            if d is not None and str(d) not in {o.value for o in spec.options}:
                raise bad_manifest(f"{what} '{spec.id}': default is not one of the options")
        if spec.type in ("text", "folder", "file") and d is not None and not isinstance(d, str):
            raise bad_manifest(f"{what} '{spec.id}': default must be text")
        if spec.min is not None and spec.max is not None and spec.min > spec.max:
            raise bad_manifest(f"{what} '{spec.id}': min is greater than max")


def validate(manifest: Manifest, folder: Path | None = None) -> Manifest:
    """Semantic checks on top of the pydantic model; raises ``plugin.bad_manifest``."""
    m = manifest
    if not ID_RE.match(m.id) or m.id in RESERVED_IDS:
        raise bad_manifest(f"id '{m.id}' must be 1-64 lowercase letters, digits, '-' or '_' "
                           f"(and not one of {', '.join(sorted(RESERVED_IDS))})")
    if not m.name.strip():
        raise bad_manifest("name is empty")
    _check_specs(m.params, "param")
    _check_specs(m.settings, "setting")
    if m.max_vocals is not None and not 1 <= m.max_vocals <= 16:
        raise bad_manifest("maxVocals must be between 1 and 16")
    if not 1 <= m.timeout_sec <= MAX_TIMEOUT:
        raise bad_manifest(f"timeoutSec must be between 1 and {int(MAX_TIMEOUT)}")
    for req in m.requirements:
        if not isinstance(req, str) or not req.strip() or req.strip().startswith("-") or "\n" in req:
            raise bad_manifest(f"requirement '{req}' is not a package name")
    for pattern in m.outputs:
        if not _relative_ok(pattern):
            raise bad_manifest(f"output pattern '{pattern}' must be relative to the working folder")
    if m.runtime == "command":
        if not m.command or not all(isinstance(a, str) and a for a in m.command):
            raise bad_manifest("runtime 'command' needs a non-empty command list")
        for i, arg in enumerate(m.command):
            for token in tokens(arg):
                _check_token(token, arg, i, m)
        toks = command_tokens(m)
        if m.kind == "mix" and not toks & {"vocal", "vocals", "params_json"}:
            raise bad_manifest("a 'mix' command must pass the vocal ({vocal} or {vocals})")
        if m.kind == "effect" and not toks & {"input", "params_json"}:
            raise bad_manifest("an 'effect' command must pass the audio ({input})")
        if not toks & OUTPUT_TOKENS and not m.outputs:
            raise bad_manifest("no way to find the result: use {output} or {outdir} in the command, or list outputs")
        script = _script_arg(m.command)
        if folder is not None and script is not None and not (folder / script).is_file():
            raise bad_manifest(f"file not found in the plugin folder: {script}")
    else:
        if not m.entry:
            raise bad_manifest("runtime 'python' needs an entry file with process(job)")
        if not _relative_ok(m.entry):
            raise bad_manifest(f"entry '{m.entry}' must be relative to the plugin folder")
        if folder is not None and not (folder / m.entry).is_file():
            raise bad_manifest(f"entry file not found: {m.entry}")
    return m


def _script_arg(command: list[str]) -> str | None:
    """The script/program the command starts, when written as ``{folder}/<path>``."""
    first = command[1] if command[0] == "{python}" and len(command) > 1 else command[0]
    if first.startswith("{folder}/") and len(tokens(first)) == 1:
        return first[len("{folder}/"):]
    return None


def _parse(data: Any, where: str, folder: Path | None) -> Manifest:
    if not isinstance(data, dict):
        raise bad_manifest(f"{where} must contain a JSON object")
    try:
        manifest = Manifest.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first.get("loc", ())) or "manifest"
        raise bad_manifest(f"{loc}: {first.get('msg', 'invalid')}", detail=str(exc)) from exc
    return validate(manifest, folder)


def read_json_file(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise bad_manifest(f"{path.name} is not valid JSON ({exc})") from exc


def load_folder_manifest(folder: Path) -> Manifest:
    """``<folder>/stellaria-plugin.json``; FileNotFoundError when the folder has none."""
    return _parse(read_json_file(folder / MANIFEST_FILE), MANIFEST_FILE, folder)


def load_wizard_manifest(path: Path) -> tuple[Manifest, Path]:
    data = read_json_file(path)
    folder = Path(str((data or {}).get("folder") or "")) if isinstance(data, dict) else Path()
    return _parse(data, path.name, folder if str(folder) not in ("", ".") else None), folder


def parse_manifest(data: dict[str, Any], folder: Path | None) -> Manifest:
    return _parse(data, "manifest", folder)


# ---- wizard ----------------------------------------------------------------------------
def _posix_relative(entry: str, folder: Path) -> str:
    text = entry.strip().replace("\\", "/")
    win = PureWindowsPath(entry)
    if Path(entry).is_absolute() or win.is_absolute():
        try:
            text = Path(entry).resolve().relative_to(folder.resolve()).as_posix()
        except (ValueError, OSError) as exc:
            raise bad_manifest("the entry file must be inside the plugin folder") from exc
    while text.startswith("./"):
        text = text[2:]
    return text


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48].strip("-")


def wizard_manifest(spec: WizardSpec, pid: str) -> Manifest:
    folder = Path(spec.folder)
    entry = _posix_relative(spec.entry, folder)
    if not entry:
        raise bad_manifest("entry is empty")
    data: dict[str, Any] = {
        "id": pid, "name": spec.name.strip(), "version": "1.0", "kind": spec.kind, "runtime": spec.runtime,
        "cwd": spec.cwd, "requirements": [r.strip() for r in spec.requirements if r.strip()],
        "timeoutSec": spec.timeout_sec or 1200.0, "maxVocals": spec.max_vocals,
    }
    if spec.runtime == "python":
        data["entry"] = entry
        data["outputs"] = list(spec.outputs or [])
    else:
        if spec.args is None:
            args = ["{backing}", "{vocal}"] if spec.kind == "mix" else ["{input}", "{output}"]
        else:
            args = [a for a in spec.args if a != ""]
        target = "{folder}/" + entry
        prefix = ["{python}", target] if Path(entry).suffix.lower() in (".py", ".pyw") else [target]
        data["command"] = prefix + args
        uses_output = any(set(tokens(a)) & OUTPUT_TOKENS for a in args)
        data["outputs"] = list(spec.outputs) if spec.outputs is not None else ([] if uses_output else list(DEFAULT_OUTPUT_GLOBS))
    return parse_manifest(data, folder)


def script_candidates(folder: Path) -> list[str]:
    """Script files in the folder for the wizard: ``.py`` with a ``__main__`` guard first, then
    other ``.py``, then programs/batch files."""
    found: list[tuple[int, str]] = []
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return []
    for path in entries:
        if not path.is_file() or path.suffix.lower() not in SCRIPT_EXTS or path.name.startswith("."):
            continue
        rank = 3
        if path.suffix.lower() in (".py", ".pyw"):
            rank = 1
            try:
                if re.search(r"^if\s+__name__\s*==", path.read_text(encoding="utf-8", errors="replace"), re.M):
                    rank = 0
            except OSError:
                pass
            if path.name in ("setup.py", "setting.py", "settings.py", "config.py"):
                rank = 2
        found.append((rank, path.name))
    return [name for _, name in sorted(found, key=lambda x: (x[0], x[1].lower()))]


def guess_requirements(folder: Path) -> list[str]:
    """Third-party distributions imported by the folder's top-level ``.py`` files (a hint)."""
    local = {p.stem for p in folder.glob("*.py")} | {p.name for p in folder.iterdir() if p.is_dir()} \
        if folder.is_dir() else set()
    stdlib = set(getattr(sys, "stdlib_module_names", ())) | {"__future__"}
    found: dict[str, None] = {}
    for path in sorted(folder.glob("*.py")) if folder.is_dir() else []:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            for name in names:
                root = name.split(".", 1)[0]
                if root in stdlib or root in local or root in ("google",):
                    continue
                found.setdefault(IMPORT_TO_DIST.get(root, root), None)
    return sorted(found, key=str.lower)
