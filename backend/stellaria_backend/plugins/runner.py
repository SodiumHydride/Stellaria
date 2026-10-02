"""Running a plugin in a subprocess: ``run_plugin`` (creation-v2 §6).

Layout of one run (under ``<data>/tmp``, removed afterwards)::

    plugin-<id>-xxxx/
        in/       WAV copies of the inputs with readable names (outside the working folder, so a
                  script that clears ./input or ./temporary cannot delete them)
        out/      {outdir}; {output} = out/output.wav
        work/     the working directory for cwd 'temp' (cwd 'folder' runs in the plugin folder)
        params.json, job.json

The process gets no console window, UTF-8 I/O and its own process group/session, so cancelling
or a timeout kills the whole tree. stdout and stderr (one pipe for commands, so their order is
kept) go to the job log line by line;
``STELLARIA_PROGRESS <0..1>`` lines set progress; python-runtime plugins talk through the host's
``STELLARIA_MSG {json}`` lines.
"""

from __future__ import annotations

import json
import locale
import os
import queue
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Mapping

from ..errors import ApiError, JobCancelled
from ..util import procs
from ..util.fsutil import safe_filename, unique_path
from .manifest import PATH_TOKENS, command_tokens, expand_arg, tokens
from .models import Manifest, ParamSpec

if TYPE_CHECKING:
    from ..context import AppContext
    from ..jobs import JobContext

HOST = Path(__file__).with_name("host.py")
PROGRESS_RE = re.compile(r"^\s*STELLARIA_PROGRESS\s+([-+0-9.eE]+)\s*$")
MSG_PREFIX = "STELLARIA_MSG "
TAIL_LINES = 60
MAX_LINE = 4000
GRACE_AFTER_EXIT = 3.0
ID_LIKE = re.compile(r"^[a-z]{2,5}_[a-z0-9]{6,}$")


@dataclass
class _Outcome:
    code: int
    tail: list[str]
    output: str | None = None  # python runtime: path returned by process(job)
    error: dict[str, Any] | None = None  # python runtime: exception reported by the host


# ---- public ----------------------------------------------------------------------------
def run_plugin(ctx: "AppContext", jctx: "JobContext", plugin_id: str, *, vocals: list[Path],
               backing: Path | None, input: Path | None, output: Path, params: dict,
               names: Mapping[str | Path, str] | None = None,
               on_progress: Callable[[float], None] | None = None,
               require_enabled: bool = True) -> Path:
    """Run plugin ``plugin_id`` and write its result to ``output`` (WAV); returns ``output``.

    ``names`` optionally maps input paths to readable file stems for the copies the plugin
    sees (default: the file's own stem, or vocal/backing/input for id-like names).
    ``on_progress`` receives the plugin's progress (0..1) instead of ``jctx.progress`` — use it
    to fold several runs (variants) into one job's progress.
    Raises ``ApiError`` with ``plugin.*`` codes, or ``JobCancelled``."""
    jctx.check_cancelled()
    target = ctx.plugins.run_target(plugin_id, require_enabled=require_enabled)
    manifest = target.manifest
    vocals = [Path(v) for v in vocals or []]
    _check_inputs(manifest, vocals, backing, input)
    if on_progress is None:
        jctx.stage("stage.plugin", 0.0, plugin=target.name)
    report = on_progress or (lambda value: jctx.progress(value))

    run_dir = Path(tempfile.mkdtemp(prefix=f"plugin-{plugin_id}-", dir=_tmp_root(ctx)))
    try:
        in_dir, out_dir = run_dir / "in", run_dir / "out"
        in_dir.mkdir()
        out_dir.mkdir()
        if manifest.cwd == "temp":
            work = run_dir / "work"
            work.mkdir()
        else:
            work = target.folder
        staged = _stage_inputs(in_dir, vocals, Path(backing) if backing else None,
                               Path(input) if input else None, names or {})
        out_file = out_dir / "output.wav"
        merged = merge_params(manifest.params, params or {})
        inputs: dict[str, Any] = ({"input": str(staged["input"])} if manifest.kind == "effect" else
                                  {"vocals": [str(v) for v in staged["vocals"]],
                                   "backing": str(staged["backing"]) if staged["backing"] else None})
        params_json = run_dir / "params.json"
        params_json.write_text(json.dumps({
            "params": merged, "settings": target.settings, "inputs": inputs, "output": str(out_file),
            "outdir": str(out_dir), "plugin": {"id": target.id, "name": target.name, "folder": str(target.folder)},
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        env = dict(target.env)
        # compiled bytecode of the plugin's modules goes to the app's data, not the user's folder
        env.setdefault("PYTHONPYCACHEPREFIX", str(ctx.options.data / "plugins" / "pycache"))
        env.update({"STELLARIA_PLUGIN_ID": target.id, "STELLARIA_PLUGIN_FOLDER": str(target.folder),
                    "STELLARIA_OUTPUT": str(out_file), "STELLARIA_OUTDIR": str(out_dir),
                    "STELLARIA_PARAMS_JSON": str(params_json)})
        if isinstance(merged.get("seed"), (int, float)) and not isinstance(merged.get("seed"), bool):
            env["STELLARIA_SEED"] = str(int(merged["seed"]))  # the mix job passes its variant seed in params
        if manifest.runtime == "python":
            job_json = run_dir / "job.json"
            job_json.write_text(json.dumps({
                "entry": str(target.folder / (manifest.entry or "")), "folder": str(target.folder),
                "kind": manifest.kind, "inputs": inputs, "output": str(out_file), "outdir": str(out_dir),
                "workdir": str(work), "params": merged, "settings": target.settings,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            argv = [str(target.python), str(HOST), str(job_json)]
        else:
            values = {
                "python": str(target.python or ""), "folder": str(target.folder),
                "backing": str(staged["backing"] or ""), "vocal": str(staged["vocals"][0]) if staged["vocals"] else "",
                "input": str(staged["input"] or ""), "output": str(out_file), "outdir": str(out_dir),
                "params_json": str(params_json),
            }
            values.update({f"setting:{k}": _setting_text(v) for k, v in target.settings.items()})
            argv = expand_command(manifest, values, [str(v) for v in staged["vocals"]], target.folder, env)
        started = time.time()
        outcome = _execute(jctx, argv, cwd=work, env=env, timeout=manifest.timeout_sec, name=target.name,
                           report=report, python_host=manifest.runtime == "python",
                           python_first=target.python is not None and argv[:1] == [str(target.python)])
        if outcome.error is not None or outcome.code != 0:
            err = outcome.error or {}
            text = (f"{target.name} failed: {err.get('type')}: {err.get('message')}" if err
                    else f"{target.name} exited with code {outcome.code}")
            raise ApiError("plugin.failed", text, status=500, detail="\n".join(outcome.tail),
                           params={"exitCode": outcome.code})
        produced = collect_output(manifest, work, out_file, out_dir, outcome.output,
                                  since=started if manifest.cwd == "folder" else None)
        if produced is None:
            raise ApiError("plugin.no_output", f"{target.name} finished but wrote no audio", status=500,
                           detail="\n".join(outcome.tail),
                           params={"outputs": ", ".join(manifest.outputs) or "{output}"})
        jctx.log(f"[stellaria] result: {produced}")
        try:
            to_wav(produced, output)
        except ApiError as exc:
            raise ApiError("plugin.no_output", f"{target.name} wrote a file that cannot be read: {produced.name}",
                           status=500, detail=f"{exc.message}\n{exc.detail}",
                           params={"outputs": produced.name}) from exc
        report(1.0)
        return output
    finally:
        if os.environ.get("STELLARIA_KEEP_PLUGIN_RUNS") != "1":
            shutil.rmtree(run_dir, ignore_errors=True)


# ---- preparation -----------------------------------------------------------------------
def _tmp_root(ctx: "AppContext") -> Path:
    root = ctx.tmp_dir / "plugins"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _check_inputs(manifest: Manifest, vocals: list[Path], backing: Path | None, input: Path | None) -> None:
    if manifest.kind == "mix":
        if not vocals:
            raise ApiError("plugin.input_missing", "The mix needs at least one vocal", status=409,
                           params={"input": "vocals"})
        if len(vocals) > manifest.vocals_limit:
            raise ApiError("plugin.too_many_vocals", f"This plugin takes at most {manifest.vocals_limit} vocal(s)",
                           status=409, params={"max": manifest.vocals_limit})
        if backing is None and "backing" in command_tokens(manifest):
            raise ApiError("plugin.input_missing", "This plugin needs a backing track", status=409,
                           params={"input": "backing"})
    elif input is None:
        raise ApiError("plugin.input_missing", "This plugin needs an input", status=409, params={"input": "input"})


def _stem(path: Path, names: Mapping[str | Path, str], fallback: str) -> str:
    given = names.get(path) or names.get(str(path))
    if given:
        return safe_filename(Path(str(given)).stem if str(given).lower().endswith(".wav") else str(given), fallback)
    stem = path.stem
    return fallback if ID_LIKE.match(stem) else safe_filename(stem, fallback)


def _stage_inputs(in_dir: Path, vocals: list[Path], backing: Path | None, input: Path | None,
                  names: Mapping[str | Path, str]) -> dict[str, Any]:
    def stage(src: Path, stem: str) -> Path:
        if not src.is_file():
            raise ApiError("plugin.input_missing", f"Input file not found: {src}", status=409,
                           params={"input": src.name})
        dest = unique_path(in_dir / f"{stem}.wav")
        to_wav(src, dest)
        return dest

    staged: dict[str, Any] = {"vocals": [], "backing": None, "input": None}
    if backing is not None:
        staged["backing"] = stage(backing, _stem(backing, names, "backing"))
    for i, vocal in enumerate(vocals):
        staged["vocals"].append(stage(vocal, _stem(vocal, names, "vocal" if i == 0 else f"vocal {i + 1}")))
    if input is not None:
        staged["input"] = stage(input, _stem(input, names, "input"))
    return staged


def _coerce(spec: ParamSpec, value: Any) -> Any:
    try:
        if value is None:
            return None
        if spec.type == "number":
            number = float(value)
            if spec.min is not None:
                number = max(spec.min, number)
            if spec.max is not None:
                number = min(spec.max, number)
            return int(number) if float(number).is_integer() and isinstance(spec.default, int) else number
        if spec.type == "boolean":
            return value if isinstance(value, bool) else str(value).lower() in ("1", "true", "yes", "on")
        if spec.type == "choice":
            text = str(value)
            return text if text in {o.value for o in spec.options or []} else spec.default
        return str(value)
    except (TypeError, ValueError):
        return spec.default


def merge_params(specs: list[ParamSpec], given: dict[str, Any]) -> dict[str, Any]:
    """Manifest defaults, overridden by ``given`` (coerced and clamped); unknown keys pass through."""
    out = {spec.id: _coerce(spec, given.get(spec.id, spec.default)) for spec in specs}
    for key, value in given.items():
        out.setdefault(key, value)
    return out


def _setting_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_program(program: str, folder: Path, path_env: str | None) -> str | None:
    """Absolute program path: as given, relative to the plugin folder, or found on PATH."""
    p = Path(program)
    if p.is_absolute():
        return str(p) if p.is_file() else None
    if (folder / p).is_file():
        return str(folder / p)
    return shutil.which(program, path=path_env)


def expand_command(manifest: Manifest, values: dict[str, str], vocals: list[str], folder: Path,
                   env: Mapping[str, str]) -> list[str]:
    argv: list[str] = []
    for arg in manifest.command or []:
        if arg == "{vocals}":
            argv.extend(vocals)
            continue
        text = expand_arg(arg, values)
        first = tokens(arg)[:1]
        if first and arg.startswith("{" + first[0] + "}") and first[0] in PATH_TOKENS and len(text) > 0:
            text = os.path.normpath(text)  # '{folder}/x.py' → native separators (scripts split on '\\')
        argv.append(text)
    if argv and manifest.command and manifest.command[0] != "{python}":
        found = resolve_program(argv[0], folder, env.get("PATH"))
        if found:
            argv[0] = found
    return argv


# ---- execution -------------------------------------------------------------------------
def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        # programs that ignore PYTHONUTF8 (e.g. a Windows .exe) print in the locale encoding
        return raw.decode(locale.getpreferredencoding(False) or "utf-8", errors="replace")


def _pump(pipe: IO[bytes], tag: str, out: "queue.Queue[tuple[str, str | None]]") -> None:
    try:
        for raw in iter(pipe.readline, b""):
            text = _decode(raw).rstrip("\r\n")
            if "\r" in text:  # progress bars redraw with \r: keep what a terminal would show
                parts = [p for p in text.split("\r") if p.strip()]
                text = parts[-1] if parts else ""
            if not text.strip():
                continue
            if len(text) > MAX_LINE:
                text = text[:MAX_LINE] + " …"
            out.put((tag, text))
    except (OSError, ValueError):
        pass
    finally:
        out.put((tag, None))


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Terminate the plugin and everything it started (its own process group / session)."""
    if proc.poll() is None:  # once reaped, its pid may belong to someone else: only the group kill below
        procs.kill_tree(proc.pid, timeout=3.0)
    if not procs.IS_WINDOWS:
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # stragglers that left the tree but not the session
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _spawn(argv: list[str], cwd: Path, env: Mapping[str, str], merge_stderr: bool) -> subprocess.Popen:
    """``merge_stderr``: one pipe keeps the program's own ordering of stdout/stderr lines (command
    plugins); the python host needs a clean stdout for its protocol, so it gets two."""
    kwargs: dict[str, Any] = {}
    if procs.IS_WINDOWS:
        kwargs["creationflags"] = procs.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:
        kwargs["start_new_session"] = True
    return subprocess.Popen(argv, cwd=str(cwd), env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE, **kwargs)


def _execute(jctx: "JobContext", argv: list[str], *, cwd: Path, env: Mapping[str, str], timeout: float, name: str,
             report: Callable[[float], None], python_host: bool, python_first: bool = False) -> _Outcome:
    jctx.log(f"[stellaria] $ {subprocess.list2cmdline(argv)}")
    jctx.log(f"[stellaria] cwd: {cwd}")
    try:
        proc = _spawn(argv, cwd, env, merge_stderr=not python_host)
    except OSError as exc:
        if python_host or python_first:
            raise ApiError("plugin.python_missing", f"Could not start {argv[0]}", status=409, detail=str(exc),
                           params={"path": argv[0], "reason": "python.failed"}) from exc
        raise ApiError("plugin.failed", f"Could not start {name}: {exc}", status=500, detail=str(exc),
                       params={"exitCode": -1}) from exc
    lines: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
    pipes = [(proc.stdout, "out")] + ([(proc.stderr, "err")] if proc.stderr is not None else [])
    for pipe, tag in pipes:
        threading.Thread(target=_pump, args=(pipe, tag, lines), name=f"plugin-{tag}", daemon=True).start()
    tail: deque[str] = deque(maxlen=TAIL_LINES)
    outcome = _Outcome(code=0, tail=[])
    previous_handler = jctx.job.cancel_handler
    killed = threading.Event()

    def kill() -> None:
        if not killed.is_set():
            killed.set()
            kill_process_tree(proc)

    jctx.on_cancel(kill)
    deadline = time.monotonic() + timeout
    open_streams, exited_at, timed_out = len(pipes), None, False
    try:
        while True:
            batch: list[tuple[str, str | None]] = []
            try:
                batch.append(lines.get(timeout=0.1))
                while len(batch) < 500:
                    batch.append(lines.get_nowait())
            except queue.Empty:
                pass
            to_log: list[str] = []
            for tag, line in batch:
                if line is None:
                    open_streams -= 1
                    continue
                _handle_line(tag, line, python_host, outcome, to_log, report)
            if to_log:
                tail.extend(to_log)
                jctx.log_lines(to_log)
            now = time.monotonic()
            if proc.poll() is not None:
                exited_at = exited_at or now
                if open_streams <= 0:
                    break
                if now - exited_at > GRACE_AFTER_EXIT:
                    kill()  # something it started still holds the pipes
                    if now - exited_at > 2 * GRACE_AFTER_EXIT:
                        break
            elif now > deadline and not timed_out:
                timed_out = True
                jctx.log(f"[stellaria] timed out after {int(timeout)} s, stopping the plugin")
                kill()
            if jctx.cancelled:
                kill()
        proc.wait()
    finally:
        if proc.poll() is None:
            kill()
        jctx.job.cancel_handler = previous_handler
    outcome.code = proc.returncode
    outcome.tail = list(tail)
    if jctx.cancelled:
        raise JobCancelled()
    if timed_out:
        raise ApiError("plugin.timeout", f"{name} did not finish within {int(timeout)} s", status=504,
                       detail="\n".join(tail), params={"seconds": int(timeout)})
    return outcome


def _handle_line(tag: str, line: str, python_host: bool, outcome: _Outcome, to_log: list[str],
                 report: Callable[[float], None]) -> None:
    if python_host and tag == "out" and line.startswith(MSG_PREFIX):
        try:
            message = json.loads(line[len(MSG_PREFIX):])
        except ValueError:
            message = None
        if isinstance(message, dict):
            if "progress" in message:
                _report(report, message["progress"])
            elif "log" in message:
                to_log.append(str(message["log"]))
            elif "result" in message:
                outcome.output = (message.get("result") or {}).get("output")
            elif "error" in message:
                outcome.error = message.get("error") or {}
            return
    m = PROGRESS_RE.match(line)
    if m:
        _report(report, m.group(1))
        return
    to_log.append(line)


def _report(report: Callable[[float], None], value: Any) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if number == number:  # not NaN
        report(max(0.0, min(1.0, number)))


# ---- output ----------------------------------------------------------------------------
def _usable(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def collect_output(manifest: Manifest, work: Path, out_file: Path, out_dir: Path, reported: str | None,
                   since: float | None = None) -> Path | None:
    """The produced file: the path process(job) returned, ``{output}``, the first match of the
    ``outputs`` globs (relative to the working folder; in cwd 'folder' only files written during
    this run), or any audio file in ``{outdir}``."""
    from ..audio.io import AUDIO_EXTS

    if reported:
        path = Path(reported)
        path = path if path.is_absolute() else work / path
        if _usable(path):
            return path
    if _usable(out_file):
        return out_file
    for pattern in manifest.outputs:
        matches = sorted((p for p in work.glob(pattern) if _usable(p)), key=lambda p: str(p).lower())
        if since is not None:
            matches = [p for p in matches if p.stat().st_mtime >= since - 2.0]
        if matches:
            return matches[0]
    for path in sorted(out_dir.rglob("*"), key=lambda p: str(p).lower()):
        if path.suffix.lower() in AUDIO_EXTS and _usable(path):
            return path
    return None


def to_wav(src: Path, dst: Path) -> None:
    """Copy a WAV as is; convert anything else (FLAC, MP3, …) to WAV at its own sample rate.
    Raises ``audio.decode_failed`` for unreadable files."""
    import soundfile as sf

    from ..audio.io import SOUNDFILE_EXTS, decode, probe

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part.wav")
    try:
        if src.suffix.lower() == ".wav":
            try:
                info = sf.info(str(src))
                if info.frames > 0:
                    shutil.copyfile(src, tmp)
                    tmp.replace(dst)
                    return
            except Exception:  # noqa: BLE001 - odd WAV flavours: convert below
                pass
        data, sr, subtype = None, 0, "PCM_24"
        if src.suffix.lower() in SOUNDFILE_EXTS:
            try:
                info = sf.info(str(src))
                data, sr = sf.read(str(src), dtype="float32", always_2d=True)
                subtype = "FLOAT" if info.subtype in ("FLOAT", "DOUBLE") else "PCM_24"
            except Exception:  # noqa: BLE001 - try ffmpeg
                data = None
        if data is None:
            sr = probe(src).sample_rate
            data = decode(src, sr=sr, mono=False)
        if len(data) == 0:
            raise ApiError("audio.decode_failed", f"'{src.name}' contains no audio", status=422)
        sf.write(str(tmp), data, sr, subtype=subtype, format="WAV")
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
