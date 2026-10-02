"""Host for ``runtime: "python"`` plugins. Standard library only, Python >= 3.8.

Run with the plugin's interpreter (never imported by the backend)::

    python host.py <job.json>

``job.json``: ``{"entry", "folder", "kind", "inputs", "output", "outdir", "workdir", "params",
"settings"}``. Imports the entry file by path and calls ``process(job)``.

Protocol: JSON lines on **stdout**, each prefixed with ``STELLARIA_MSG ``:
``{"progress": 0.5}``, ``{"log": "text"}``, finally ``{"result": {"output": path | null}}`` (exit 0)
or ``{"error": {"type", "message"}}`` (exit 1, traceback on stderr). Everything the plugin prints
(``print``, native libraries writing to fd 1) is redirected to stderr, so it shows up in the job
log as plain lines and can never be mistaken for a protocol message.
"""

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path

PREFIX = "STELLARIA_MSG "


class _Channel:
    def __init__(self):
        # keep a private copy of the original stdout for the protocol, then point fd 1 and
        # sys.stdout at stderr for the plugin
        sys.stdout.flush()
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8", errors="replace", newline="\n")
        try:
            os.dup2(2, 1)
        except OSError:
            pass
        sys.stdout = sys.stderr

    def send(self, obj):
        sys.stderr.flush()
        self._out.write(PREFIX + json.dumps(obj, ensure_ascii=False) + "\n")
        self._out.flush()


class Job:
    """What ``process(job)`` receives."""

    def __init__(self, spec, channel):
        self._channel = channel
        self.kind = spec.get("kind", "mix")
        inputs = spec.get("inputs") or {}
        if "input" in inputs:
            self.inputs = {"input": Path(inputs["input"]) if inputs["input"] else None}
        else:
            self.inputs = {
                "vocals": [Path(p) for p in inputs.get("vocals") or []],
                "backing": Path(inputs["backing"]) if inputs.get("backing") else None,
            }
        self.output = Path(spec["output"])
        self.outdir = Path(spec["outdir"])
        self.workdir = Path(spec.get("workdir") or os.getcwd())
        self.folder = Path(spec["folder"])
        self.params = dict(spec.get("params") or {})
        self.settings = dict(spec.get("settings") or {})

    def progress(self, value, text=None):
        try:
            value = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return
        self._channel.send({"progress": value})
        if text:
            self.log(text)

    def log(self, text):
        for line in str(text).splitlines() or [""]:
            self._channel.send({"log": line})

    def __repr__(self):
        return "<stellaria Job kind=%s output=%s>" % (self.kind, self.output)


def _load_entry(path, folder):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))  # sibling modules of the entry import as usual
    spec = importlib.util.spec_from_file_location("stellaria_plugin_entry", str(path))
    if spec is None or spec.loader is None:
        raise ImportError("cannot import %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv):
    if len(argv) < 2:
        sys.stderr.write("usage: host.py <job.json>\n")
        return 2
    # the host's own folder (the backend's plugins package) must not shadow the plugin's modules
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    with open(argv[1], "r", encoding="utf-8") as f:
        spec = json.load(f)
    channel = _Channel()
    try:
        folder = Path(spec["folder"])
        module = _load_entry(Path(spec["entry"]), folder)
        process = getattr(module, "process", None)
        if not callable(process):
            raise AttributeError("the entry file has no process(job) function")
        job = Job(spec, channel)
        result = process(job)
        output = None
        if isinstance(result, (str, os.PathLike)):
            output = str(Path(result).resolve())
        channel.send({"result": {"output": output}})
        return 0
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
        if code:
            channel.send({"error": {"type": "SystemExit", "message": "exit code %s" % code}})
        else:
            channel.send({"result": {"output": None}})
        return code
    except BaseException as exc:  # noqa: BLE001 - report everything, including KeyboardInterrupt
        traceback.print_exc()
        channel.send({"error": {"type": type(exc).__name__, "message": str(exc)[:2000]}})
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
