"""Short-lived worker that touches third-party plugins (VST3 / Audio Unit) through pedalboard.

Run by path with the app runtime (never imported by the backend), one plugin per process, so a
plugin that crashes or hangs only takes this process down::

    python worker.py probe
    python worker.py scan <path> [<pluginName>]   # one plugin; or the names of a shell bundle
    python worker.py inspect <job.json>           # {path, pluginName, state} → current values
    python worker.py editor <job.json>            # shows the plugin's window until it is closed

Answer: one line ``STELLARIA_VST {json}`` on stdout — ``{"ok": {...}}`` or
``{"error": {"code", "type", "message"}}``. Anything plugins print (stdout or native code
writing to fd 1) is moved to stderr so it can never be mistaken for the answer.
Needs only the standard library, numpy and pedalboard.
"""

import base64
import json
import os
import sys
import traceback

PREFIX = "STELLARIA_VST "
SR = 44100


class _Answer:
    def __init__(self):
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


def _fail(code, exc=None, message=None):
    return {"error": {"code": code, "type": type(exc).__name__ if exc else None,
                      "message": (message or str(exc) or type(exc).__name__)[:2000] if (exc or message) else ""}}


def _plugin_class(path):
    import pedalboard

    if path.lower().rstrip("/\\").endswith(".component"):
        cls = getattr(pedalboard, "AudioUnitPlugin", None)
        if cls is None:
            raise ImportError("Audio Units are only supported on macOS")
        return cls
    return pedalboard.VST3Plugin


def _load(path, plugin_name):
    import pedalboard

    return pedalboard.load_plugin(path, plugin_name=plugin_name or None, initialization_timeout=10.0)


def _display_for_raw(parameter, raw):
    """The displayed value at a raw (0..1) value. pedalboard's ranges are contiguous and touch at
    their ends, so the one starting last at or before ``raw`` is the right one."""
    best = None
    for (low, _high), value in parameter.ranges.items():
        if low <= raw and (best is None or low >= best[0]):
            best = (low, value)
    return best[1] if best is not None else None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def describe_params(plugin):
    out = []
    for pid, parameter in plugin.parameters.items():
        kind = "float" if parameter.type is float else "bool" if parameter.type is bool else "choice"
        default_raw = getattr(parameter, "default_raw_value", None)
        default = _display_for_raw(parameter, default_raw) if default_raw is not None else None
        entry = {"id": pid, "name": parameter.name or pid, "kind": kind, "min": None, "max": None,
                 "default": None, "unit": parameter.label or None, "choices": None}
        if kind == "float":
            entry.update(min=_number(parameter.min_value), max=_number(parameter.max_value), default=_number(default))
        elif kind == "bool":
            entry.update(default=bool(default) if default is not None else None)
        else:
            entry.update(choices=[str(v) for v in parameter.valid_values],
                         default=str(default) if default is not None else None)
        out.append(entry)
    return out


def current_values(plugin):
    values = {}
    for pid, parameter in plugin.parameters.items():
        try:
            value = getattr(plugin, pid)
        except Exception:  # noqa: BLE001 - a parameter that fails to read is left out
            continue
        if parameter.type is float:
            number = _number(value)
            if number is not None:
                values[pid] = number
        elif parameter.type is bool:
            values[pid] = bool(value)
        else:
            values[pid] = str(value)
    return values


def flush(plugin):
    """VST3 hands parameter changes to the audio processor inside process(): run a short block
    of silence so the state we read includes what was just set (by us or in the editor)."""
    import numpy as np

    try:
        plugin.process(np.zeros((2, 512), dtype=np.float32), SR, reset=False)
    except Exception:  # noqa: BLE001 - instruments / odd channel layouts: the state is still useful
        pass


def describe(plugin, plugin_name):
    is_instrument = bool(getattr(plugin, "is_instrument", False))
    is_effect = bool(getattr(plugin, "is_effect", not is_instrument))
    return {
        "name": str(getattr(plugin, "name", "") or plugin_name or ""),
        "vendor": str(getattr(plugin, "manufacturer_name", "") or "") or None,
        "version": str(getattr(plugin, "version", "") or "") or None,
        "category": "instrument" if is_instrument and not is_effect else "effect",
        "pluginName": plugin_name or None,
        "latency": int(getattr(plugin, "reported_latency_samples", 0) or 0),
        "params": describe_params(plugin),
    }


def restore(plugin, state):
    if state:
        plugin.raw_state = base64.b64decode(state)


def main(argv):
    here = os.path.dirname(os.path.abspath(__file__))  # the backend package must not shadow anything
    sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != here]
    answer = _Answer()
    if len(argv) < 2:
        answer.send(_fail("vst.bad_request", message="usage: worker.py probe|scan|inspect|editor ..."))
        return 2
    mode = argv[1]
    try:
        import pedalboard
    except Exception as exc:  # noqa: BLE001
        answer.send(_fail("vst.host_missing", exc))
        return 0
    try:
        if mode == "probe":
            formats = ["vst3"] + (["au"] if hasattr(pedalboard, "AudioUnitPlugin") else [])
            answer.send({"ok": {"version": getattr(pedalboard, "__version__", None), "formats": formats}})
            return 0
        if mode == "scan":
            path = argv[2]
            plugin_name = argv[3] if len(argv) > 3 and argv[3] else None
            if plugin_name is None:
                names = list(_plugin_class(path).get_plugin_names_for_file(path))
                if len(names) > 1:
                    answer.send({"ok": {"names": names}})
                    return 0
            plugin = _load(path, plugin_name)
            answer.send({"ok": {"plugins": [describe(plugin, plugin_name)]}})
            return 0
        with open(argv[2], "r", encoding="utf-8") as f:
            job = json.load(f)
        plugin = _load(job["path"], job.get("pluginName"))
        restore(plugin, job.get("state"))
        if mode == "editor":
            plugin.show_editor()
        flush(plugin)
        result = {"params": current_values(plugin)}
        if mode == "editor":
            result["state"] = base64.b64encode(plugin.raw_state).decode("ascii")
        answer.send({"ok": result})
        return 0
    except Exception as exc:  # noqa: BLE001 - every load problem becomes vst.load_failed
        traceback.print_exc()
        answer.send(_fail("vst.load_failed", exc))
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
