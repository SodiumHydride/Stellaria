"""Stand-in for ``stellaria_backend/vst/worker.py`` in tests (no pedalboard needed).

A fake plugin is a ``*.vst3`` file (or a bundle folder with ``fake.json``) holding JSON:
``{"behavior": "ok" | "crash" | "hang" | "error", "plugins": [describe() dicts], "names": [...],
"perName": {name: behavior}, "editor": "ok" | "hang"}``. Every load appends a line to
``$FAKE_VST_LOG``; ``FAKE_VST_NO_HOST=1`` answers like a runtime without pedalboard.
"""

import base64
import json
import os
import sys
import time

PREFIX = "STELLARIA_VST "


def send(obj):
    sys.stdout.write(PREFIX + json.dumps(obj) + "\n")
    sys.stdout.flush()


def spec_of(path):
    target = os.path.join(path, "fake.json") if os.path.isdir(path) else path
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def behave(behavior):
    if behavior == "crash":
        sys.stderr.write("plugin exploded\n")
        sys.stderr.flush()
        os._exit(139)
    if behavior == "hang":
        time.sleep(120)
    if behavior == "error":
        send({"error": {"code": "vst.load_failed", "type": "ImportError", "message": "no factory in module"}})
        sys.exit(0)


def log(line):
    target = os.environ.get("FAKE_VST_LOG")
    if target:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main(argv):
    mode = argv[1]
    if os.environ.get("FAKE_VST_NO_HOST") == "1":
        send({"error": {"code": "vst.host_missing", "type": "ModuleNotFoundError", "message": "No module named 'pedalboard'"}})
        return 0
    if mode == "probe":
        send({"ok": {"version": "0.0-fake", "formats": ["vst3"]}})
        return 0
    if mode == "scan":
        path = argv[2]
        name = argv[3] if len(argv) > 3 else None
        spec = spec_of(path)
        log("scan %s %s pid=%d" % (os.path.basename(path.rstrip("/")), name or "", os.getpid()))
        print("noise a plugin prints on stdout")
        if name is None and len(spec.get("names") or []) > 1:
            send({"ok": {"names": spec["names"]}})
            return 0
        behave((spec.get("perName") or {}).get(name, spec.get("behavior", "ok")))
        plugins = [p for p in spec.get("plugins", []) if name is None or p.get("name") == name]
        send({"ok": {"plugins": plugins}})
        return 0
    with open(argv[2], "r", encoding="utf-8") as f:
        job = json.load(f)
    spec = spec_of(job["path"])
    log("%s %s" % (mode, os.path.basename(job["path"].rstrip("/"))))
    values = json.loads(base64.b64decode(job["state"]).decode("utf-8")) if job.get("state") else {"gain": 1.0}
    if mode == "editor":
        behave("hang" if spec.get("editor") == "hang" else "ok")
        values["gain"] = values.get("gain", 1.0) / 2  # "the user turned it down"
        state = base64.b64encode(json.dumps(values).encode("utf-8")).decode("ascii")
        send({"ok": {"params": values, "state": state}})
        return 0
    send({"ok": {"params": values}})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
