"""VST3 / AU host: ids, discovery, scanning (fake worker: crash, hang, errors, shell bundles,
cache invalidation), cancel, editor/inspect, host detection, the API, processing helpers, and —
when an interpreter with pedalboard and a test plugin are given — a real round trip.

Real pedalboard tests run when ``STELLARIA_TEST_PEDALBOARD_PYTHON`` (a python with pedalboard,
numpy, soxr) and ``STELLARIA_TEST_VST3`` (a ``.vst3`` bundle) are set; they are skipped otherwise.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import pytest

from conftest import BACKEND, wait_for, wait_job
from stellaria_backend.errors import ApiError
from stellaria_backend.vst import processing, scanner
from stellaria_backend.vst.scanner import WorkerRun, plugin_id

FAKE_WORKER = Path(__file__).with_name("fake_vst_worker.py")
PB_PYTHON = os.environ.get("STELLARIA_TEST_PEDALBOARD_PYTHON")
TEST_VST3 = os.environ.get("STELLARIA_TEST_VST3")
real = pytest.mark.skipif(not (PB_PYTHON and TEST_VST3 and Path(TEST_VST3).exists()),
                          reason="set STELLARIA_TEST_PEDALBOARD_PYTHON and STELLARIA_TEST_VST3")

PARAMS = [
    {"id": "gain", "name": "Gain", "kind": "float", "min": -24.0, "max": 24.0, "default": 0.0, "unit": "dB", "choices": None},
    {"id": "bypass", "name": "Bypass", "kind": "bool", "min": None, "max": None, "default": False, "unit": None, "choices": None},
    {"id": "mode", "name": "Mode", "kind": "choice", "min": None, "max": None, "default": "Warm", "unit": None,
     "choices": ["Clean", "Warm"]},
]


def fake(folder: Path, name: str, spec: dict, bundle: bool = False) -> Path:
    path = folder / name
    if bundle:
        (path / "Contents" / "x86_64-linux").mkdir(parents=True, exist_ok=True)
        (path / "Contents" / "x86_64-linux" / "plugin.so").write_bytes(b"\0")
        (path / "fake.json").write_text(json.dumps(spec), encoding="utf-8")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(spec), encoding="utf-8")
    return path


def plugin_info(name: str, **kw) -> dict:
    return {"name": name, "vendor": "Fake Audio", "version": "1.2.3", "category": "effect", "pluginName": None,
            "latency": 0, "params": PARAMS, **kw}


@pytest.fixture
def vst_client(client, tmp_path, monkeypatch):
    ctx = client.ctx
    ctx.vst.worker = WorkerRun(python=sys.executable, script=FAKE_WORKER)
    log = tmp_path / "loads.log"
    monkeypatch.setenv("FAKE_VST_LOG", str(log))
    monkeypatch.setattr(scanner, "SCAN_TIMEOUT", 3.0)
    monkeypatch.setattr(scanner, "default_folders", lambda: [str(tmp_path / "no default folder")])
    client.loads = lambda: log.read_text(encoding="utf-8").splitlines() if log.exists() else []  # type: ignore[attr-defined]
    events: list[tuple[str, object]] = []
    original = ctx.hub.publish

    def record(type_, data, **kw):
        events.append((type_, data))
        return original(type_, data, **kw)

    monkeypatch.setattr(ctx.hub, "publish", record)
    client.events = events  # type: ignore[attr-defined]
    return client


# ---- ids, discovery ------------------------------------------------------------------------
def test_ids_are_stable(tmp_path):
    path = tmp_path / "Pro-Q 3.vst3"
    expected = hashlib.sha1((scanner.path_key(path) + "").encode("utf-8")).hexdigest()
    assert plugin_id(path, None) == expected == plugin_id(str(path) + os.sep, None)
    assert plugin_id(tmp_path / "x" / ".." / "Pro-Q 3.vst3", None) == expected
    shell = plugin_id(path, "C1 comp")
    assert shell == hashlib.sha1((scanner.path_key(path) + "C1 comp").encode("utf-8")).hexdigest() != expected
    assert len(expected) == 40


def test_discover_and_signature(tmp_path):
    root = tmp_path / "VST3"
    a = fake(root, "A.vst3", {}, bundle=True)
    b = fake(root / "Vendor" / "Deep", "B.vst3", {})
    fake(root / "Vendor", "readme.txt", {})
    (a / "Contents" / "Inner.vst3").mkdir()  # never looked for inside a bundle
    found = scanner.discover([str(root), str(root), str(tmp_path / "missing")])
    assert [p.name for p in found] == ["A.vst3", "B.vst3"]
    assert scanner.discover([str(b)]) == [b]  # a plugin path given directly
    sig = scanner.signature(a)
    inner = a / "Contents" / "x86_64-linux" / "plugin.so"
    os.utime(inner, (time.time() + 100, time.time() + 100))
    assert scanner.signature(a) > sig
    assert scanner.signature(tmp_path / "gone.vst3") == 0.0


# ---- scanning ------------------------------------------------------------------------------
def test_scan_with_crashes_timeouts_errors_and_shell_bundles(vst_client, tmp_path):
    client = vst_client
    folder = tmp_path / "my plugins 插件"
    fake(folder, "Gain.vst3", {"plugins": [plugin_info("Fake Gain")]}, bundle=True)
    fake(folder / "Vendor", "Shell.vst3", {"names": ["Shell A", "Shell B"], "perName": {"Shell B": "crash"},
                                           "plugins": [plugin_info("Shell A"), plugin_info("Shell B")]})
    fake(folder, "Crash.vst3", {"behavior": "crash"})
    fake(folder, "Hang.vst3", {"behavior": "hang"})
    fake(folder, "Error.vst3", {"behavior": "error"})
    fake(folder, "Synth.vst3", {"plugins": [plugin_info("Fake Synth", category="instrument", params=[])]})

    status = client.patch("/api/vst/settings", json={"folders": [str(folder), str(folder) + "/", "  "]}).json()
    assert status["folders"] == [str(folder)] and status["hostInstalled"] is True and status["hostVersion"] == "0.0-fake"
    assert set(status) == {"hostInstalled", "hostVersion", "formats", "folders", "defaultFolders", "editorSupported",
                           "lastScan", "scanning"}
    assert status["lastScan"] is None and status["scanning"] is False
    assert client.patch("/api/vst/settings", json={"folders": ["relative/dir"]}).status_code == 422

    started = time.monotonic()
    job = client.post("/api/vst/scan", json={}).json()
    assert job["type"] == "vst_scan"
    job = wait_job(client, job["id"], timeout=60)
    assert job["status"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").json()
    assert job["result"] == {"found": 3, "failed": 4, "scanned": 6}
    assert time.monotonic() - started < 20  # the hang was cut at the (test) timeout, in parallel

    plugins = {p["name"]: p for p in client.get("/api/vst/plugins").json()}
    assert sorted(plugins) == ["Crash", "Error", "Fake Gain", "Fake Synth", "Hang", "Shell A", "Shell B"]
    gain = plugins["Fake Gain"]
    assert set(gain) == {"id", "name", "vendor", "version", "format", "path", "pluginName", "category", "params",
                         "status", "error", "scannedAt"}
    assert gain["id"] == plugin_id(folder / "Gain.vst3", None) and gain["pluginName"] is None
    assert gain["format"] == "vst3" and gain["vendor"] == "Fake Audio" and gain["version"] == "1.2.3"
    assert gain["status"] == "ok" and gain["error"] is None and gain["params"] == PARAMS
    assert plugins["Fake Synth"]["category"] == "instrument"
    assert plugins["Shell A"]["pluginName"] == "Shell A" and plugins["Shell A"]["status"] == "ok"
    assert plugins["Shell A"]["id"] == plugin_id(folder / "Vendor" / "Shell.vst3", "Shell A")
    assert plugins["Shell B"]["status"] == "failed" and plugins["Shell B"]["error"] == {
        "code": "vst.crashed", "params": {"name": "Shell B"}}
    assert plugins["Crash"]["error"]["code"] == "vst.crashed" and plugins["Crash"]["params"] == []
    assert plugins["Hang"]["error"] == {"code": "vst.timeout", "params": {"name": "Hang", "seconds": 3}}
    assert plugins["Error"]["error"]["code"] == "vst.load_failed"
    assert plugins["Error"]["error"]["params"]["reason"] == "ImportError: no factory in module"
    assert client.get(f"/api/vst/plugins/{gain['id']}").json()["name"] == "Fake Gain"
    assert client.get("/api/vst/plugins/nope").json()["error"]["code"] == "vst.not_found"

    status = client.get("/api/vst/status").json()
    assert status["lastScan"] and status["scanning"] is False
    kinds = [t for t, _ in client.events]
    assert "vst.plugins" in kinds and kinds.count("vst.status") >= 3  # settings, scan start, scan end
    scanning_flags = [d["scanning"] for t, d in client.events if t == "vst.status"]
    assert True in scanning_flags and scanning_flags[-1] is False
    cache = json.loads((client.ctx.options.data / "vst" / "cache.json").read_text(encoding="utf-8"))
    assert len(cache["files"]) == 6

    # incremental: nothing changed → nothing loaded
    before = len(client.loads())
    job = wait_job(client, client.post("/api/vst/scan").json()["id"])
    assert job["result"]["scanned"] == 0 and job["result"]["found"] == 3 and len(client.loads()) == before
    # one bundle changed inside, one plugin removed, one added → only those are loaded
    inner = folder / "Gain.vst3" / "Contents" / "x86_64-linux" / "plugin.so"
    os.utime(inner, (time.time() + 50, time.time() + 50))
    (folder / "Error.vst3").unlink()
    fake(folder, "New.vst3", {"plugins": [plugin_info("Fake New")]})
    job = wait_job(client, client.post("/api/vst/scan").json()["id"])
    assert job["result"] == {"found": 4, "failed": 3, "scanned": 2}
    assert sorted(line.split()[1] for line in client.loads()[before:]) == ["Gain.vst3", "New.vst3"]
    assert "Error" not in {p["name"] for p in client.get("/api/vst/plugins").json()}
    # full rescan loads everything again
    before = len(client.loads())
    job = wait_job(client, client.post("/api/vst/scan", json={"full": True}).json()["id"], timeout=60)
    assert job["result"]["scanned"] == 6 and len(client.loads()) - before >= 6


def test_cancel_scan_kills_workers(vst_client, tmp_path, monkeypatch):
    client = vst_client
    monkeypatch.setattr(scanner, "SCAN_TIMEOUT", 60.0)
    folder = tmp_path / "hangs"
    for i in range(3):
        fake(folder, f"Hang{i}.vst3", {"behavior": "hang"})
    client.patch("/api/vst/settings", json={"folders": [str(folder)]})
    job = client.post("/api/vst/scan").json()
    # all three workers are up (each logs its pid before hanging)
    pids = wait_for(lambda: (lambda found: found if len(found) == 3 else None)(
        [int(line.rsplit("pid=", 1)[1]) for line in client.loads() if "pid=" in line]), timeout=30)
    started = time.monotonic()
    client.post(f"/api/jobs/{job['id']}/cancel")
    job = wait_job(client, job["id"], timeout=30)
    assert job["status"] == "cancelled", job
    assert time.monotonic() - started < 15

    def gone(pid: int) -> bool:
        try:
            return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return True

    wait_for(lambda: all(gone(pid) for pid in pids), timeout=15)
    assert client.get("/api/vst/status").json()["scanning"] is False


def test_kill_tree_tolerates_concurrent_kills():
    """Several threads killing one process tree at once (cancel handler + worker loop) must not raise."""
    import threading

    from stellaria_backend.util import procs

    for _ in range(5):
        proc = subprocess.Popen([sys.executable, "-c", "import subprocess, sys, time; "
                                 "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); time.sleep(60)"])
        wait_for(lambda: psutil.Process(proc.pid).children(), timeout=10)
        errors: list[BaseException] = []

        def kill() -> None:
            try:
                procs.kill_tree(proc.pid, timeout=3)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=kill) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        proc.wait(10)
        assert errors == []
        procs.kill_tree(proc.pid)  # already reaped: a no-op


def test_editor_and_inspect(vst_client, tmp_path):
    client = vst_client
    folder = tmp_path / "fx"
    fake(folder, "Gain.vst3", {"plugins": [plugin_info("Fake Gain")]})
    fake(folder, "Slow.vst3", {"plugins": [plugin_info("Slow Editor")], "editor": "hang"})
    client.patch("/api/vst/settings", json={"folders": [str(folder)]})
    wait_job(client, client.post("/api/vst/scan").json()["id"])
    gid = plugin_id(folder / "Gain.vst3", None)
    sid = plugin_id(folder / "Slow.vst3", None)
    state = base64.b64encode(json.dumps({"gain": 0.8}).encode()).decode()

    assert client.post(f"/api/vst/plugins/{gid}/inspect", json={"state": state}).json() == {"params": {"gain": 0.8}}
    assert client.post(f"/api/vst/plugins/{gid}/inspect", json={"state": None}).json() == {"params": {"gain": 1.0}}
    assert client.post(f"/api/vst/plugins/{gid}/inspect", json={"state": "%%%"}).status_code == 422
    assert client.post("/api/vst/plugins/nope/inspect", json={}).json()["error"]["code"] == "vst.not_found"

    client.ctx.vst.editor_override = False
    r = client.post(f"/api/vst/plugins/{gid}/editor", json={"state": state})
    assert r.status_code == 409 and r.json()["error"]["code"] == "vst.editor_unsupported"
    assert client.get("/api/vst/status").json()["editorSupported"] is False
    client.ctx.vst.editor_override = True
    job = client.post(f"/api/vst/plugins/{gid}/editor", json={"state": state}).json()
    assert job["type"] == "vst_editor" and job["target"] == {"vstPluginId": gid}
    job = wait_job(client, job["id"])
    assert job["status"] == "succeeded" and job["result"]["params"] == {"gain": 0.4}
    assert json.loads(base64.b64decode(job["result"]["state"])) == {"gain": 0.4}

    job = client.post(f"/api/vst/plugins/{sid}/editor", json={}).json()
    wait_for(lambda: any(line.startswith("editor Slow") for line in client.loads()), timeout=20)
    client.post(f"/api/jobs/{job['id']}/cancel")
    assert wait_job(client, job["id"], timeout=15)["status"] == "cancelled"


def test_host_missing(vst_client, tmp_path, monkeypatch):
    client = vst_client
    monkeypatch.setenv("FAKE_VST_NO_HOST", "1")
    status = client.get("/api/vst/status").json()
    assert status["hostInstalled"] is False and status["hostVersion"] is None
    r = client.post("/api/vst/scan")
    assert r.status_code == 409 and r.json()["error"]["code"] == "vst.host_missing"


def test_status_defaults_and_real_host_detection(client):
    vst = client.ctx.vst
    status = client.get("/api/vst/status").json()
    assert status["defaultFolders"] == scanner.default_folders() and status["folders"] == []
    assert status["formats"] == (["vst3", "au"] if sys.platform == "darwin" else ["vst3"])
    assert status["editorSupported"] == (sys.platform in ("win32", "darwin"))
    import importlib.util

    assert status["hostInstalled"] == (importlib.util.find_spec("pedalboard") is not None)
    assert "pedalboard" not in sys.modules  # detection never imports it into the API process
    assert vst.status()["hostInstalled"] == processing.host_available()


def test_vsthost_resource_is_an_addon(client):
    res = {r["id"]: r for r in client.get("/api/resources").json()}["vsthost"]
    assert res["kind"] == "addon" and res["required"] is False
    rdef = client.ctx.resources.definition("vsthost")
    assert rdef.packages == ["pedalboard==0.9.25"] and rdef.module_name == "pedalboard"
    manifest = json.loads(Path(client.ctx.options.resources).read_text(encoding="utf-8"))
    entry = next(r for r in manifest["resources"] if r["id"] == "vsthost")
    assert entry["license"] == "GPL-3.0" and entry["name"]["zh-CN"] and entry["description"]["en"]


def test_addon_constraints_path_with_spaces(client, tmp_path):
    """uv splits --constraint values on spaces: the installer passes a bare name and its folder as cwd."""
    from stellaria_backend.resources.manager import ResourceManager

    folder = tmp_path / "temp dir with space"
    folder.mkdir()
    cmd = ResourceManager._pip_command(client.ctx.resources, ["pedalboard==0.9.25"], None, folder / "c.txt")
    assert cmd[-2:] == ["--constraint", "c.txt"]


# ---- processing helpers (no pedalboard needed) ---------------------------------------------
class FakeParameter:
    def __init__(self, kind, lo=None, hi=None, values=()):
        self.type, self.min_value, self.max_value, self.valid_values = kind, lo, hi, list(values)


class FakeInstance:
    """Behaves like a pedalboard plugin: rejects 44.1 kHz, adds a 0.5 s decaying tail."""

    def __init__(self):
        self.raw_state = b"default"
        self.parameters = {"gain": FakeParameter(float, -12.0, 12.0), "on": FakeParameter(bool),
                           "mode": FakeParameter(str, values=["A", "B"])}
        self.gain, self.on, self.mode = 0.0, False, "A"
        self.rates: list[float] = []
        self.is_instrument, self.is_effect = False, True

    def process(self, x, sr, reset=True):
        self.rates.append(sr)
        if sr == 44100.0:
            raise RuntimeError("unsupported sample rate")
        out = x * np.float32(10 ** (self.gain / 20))
        loud = np.nonzero(np.abs(x).max(axis=0) > 1e-3)[0]
        if len(loud):  # a decaying "reverb" tail after the last loud sample
            end = int(loud[-1]) + 1
            tail = min(int(0.5 * sr), out.shape[1] - end)
            out[:, end:end + tail] += 0.1 * np.exp(-np.arange(tail) / (0.05 * sr)).astype(np.float32)
        return out


def test_processing_with_a_fake_instance(monkeypatch):
    instance = FakeInstance()
    plugin = {"id": "p", "name": "Fake", "path": "/fake/Fake.vst3", "pluginName": None, "category": "effect"}
    monkeypatch.setitem(processing._loaded, processing._key(plugin), (instance, b"default"))
    x = np.full((2, 44100), 0.25, np.float32)
    slot = {"state": base64.b64encode(b"slot state").decode(), "params": {"gain": 99, "on": "true", "mode": "Z"}}
    y = processing.apply(x, 44100, slot, plugin)
    assert instance.raw_state == b"slot state" and instance.gain == 12.0 and instance.on is True and instance.mode == "A"
    assert instance.rates == [44100.0, 48000.0]  # rejected 44.1 kHz → 48 kHz with resampling
    assert y.dtype == np.float32 and y.shape[0] == 2
    assert 44100 + 0.3 * 44100 < y.shape[1] < 44100 + 6 * 44100  # tail kept, trimmed at −90 dBFS
    assert abs(float(y[0, 22050]) - 0.25 * 10 ** (12 / 20)) < 0.01
    processing.apply(x, 44100, {"state": None, "params": {}}, plugin)
    assert instance.raw_state == b"default"  # a slot without state gets the plugin's defaults back
    with pytest.raises(ApiError) as exc:
        processing.apply(x, 44100, {}, {**plugin, "category": "instrument"})
    assert exc.value.code == "vst.instrument" and exc.value.params == {"name": "Fake"}
    instance.is_instrument, instance.is_effect = True, False
    with pytest.raises(ApiError) as exc:
        processing.apply(x, 44100, {}, plugin)
    assert exc.value.code == "vst.instrument"


def test_processing_errors_and_helpers(monkeypatch):
    assert processing.trim_tail(np.zeros((2, 1500), np.float32), 1000).shape == (2, 1000)  # silent tail dropped
    z = np.zeros((2, 2000), np.float32)
    z[0, 1500] = 1e-3
    assert processing.trim_tail(z, 1000).shape == (2, 1501)
    sine = np.sin(np.arange(44100) / 44100 * 2 * np.pi * 440).astype(np.float32)[None, :]
    up = processing.resample(sine, 44100, 48000)
    assert up.shape == (1, 48000) and abs(processing.resample(up, 48000, 44100).shape[1] - 44100) <= 1
    if not processing.host_available():
        with pytest.raises(ApiError) as exc:
            processing.load({"name": "X", "path": "/nope/X.vst3"})
        assert exc.value.code == "vst.host_missing"
    monkeypatch.setattr(processing, "host_available", lambda: True)
    with pytest.raises(ApiError) as exc:
        processing.load({"name": "Gone", "path": "/nope/Gone.vst3"})
    assert exc.value.code == "vst.not_found" and exc.value.params == {"name": "Gone"}


# ---- real pedalboard round trip (optional) -------------------------------------------------
EVIL = {  # VST3 "modules" whose factory segfaults or never returns
    "Crash": "#include <stddef.h>\nvoid* GetPluginFactory(void) { volatile int* p = NULL; *p = 42; return NULL; }\n",
    "Hang": "#include <unistd.h>\nvoid* GetPluginFactory(void) { for (;;) sleep(1); return 0; }\n",
}


def _evil_bundles(folder: Path) -> bool:
    import shutil

    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None or sys.platform != "linux":
        return False
    for name, code in EVIL.items():
        binary = folder / f"{name}.vst3" / "Contents" / "x86_64-linux" / f"{name}.so"
        binary.parent.mkdir(parents=True)
        source = folder / f"{name}.c"
        source.write_text(code + "int ModuleEntry(void* h) { return 1; }\nint ModuleExit(void) { return 1; }\n")
        subprocess.run([cc, "-shared", "-fPIC", "-o", str(binary), str(source)], check=True)
    return True


@real
def test_real_plugin_scan_inspect_and_process(client, tmp_path, monkeypatch):
    vst = client.ctx.vst
    vst.worker = WorkerRun(python=PB_PYTHON)  # the real worker.py, run with an interpreter that has pedalboard
    monkeypatch.setattr(scanner, "SCAN_TIMEOUT", 8.0)
    status = client.get("/api/vst/status").json()
    assert status["hostInstalled"] is True and status["hostVersion"]
    folder = Path(TEST_VST3).parent
    evil = tmp_path / "evil"
    has_evil = _evil_bundles(evil)
    client.patch("/api/vst/settings", json={"folders": [str(folder), str(evil)]})
    job = wait_job(client, client.post("/api/vst/scan").json()["id"], timeout=120)
    assert job["status"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").json()
    found = client.get("/api/vst/plugins").json()
    plugin = next(p for p in found if p["path"] == str(Path(TEST_VST3)))
    assert plugin["status"] == "ok", plugin
    params = {p["id"]: p for p in plugin["params"]}
    assert plugin["category"] == "effect" and plugin["format"] == "vst3"
    if plugin["name"] == "Stellaria Test Gain":  # the scratch test plugin: gain + invert + mode, 32 samples latency
        assert params["gain"] == {"id": "gain", "name": "Gain", "kind": "float", "min": 0.0, "max": 1.0,
                                  "default": 1.0, "unit": None, "choices": None}
        assert params["invert"]["kind"] == "bool" and params["invert"]["default"] is False
        assert params["mode"]["kind"] == "choice" and params["mode"]["choices"] == ["Clean", "Warm", "Hot"]
        assert plugin["vendor"] == "Stellaria Tests"
    if has_evil:
        by_name = {p["name"]: p for p in found}
        assert by_name["Crash"]["error"] == {"code": "vst.crashed", "params": {"name": "Crash"}}
        assert by_name["Hang"]["error"] == {"code": "vst.timeout", "params": {"name": "Hang", "seconds": 8}}

    # make a state with the real plugin, then read it back through inspect and use it in processing
    script = r'''
import base64, json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from pedalboard import load_plugin
from stellaria_backend.vst import processing
plugin = json.loads(sys.argv[2])
p = load_plugin(plugin["path"], plugin_name=plugin["pluginName"])
first = next(k for k, v in p.parameters.items() if v.type is float)
lo, hi = p.parameters[first].min_value, p.parameters[first].max_value
setattr(p, first, lo + (hi - lo) * 0.5)
p.process(np.zeros((2, 512), np.float32), 44100, reset=False)
state = base64.b64encode(p.raw_state).decode()
x = np.zeros((2, 44100), np.float32); x[:, 1000] = 0.5
y = processing.apply(x, 44100, {"state": state, "params": {}}, plugin)
y2 = processing.apply(x, 44100, {"state": None, "params": {}}, plugin)
print(json.dumps({"state": state, "param": first, "value": float(getattr(p, first)),
                  "peak": int(np.argmax(np.abs(y[0]))), "shape": list(y.shape), "differs": bool(np.abs(y - y2[:, :y.shape[1]]).max() > 1e-6)}))
'''
    out = subprocess.run([PB_PYTHON, "-c", script, str(BACKEND), json.dumps(plugin)], capture_output=True, text=True,
                         timeout=120)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout.strip().splitlines()[-1])
    assert data["peak"] == 1000  # latency compensated by pedalboard: still aligned
    assert data["shape"][1] >= 44100
    inspected = client.post(f"/api/vst/plugins/{plugin['id']}/inspect", json={"state": data["state"]}).json()
    assert inspected["params"][data["param"]] == pytest.approx(data["value"], abs=1e-3)
    assert set(inspected["params"]) == set(params)
