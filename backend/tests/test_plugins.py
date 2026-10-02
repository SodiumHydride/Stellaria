"""Plugins: manifests, wizard, command/python runtimes, timeouts, cancel, failures, interpreter
checks, custom interpreters, registry persistence, installs and the HTTP API."""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import time
import zipfile
from importlib import metadata
from pathlib import Path

import numpy as np
import psutil
import pytest
import soundfile as sf

from conftest import wait_for, wait_job
from stellaria_backend.errors import ApiError
from stellaria_backend.plugins import PluginRegistry, check_python, run_plugin
from stellaria_backend.plugins.envs import env_dir, provided_by_runtime, venv_python
from stellaria_backend.plugins.manifest import MANIFEST_FILE, load_folder_manifest
from stellaria_backend.plugins.models import PluginPatch, WizardSpec
from stellaria_backend.plugins.testdata import ensure_testdata
from stellaria_backend.resources.manifest import Source
from stellaria_backend.schemas import msg

POSIX = os.name == "posix"

# A stand-in for a friend's auto-mixing script: argv = <backing> <vocal...>, creates and clears
# input/ output/ temporary/ in its cwd, imports setting.py from next to it, and writes
# output/mixdown_<vocal>.flac. (Uses os.path.join so it also runs off Windows.)
FRIEND_SCRIPT = r'''
import os, shutil, sys
import numpy as np
import soundfile as sf
from setting import CONFIG


def prepare():
    for d in ("input", "output", "temporary"):
        os.makedirs(d, exist_ok=True)
    for d in ("input", "temporary"):
        for name in os.listdir(d):
            os.remove(os.path.join(d, name))
    print("Necessary folders have been created.")
    if CONFIG.recieve_mode != 1:
        print("Test mode, skip file paste.")
        return
    if len(sys.argv) < 3:
        print("Not enough paths are provided.")
        os._exit(0)
    inst = sys.argv[1]
    stem, ext = os.path.splitext(os.path.basename(inst))
    shutil.copy(inst, os.path.join("input", stem + "_instrument" + ext))
    for path in sys.argv[2:]:
        shutil.copy(path, "input")
    print("File paste successfully.")


def main():
    prepare()
    names = sorted(os.listdir("input"))
    inst = [n for n in names if "instrument" in n.lower()]
    vocals = [n for n in names if "instrument" not in n.lower()]
    backing, sr = sf.read(os.path.join("input", inst[0]), always_2d=True)
    print("STELLARIA_PROGRESS 0.3")
    for name in vocals:
        vocal, _ = sf.read(os.path.join("input", name), always_2d=True)
        n = min(len(backing), len(vocal))
        mix = backing[:n] * 0.6 + vocal[:n, :1] * 0.6
        sf.write(os.path.join("temporary", "temporary_stereo.wav"), mix, sr)
        out = os.path.join("output", "mixdown_" + os.path.splitext(name)[0] + ".flac")
        sf.write(out, mix, sr, subtype="PCM_16")
        print("Mixing finished! Output saved to: " + out)
    print("STELLARIA_PROGRESS 0.9")


if __name__ == "__main__":
    main()
'''

PROCESS_ENTRY = r'''
import json, os
import numpy as np
import soundfile as sf


def process(job):
    job.log("params " + json.dumps(job.params, sort_keys=True))
    job.log("seed " + os.environ.get("STELLARIA_SEED", "-"))
    job.log("settings " + json.dumps(job.settings, sort_keys=True))
    print("a plain print ends up in the log")
    vocal, sr = sf.read(str(job.inputs["vocals"][0]), always_2d=True)
    backing, _ = sf.read(str(job.inputs["backing"]), always_2d=True)
    for i in range(1, 5):
        job.progress(i / 5)
    n = min(len(vocal), len(backing))
    sf.write(str(job.output), backing[:n] * 0.5 + vocal[:n, :1] * job.params["gain"], sr)
'''

STDLIB_ENTRY = r'''
import shutil, sys


def process(job):
    job.log("interpreter " + sys.executable)
    job.progress(0.5)
    out = job.outdir / "copied.wav"
    shutil.copyfile(str(job.inputs["input"]), str(out))
    return out
'''


def write_plugin(folder: Path, files: dict[str, str], manifest: dict | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    if manifest is not None:
        (folder / MANIFEST_FILE).write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return folder


def friend_folder(root: Path) -> Path:
    return write_plugin(root / "Sodium 混音", {
        "auto_mixing.py": FRIEND_SCRIPT,
        "setting.py": "class CONFIG:\n    recieve_mode = 1\n    random_mode = False\n",
        "file_resource.py": "import os\nimport soundfile\n",
    })


def command_manifest(pid: str, script: str, **extra) -> dict:
    data = {"id": pid, "name": pid, "kind": "mix", "runtime": "command",
            "command": ["{python}", "{folder}/" + script, "{backing}", "{vocal}"], "outputs": ["out/*.wav"]}
    data.update(extra)
    return data


def run_job(ctx, fn, timeout: float = 90.0):
    job = ctx.jobs.submit("plugin_test", msg("job.plugin_test"), fn, target={"pluginId": "test"})
    wait_for(lambda: not ctx.jobs.get(job.id).active, timeout=timeout)
    return ctx.jobs.get(job.id)


def run_capturing(ctx, pid: str, output: Path, **kwargs):
    """Run a plugin inside a job; returns (job, result-or-error dict)."""
    back, voc = ensure_testdata(ctx.options.data)
    captured: dict = {}

    def fn(jctx):
        try:
            args = {"vocals": [voc], "backing": back, "input": None, "params": {}}
            args.update(kwargs)
            run_plugin(ctx, jctx, pid, output=output, **args)
            captured["ok"] = True
        except ApiError as exc:
            captured.update(code=exc.code, params=exc.params, detail=exc.detail, message=exc.message)
            raise
        return {}

    return run_job(ctx, fn), captured


# ---- manifests -----------------------------------------------------------------------------
def test_manifest_validation(tmp_path):
    folder = write_plugin(tmp_path / "p", {"mix.py": "print(1)\n"}, command_manifest("good-one", "mix.py", settings=[
        {"id": "vst3", "label": "VST3 文件夹", "type": "folder", "default": None}],
        params=[{"id": "wet", "label": "Wet", "type": "number", "default": 0.3, "min": 0, "max": 1}]))
    m = load_folder_manifest(folder)
    assert m.id == "good-one" and m.max_vocals is None and m.vocals_limit == 1 and m.timeout_sec == 1200
    wire = m.params[0].wire()
    assert wire == {"id": "wet", "label": "Wet", "type": "number", "default": 0.3, "min": 0.0, "max": 1.0}
    assert "options" not in m.settings[0].wire() and m.settings[0].wire()["default"] is None

    cases = {
        "unknown token {nope}": {"command": ["{python}", "{folder}/mix.py", "{nope}", "{vocal}"]},
        "{vocals} must be a whole argument": {"command": ["{python}", "{folder}/mix.py", "--v={vocals}"]},
        "{input} is only for 'effect' plugins": {"command": ["{python}", "{folder}/mix.py", "{input}", "{vocal}"]},
        "there is no setting 'vst'": {"command": ["{python}", "{folder}/mix.py", "{vocal}", "{setting:vst}"]},
        "no way to find the result": {"outputs": []},
        "must be relative": {"outputs": ["../escape/*.wav"]},
        "file not found in the plugin folder: missing.py": {"command": ["{python}", "{folder}/missing.py", "{vocal}"]},
        "id 'Bad Id'": {"id": "Bad Id"},
        "id 'builtin'": {"id": "builtin"},
        "needs a non-empty command": {"command": []},
        "kind": {"kind": "video"},
        "is not a package name": {"requirements": ["--index-url=http://evil"]},
        "a choice needs options": {"params": [{"id": "x", "label": "X", "type": "choice", "default": "a"}]},
        "duplicate param id 'x'": {"params": [{"id": "x", "label": "X", "type": "boolean", "default": True},
                                              {"id": "x", "label": "X", "type": "boolean", "default": False}]},
        "entry file not found": {"runtime": "python", "entry": "nope.py", "command": None},
    }
    for i, (reason, change) in enumerate(cases.items()):
        bad = write_plugin(tmp_path / f"bad{i}", {"mix.py": ""}, {**command_manifest("bad", "mix.py"), **change})
        with pytest.raises(ApiError) as exc:
            load_folder_manifest(bad)
        assert exc.value.code == "plugin.bad_manifest" and exc.value.status == 422
        assert reason in exc.value.params["reason"], (reason, exc.value.params)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(ApiError) as exc:
        load_folder_manifest(broken)
    assert "not valid JSON" in exc.value.params["reason"]
    with pytest.raises(FileNotFoundError):
        load_folder_manifest(tmp_path)


def test_wizard_manifest_and_hints(ctx, tmp_path):
    folder = friend_folder(tmp_path)
    with pytest.raises(ApiError) as exc:
        ctx.plugins.add_folder(str(folder))
    assert exc.value.code == "plugin.no_manifest" and exc.value.status == 409
    assert exc.value.detail["candidates"][0] == "auto_mixing.py"  # __main__ guard first; setting.py last
    assert exc.value.detail["candidates"][-1] == "setting.py"
    assert exc.value.detail["requirements"] == ["numpy", "soundfile"]  # setting/file_resource are local modules

    plugin = ctx.plugins.add_wizard(WizardSpec(folder=str(folder), name="Sodium 自动混音", kind="mix",
                                               entry="auto_mixing.py", args=["{backing}", "{vocal}"],
                                               outputs=["output/*.flac", "output/*.wav"], cwd="temp",
                                               requirements=["numpy", "soundfile"]))
    assert plugin.id == "sodium" and plugin.manifest_source == "wizard" and plugin.status == "ready", plugin
    stored = json.loads((ctx.options.data / "plugins" / "sodium.json").read_text(encoding="utf-8"))
    assert stored["command"] == ["{python}", "{folder}/auto_mixing.py", "{backing}", "{vocal}"]
    assert stored["folder"] == str(folder.resolve()) and stored["outputs"] == ["output/*.flac", "output/*.wav"]
    assert not (folder / MANIFEST_FILE).exists()  # the user's folder is not written to
    # same name again → a fresh id; defaults for args/outputs
    again = ctx.plugins.add_wizard(WizardSpec(folder=str(folder), name="Sodium 自动混音", entry=str(folder / "auto_mixing.py")))
    assert again.id == "sodium-2"
    stored = json.loads((ctx.options.data / "plugins" / "sodium-2.json").read_text(encoding="utf-8"))
    assert stored["command"][2:] == ["{backing}", "{vocal}"] and stored["outputs"][0] == "output/*.flac"
    with pytest.raises(ApiError) as exc:
        ctx.plugins.add_wizard(WizardSpec(folder=str(folder), name="x", entry="auto_mixing.py", args=["{input}"]))
    assert exc.value.code == "plugin.bad_manifest"
    with pytest.raises(ApiError) as exc:
        ctx.plugins.add_wizard(WizardSpec(folder=str(folder), name="x", entry="nope.py"))
    assert "file not found" in exc.value.params["reason"]


# ---- running -------------------------------------------------------------------------------
def test_friend_style_command_plugin_runs_in_a_temp_cwd(ctx, tmp_path):
    folder = friend_folder(tmp_path)
    before = sorted(p.name for p in folder.iterdir())
    plugin = ctx.plugins.add_wizard(WizardSpec(folder=str(folder), name="Sodium 自动混音", entry="auto_mixing.py",
                                               outputs=["output/*.flac", "output/*.wav"], requirements=["numpy", "soundfile"]))
    assert plugin.interpreter == "builtin" and plugin.status == "ready"
    py = venv_python(env_dir(ctx.options.data, plugin.id))
    assert plugin.check and Path(plugin.check.executable) == py
    source = tmp_path / "med_abc123xyz9.flac"  # id-like name, not a WAV: staged as a converted WAV
    sf.write(str(source), np.zeros((44100, 1), dtype=np.float32) + 0.01, 44100)
    backing, _ = ensure_testdata(ctx.options.data)
    out = tmp_path / "result" / "mix.wav"
    progress: list[float] = []
    job, captured = run_capturing(ctx, plugin.id, out, vocals=[source], backing=backing,
                                  names={str(backing): "我的歌 - 伴奏"}, on_progress=progress.append)
    assert job.status == "succeeded", (job.error, list(job.log))
    info = sf.info(str(out))
    assert info.format == "WAV" and info.channels == 2 and abs(info.duration - 1.0) < 0.01
    log = "\n".join(job.log)
    assert "File paste successfully." in log and "Mixing finished!" in log
    assert "我的歌 - 伴奏.wav" in log and "vocal.wav" in log  # readable input names
    assert "STELLARIA_PROGRESS" not in log and progress[:2] == [0.3, 0.9] and progress[-1] == 1.0
    assert "mixdown_vocal.flac" in log
    # the script's input/ output/ temporary/ went to a temp cwd, never into the plugin folder
    assert sorted(p.name for p in folder.iterdir()) == before
    assert not list((ctx.tmp_dir / "plugins").iterdir())  # run folder cleaned up


def test_python_runtime_plugin_reports_progress(ctx, tmp_path):
    folder = write_plugin(tmp_path / "py plugin", {"entry.py": PROCESS_ENTRY}, {
        "id": "pyproc", "name": "Py", "kind": "mix", "runtime": "python", "entry": "entry.py", "maxVocals": 2,
        "params": [{"id": "gain", "label": "Gain", "type": "number", "default": 0.5, "min": 0, "max": 1}],
        "settings": [{"id": "flavour", "label": "Flavour", "type": "text", "default": "vanilla"}],
        "requirements": ["numpy", "soundfile"],
    })
    plugin = ctx.plugins.add_folder(str(folder))
    assert plugin.status == "ready" and plugin.runtime == "python" and plugin.max_vocals == 2
    assert plugin.setting_values == {"flavour": "vanilla"}
    ctx.plugins.update(plugin.id, PluginPatch(setting_values={"flavour": "mint"}))
    out = tmp_path / "out.wav"
    progress: list[float] = []
    job, _ = run_capturing(ctx, plugin.id, out, params={"gain": 7, "extra": "x", "seed": 42},
                           on_progress=progress.append)
    assert job.status == "succeeded", (job.error, list(job.log))
    log = "\n".join(job.log)
    assert 'params {"extra": "x", "gain": 1.0, "seed": 42}' in log  # clamped to max, unknown keys pass through
    assert "seed 42" in log
    assert 'settings {"flavour": "mint"}' in log
    assert "a plain print ends up in the log" in log and "STELLARIA_MSG" not in log
    assert progress == [0.2, 0.4, 0.6, 0.8, 1.0]
    assert sf.info(str(out)).duration == pytest.approx(8.0, abs=0.01)
    # a job without on_progress: progress and stage go to the job itself
    job, _ = run_capturing(ctx, plugin.id, tmp_path / "out2.wav")
    assert job.status == "succeeded" and job.stage.code == "stage.plugin" and job.progress == 1.0

    # the author adds a requirement: the stored check is stale until listed (or run) again
    manifest = json.loads((folder / MANIFEST_FILE).read_text(encoding="utf-8"))
    manifest["requirements"].append("definitely-not-installed-xyz")
    (folder / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    stale = ctx.plugins.get(plugin.id)
    assert stale.status == "needs_setup" and stale.error.code == "plugin.not_checked"
    listed = ctx.plugins.list()[0]
    assert listed.status == "needs_setup" and listed.error.code == "plugin.packages_missing"
    assert listed.check.missing == ["definitely-not-installed-xyz"]
    assert ctx.plugins.mix_processors()[0]["available"] is False
    job, captured = run_capturing(ctx, plugin.id, tmp_path / "out3.wav")
    assert captured["code"] == "plugin.packages_missing" and captured["params"] == {"packages": "definitely-not-installed-xyz"}


def test_python_plugin_exception_is_plugin_failed(ctx, tmp_path):
    folder = write_plugin(tmp_path / "boom", {"entry.py": "def process(job):\n    raise ValueError('kaputt')\n"},
                          {"id": "boom", "name": "Boom", "kind": "effect", "runtime": "python", "entry": "entry.py"})
    ctx.plugins.add_folder(str(folder))
    job, captured = run_capturing(ctx, "boom", tmp_path / "o.wav", vocals=[], backing=None,
                                  input=ensure_testdata(ctx.options.data)[1])
    assert job.status == "failed" and job.error.code == "plugin.failed"
    assert captured["params"] == {"exitCode": 1} and "ValueError: kaputt" in captured["message"]
    assert "Traceback" in captured["detail"]


def test_timeout(ctx, tmp_path):
    folder = write_plugin(tmp_path / "slow", {"slow.py": "import time\nprint('sleeping', flush=True)\ntime.sleep(60)\n"},
                          command_manifest("slow", "slow.py", timeoutSec=1))
    ctx.plugins.add_folder(str(folder))
    started = time.monotonic()
    job, captured = run_capturing(ctx, "slow", tmp_path / "o.wav")
    assert job.status == "failed" and job.error.code == "plugin.timeout"
    assert captured["params"] == {"seconds": 1} and "sleeping" in captured["detail"]
    assert time.monotonic() - started < 20


def test_cancel_kills_the_process_tree(ctx, tmp_path):
    pids = tmp_path / "pids.txt"
    script = (
        "import os, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(pids)!r}, 'w').write('%d %d' % (os.getpid(), child.pid))\n"
        "print('running', flush=True)\n"
        "time.sleep(120)\n"
    )
    folder = write_plugin(tmp_path / "sleepy", {"sleepy.py": script}, command_manifest("sleepy", "sleepy.py"))
    ctx.plugins.add_folder(str(folder))
    back, voc = ensure_testdata(ctx.options.data)

    def fn(jctx):
        run_plugin(ctx, jctx, "sleepy", vocals=[voc], backing=back, input=None, output=tmp_path / "o.wav", params={})

    job = ctx.jobs.submit("plugin_test", msg("job.plugin_test"), fn)
    wait_for(lambda: pids.exists() and pids.read_text().strip(), timeout=30)
    parent, child = (int(x) for x in pids.read_text().split())
    assert psutil.pid_exists(parent) and psutil.pid_exists(child)
    ctx.jobs.cancel(job.id)
    wait_for(lambda: not ctx.jobs.get(job.id).active, timeout=20)
    assert ctx.jobs.get(job.id).status == "cancelled"

    def gone(pid: int) -> bool:
        try:
            return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            return True

    wait_for(lambda: gone(parent) and gone(child), timeout=10)


@pytest.mark.skipif(not POSIX, reason="process groups")
def test_background_child_holding_the_pipes_does_not_hang(ctx, tmp_path):
    pids = tmp_path / "bg.txt"
    script = (
        "import os, shutil, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"  # inherits stdout
        f"open({str(pids)!r}, 'w').write(str(child.pid))\n"
        "shutil.copyfile(sys.argv[2], sys.argv[3])\n"
        "print('main done', flush=True)\n"
    )
    folder = write_plugin(tmp_path / "bg", {"bg.py": script}, command_manifest(
        "bg", "bg.py", outputs=[], command=["{python}", "{folder}/bg.py", "{backing}", "{vocal}", "{output}"]))
    ctx.plugins.add_folder(str(folder))
    started = time.monotonic()
    job, _ = run_capturing(ctx, "bg", tmp_path / "o.wav")
    assert job.status == "succeeded", (job.error, list(job.log))
    assert time.monotonic() - started < 20 and sf.info(str(tmp_path / "o.wav")).duration == pytest.approx(8.0, abs=0.01)
    child = int(pids.read_text())
    wait_for(lambda: not psutil.pid_exists(child) or psutil.Process(child).status() == psutil.STATUS_ZOMBIE, timeout=10)


def test_missing_output_and_nonzero_exit(ctx, tmp_path):
    quiet = write_plugin(tmp_path / "quiet", {"quiet.py": "print('Not enough paths are provided.')\n"},
                         command_manifest("quiet", "quiet.py"))
    ctx.plugins.add_folder(str(quiet))
    job, captured = run_capturing(ctx, "quiet", tmp_path / "o.wav")
    assert job.status == "failed" and job.error.code == "plugin.no_output"
    assert "Not enough paths are provided." in job.error.detail

    noisy = "import sys\nfor i in range(100):\n    print('line %d' % i)\nsys.stderr.write('bad things\\n')\nsys.exit(3)\n"
    failing = write_plugin(tmp_path / "failing", {"failing.py": noisy}, command_manifest("failing", "failing.py"))
    ctx.plugins.add_folder(str(failing))
    job, captured = run_capturing(ctx, "failing", tmp_path / "o.wav")
    assert job.status == "failed" and job.error.code == "plugin.failed"
    assert captured["params"] == {"exitCode": 3} and "code 3" in job.error.message
    lines = job.error.detail.splitlines()
    assert len(lines) == 60 and "bad things" in lines and "line 99" in lines and "line 10" not in lines


def test_inputs_settings_and_disabled(ctx, tmp_path):
    folder = write_plugin(tmp_path / "s", {"s.py": "print(1)\n"}, command_manifest(
        "needs-setting", "s.py", command=["{python}", "{folder}/s.py", "{vocal}", "{setting:vst3}", "{output}"],
        outputs=[], settings=[{"id": "vst3", "label": "VST3", "type": "folder", "default": None}]))
    plugin = ctx.plugins.add_folder(str(folder))
    assert plugin.status == "needs_setup" and plugin.error.code == "plugin.setting_missing"
    job, captured = run_capturing(ctx, plugin.id, tmp_path / "o.wav")
    assert captured["code"] == "plugin.setting_missing" and captured["params"] == {"setting": "vst3"}
    plugin = ctx.plugins.update(plugin.id, PluginPatch(setting_values={"vst3": str(tmp_path)}))
    assert plugin.status == "ready" and plugin.setting_values == {"vst3": str(tmp_path)}
    job, captured = run_capturing(ctx, plugin.id, tmp_path / "o.wav", vocals=[])
    assert captured["code"] == "plugin.input_missing"
    back, voc = ensure_testdata(ctx.options.data)
    job, captured = run_capturing(ctx, plugin.id, tmp_path / "o.wav", vocals=[voc, voc])
    assert captured["code"] == "plugin.too_many_vocals" and captured["params"] == {"max": 1}
    ctx.plugins.update(plugin.id, PluginPatch(enabled=False))
    job, captured = run_capturing(ctx, plugin.id, tmp_path / "o.wav")
    assert captured["code"] == "plugin.disabled"
    job, captured = run_capturing(ctx, "nope", tmp_path / "o.wav")
    assert captured["code"] == "plugin.not_found"


# ---- interpreters --------------------------------------------------------------------------
def test_check_python(tmp_path):
    venv = Path(sys.executable)
    result = check_python(venv, ["numpy", "SoundFile>=0.1", "definitely-not-installed-xyz",
                                 "pywin32; sys_platform == 'win32'" if POSIX else "pyobjc; sys_platform == 'darwin'",
                                 "numpy>=999"])
    assert result["ok"] is False and result["error"] is None
    assert result["version"].startswith("%d.%d." % sys.version_info[:2])
    assert result["missing"] == ["definitely-not-installed-xyz", "numpy>=999"]
    assert result["torch"] is None or result["torch"]["version"]  # torch is optional
    assert set(result) == {"ok", "executable", "version", "missing", "error", "torch"}
    ok = check_python(venv.parent.parent if venv.parent.name in ("bin", "Scripts") else venv, ["numpy"], torch=False)
    assert ok["ok"] is True and ok["missing"] == [] and ok["torch"] is None  # an env folder works too

    bogus = check_python(tmp_path / "no" / "python.exe")
    assert bogus["ok"] is False and bogus["error"]["code"] == "python.not_found"
    assert check_python("")["error"]["code"] == "python.not_found"
    if POSIX:
        fake = tmp_path / "fakepython"
        fake.write_text("#!/bin/sh\necho hello\n")
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        assert check_python(fake)["error"]["code"] == "python.failed"
        hang = tmp_path / "hangpython"
        hang.write_text("#!/bin/sh\nsleep 30\n")
        hang.chmod(hang.stat().st_mode | stat.S_IEXEC)
        started = time.monotonic()
        res = check_python(hang, timeout=1.0)
        assert res["error"]["code"] == "python.timeout" and time.monotonic() - started < 10
    text = tmp_path / "notes.txt"
    text.write_text("x")
    assert check_python(text)["error"]["code"] == "python.failed"
    old = check_python(venv, [], torch=False, min_version=(9, 0))
    assert old["error"]["code"] == "python.too_old" and old["ok"] is False


def _other_python() -> str:
    """Some interpreter that is not the test runner's (falls back to the runner itself)."""
    own = os.path.realpath(sys.executable)
    for name in ("python3.12", "python3.13", "python3.10", "python3.11", "python3", "python"):
        found = shutil.which(name)
        if found and os.path.realpath(found) != own and "venv" not in found:
            return found
    return sys.executable


def test_custom_interpreter_selection(ctx, tmp_path):
    folder = write_plugin(tmp_path / "std", {"entry.py": STDLIB_ENTRY}, {
        "id": "stdlib", "name": "Stdlib", "kind": "effect", "runtime": "python", "entry": "entry.py"})
    plugin = ctx.plugins.add_folder(str(folder))
    assert plugin.interpreter == "builtin" and plugin.status == "ready"
    with pytest.raises(ApiError) as exc:
        ctx.plugins.update("stdlib", PluginPatch(interpreter="custom"))
    assert exc.value.code == "plugin.python_missing"
    plugin = ctx.plugins.update("stdlib", PluginPatch(interpreter="custom", python_path=str(tmp_path / "nope" / "python")))
    assert plugin.status == "error" and plugin.check.error.code == "python.not_found"
    job, captured = run_capturing(ctx, "stdlib", tmp_path / "o.wav", vocals=[], backing=None,
                                  input=ensure_testdata(ctx.options.data)[1])
    assert captured["code"] == "plugin.python_missing"

    other = _other_python()
    plugin = ctx.plugins.update("stdlib", PluginPatch(python_path=other))
    assert plugin.status == "ready" and plugin.python_path == other and plugin.check.ok
    out = tmp_path / "copy.wav"
    job, _ = run_capturing(ctx, "stdlib", out, vocals=[], backing=None, input=ensure_testdata(ctx.options.data)[1])
    assert job.status == "succeeded", (job.error, list(job.log))
    log = "\n".join(job.log)
    assert f"interpreter {other}" in log or f"interpreter {os.path.realpath(other)}" in log, log
    assert sf.info(str(out)).duration == pytest.approx(8.0, abs=0.01)
    # back to the builtin env
    plugin = ctx.plugins.update("stdlib", PluginPatch(interpreter="builtin"))
    assert plugin.status == "ready" and Path(plugin.check.executable) == venv_python(env_dir(ctx.options.data, "stdlib"))


def test_registry_persistence(ctx, tmp_path):
    folder = friend_folder(tmp_path)
    plugin = ctx.plugins.add_wizard(WizardSpec(folder=str(folder), name="Persist Me", entry="auto_mixing.py"))
    ctx.plugins.update(plugin.id, PluginPatch(enabled=False, name="改名", interpreter="custom",
                                              python_path=sys.executable))
    fresh = PluginRegistry(ctx)
    again = fresh.get(plugin.id)
    assert again.enabled is False and again.name == "改名" and again.interpreter == "custom"
    assert again.python_path == sys.executable and again.check is not None and again.check.ok
    assert again.added_at == plugin.added_at and again.manifest_source == "wizard"
    data = json.loads((ctx.options.data / "plugins" / "registry.json").read_text(encoding="utf-8"))
    assert data["plugins"][0]["id"] == plugin.id and data["plugins"][0]["pythonPath"] == sys.executable
    # a broken manifest keeps the plugin listed with an error
    (ctx.options.data / "plugins" / f"{plugin.id}.json").write_text("{", encoding="utf-8")
    broken = fresh.get(plugin.id)
    assert broken.status == "error" and broken.error.code == "plugin.bad_manifest" and broken.name == "改名"


def test_provided_by_runtime():
    versions = {"numpy": "1.26.4", "torch": "2.9.1+cu128"}
    assert provided_by_runtime("numpy==1.26.4", versions)
    assert provided_by_runtime("torch==2.9.1", versions)
    assert not provided_by_runtime("numpy==2.0.0", versions)
    assert not provided_by_runtime("pedalboard==0.9.25", versions)
    assert not provided_by_runtime("pkg @ https://example.com/pkg.whl", versions)


def _wheel(index: Path, name: str, version: str, requires: list[str]) -> None:
    dist = name.replace("-", "_")
    folder = index / name
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"{dist}-{version}-py3-none-any.whl"
    info = f"{dist}-{version}.dist-info"
    files = {
        f"{dist}/__init__.py": f"VALUE = {version!r}\n",
        f"{info}/METADATA": "Metadata-Version: 2.1\nName: %s\nVersion: %s\n%s" % (
            name, version, "".join(f"Requires-Dist: {r}\n" for r in requires)),
        f"{info}/WHEEL": "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{info}/RECORD"] = "".join(f"{k},,\n" for k in files) + f"{info}/RECORD,,\n"
    with zipfile.ZipFile(folder / filename, "w") as zf:
        for path, text in files.items():
            zf.writestr(path, text)
    (folder / "index.html").write_text(f'<html><body><a href="{filename}">{filename}</a></body></html>')


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
def test_install_requirements_into_builtin_env_offline(client, tmp_path, monkeypatch):
    ctx = client.ctx
    index = tmp_path / "index" / "simple"
    soundfile_version = metadata.version("soundfile")
    _wheel(index, "stellaria-dummy", "1.0", ["soundfile"])
    _wheel(index, "soundfile", soundfile_version, [])  # resolvable, but provided by the runtime → not installed
    (index / "index.html").write_text("<html></html>")
    monkeypatch.setattr(ctx.resources.manifest, "pypi", [Source(id="local", region="any", url=index.as_uri())])
    monkeypatch.setattr(ctx.resources, "ordered", lambda sources, preferred=None: list(sources))
    folder = write_plugin(tmp_path / "dummy", {"entry.py": "import stellaria_dummy\ndef process(job):\n    pass\n"}, {
        "id": "dummy", "name": "Dummy", "kind": "effect", "runtime": "python", "entry": "entry.py",
        "requirements": ["stellaria-dummy", "soundfile"]})
    plugin = client.post("/api/plugins", json={"folder": str(folder)}).json()
    assert plugin["status"] == "needs_setup" and plugin["check"]["missing"] == ["stellaria-dummy"]
    assert plugin["error"] == {"code": "plugin.packages_missing", "params": {"packages": "stellaria-dummy"}}
    assert ctx.plugins.mix_processors() == []  # an effect, not a mix processor
    job = client.post("/api/plugins/dummy/install-requirements", json={"target": "builtin"}).json()
    assert job["type"] == "install_plugin" and job["target"] == {"pluginId": "dummy"}
    job = wait_job(client, job["id"], timeout=120)
    log = client.get(f"/api/jobs/{job['id']}/log").json()["lines"]
    assert job["status"] == "succeeded", (job, log)
    assert job["result"]["installed"] == ["stellaria-dummy==1.0"] and job["result"]["status"] == "ready"
    plugin = client.get("/api/plugins/dummy").json()
    assert plugin["status"] == "ready" and plugin["check"]["missing"] == []
    site = venv_python(env_dir(ctx.options.data, "dummy")).parent.parent
    assert list(site.rglob("stellaria_dummy/__init__.py")) and not list(site.rglob("soundfile.py"))


# ---- API -----------------------------------------------------------------------------------
def test_plugins_api(client, tmp_path):
    ctx = client.ctx
    folder = friend_folder(tmp_path)
    r = client.post("/api/plugins", json={"folder": str(folder)})
    assert r.status_code == 409 and r.json()["error"]["code"] == "plugin.no_manifest"
    assert r.json()["error"]["detail"]["candidates"][0] == "auto_mixing.py"
    assert client.post("/api/plugins", json={"folder": "relative/path"}).json()["error"]["code"] == "request.relative_path"
    assert client.post("/api/plugins", json={"folder": str(tmp_path / "gone")}).status_code == 404

    body = {"folder": str(folder), "name": "Sodium 自动混音", "kind": "mix", "entry": "auto_mixing.py",
            "args": ["{backing}", "{vocal}"], "outputs": ["output/*.flac", "output/*.wav"], "cwd": "temp",
            "requirements": ["numpy", "soundfile"]}
    plugin = client.post("/api/plugins/wizard", json=body).json()
    assert plugin["id"] == "sodium" and plugin["status"] == "ready" and plugin["kind"] == "mix"
    assert set(plugin) >= {"id", "name", "version", "kind", "runtime", "folder", "manifestSource", "interpreter",
                           "pythonPath", "status", "check", "params", "settings", "settingValues", "enabled", "addedAt"}
    assert plugin["check"]["executable"] and plugin["check"]["torch"] is None  # torch only probed when required
    assert [p["id"] for p in client.get("/api/plugins").json()] == ["sodium"]
    assert ctx.plugins.mix_processors() == [{"id": "sodium", "name": "Sodium 自动混音", "kind": "plugin", "presets": [],
                                             "params": [], "maxVocals": 1, "available": True, "error": None}]

    bad = write_plugin(tmp_path / "bad", {"x.py": ""}, {"id": "bad", "name": "Bad", "command": ["{python}", "{wat}"]})
    r = client.post("/api/plugins", json={"folder": str(bad)})
    assert r.status_code == 422 and r.json()["error"]["code"] == "plugin.bad_manifest" and "{wat}" in r.json()["error"]["message"]

    good = write_plugin(tmp_path / "good", {"mix.py": "print(1)\n"}, command_manifest("good", "mix.py"))
    assert client.post("/api/plugins", json={"folder": str(good)}).json()["id"] == "good"
    assert client.post("/api/plugins", json={"folder": str(good)}).json()["id"] == "good"  # same folder: idempotent
    other = write_plugin(tmp_path / "other", {"mix.py": "print(1)\n"}, command_manifest("good", "mix.py"))
    r = client.post("/api/plugins", json={"folder": str(other)})
    assert r.status_code == 409 and r.json()["error"]["code"] == "plugin.exists"

    r = client.patch("/api/plugins/sodium", json={"enabled": False, "name": "混音 2"})
    assert r.json()["enabled"] is False and r.json()["name"] == "混音 2"
    assert [p["id"] for p in ctx.plugins.mix_processors()] == ["good"]
    client.patch("/api/plugins/sodium", json={"enabled": True})
    checked = client.post("/api/plugins/sodium/check").json()
    assert checked["status"] == "ready" and checked["checkedAt"]
    assert client.get("/api/plugins/nope").json()["error"]["code"] == "plugin.not_found"
    assert client.post("/api/plugins/nope/check").status_code == 404

    job = client.post("/api/plugins/sodium/test").json()
    assert job["type"] == "plugin_test" and job["target"] == {"pluginId": "sodium"}
    job = wait_job(client, job["id"])
    assert job["status"] == "succeeded", client.get(f"/api/jobs/{job['id']}/log").json()
    url = job["result"]["mediaUrl"]
    assert url == f"/plugins/sodium/test/{job['id']}/audio" and job["result"]["duration"] == pytest.approx(8.0, abs=0.01)
    audio = client.get("/api" + url)
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav" and audio.content[:4] == b"RIFF"
    assert client.get(f"/api/plugins/sodium/test/{job['id']}/peaks?n=50").json()["duration"] == pytest.approx(8.0, abs=0.01)
    assert client.get("/api/plugins/sodium/test/job_nope/audio").status_code == 404

    job = client.post("/api/plugins/sodium/install-requirements", json={"target": "builtin"}).json()
    job = wait_job(client, job["id"])
    assert job["status"] == "succeeded" and job["result"]["installed"] == []  # all provided by the app runtime
    r = client.post("/api/plugins/sodium/install-requirements", json={"target": "custom"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "plugin.python_missing"

    check = client.post("/api/python/check", json={"path": sys.executable, "packages": ["numpy", "nope-nope-nope"]}).json()
    assert check["missing"] == ["nope-nope-nope"] and check["version"] and check["error"] is None
    assert client.post("/api/python/check", json={"path": str(tmp_path / "x")}).json()["error"]["code"] == "python.not_found"

    envdir = env_dir(ctx.options.data, "sodium")
    assert envdir.is_dir()
    assert client.delete("/api/plugins/sodium").json() == {"ok": True}
    assert not envdir.exists() and not (ctx.options.data / "plugins" / "sodium.json").exists()
    assert (folder / "auto_mixing.py").is_file()  # never the user's folder
    assert [p["id"] for p in client.get("/api/plugins").json()] == ["good"]
    assert client.delete("/api/plugins/sodium").status_code == 404
