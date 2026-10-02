"""The process contract the shell relies on: one READY line on stdout, token auth, clean shutdown."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

from conftest import BACKEND, ENGINE, MANIFEST


def start(tmp_path, *extra: str) -> tuple[subprocess.Popen, dict]:
    env = {**os.environ, "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "stellaria_backend", "--library", str(tmp_path / "库 lib"), "--data", str(tmp_path / "data"),
         "--engine", str(ENGINE), "--resources", str(MANIFEST), "--port", "0", "--fake-engine", *extra],
        cwd=BACKEND, env=env, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    line = proc.stdout.readline().strip()
    assert line.startswith("STELLARIA_READY "), line
    return proc, json.loads(line.split(" ", 1)[1])


def test_ready_line_auth_and_shutdown(tmp_path):
    proc, info = start(tmp_path)
    try:
        assert info["port"] > 0 and len(info["token"]) >= 20 and info["version"] == "0.1.0"
        base = f"http://127.0.0.1:{info['port']}/api"
        assert httpx.get(f"{base}/health", timeout=10).json()["ok"] is True
        assert httpx.get(f"{base}/system", timeout=10).status_code == 401
        headers = {"Authorization": f"Bearer {info['token']}"}
        assert httpx.post(f"{base}/shutdown", headers=headers, timeout=10).json() == {"ok": True}
        assert proc.wait(30) == 0
        assert proc.stdout.read() == ""  # nothing but the READY line on stdout
        assert (tmp_path / "data" / "logs" / "backend.log").is_file()
    finally:
        if proc.poll() is None:
            proc.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal")
def test_sigterm_runs_graceful_shutdown(tmp_path):
    proc, _ = start(tmp_path, "--token", "fixed")
    proc.send_signal(signal.SIGTERM)
    started = time.time()
    proc.wait(30)
    assert time.time() - started < 25
    log = (tmp_path / "data" / "logs" / "backend.log").read_text(encoding="utf-8")
    assert "pausing trainings" in log and "Application shutdown complete" in log
