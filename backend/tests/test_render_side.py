"""Render worker caching (no torch needed) and safe snapshots of checkpoints being written."""

from __future__ import annotations

import os
import threading
import time

from stellaria_backend.engine.checkpoints import snapshot_latest
from stellaria_backend.engine.render_worker import BadCheckpoint, DeviceCache, classify


def test_cpu_requests_do_not_evict_gpu_models():
    cache, built = DeviceCache(), []

    def build(tag):
        built.append(tag)
        return tag

    assert cache.get("cuda", "model", "k1", lambda: build("cuda-k1")) == "cuda-k1"
    assert cache.get("cpu", "model", "k1", lambda: build("cpu-k1")) == "cpu-k1"
    assert cache.get("cuda", "model", "k1", lambda: build("again")) == "cuda-k1"
    cache.get("cpu", "model", "k2", lambda: build("cpu-k2"))  # another model on the CPU side only
    assert cache.get("cuda", "model", "k1", lambda: build("again")) == "cuda-k1"
    assert built == ["cuda-k1", "cpu-k1", "cpu-k2"]


def test_idle_cpu_models_are_released():
    cache = DeviceCache()
    cache.get("cuda", "model", "k", lambda: "gpu")
    cache.get("cpu", "encoder", "k", lambda: "cpu")
    now = time.monotonic()
    assert not cache.release_idle("cpu", 120, now=now + 60)
    assert cache.release_idle("cpu", 120, now=now + 121)
    assert cache.devices() == {"cuda"}


def test_unreadable_checkpoint_has_its_own_code():
    assert classify(BadCheckpoint("model_10.pt"))[0] == "render.bad_checkpoint"


def _experiment(tmp_path):
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    old = exp / "model_100.pt"
    old.write_bytes(b"o" * 1000)
    os.utime(old, (time.time() - 30, time.time() - 30))
    return exp


def test_snapshot_waits_for_the_checkpoint_being_written(tmp_path):
    exp = _experiment(tmp_path)
    growing = exp / "model_200.pt"

    def writer():
        with open(growing, "wb") as f:
            for _ in range(6):
                f.write(b"n" * 1000)
                f.flush()
                time.sleep(0.1)

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.05)
    snap = snapshot_latest(exp, tmp_path / "snap", settle=0.4, timeout=5)
    thread.join()
    assert snap.name == "model_200.pt" and snap.read_bytes() == b"n" * 6000
    assert (tmp_path / "snap" / "config.yaml").is_file()
    older = snapshot_latest(exp, tmp_path / "snap", skip={200}, settle=0.4)
    assert older.name == "model_100.pt" and not (tmp_path / "snap" / "model_200.pt").exists()


def test_snapshot_falls_back_when_the_newest_never_settles(tmp_path):
    exp = _experiment(tmp_path)
    stop = threading.Event()

    def writer():
        with open(exp / "model_200.pt", "wb") as f:
            while not stop.is_set():
                f.write(b"n")
                f.flush()
                time.sleep(0.05)

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        snap = snapshot_latest(exp, tmp_path / "snap", settle=0.4, timeout=1.0)
    finally:
        stop.set()
        thread.join()
    assert snap.name == "model_100.pt"
