"""Shared fixtures: an app context on a library path with CJK characters and spaces."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
ENGINE = ROOT / "engine"
MANIFEST = ROOT / "resources" / "manifest.json"
sys.path.insert(0, str(BACKEND))
sys.path.insert(0, str(Path(__file__).parent))

from stellaria_backend.context import AppContext, Options  # noqa: E402
from stellaria_backend.schemas import TorchInfo  # noqa: E402

TOKEN = "test-token"


def make_options(root: Path, fake: bool = True, **kw) -> Options:
    return Options(library=root / "星籁 库", data=root / "数据 data", engine=ENGINE, resources=kw.pop("resources", MANIFEST),
                   token=TOKEN, dev=False, fake_engine=fake, fake_speed=kw.pop("fake_speed", 400.0), **kw)


def make_ctx(root: Path, fake: bool = True, torch: TorchInfo | None = None, **kw) -> AppContext:
    ctx = AppContext(make_options(root, fake, **kw))
    ctx.system.set_torch(torch or TorchInfo(available=True, version="2.9.1", device="cpu"))
    return ctx


@pytest.fixture
def ctx(tmp_path: Path):
    c = make_ctx(tmp_path)
    yield c
    c.stop(timeout=5)


@pytest.fixture
def client(tmp_path: Path):
    from fastapi.testclient import TestClient

    from stellaria_backend.app import create_app

    c = make_ctx(tmp_path)
    app = create_app(c.options, ctx=c)
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as tc:
        tc.ctx = c  # type: ignore[attr-defined]
        yield tc


def wait_job(client, job_id: str, timeout: float = 60.0, statuses=("succeeded", "failed", "cancelled")) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def wait_for(predicate, timeout: float = 30.0, interval: float = 0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition not met in time")


@pytest.fixture(scope="session")
def material_dir(tmp_path_factory) -> Path:
    """~6 minutes of synthetic singing in four files (wav with a CJK name, mp3, flac, wav)."""
    from synth import song, write

    d = tmp_path_factory.mktemp("material")
    write(d / "歌声 一.wav", song(92, 220, seed=1))
    write(d / "song two.mp3", song(92, 247, seed=2))
    write(d / "song3.flac", song(92, 196, seed=3))
    write(d / "song4.wav", song(92, 262, seed=4))
    return d


