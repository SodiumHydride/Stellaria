"""Downloads: mirror fallback, HTTP Range resume, sha256 verification, zip install, region order."""

from __future__ import annotations

import hashlib
import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from conftest import MANIFEST, TOKEN, wait_job
from stellaria_backend.errors import ApiError
from stellaria_backend.resources.downloader import download, order_sources
from stellaria_backend.resources.manifest import Source

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB
SHA = hashlib.sha256(PAYLOAD).hexdigest()


class Server:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[tuple[str, str | None]] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:  # quiet
                pass

            def do_HEAD(self) -> None:
                self.send_response(200)
                self.end_headers()

            def do_GET(self) -> None:
                rng = self.headers.get("Range")
                server.requests.append((self.path, rng))
                name = self.path.split("?")[0].lstrip("/")
                if name.startswith("missing"):
                    self.send_error(404)
                    return
                data = server.files[name.split("/", 1)[-1]]
                if name.startswith("corrupt/"):
                    data = data[:-10] + b"x" * 10
                cut = name.startswith("flaky/")
                start = int(rng.split("=")[1].split("-")[0]) if rng else 0
                body = data[start:]
                self.send_response(206 if rng else 200)
                if rng:
                    self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if cut:  # drop the connection half way through
                    self.wfile.write(body[: len(body) // 2])
                    self.wfile.flush()
                    self.connection.close()
                    return
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.httpd.shutdown()


@pytest.fixture
def server():
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("pack/model.pt", PAYLOAD[:1000])
        zf.writestr("pack/config.json", json.dumps({"ok": True}))
        zf.writestr("pack/unused.txt", "x")
    s = Server({"model.bin": PAYLOAD, "bundle.zip": archive.getvalue()})
    yield s
    s.close()


def test_fallback_to_next_mirror(server, tmp_path):
    dest = tmp_path / "模型" / "model.bin"
    used = download([Source("a", "global", f"{server.url}/missing/model.bin"),
                     Source("b", "cn", f"{server.url}/good/model.bin")], dest, size=len(PAYLOAD), sha256=SHA)
    assert used.id == "b" and dest.read_bytes() == PAYLOAD and not dest.with_name("model.bin.part").exists()


def test_resume_with_range(server, tmp_path):
    dest = tmp_path / "model.bin"
    part = dest.with_name("model.bin.part")
    part.write_bytes(PAYLOAD[:300_000])
    progress = []
    download([Source("a", "global", f"{server.url}/good/model.bin")], dest, size=len(PAYLOAD), sha256=SHA,
             on_progress=progress.append)
    assert dest.read_bytes() == PAYLOAD
    assert server.requests[-1] == ("/good/model.bin", "bytes=300000-")
    assert progress[-1].bytes_done == len(PAYLOAD) and progress[-1].total == len(PAYLOAD)


def test_interrupted_transfer_resumes_on_next_source(server, tmp_path):
    dest = tmp_path / "model.bin"
    used = download([Source("flaky", "global", f"{server.url}/flaky/model.bin"),
                     Source("good", "global", f"{server.url}/good/model.bin")], dest, size=len(PAYLOAD), sha256=SHA)
    assert used.id == "good" and dest.read_bytes() == PAYLOAD
    good = [r for r in server.requests if r[0] == "/good/model.bin"]
    assert good[0][1] is not None and good[0][1].startswith("bytes=")  # continued, not restarted


def test_sha256_mismatch_falls_over(server, tmp_path):
    dest = tmp_path / "model.bin"
    used = download([Source("bad", "global", f"{server.url}/corrupt/model.bin"),
                     Source("good", "global", f"{server.url}/good/model.bin")], dest, sha256=SHA)
    assert used.id == "good" and dest.read_bytes() == PAYLOAD
    with pytest.raises(ApiError) as err:
        download([Source("bad", "global", f"{server.url}/corrupt/model.bin")], tmp_path / "x.bin", sha256=SHA)
    assert err.value.code == "download.failed" and "sha256" in err.value.detail


def test_region_ordering():
    srcs = [Source("gh", "global", "https://a"), Source("cn1", "cn", "https://b"), Source("any", "any", "https://c")]
    assert [s.id for s in order_sources(srcs, "cn")] == ["cn1", "any", "gh"]
    assert [s.id for s in order_sources(srcs, "global")] == ["gh", "any", "cn1"]
    assert [s.id for s in order_sources(srcs, "auto", {"gh": 300.0, "cn1": 20.0, "any": None})] == ["cn1", "gh", "any"]


def test_resource_install_through_api(server, tmp_path):
    from fastapi.testclient import TestClient

    from conftest import make_ctx
    from stellaria_backend.app import create_app

    manifest = json.loads((MANIFEST).read_text(encoding="utf-8"))
    manifest["resources"] = [
        {"id": "nsf_hifigan", "kind": "model", "required": True, "name": {"en": "Vocoder"}, "description": {"en": "d"},
         "size": None, "sha256": None,
         "install": {"type": "zip", "dest": "nsf_hifigan", "files": {"pack/model.pt": "model.ckpt", "pack/config.json": "config.json"}},
         "sources": [{"id": "dead", "region": "global", "url": f"{server.url}/missing/bundle.zip"},
                     {"id": "mirror", "region": "cn", "url": f"{server.url}/good/bundle.zip"}]},
        {"id": "hubertsoft", "kind": "model", "required": False, "name": {"en": "H"}, "description": {"en": "d"},
         "size": len(PAYLOAD), "sha256": SHA, "install": {"type": "file", "dest": "hubertsoft/h.pt"},
         "sources": [{"id": "good", "region": "global", "url": f"{server.url}/good/model.bin"}]},
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    ctx = make_ctx(tmp_path, fake=False, resources=path)
    ctx.prefs.update({"mirrorRegion": "cn"})
    with TestClient(create_app(ctx.options, ctx=ctx), headers={"Authorization": f"Bearer {TOKEN}"}) as client:
        resources = {r["id"]: r for r in client.get("/api/resources").json()}
        assert resources["nsf_hifigan"]["status"] == "missing" and resources["nsf_hifigan"]["required"]
        assert resources["nsf_hifigan"]["name"] == {"code": "resource.nsf_hifigan.name", "params": {"en": "Vocoder"}}
        jobs = client.post("/api/resources/install-required").json()
        assert [j["type"] for j in jobs] == ["download"]
        done = wait_job(client, jobs[0]["id"])
        assert done["status"] == "succeeded", done
        vocoder = ctx.options.data / "models" / "nsf_hifigan"
        assert sorted(p.name for p in vocoder.iterdir()) == ["config.json", "model.ckpt"]
        assert (vocoder / "model.ckpt").read_bytes() == PAYLOAD[:1000]
        assert server.requests[0][0] == "/good/bundle.zip"  # cn region tries the cn mirror first
        r = next(r for r in client.get("/api/resources").json() if r["id"] == "nsf_hifigan")
        assert r["status"] == "ready" and r["path"].endswith("model.ckpt")
        done = wait_job(client, client.post("/api/resources/hubertsoft/install", json={}).json()["id"])
        assert done["status"] == "succeeded" and Path(done["result"]["path"]).read_bytes() == PAYLOAD
        assert client.delete("/api/resources/hubertsoft").json()["status"] == "missing"
        info = client.get("/api/resources/sources").json()
        assert info["region"] == "cn" and {m["sourceId"] for m in info["measured"]} >= {"dead", "mirror", "good"}
