"""Organize pipeline on synthetic audio: decode (wav/mp3/CJK names) → slice → split → drop → name → health."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from conftest import wait_job
from synth import note, song, write


def _import(client, cid, paths):
    job = client.post(f"/api/characters/{cid}/raw", json={"paths": [str(p) for p in paths]}).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "succeeded", done
    return done["result"]["rawIds"]


def test_organize_pipeline(client, tmp_path):
    src = tmp_path / "素材 in"
    files = [
        write(src / "歌声 测试.wav", song(40, 220, seed=11)),
        write(src / "mp3 song.mp3", song(30, 330, seed=12)),
        write(src / "short.wav", note(1.0, 300)),  # below minDuration → dropped
        write(src / "clipped.wav", np.clip(song(12, 260, seed=13) * 6.0, -1.0, 1.0)),
    ]
    cid = client.post("/api/characters", json={"name": "小星"}).json()["id"]
    raw_ids = _import(client, cid, files)
    raws = client.get(f"/api/characters/{cid}/raw").json()
    assert [r["status"] for r in raws] == ["imported"] * 4
    assert raws[1]["ext"] == "mp3" and abs(raws[1]["duration"] - 30) < 0.2

    defaults = client.get(f"/api/characters/{cid}/organize/defaults").json()
    assert defaults["slice"]["enabled"] is True and defaults["minDuration"] == 2.0
    job = client.post(f"/api/characters/{cid}/organize", json={"maxLength": 10, "dropClipping": True}).json()
    done = wait_job(client, job["id"], timeout=120)
    assert done["status"] == "succeeded", done
    result = done["result"]
    assert result["dropped"]["tooShort"] >= 1
    assert result["dropped"]["clipping"] >= 1
    assert set(result["rawIds"]) == set(raw_ids)

    clips = client.get(f"/api/characters/{cid}/clips").json()
    assert len(clips) >= 6
    for c in clips:
        assert 2.0 <= c["duration"] <= 10.0 + 1e-6
        assert c["name"].startswith("小星_") and c["file"] == f"clips/{c['name']}.wav"
        assert c["metrics"] is not None and c["metrics"]["f0Median"] is not None
        data, sr = sf.read(client.ctx.characters.dir(cid) / c["file"])
        assert sr == 44100 and data.ndim == 1
        assert np.max(np.abs(data)) <= 10 ** (-1 / 20) + 1e-3  # peak safety
    names = [c["name"] for c in clips]
    assert len(set(names)) == len(names)
    assert {c["rawId"] for c in clips} <= set(raw_ids)
    # the mp3 voice is higher than the CJK-named one
    by_raw = {}
    for c in clips:
        by_raw.setdefault(c["rawName"], []).append(c["metrics"]["f0Median"])
    assert np.median(by_raw["mp3 song"]) > np.median(by_raw["歌声 测试"])

    val = [c for c in clips if c["split"] == "val"]
    assert len(val) == 2 and all(not c["flags"] for c in val)
    report = client.get(f"/api/characters/{cid}/health").json()
    assert report["checks"]["basic"] and report["checks"]["pitch"]
    codes = {i["code"] for i in report["issues"]}
    assert "data.too_little" in codes and report["level"] == "blocked"
    assert "checks.music_unavailable" in codes
    character = client.get(f"/api/characters/{cid}").json()
    assert character["stage"] == "material" and character["stats"]["clipCount"] == len(clips)
    assert character["health"]["stale"] is False and character["pitch"]["median"] > 0
    raws = client.get(f"/api/characters/{cid}/raw").json()
    assert {r["status"] for r in raws} == {"organized"}


def test_organize_without_slicing_keeps_files(client, tmp_path):
    src = tmp_path / "cut"
    files = [write(src / f"take {i}.wav", song(5.5, 200 + 20 * i, seed=i, gap=0.2)) for i in range(3)]
    cid = client.post("/api/characters", json={"name": "Cut"}).json()["id"]
    _import(client, cid, files)
    job = client.post(f"/api/characters/{cid}/organize",
                      json={"slice": {"enabled": False}, "normalize": False, "namePattern": "{name}-{index}"}).json()
    assert wait_job(client, job["id"], timeout=60)["status"] == "succeeded"
    clips = client.get(f"/api/characters/{cid}/clips").json()
    assert sorted(c["name"] for c in clips) == ["Cut-1", "Cut-2", "Cut-3"]
    assert all(abs(c["duration"] - 5.5) < 0.01 for c in clips)


def test_decode_failure_is_reported(client, tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"not audio at all" * 100)
    cid = client.post("/api/characters", json={"name": "Bad"}).json()["id"]
    _import(client, cid, [bad])
    raw = client.get(f"/api/characters/{cid}/raw").json()[0]
    assert raw["status"] == "failed" and raw["error"]["code"] == "audio.decode_failed"


def test_bad_pattern_rejected(client, tmp_path):
    cid = client.post("/api/characters", json={"name": "P"}).json()["id"]
    job = client.post(f"/api/characters/{cid}/organize", json={"namePattern": "{nope}"}).json()
    done = wait_job(client, job["id"])
    assert done["status"] == "failed" and done["error"]["code"] == "organize.bad_pattern"


def test_slicing_preview_matches_organize(client, tmp_path):
    import time

    # a lone 1 s note, 6 s of silence, then a 233 s song: 240 s in total
    src = write(tmp_path / "long take.mp3", np.concatenate([note(1.0, 300), np.zeros(44100 * 6, dtype=np.float32), song(233, 220, seed=31)]))
    cid = client.post("/api/characters", json={"name": "Pre"}).json()["id"]
    raw_id = _import(client, cid, [src])[0]
    body = {"rawId": raw_id, "slice": {"enabled": True, "thresholdDb": -40, "minLength": 5, "minInterval": 0.3,
                                       "hopSize": 0.01, "maxSilKept": 0.5}, "maxLength": 10, "n": 300}
    first = client.post(f"/api/characters/{cid}/organize/preview", json=body)
    assert first.status_code == 200, first.text
    started = time.perf_counter()
    prev = client.post(f"/api/characters/{cid}/organize/preview", json=body).json()  # decoded audio is cached now
    elapsed = time.perf_counter() - started
    assert elapsed < 3.0, elapsed  # target ≈ 1.5 s for 4 minutes; leave room for slow CI machines
    assert abs(prev["duration"] - 240.0) < 0.1 and len(prev["peaks"]) == 300 and max(prev["peaks"]) <= 1
    segs = prev["segments"]
    assert segs and all(0 <= s["start"] < s["end"] <= prev["duration"] + 1e-6 for s in segs)
    assert all(s["end"] - s["start"] <= 10.0 + 1e-3 for s in segs)
    assert {s["dropped"] for s in segs} <= {None, "too_short", "silent", "clipping"}
    assert any(s["dropped"] == "too_short" for s in segs)

    job = client.post(f"/api/characters/{cid}/organize", json={"maxLength": 10, "dedupe": False}).json()
    assert wait_job(client, job["id"], timeout=180)["status"] == "succeeded"
    clips = client.get(f"/api/characters/{cid}/clips").json()
    kept = [(s["start"], s["end"]) for s in segs if s["dropped"] is None]
    assert sorted((c["start"], c["end"]) for c in clips) == kept  # the preview predicted organize exactly

    off = client.post(f"/api/characters/{cid}/organize/preview", json={**body, "slice": {"enabled": False}, "maxLength": 15}).json()
    assert all(s["end"] - s["start"] <= 15.0 + 1e-3 for s in off["segments"]) and off["segments"][0]["start"] == 0
    missing = client.post(f"/api/characters/{cid}/organize/preview", json={**body, "rawId": "raw_nope"})
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "raw_not_found"
