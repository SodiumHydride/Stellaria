"""Timeline mixdown: clip mute/fades/offsets, splits, master gain, ranges, stems, and the export route."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import wait_job
from stellaria_backend.engine.mixdown import SR, render_stems, render_timeline

LEN = 4.0


@pytest.fixture
def media(tmp_path: Path):
    """``dc``: a constant 0.5 (mono); ``ramp``: sample i = i / n (stereo, both channels)."""
    n = int(LEN * SR)
    sf.write(str(tmp_path / "dc.wav"), np.full(n, 0.5, dtype=np.float32), SR, subtype="FLOAT")
    ramp = np.arange(n, dtype=np.float32) / n
    sf.write(str(tmp_path / "ramp.wav"), np.stack([ramp, ramp], axis=1), SR, subtype="FLOAT")
    return lambda mid: tmp_path / f"{mid}.wav"


def clip(cid: str, mid: str = "dc", start: float = 0.0, offset: float = 0.0, duration: float = 2.0, **kw) -> dict:
    return {"id": cid, "mediaId": mid, "nodeId": None, "start": start, "offset": offset, "duration": duration,
            "gainDb": 0, "fadeIn": 0, "fadeOut": 0, **kw}


def track(tid: str, *clips: dict, **kw) -> dict:
    return {"id": tid, "name": tid, "kind": "other", "gainDb": 0, "pan": 0, "mute": False, "solo": False,
            "clips": list(clips), **kw}


def timeline(*tracks: dict, **kw) -> dict:
    return {"tracks": list(tracks), "length": 0, **kw}


def at(x: np.ndarray, seconds: float) -> np.ndarray:
    return x[int(round(seconds * SR))]


def test_offsets_fades_and_declick(media):
    tl = timeline(track("a", clip("c", "ramp", start=1.0, offset=0.5, duration=2.0, fadeIn=0.2, fadeOut=0.4)))
    out = render_timeline(tl, media)
    assert out.shape == (int(3.0 * SR), 2)
    assert np.all(out[: SR] == 0)
    # 1 s into the clip: the file 1.5 s in, no fade
    assert at(out, 2.0)[0] == pytest.approx(1.5 / LEN, abs=1e-4)
    # halfway through the fade-in / fade-out
    assert at(out, 1.1)[0] == pytest.approx(0.5 * 0.6 / LEN, abs=1e-3)
    assert at(out, 2.8)[0] == pytest.approx(0.5 * 2.3 / LEN, abs=1e-3)
    # no fades: a 2 ms ramp where the clip cuts into its audio
    tl = timeline(track("a", clip("c", "dc", start=1.0, duration=1.0)))
    out = render_timeline(tl, media)
    assert at(out, 1.0)[0] == 0 and at(out, 1.001)[0] == pytest.approx(0.25, abs=0.01) and at(out, 1.01)[0] == pytest.approx(0.5)


def test_split_clip_plays_as_one(media):
    whole = render_timeline(timeline(track("a", clip("c", "ramp", start=0.5, offset=0.25, duration=3.0))), media)
    halves = timeline(track("a", clip("l", "ramp", start=0.5, offset=0.25, duration=1.2),
                            clip("r", "ramp", start=1.7, offset=1.45, duration=1.8)))
    assert np.allclose(render_timeline(halves, media), whole, atol=1e-6)


def test_clip_and_track_mute_solo_master_gain(media):
    a = track("a", clip("c1", start=0.0, duration=1.0), clip("c2", start=1.0, duration=1.0, mute=True))
    b = track("b", clip("c3", start=0.0, duration=2.0), gainDb=-60)          # −∞ on the fader
    out = render_timeline(timeline(a, b), media)
    assert at(out, 0.5)[0] == pytest.approx(0.5) and np.all(out[int(1.01 * SR):] == 0)
    out = render_timeline(timeline(a, b, masterGainDb=-6.0206), media)
    assert at(out, 0.5)[0] == pytest.approx(0.25, abs=1e-4)
    loud = track("c", clip("c4", start=0.0, duration=2.0), solo=True, pan=-1)
    out = render_timeline(timeline(a, loud), media)
    assert at(out, 1.5).tolist() == pytest.approx([0.5, 0.0])
    assert not render_timeline(timeline(track("m", clip("c5"), mute=True)), media).any()


def test_range_cuts_like_the_whole_mix(media):
    tl = timeline(track("a", clip("c", "ramp", start=1.0, offset=0.5, duration=2.5, fadeIn=0.6, fadeOut=0.5)),
                  track("b", clip("d", start=0.2, duration=1.0)))
    whole = render_timeline(tl, media)
    part = render_timeline(tl, media, start=1.3, end=3.2)
    assert len(part) == int(round(1.9 * SR))
    i = int(round(1.3 * SR))
    assert np.allclose(part, whole[i:i + len(part)], atol=1e-5)          # starts inside the fade-in, keeps it
    # a range past the end is silence there; an empty range is empty
    assert len(render_timeline(tl, media, start=3.0, end=5.0)) == 2 * SR
    assert len(render_timeline(tl, media, start=2.0, end=2.0)) == 0


def test_stems_add_up_to_the_mix(media):
    tl = timeline(track("a", clip("c1", start=0.0, duration=2.0), pan=0.5, gainDb=-3),
                  track("b", clip("c2", "ramp", start=1.0, duration=2.0)),
                  track("muted", clip("c3"), mute=True),
                  track("empty"),
                  masterGainDb=-2)
    stems = render_stems(tl, media)
    assert [t["id"] for t, _ in stems] == ["a", "b"]
    assert np.allclose(sum(x for _, x in stems), render_timeline(tl, media), atol=1e-6)
    ranged = render_stems(tl, media, start=2.5, end=3.0)
    assert [t["id"] for t, _ in ranged] == ["b"]


def test_normalize_peak(media):
    out = render_timeline(timeline(track("a", clip("c", duration=1.0))), media, normalize=True)
    assert float(np.max(np.abs(out))) == pytest.approx(10 ** (-1 / 20), abs=1e-5)


def _project_with_audio(client, tmp_path: Path) -> tuple[str, dict]:
    p = client.post("/api/projects", json={"name": "Mix"}).json()
    t = np.arange(int(3 * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    path = tmp_path / "tone.wav"
    sf.write(str(path), np.stack([tone, tone], axis=1), SR)
    done = wait_job(client, client.post(f"/api/projects/{p['id']}/media", json={"paths": [str(path)]}).json()["id"])
    mid = done["result"]["mediaIds"][0]
    tl = {"tracks": [track("t1", clip("c1", mid, start=0.0, duration=3.0)) | {"name": "Lead"},
                     track("t2", clip("c2", mid, start=1.0, duration=2.0)) | {"name": "Harmony"},
                     track("t3", clip("c3", mid, start=0.0, duration=1.0), mute=True)],
          "length": 3.0, "masterGainDb": 0}
    client.put(f"/api/projects/{p['id']}", json={"timeline": tl})
    return p["id"], tl


def test_mixdown_route_formats_range_loudness(client, tmp_path):
    pid, _ = _project_with_audio(client, tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    body = {"format": "flac", "path": str(out), "sampleRate": 48000, "bitDepth": 16, "start": 0.5, "end": 2.5,
            "normalize": False, "loudness": -16}
    done = wait_job(client, client.post(f"/api/projects/{pid}/mixdown", json=body).json()["id"])
    assert done["status"] == "succeeded", done
    info = sf.info(done["result"]["path"])
    assert info.samplerate == 48000 and info.subtype == "PCM_16" and abs(info.duration - 2.0) < 0.01
    assert "mediaId" not in done["result"]                                 # a range is not kept with the project
    from stellaria_backend.mixing import loudness as ld

    data, sr = sf.read(done["result"]["path"], always_2d=True)
    assert ld.integrated_loudness(data.T, sr) == pytest.approx(-16, abs=0.3)
    # the same folder again: a new name, never an overwrite
    again = wait_job(client, client.post(f"/api/projects/{pid}/mixdown", json=body).json()["id"])
    assert again["result"]["path"] != done["result"]["path"] and Path(again["result"]["path"]).exists()
    # browser mode: written to the exports folder with a download URL
    web = wait_job(client, client.post(f"/api/projects/{pid}/mixdown", json={"format": "wav", "bitDepth": 32}).json()["id"])
    assert web["status"] == "succeeded" and web["result"]["downloadUrl"].startswith("/files/") and web["result"]["mediaId"]
    assert sf.info(web["result"]["path"]).subtype == "FLOAT"


def test_mixdown_route_stems(client, tmp_path):
    pid, tl = _project_with_audio(client, tmp_path)
    out = tmp_path / "stems"
    out.mkdir()
    done = wait_job(client, client.post(f"/api/projects/{pid}/mixdown", json={"format": "wav", "path": str(out), "stems": True}).json()["id"])
    assert done["status"] == "succeeded", done
    names = sorted(Path(p).name for p in done["result"]["paths"])
    assert names == ["Mix - 01 Lead.wav", "Mix - 02 Harmony.wav"]               # the muted track has no file
    lead, _ = sf.read(str(out / "Mix - 01 Lead.wav"), always_2d=True)
    harmony, _ = sf.read(str(out / "Mix - 02 Harmony.wav"), always_2d=True)
    # one shared gain: the peak of their sum is −1 dBFS
    assert float(np.max(np.abs(lead + harmony))) == pytest.approx(10 ** (-1 / 20), abs=2e-3)
    web = wait_job(client, client.post(f"/api/projects/{pid}/mixdown", json={"format": "mp3", "stems": True}).json()["id"])
    assert web["status"] == "succeeded", web
    with zipfile.ZipFile(web["result"]["path"]) as z:
        assert sorted(z.namelist()) == ["Mix - 01 Lead.mp3", "Mix - 02 Harmony.mp3"]
    assert web["result"]["downloadUrl"].startswith("/files/")
    # the arrangement as on screen (not saved yet) wins over the stored one
    shown = {**tl, "tracks": [tl["tracks"][1]]}
    body = {"format": "wav", "path": str(out), "stems": True, "timeline": shown}
    fresh = wait_job(client, client.post(f"/api/projects/{pid}/mixdown", json=body).json()["id"])
    assert [Path(p).name for p in fresh["result"]["paths"]] == ["Mix - 01 Harmony.wav"]
