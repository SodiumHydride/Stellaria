"""The pitch editor: pitch lines of render inputs, edits applied when singing, phrase previews."""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from conftest import wait_job
from stellaria_backend.engine import pitch_edit as pe
from stellaria_backend.engine import pitching
from synth import write
from test_creation import _trained_model, import_media, node_of, put_canvas, render_node, source_node

FS = pe.ANALYSIS_HOP_S


def _hz(midi):
    return 440.0 * 2 ** ((np.asarray(midi, dtype=float) - 69) / 12)


def _line():
    """C4 0.5 s, a gap, E4 with vibrato 0.6 s, a slide up to a held G4 0.5 s."""
    vib = 64 + 0.3 * np.sin(2 * np.pi * 5.5 * np.arange(int(0.6 / FS)) * FS)
    return np.concatenate([np.full(int(0.5 / FS), _hz(60)), np.zeros(int(0.1 / FS)), _hz(vib),
                           _hz(np.linspace(64, 67, 6)), np.full(int(0.5 / FS), _hz(67))])


# ---- the math --------------------------------------------------------------------------------
def test_notes_and_key_of_a_line():
    a = pe.analysis(_line(), FS)
    notes = [(n["start"], round(n["pitch"])) for n in a["notes"]]
    assert [p for _, p in notes] == [60, 64, 67]  # vibrato stays one note, the slide joins G
    assert a["midi"][int(0.55 / FS)] is None and abs(a["midi"][5] - 60) < 1e-3
    assert a["key"]["mode"] in ("major", "minor") and 0 <= a["key"]["tonic"] < 12


def test_edits_are_offsets_in_time():
    f0 = np.full(200, 220.0)
    edit = pe.parse({"rate": 100, "segments": [{"start": 0.5, "values": [12.0] * 51}]})
    out = pe.apply(f0, edit, 0.01)  # frames every 10 ms
    assert np.allclose(out[:50], 220) and np.allclose(out[50:101], 440) and np.allclose(out[101:], 220)
    # a region rendered from 0.4 s sees the same edit at the same song time
    region = pe.apply(f0[:50], edit, 0.01, start_s=0.4)
    assert np.allclose(region[:10], 220) and np.allclose(region[10:], 440)
    assert pe.apply(f0, None, 0.01) is f0 and pe.parse({"segments": [{"start": 0, "values": []}]}) is None
    with pytest.raises(ValueError):
        pe.parse({"rate": 5, "segments": []})
    # the editor's own grid (its analysis frames) is kept
    assert pe.parse({"rate": 44100 / 512, "segments": [{"start": 0, "values": [1]}]})["rate"] == 44100 / 512
    with pytest.raises(ValueError):
        pe.parse({"segments": [{"start": -1, "values": [1]}]})


def test_unvoiced_interpolation_matches_the_engine():
    f0 = np.array([0, 0, 200, 0, 0, 400, 0, 30], dtype=float)
    out = pe.interp_unvoiced(f0, 50)
    assert out.tolist() == [200, 200, 200, 200 + 200 / 3, 200 + 400 / 3, 400, 50.0 + 165, 50.0]


def test_render_params_take_checked_edits():
    from stellaria_backend.engine.rendering import parse_render_params
    from stellaria_backend.errors import ApiError

    p = parse_render_params({"pitchEdit": {"rate": 100, "segments": [{"start": 1, "values": [1, 99]}]}})
    assert p.pitch_edit == {"rate": 100.0, "segments": [{"start": 1.0, "values": [1.0, 24.0]}]}
    assert parse_render_params({"pitchEdit": {"segments": []}}).pitch_edit is None
    with pytest.raises(ApiError):
        parse_render_params({"pitchEdit": {"rate": 3}})


# ---- the API (fake engine: WORLD pitch tracking, renders = input −2 dB) --------------------------
def test_analyze_then_preview_a_phrase(client, tmp_path):
    ctx = client.ctx
    cid, mid = _trained_model(client, ctx)
    p = client.post("/api/projects", json={"name": "pitch"}).json()
    sr = 44100
    t = np.arange(int(2.0 * sr)) / sr
    voice = 0.3 * np.sin(2 * np.pi * 220 * t) * (t < 1.4)  # A3 for 1.4 s, then silence
    media_id = import_media(client, p["id"], write(tmp_path / "v.wav", np.stack([voice] * 2, axis=1), sr))
    put_canvas(client, p["id"], [source_node(media_id), render_node("nd_r", "mix", characterId=cid, modelId=mid)])

    missing = client.get(f"/api/projects/{p['id']}/nodes/nd_r/pitch")
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "pitch.not_analyzed"
    job = wait_job(client, client.post(f"/api/projects/{p['id']}/nodes/nd_r/pitch/analyze").json()["id"])
    assert job["status"] == "succeeded", job
    a = client.get(f"/api/projects/{p['id']}/nodes/nd_r/pitch").json()
    assert a["current"] is True and a["madeFrom"]["mediaId"] == media_id
    assert len(a["notes"]) == 1 and round(a["notes"][0]["pitch"]) == 57  # A3
    assert 1.2 < a["notes"][0]["end"] < 1.6 and abs(a["frameS"] - FS) < 1e-9

    # changing how the pitch is tracked makes the stored line out of date
    node = node_of(client, p["id"], "nd_r")
    node["data"]["params"] = {"f0Extractor": "fcpe"}
    put_canvas(client, p["id"], [node_of(client, p["id"], "nd_s"), node])
    assert client.get(f"/api/projects/{p['id']}/nodes/nd_r/pitch").json()["current"] is False

    edit = {"rate": 100, "segments": [{"start": 0.2, "values": [2.0] * 40}]}
    preview = client.post(f"/api/projects/{p['id']}/nodes/nd_r/pitch/preview",
                          json={"start": 0.1, "end": 0.9, "pitchEdit": edit}).json()
    done = wait_job(client, preview["id"])
    assert done["status"] == "succeeded", done
    url = done["result"]["url"]
    assert done["result"]["start"] == 0.1 and done["result"]["end"] == 0.9
    audio = client.get(f"/api{url}")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    out = tmp_path / "preview.wav"
    out.write_bytes(audio.content)
    assert abs(sf.info(out).duration - 0.8) < 0.02
    assert node_of(client, p["id"], "nd_r")["data"]["status"] == "draft"  # the node is untouched

    bad = client.post(f"/api/projects/{p['id']}/nodes/nd_r/pitch/preview", json={"start": 1, "end": 1})
    assert bad.status_code == 422

    # only the newest previews are kept, and a deleted project takes its caches with it
    folder = pitching.preview_path(ctx, p["id"], "x").parent
    for i in range(pitching.PREVIEWS_KEPT + 3):
        f = folder / f"old{i}.wav"
        f.write_bytes(b"")
        os.utime(f, (1000 + i, 1000 + i))
    pitching.prune_previews(ctx, p["id"])
    kept = sorted(f.name for f in folder.glob("*.wav"))
    assert len(kept) == pitching.PREVIEWS_KEPT and "old0.wav" not in kept and f"{done['id']}.wav" in kept
    assert client.delete(f"/api/projects/{p['id']}").status_code == 200
    assert not folder.exists() and not list((ctx.options.data / "cache" / "pitch").glob(f"{p['id']}_*"))
