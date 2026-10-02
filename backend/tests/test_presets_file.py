"""Preset files: the user's mix and effect-chain presets written out and read back."""

from __future__ import annotations

import json


def _file(**kinds):
    return {"format": "stellaria-presets", "version": 1, "mix": [], "fx": [], **kinds}


def test_export_then_import(client, tmp_path):
    mix = [{"name": "我的人声", "processor": "builtin", "preset": "classic", "params": {"compRatio": 4.5}}]
    fx = [{"name": "Lead", "chain": [{"kind": "builtin", "effect": "eq", "params": {}, "mix": 1, "bypass": False}]}]
    r = client.post("/api/presets/export", json={"path": str(tmp_path / "out" / "mine"), "file": _file(mix=mix, fx=fx)})
    assert r.status_code == 200, r.text
    path = r.json()["path"]
    assert path.endswith("mine.stellapreset")  # the suffix is added
    assert json.loads(open(path, encoding="utf-8").read())["mix"][0]["name"] == "我的人声"

    back = client.post("/api/presets/import", json={"path": path}).json()
    assert back["mix"] == mix and back["fx"] == fx and back["version"] == 1


def test_only_our_files(client, tmp_path):
    bad = client.post("/api/presets/export", json={"path": str(tmp_path / "x"), "file": {"format": "other"}})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "preset.invalid_file"
    nameless = client.post("/api/presets/export", json={"path": str(tmp_path / "x"), "file": _file(mix=[{"name": " "}])})
    assert nameless.status_code == 400
    relative = client.post("/api/presets/export", json={"path": "x.stellapreset", "file": _file()})
    assert relative.status_code == 400 and relative.json()["error"]["code"] == "preset.bad_path"

    (tmp_path / "junk.stellapreset").write_text("{not json", encoding="utf-8")
    junk = client.post("/api/presets/import", json={"path": str(tmp_path / "junk.stellapreset")})
    assert junk.status_code == 400 and junk.json()["error"]["params"]["reason"] == "json"
    newer = tmp_path / "newer.stellapreset"
    newer.write_text(json.dumps(_file(version=9)), encoding="utf-8")
    assert client.post("/api/presets/import", json={"path": str(newer)}).json()["error"]["params"]["reason"] == "version"
    missing = client.post("/api/presets/import", json={"path": str(tmp_path / "nope.stellapreset")})
    assert missing.status_code == 404
