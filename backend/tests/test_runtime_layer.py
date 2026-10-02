"""The shell's layer over the user's own Python (desktop/src-tauri/src/layer.rs): plugin
environments see the user's packages behind it, and add-ons install into the layer only what
the user's environment lacks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stellaria_backend.errors import ApiError
from stellaria_backend.plugins import envs


def test_outside_a_layer_nothing_changes():
    assert envs.layer_marker() is None
    assert envs.layer_sites() == []
    assert envs._wanted_marker()["layer"] is None


def test_layer_sites_follow_the_runtime_own(monkeypatch, tmp_path):
    layer = tmp_path / "layer"
    layer.mkdir()
    theirs = tmp_path / "conda" / "envs" / "svc" / "Lib" / "site-packages"
    theirs.mkdir(parents=True)
    (layer / envs.LAYER_MARKER).write_text(
        json.dumps({"base": str(tmp_path / "conda/envs/svc/python.exe"), "version": "3.11",
                    "sites": [str(theirs), str(tmp_path / "gone")], "createdAt": "2026-10-02T00:00:00Z"}),
        encoding="utf-8")
    monkeypatch.setattr(sys, "prefix", str(layer))

    assert envs.layer_sites() == [str(theirs)]  # folders that no longer exist are left out
    dirs = envs.runtime_site_dirs()
    assert dirs[-1] == str(theirs) and dirs.index(str(theirs)) > 0
    # a plugin env made in one layer is rebuilt when the layer moves to another Python
    assert envs._wanted_marker()["layer"] == str(tmp_path / "conda/envs/svc/python.exe")


def _jctx():
    return SimpleNamespace(stage=lambda *a, **k: None, log=lambda *a, **k: None)


def test_addons_install_into_the_layer(client, monkeypatch):
    calls = []
    monkeypatch.setattr(envs, "layer_marker", lambda: {"base": "/envs/svc/bin/python", "sites": []})
    monkeypatch.setattr(envs, "install_into_env",
                        lambda ctx, jctx, python, venv, requirements: calls.append((python, venv, requirements)))
    manager = client.ctx.resources
    rdef = manager.definition("vsthost")

    manager._install_addon(_jctx(), rdef)

    assert calls == [(Path(sys.executable), Path(sys.prefix), ["pedalboard==0.9.25"])]
    assert (manager._addon_dir("vsthost") / ".ready").is_file()


def test_layer_install_failures_are_addon_failures(client, monkeypatch):
    def fail(*_args):
        raise ApiError("plugin.install_failed", "x", status=502, detail="no mirror reachable")

    monkeypatch.setattr(envs, "layer_marker", lambda: {"base": "/envs/svc/bin/python", "sites": []})
    monkeypatch.setattr(envs, "install_into_env", fail)
    manager = client.ctx.resources
    with pytest.raises(ApiError) as exc:
        manager._install_addon(_jctx(), manager.definition("vsthost"))
    assert exc.value.code == "addon.install_failed" and exc.value.detail == "no mirror reachable"


def test_the_users_python_without_a_layer_is_never_installed_into(client, monkeypatch):
    monkeypatch.setenv("STELLARIA_CUSTOM_RUNTIME", "1")
    ran = []
    monkeypatch.setattr("stellaria_backend.util.procs.popen", lambda *a, **k: ran.append(a))
    manager = client.ctx.resources
    with pytest.raises(ApiError) as err:
        manager._install_addon(_jctx(), manager.definition("vsthost"))
    assert err.value.code == "addon.needs_layer" and not ran


def test_no_plugin_envs_in_a_shadow_layer(client, monkeypatch):
    monkeypatch.setattr(envs, "layer_marker", lambda: {"base": "C:/bundle/runtime/python.exe", "kind": "shadow"})
    assert envs.in_shadow_layer()
    with pytest.raises(ApiError) as err:
        envs.ensure_env(client.ctx, "some-plugin")
    assert err.value.code == "plugin.env_unsupported"
    monkeypatch.setattr(envs, "layer_marker", lambda: {"base": "/envs/svc/bin/python"})  # older markers: venv
    assert not envs.in_shadow_layer()
