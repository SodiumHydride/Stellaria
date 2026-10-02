"""Preset files (``.stellapreset``): the user's mix and effect-chain presets, saved to share and
loaded back. The presets themselves live in the frontend; this only writes and reads the file
(the desktop page cannot), checking that it is one of ours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter

from ..errors import ApiError
from ..util.jsonio import write_json

router = APIRouter()

FORMAT = "stellaria-presets"
VERSION = 1
SUFFIX = ".stellapreset"
MAX_BYTES = 2 * 1024 * 1024
MAX_PRESETS = 500


def _invalid(reason: str) -> ApiError:
    return ApiError("preset.invalid_file", "Not a Stellaria preset file", status=400, params={"reason": reason})


def check(data: Any) -> dict:
    """``{format, version, mix: [...], fx: [...]}`` with every preset an object that has a name."""
    if not isinstance(data, dict) or data.get("format") != FORMAT:
        raise _invalid("format")
    if not isinstance(data.get("version"), int) or data["version"] > VERSION:
        raise _invalid("version")
    out: dict[str, Any] = {"format": FORMAT, "version": VERSION}
    for kind in ("mix", "fx"):
        items = data.get(kind) or []
        if not isinstance(items, list) or len(items) > MAX_PRESETS:
            raise _invalid(kind)
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"].strip():
                raise _invalid(kind)
        out[kind] = items
    return out


def _path(raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ApiError("preset.bad_path", "Choose where the preset file goes", status=400)
    path = Path(raw.strip()).expanduser()
    if not path.is_absolute():
        raise ApiError("preset.bad_path", "Choose where the preset file goes", status=400)
    return path if path.suffix.lower() == SUFFIX else path.with_name(path.name + SUFFIX)


@router.post("/presets/export")
def export_presets(body: dict) -> dict:
    """``{path, file}`` → writes ``file`` (checked) to ``path`` (``.stellapreset`` added if missing)."""
    data = check(body.get("file"))
    path = _path(body.get("path"))
    if len(json.dumps(data)) > MAX_BYTES:
        raise _invalid("size")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, data)
    except OSError as exc:
        raise ApiError("preset.write_failed", "Could not save the preset file", status=500, detail=str(exc)) from exc
    return {"path": str(path)}


@router.post("/presets/import")
def import_presets(body: dict) -> dict:
    """``{path}`` → the file's presets ``{format, version, mix, fx}`` (the frontend adds them)."""
    raw = body.get("path")
    if not isinstance(raw, str) or not raw.strip():
        raise ApiError("preset.bad_path", "Choose a preset file", status=400)
    path = Path(raw.strip()).expanduser()
    try:
        if path.stat().st_size > MAX_BYTES:
            raise _invalid("size")
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ApiError("preset.not_found", "The preset file was not found", status=404) from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid("json") from exc
    return check(data)
