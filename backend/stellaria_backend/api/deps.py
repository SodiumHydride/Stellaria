"""Request helpers shared by the routers."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Request
from starlette.datastructures import UploadFile

from ..context import AppContext
from ..errors import ApiError
from ..util.fsutil import safe_filename, unique_path


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


@dataclass
class Incoming:
    """Files referenced by a request: absolute ``paths`` (desktop) or uploaded files (browser)."""

    paths: list[Path] = field(default_factory=list)
    names: dict[Path, str] = field(default_factory=dict)  # display names of uploaded files
    cleanup: list[Path] = field(default_factory=list)  # temp folders to delete after the job
    fields: dict[str, Any] = field(default_factory=dict)  # other JSON/form fields


async def read_incoming(request: Request, ctx: AppContext) -> Incoming:
    """Accept JSON ``{paths: [...]}`` / ``{path}`` or multipart ``files[]`` / ``file`` (+ form fields)."""
    ctype = request.headers.get("content-type", "")
    incoming = Incoming()
    if ctype.startswith("multipart/form-data"):
        form = await request.form()
        tmp = Path(tempfile.mkdtemp(prefix="upload-", dir=ctx.tmp_dir))
        incoming.cleanup.append(tmp)
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                original = value.filename or "upload"
                dest = unique_path(tmp / safe_filename(Path(original).name, "upload"))
                with open(dest, "wb") as out:
                    while chunk := await value.read(1 << 20):
                        out.write(chunk)
                incoming.paths.append(dest)
                incoming.names[dest] = original
            else:
                incoming.fields[key] = _maybe_json(value)
        return incoming
    raw = await request.body()
    body = json.loads(raw) if raw.strip() else {}
    if not isinstance(body, dict):
        raise ApiError("request.invalid", "Expected a JSON object", status=422)
    incoming.fields = body
    paths = body.get("paths") or ([body["path"]] if body.get("path") else [])
    incoming.paths = [Path(p) for p in paths]
    return incoming


def _maybe_json(value: str) -> Any:
    text = value.strip()
    if text[:1] in ("{", "[") or text in ("true", "false", "null"):
        try:
            return json.loads(text)
        except ValueError:
            return value
    return value


def require_paths(incoming: Incoming) -> None:
    if not incoming.paths:
        raise ApiError("request.no_files", "No files or paths given", status=422)
    for p in incoming.paths:
        if not p.is_absolute():
            raise ApiError("request.relative_path", f"Paths must be absolute: {p}", status=422)
