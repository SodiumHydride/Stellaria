"""``GET /files/{token}``: export results and quick renders for browser mode."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse, Response

from ..context import AppContext
from ..errors import not_found
from .deps import get_ctx

router = APIRouter()

MEDIA_TYPES = {".wav": "audio/wav", ".stella": "application/zip", ".mp3": "audio/mpeg", ".flac": "audio/flac"}


@router.get("/files/{token}")
def get_file(token: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    path = ctx.files.resolve(token)
    if path is None or not path.is_file():
        raise not_found("file", token)
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"),
                        filename=path.name)
