"""Where export jobs write: a user-chosen path (desktop) or the exports folder + a
``/files/{token}`` download URL (browser mode)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .util.fsutil import safe_filename
from .util.ids import new_id

if TYPE_CHECKING:
    from .context import AppContext


def destination(ctx: "AppContext", path: str | None, stem: str, ext: str) -> tuple[Path, bool]:
    """(file to write, browser mode?). A folder ``path`` gets ``<stem>.<ext>`` inside it."""
    if path:
        dest = Path(path)
        if dest.is_dir():
            dest = dest / f"{safe_filename(stem)}.{ext}"
        return dest, False
    return ctx.options.data / "exports" / f"{safe_filename(stem)}-{new_id('x', 6)[2:]}.{ext}", True


def finish(ctx: "AppContext", dest: Path, browser: bool) -> dict[str, Any]:
    if browser:
        return {"downloadUrl": f"/files/{ctx.files.register(dest)}", "path": str(dest)}
    return {"path": str(dest)}
