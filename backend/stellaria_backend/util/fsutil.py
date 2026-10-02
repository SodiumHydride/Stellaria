"""Filesystem helpers: hard links with copy fallback, sizes, safe names."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}


def safe_filename(name: str, fallback: str = "untitled") -> str:
    """Make ``name`` usable as a file name on Windows and macOS (keeps CJK characters)."""
    cleaned = _INVALID_CHARS.sub("_", name).strip().rstrip(". ")
    if not cleaned:
        cleaned = fallback
    if cleaned.upper() in _RESERVED:
        cleaned = f"_{cleaned}"
    return cleaned[:120]


def link_or_copy(src: Path, dst: Path) -> bool:
    """Hard-link ``src`` to ``dst``; copy when linking is impossible (other volume, FAT/exFAT).
    Returns True when a hard link was made."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return True
    except OSError:
        shutil.copy2(src, dst)
        return False


def dir_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=False, onerror=_retry_readonly)


def _retry_readonly(func, path, _exc_info) -> None:  # pragma: no cover - Windows read-only files
    os.chmod(path, 0o666)
    func(path)


def unique_path(path: Path) -> Path:
    """Return ``path`` or ``path`` with a numeric suffix so it does not exist yet."""
    if not path.exists():
        return path
    for i in range(2, 10000):
        candidate = path.with_name(f"{path.stem} ({i}){path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(str(path))
