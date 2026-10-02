"""Engine output: tailing log files and turning failures into stable error codes."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

from ..errors import ApiError

OOM_PATTERNS = ("CUDA out of memory", "OutOfMemoryError", "CUBLAS_STATUS_ALLOC_FAILED",
                "CUDNN_STATUS_ALLOC_FAILED", "MPS backend out of memory", "DefaultCPUAllocator: can't allocate memory")
MISSING_PATTERN = re.compile(r"(No such file or directory|FileNotFoundError|cannot find the (?:file|path))[^\n]*", re.I)
_SPLIT = re.compile(r"\r\n|\r|\n")


def split_lines(buffer: str) -> tuple[list[str], str]:
    """Split on \\n and \\r (tqdm redraws); returns complete lines and the trailing remainder."""
    parts = _SPLIT.split(buffer)
    return [p for p in parts[:-1] if p.strip()], parts[-1]


class LogTail:
    """Reads lines appended to a file since a given offset (works for files written by other processes)."""

    def __init__(self, path: Path, offset: int | None = None) -> None:
        self.path = path
        self.offset = offset if offset is not None else (path.stat().st_size if path.exists() else 0)
        self._rest = ""

    def read(self) -> list[str]:
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                data = f.read()
                self.offset = f.tell()
        except FileNotFoundError:
            return []
        lines, self._rest = split_lines(self._rest + data.decode("utf-8", errors="replace"))
        return lines


def tail_lines(path: Path, n: int = 500, max_bytes: int = 4 << 20) -> list[str]:
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - max_bytes))
            text = f.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return []
    lines, rest = split_lines(text)
    if rest.strip():
        lines.append(rest)
    return list(deque(lines, maxlen=max(1, n)))


def classify_failure(lines: list[str], default_code: str, default_message: str) -> ApiError:
    """Map engine output to ``engine.out_of_memory`` / ``engine.missing_model`` / a default code."""
    text = "\n".join(lines)
    detail = "\n".join(lines[-80:])
    if any(p in text for p in OOM_PATTERNS):
        return ApiError("engine.out_of_memory", "The GPU ran out of memory", status=500, detail=detail)
    m = MISSING_PATTERN.search(text)
    if m and any(k in m.group(0).lower() for k in (".pt", ".ckpt", ".bin", "pretrain", "models")):
        return ApiError("engine.missing_model", "A pretrained model file is missing", status=500, detail=detail)
    return ApiError(default_code, default_message, status=500, detail=detail)
