"""Export file names: template filling, Windows/macOS-safe sanitising, never-overwrite claiming."""

from __future__ import annotations

import datetime as _dt
import os
import re
import unicodedata
from pathlib import Path
from typing import Any

TOKENS = ("project", "song", "voice", "label", "date", "index")
DEFAULT_TEMPLATE = "{project} - {label}"
FALLBACK_NAME = "export"
MAX_NAME_BYTES = 200          # UTF-8 bytes of the stem: leaves room for " (9999).flac" under 255
MAX_PATH_CHARS = 250          # keep full paths under Windows' classic MAX_PATH (260)
MIN_STEM_CHARS = 16

_TOKEN = re.compile(r"\{(\w+)(?::0?(\d{1,2})d?)?\}")
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")   # zero-width, bidi controls
_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
             *(f"COM{i}" for i in "123456789¹²³"), *(f"LPT{i}" for i in "123456789¹²³")}
_SEPARATORS = " -_.·,;~+&|"


def fill(template: str, names: dict[str, Any], *, now: _dt.datetime | None = None) -> str:
    """Replace ``{project} {song} {voice} {label} {date} {index}`` (``{index:03}`` pads) from
    ``names``. Unknown tokens stay as written; empty values collapse the separators around them."""
    now = now or _dt.datetime.now()
    values = {k: names.get(k) for k in TOKENS}
    if not values.get("date"):
        values["date"] = now.strftime("%Y-%m-%d")
    if values.get("index") in (None, ""):
        values["index"] = 1

    def sub(m: re.Match) -> str:
        key, width = m.group(1), m.group(2)
        if key not in TOKENS:
            return m.group(0)
        v = values.get(key)
        if v is None:
            return "\x00"                              # marks an empty token for the cleanup below
        if key == "index" and width:
            try:
                return str(int(v)).zfill(int(width))
            except (TypeError, ValueError):
                pass
        text = str(v).strip()
        return text if text else "\x00"

    out = _TOKEN.sub(sub, template or "")
    # an empty token takes the separator run before it (or after it, at the start) with it
    out = re.sub(r"[ \-_.·,;~+&|]*\x00", "\x00", out)
    out = re.sub(r"^\x00[ \-_.·,;~+&|]*", "", out)
    out = out.replace("\x00", "")
    return re.sub(r"\s{2,}", " ", out).strip(_SEPARATORS + "\t")


def _truncate_utf8(text: str, limit: int) -> str:
    data = text.encode("utf-8")
    if len(data) <= limit:
        return text
    return data[:limit].decode("utf-8", errors="ignore")


def sanitize(name: str, *, fallback: str = FALLBACK_NAME, max_bytes: int = MAX_NAME_BYTES) -> str:
    """A file *stem* that is valid on Windows and macOS: NFC, no illegal or control characters, no
    zero-width/bidi controls, no leading dots (hidden on macOS) or trailing dots/spaces (dropped by
    Windows), not a reserved device name (``CON``, ``COM1``, ``LPT1`` … also with an extension),
    at most ``max_bytes`` UTF-8 bytes. CJK and other scripts are kept."""
    s = unicodedata.normalize("NFC", str(name or ""))
    s = _INVISIBLE.sub("", s)
    s = re.sub(r"[\t\r\n\v\f]+", " ", s)
    s = _ILLEGAL.sub("_", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.lstrip(". ").rstrip(". ")
    s = _truncate_utf8(s, max_bytes).rstrip(". ")
    if not s or set(s) <= set("_"):
        s = fallback
    if s.split(".")[0].strip().upper() in _RESERVED:
        s = f"_{s}"
    return s


def file_stem(template: str | None, names: dict[str, Any], *, now: _dt.datetime | None = None) -> str:
    filled = fill(template or DEFAULT_TEMPLATE, names, now=now)
    fallback = sanitize(str(names.get("label") or names.get("song") or names.get("project") or FALLBACK_NAME))
    return sanitize(filled, fallback=fallback)


def claim(folder: Path, stem: str, ext: str) -> Path:
    """Create (empty, exclusively) and return ``folder/stem.ext`` or ``stem (2).ext`` … — the first
    name nobody has. Exclusive creation makes it race-free: an existing file is never touched."""
    folder = Path(folder)
    room = MAX_PATH_CHARS - len(str(folder)) - len(ext) - 10
    if len(stem) > room:
        stem = sanitize(stem[: max(MIN_STEM_CHARS, room)])
    for i in range(1, 10000):
        name = f"{stem}.{ext}" if i == 1 else f"{stem} ({i}).{ext}"
        path = folder / name
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        except FileExistsError:
            continue
        os.close(fd)
        return path
    raise FileExistsError(str(folder / f"{stem}.{ext}"))
