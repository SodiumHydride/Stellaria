"""Library root: folder layout, ``library.json`` and per-object locks."""

from __future__ import annotations

import threading
from pathlib import Path

from ..util.jsonio import read_json, write_json

LIBRARY_VERSION = 1


class Library:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.characters_dir = root / "characters"
        self.models_dir = root / "models"
        self.projects_dir = root / "projects"
        for d in (self.characters_dir, self.models_dir, self.projects_dir):
            d.mkdir(parents=True, exist_ok=True)
        meta_path = root / "library.json"
        if read_json(meta_path) is None:
            write_json(meta_path, {"version": LIBRARY_VERSION})
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    def lock(self, key: str) -> threading.RLock:
        """A re-entrant lock per object (``chr:<id>``, ``mdl:<id>``, ``prj:<id>``)."""
        with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = self._locks[key] = threading.RLock()
            return lock
