"""``<data>/preferences.json`` with change events.

Only keys the user set explicitly are stored, so changing a default later (e.g. ``gpuPolicy``)
reaches everyone who never touched that setting. Files written before this rule contain every
key; such stored values are kept as they are, since an explicit choice cannot be told apart.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .errors import ApiError
from .events import EventHub
from .schemas import Preferences
from .util.jsonio import read_json, write_json


class PreferencesStore:
    def __init__(self, path: Path, hub: EventHub) -> None:
        self._path = path
        self._hub = hub
        self._lock = threading.Lock()
        stored = read_json(path, {})
        known = Preferences().wire()
        # keys of removed preferences (e.g. ``starfield``) are dropped
        self._explicit: dict[str, Any] = {k: v for k, v in stored.items() if k in known} if isinstance(stored, dict) else {}
        try:
            self._prefs = Preferences.model_validate(self._explicit)
        except Exception:  # noqa: BLE001 - a broken file falls back to defaults
            self._explicit, self._prefs = {}, Preferences()

    def get(self) -> Preferences:
        return self._prefs

    def update(self, patch: dict[str, Any]) -> Preferences:
        with self._lock:
            known = {k: v for k, v in patch.items() if k in Preferences().wire()}
            try:
                prefs = Preferences.model_validate({**self._explicit, **known})
            except Exception as exc:  # noqa: BLE001
                raise ApiError("request.invalid", "Invalid preferences", status=422, detail=str(exc)) from exc
            self._explicit = {**self._explicit, **known}
            self._prefs = prefs
            write_json(self._path, self._explicit)
        self._hub.publish("preferences", self._prefs.wire())
        return self._prefs
