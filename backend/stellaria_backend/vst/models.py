"""Wire models of the VST3 / Audio Unit host (canvas-v3 §3.2), camelCase on the wire."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..schemas import Msg, Wire


class VstParam(Wire):
    id: str  # pedalboard's Python attribute name
    name: str
    kind: Literal["float", "bool", "choice"]
    min: float | None = None
    max: float | None = None
    default: float | int | str | bool | None = None
    unit: str | None = None
    choices: list[str] | None = None


class VstPlugin(Wire):
    id: str  # sha1(path + pluginName)
    name: str
    vendor: str | None = None
    version: str | None = None
    format: Literal["vst3", "au"]
    path: str
    plugin_name: str | None = None
    category: Literal["effect", "instrument"] = "effect"
    params: list[VstParam] = Field(default_factory=list)
    status: Literal["ok", "failed"] = "ok"
    error: Msg | None = None
    scanned_at: str


class VstStatus(Wire):
    host_installed: bool
    host_version: str | None
    formats: list[Literal["vst3", "au"]]
    folders: list[str]
    default_folders: list[str]
    editor_supported: bool
    last_scan: str | None
    scanning: bool
