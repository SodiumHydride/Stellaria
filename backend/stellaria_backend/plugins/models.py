"""Wire and on-disk models of the plugin system (``docs/api.md`` → Plugins).

Kept out of ``schemas.py`` on purpose: everything plugin-shaped lives in this package.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_serializer

from ..schemas import Msg, Wire

ParamType = Literal["number", "boolean", "choice", "text", "folder", "file"]
PluginKind = Literal["mix", "effect"]
PluginRuntime = Literal["command", "python"]
Interpreter = Literal["builtin", "custom"]
PluginStatus = Literal["ready", "needs_setup", "error"]

DEFAULT_TIMEOUT_SEC = 1200.0


class ParamOption(Wire):
    value: str
    label: Msg | str


class ParamSpec(Wire):
    """One parameter (``params``) or setting (``settings``) of a processor."""

    id: str
    label: Msg | str
    type: ParamType
    default: bool | int | float | str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    options: list[ParamOption] | None = None

    @model_serializer(mode="wrap")
    def _drop_unset_optionals(self, handler: Any) -> dict[str, Any]:
        # `min?: number` etc. are optional on the wire: leave them out instead of sending null
        data = handler(self)
        for key in ("min", "max", "step", "unit", "options"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class Manifest(Wire):
    """``stellaria-plugin.json`` (or a wizard manifest in ``<data>/plugins/<id>.json``)."""

    id: str
    name: str
    version: str = "0"
    kind: PluginKind = "mix"
    runtime: PluginRuntime = "command"
    command: list[str] | None = None
    entry: str | None = None
    cwd: Literal["temp", "folder"] = "temp"
    outputs: list[str] = Field(default_factory=list)
    max_vocals: int | None = None
    params: list[ParamSpec] = Field(default_factory=list)
    settings: list[ParamSpec] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    timeout_sec: float = DEFAULT_TIMEOUT_SEC
    description: str | None = None

    @field_validator("version", mode="before")
    @classmethod
    def _version_text(cls, value: Any) -> Any:
        return str(value) if isinstance(value, (int, float)) else value

    @property
    def vocals_limit(self) -> int:
        return self.max_vocals or 1


class TorchCheck(Wire):
    version: str
    cuda: str | None = None
    cuda_available: bool = False


class PythonCheck(Wire):
    ok: bool
    executable: str
    version: str | None = None
    missing: list[str] = Field(default_factory=list)
    error: Msg | None = None
    torch: TorchCheck | None = None


class RegistryEntry(Wire):
    """What ``<data>/plugins/registry.json`` stores per plugin (the user's choices)."""

    id: str
    folder: str
    manifest_source: Literal["folder", "wizard"]
    enabled: bool = True
    interpreter: Interpreter = "builtin"
    python_path: str | None = None
    setting_values: dict[str, Any] = Field(default_factory=dict)
    name: str | None = None  # user's rename; None = the manifest's name
    added_at: str
    check: PythonCheck | None = None
    checked_at: str | None = None
    checked_for: str | None = None  # interpreter + requirements the check was made for (stale when they change)
    last_manifest: dict[str, Any] | None = None  # last valid manifest, shown while the current one is broken


class Plugin(Wire):
    id: str
    name: str
    version: str
    kind: PluginKind
    runtime: PluginRuntime
    folder: str
    manifest_source: Literal["folder", "wizard"]
    interpreter: Interpreter
    python_path: str | None
    status: PluginStatus
    check: PythonCheck | None
    params: list[ParamSpec]
    settings: list[ParamSpec]
    setting_values: dict[str, Any]
    enabled: bool
    added_at: str
    # additions to the contract
    error: Msg | None = None  # why the status is not 'ready'
    requirements: list[str] = Field(default_factory=list)
    max_vocals: int = 1
    checked_at: str | None = None


class WizardSpec(Wire):
    """Body of ``POST /plugins/wizard``: wrap a script that has no manifest."""

    folder: str
    name: str
    kind: PluginKind = "mix"
    entry: str
    args: list[str] | None = None  # default: ['{backing}', '{vocal}'] (mix) / ['{input}', '{output}'] (effect)
    outputs: list[str] | None = None  # default: output/*.flac|wav|mp3 unless args use {output}/{outdir}
    cwd: Literal["temp", "folder"] = "temp"
    runtime: PluginRuntime = "command"
    requirements: list[str] = Field(default_factory=list)
    timeout_sec: float | None = None
    max_vocals: int | None = None


class PluginPatch(Wire):
    enabled: bool | None = None
    interpreter: Interpreter | None = None
    python_path: str | None = None
    setting_values: dict[str, Any] | None = None
    name: str | None = None
