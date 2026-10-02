"""``ParamSpec`` definitions shared by the mixer and the effects: wire dicts and value parsing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


def msg(code: str, **params: Any) -> dict:
    return {"code": code, "params": params}


@dataclass(frozen=True)
class Param:
    """One ``ParamSpec``. ``default=None`` means "decided elsewhere" (e.g. by the preset)."""
    id: str
    type: str = "number"                       # 'number' | 'boolean' | 'choice'
    default: float | bool | str | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    unit: str | None = None
    options: tuple[str, ...] = ()
    option_prefix: str | None = None           # Msg code prefix of option labels
    group: str | None = None                   # fine-grained settings, shown folded (label prefix + ".group.")

    def wire(self, label_prefix: str) -> dict:
        out: dict[str, Any] = {"id": self.id, "label": msg(f"{label_prefix}.{self.id}"), "type": self.type,
                               "default": self.default}
        if self.type == "number":
            out.update(min=self.min, max=self.max, step=self.step)
            if self.unit:
                out["unit"] = self.unit
        if self.type == "choice":
            prefix = self.option_prefix or f"{label_prefix}.{self.id}"
            out["options"] = [{"value": v, "label": msg(f"{prefix}.{v}")} for v in self.options]
        if self.group:
            out["group"] = self.group
        return out

    def parse(self, raw: Any, fallback: Any = None) -> Any:
        """Coerce a wire value; missing/invalid → ``fallback`` (or the default); numbers clamped."""
        fallback = self.default if fallback is None else fallback
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            return fallback
        if self.type == "boolean":
            if isinstance(raw, bool):
                return raw
            if isinstance(raw, (int, float)) and math.isfinite(raw):
                return bool(raw)
            if isinstance(raw, str) and raw.strip().lower() in ("true", "false", "1", "0", "on", "off"):
                return raw.strip().lower() in ("true", "1", "on")
            return fallback
        if self.type == "choice":
            value = str(raw)
            return value if value in self.options else fallback
        if isinstance(raw, bool):
            return fallback
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return fallback
        if not math.isfinite(value):
            return fallback
        if self.min is not None:
            value = max(value, self.min)
        if self.max is not None:
            value = min(value, self.max)
        return value


def number(pid: str, default: float | None, lo: float, hi: float, step: float, unit: str | None = None,
           group: str | None = None) -> Param:
    return Param(pid, "number", default, lo, hi, step, unit, group=group)


def boolean(pid: str, default: bool | None) -> Param:
    return Param(pid, "boolean", default)


def choice(pid: str, default: str, options: tuple[str, ...], option_prefix: str | None = None) -> Param:
    return Param(pid, "choice", default, options=options, option_prefix=option_prefix)
