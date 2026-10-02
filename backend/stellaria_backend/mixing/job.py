"""Mix job JSON (v1 and v2) → :class:`MixJob`.

v2 (``docs/canvas-v3.md`` §4)::

    {"inputs": [{"path", "role": "lead|harmony|backing|other", "gainDb", "pan", "offset", "mute"?, "solo"?}],
     "reference": {"vocals": str | null, "mix": str | null, "offset": float} | null,
     "processor": "builtin|bus", "preset", "params", "seed", "output"}

v1 (``creation-v2.md`` §6): ``vocals[]`` become ``lead`` inputs, ``backing`` a ``backing`` input,
all at offset 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import spec
from .audioio import MixError

ROLES = ("lead", "harmony", "backing", "other")
VOCAL_ROLES = ("lead", "harmony")


@dataclass
class MixInput:
    path: str
    role: str = "lead"
    gain_db: float = 0.0
    pan: float = 0.0
    offset: float = 0.0
    mute: bool = False
    solo: bool = False
    index: int = 0

    @property
    def vocal(self) -> bool:
        return self.role in VOCAL_ROLES


@dataclass
class Reference:
    vocals: str | None
    mix: str | None
    offset: float = 0.0


@dataclass
class MixJob:
    inputs: list[MixInput]
    output: str
    processor: str = spec.PROCESSOR_ID
    preset: str = spec.DEFAULT_PRESET
    params: dict = field(default_factory=dict)
    seed: int = 0
    reference: Reference | None = None
    version: int = 2

    def active(self) -> list[MixInput]:
        """Inputs that play: not muted and, when anything is soloed, soloed."""
        playing = [i for i in self.inputs if not i.mute]
        soloed = [i for i in playing if i.solo]
        return soloed or playing


def _float(value: Any, name: str, lo: float, hi: float, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        raise MixError("mix.bad_job", f"'{name}' must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise MixError("mix.bad_job", f"'{name}' must be a number") from None
    if not math.isfinite(v):
        raise MixError("mix.bad_job", f"'{name}' must be finite")
    return min(max(v, lo), hi)


def _path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MixError("mix.bad_job", f"{name} is missing")
    return value


def _input(raw: Any, i: int, role: str | None = None) -> MixInput:
    if not isinstance(raw, dict):
        raise MixError("mix.bad_job", f"inputs[{i}] must be an object")
    r = role or raw.get("role") or "lead"
    if r not in ROLES:
        raise MixError("mix.bad_job", f"inputs[{i}].role must be one of {', '.join(ROLES)}")
    return MixInput(
        path=_path(raw.get("path"), f"inputs[{i}].path"), role=r,
        gain_db=_float(raw.get("gainDb"), f"inputs[{i}].gainDb", -60.0, 24.0),
        pan=_float(raw.get("pan"), f"inputs[{i}].pan", -1.0, 1.0),
        offset=_float(raw.get("offset"), f"inputs[{i}].offset", -1e6, 1e6),
        mute=bool(raw.get("mute") or False), solo=bool(raw.get("solo") or False), index=i,
    )


def parse_job(job: Any) -> MixJob:
    if not isinstance(job, dict):
        raise MixError("mix.bad_job", "The job must be a JSON object")
    output = job.get("output")
    if not isinstance(output, str) or not output:
        raise MixError("mix.bad_job", "'output' is missing")
    params = job.get("params") or {}
    if not isinstance(params, dict):
        raise MixError("mix.bad_job", "'params' must be an object")
    preset = job.get("preset")
    try:
        seed = abs(int(job.get("seed") or 0))
    except (TypeError, ValueError):
        raise MixError("mix.bad_job", "'seed' must be an integer") from None
    processor = job.get("processor") or spec.PROCESSOR_ID
    if processor not in (spec.PROCESSOR_ID, spec.BUS_ID):
        raise MixError("mix.bad_job", f"Unknown processor '{processor}'")

    if "inputs" in job:
        raw_inputs = job.get("inputs")
        if not isinstance(raw_inputs, list):
            raise MixError("mix.bad_job", "'inputs' must be a list")
        inputs = [_input(raw, i) for i, raw in enumerate(raw_inputs)]
        version = 2
    else:
        vocals = job.get("vocals")
        if not isinstance(vocals, list):
            raise MixError("mix.bad_job", "'vocals' must be a list")
        inputs = [_input(raw, i, "lead") for i, raw in enumerate(vocals)]
        backing = job.get("backing")
        if backing is not None:
            if not isinstance(backing, dict):
                raise MixError("mix.bad_job", "backing.path is missing")
            b = _input(backing, len(inputs), "backing")
            b.pan = 0.0
            inputs.append(b)
        version = 1

    reference = None
    ref = job.get("reference")
    if isinstance(ref, dict) and (ref.get("vocals") or ref.get("mix")):
        reference = Reference(
            vocals=ref.get("vocals") if isinstance(ref.get("vocals"), str) and ref.get("vocals") else None,
            mix=ref.get("mix") if isinstance(ref.get("mix"), str) and ref.get("mix") else None,
            offset=_float(ref.get("offset"), "reference.offset", -1e6, 1e6),
        )
    j = MixJob(inputs=inputs, output=output, processor=processor,
               preset=preset if isinstance(preset, str) else spec.DEFAULT_PRESET,
               params=params, seed=seed, reference=reference, version=version)
    validate(j)
    return j


def validate(j: MixJob) -> None:
    if j.processor == spec.BUS_ID:
        if not j.inputs:
            raise MixError("mix.no_inputs", "Connect at least one input")
        if len(j.inputs) > spec.BUS_MAX_INPUTS:
            raise MixError("mix.too_many_inputs", f"At most {spec.BUS_MAX_INPUTS} inputs are supported",
                           params={"max": spec.BUS_MAX_INPUTS})
        return
    vocals = [i for i in j.active() if i.vocal]
    if not vocals:
        raise MixError("mix.no_vocals", "At least one vocal is needed")
    if len(vocals) > spec.MAX_VOCALS:
        raise MixError("mix.too_many_vocals", f"At most {spec.MAX_VOCALS} vocals are supported",
                       params={"max": spec.MAX_VOCALS})
    if len(j.inputs) > spec.MAX_INPUTS:
        raise MixError("mix.too_many_inputs", f"At most {spec.MAX_INPUTS} inputs are supported",
                       params={"max": spec.MAX_INPUTS})
