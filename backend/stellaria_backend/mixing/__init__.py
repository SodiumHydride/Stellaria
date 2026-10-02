"""Built-in mixer ("一键混音"): numpy/scipy DSP (+ librosa for tempo), no torch, no GPL code.

Public interface (docs/creation-v2.md §6)::

    builtin_processor() -> dict    # MixProcessor wire dict for 'builtin' (auto mix)
    bus_processor() -> dict        # MixProcessor wire dict for 'bus' (plain sum)
    analyze(path) -> dict          # {"bpm", "lufs", "peakDb", "duration"}
    python -m stellaria_backend.mixing.run <job.json>    # JSON-lines CLI (job v1/v2), see run.py

Mixer v2 (``docs/canvas-v3.md`` §4): roles and offsets (``job``), reference following
(``match``), unmasking, gap expander, two-stage dynamics and bus glue (``chains``, ``master``),
throws, doubler, warmth, 4× true-peak master, and the plain-sum ``bus`` processor (``bus``).

Importing this package is cheap (no scipy/librosa); the DSP modules load on first use.
"""

from __future__ import annotations

from pathlib import Path

from .spec import MAX_VOCALS, PRESETS, builtin_processor, bus_processor

__all__ = ["MAX_VOCALS", "PRESETS", "analyze", "builtin_processor", "bus_processor"]


def analyze(path: Path | str) -> dict:
    """``{"bpm": float | None, "lufs": float, "peakDb": float, "duration": float}`` of an audio file."""
    from .analysis import analyze as _analyze

    return _analyze(path)
