"""Process-node DSP ops and song analysis (docs/canvas-v3.md §2, §6). numpy/scipy, no torch.

Public interface::

    python -m stellaria_backend.dsp.run <job.json>      # trim|split|pitch|fx|comp|concat|export
    sections(path, *, vocals_path=None) -> dict          # {bpm, beats, downbeats, phrases, sections}
    key(path) -> str | None                              # "C#m", "F", … (Krumhansl–Schmuckler)
    pitch_engines() -> list[str]                          # pitch engines available here, preferred first

Importing this package is cheap (no scipy/librosa); the DSP modules load on first use.
"""

from __future__ import annotations

from pathlib import Path

__all__ = ["key", "pitch_engines", "sections"]


def sections(path: Path | str, *, vocals_path: Path | str | None = None) -> dict:
    """Beats, downbeats, vocal phrases and labelled sections of a song (see ``analysis.sections``)."""
    from .analysis import sections as _sections

    return _sections(path, vocals_path=vocals_path)


def key(path: Path | str) -> str | None:
    """Musical key (``"C#m"``, ``"F"``) or None when the audio has no clear tonality."""
    from .analysis import key as _key

    return _key(path)


def pitch_engines() -> list[str]:
    from .pitch import available_engines

    return available_engines()
