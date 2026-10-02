"""Slicing with the engine's ``Slicer`` and splitting over-long slices at quiet points.

``slicer.py`` imports torch/torchaudio at module level (only for helpers we do not use),
so the ``Slicer`` class is compiled from the engine source on its own, keeping the API
process torch-free while still running the engine's exact slicing code.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..audio.dsp import frame_rms
from ..errors import ApiError
from ..schemas import SliceOptions


@lru_cache(maxsize=4)
def engine_slicer_class(engine_root: str) -> type:
    import librosa

    path = Path(engine_root) / "slicer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Slicer"), None)
    if cls is None:
        raise RuntimeError(f"no Slicer class in {path}")
    namespace: dict = {"librosa": librosa, "np": np}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["Slicer"]


def voiced_segments(x: np.ndarray, sr: int, opts: SliceOptions, engine_root: Path) -> list[tuple[int, int]]:
    """Sample ranges of the non-silent chunks found by the engine's Slicer."""
    if not opts.enabled:
        return [(0, len(x))]
    ms = lambda s: int(round(s * 1000))  # noqa: E731
    try:
        slicer = engine_slicer_class(str(engine_root))(
            sr=sr, threshold=opts.threshold_db, min_length=ms(opts.min_length), min_interval=ms(opts.min_interval),
            hop_size=max(1, ms(opts.hop_size)), max_sil_kept=ms(opts.max_sil_kept),
        )
    except ValueError as exc:
        raise ApiError("organize.bad_slice_options", str(exc), status=422) from exc
    chunks = slicer.slice(x)
    out = []
    for chunk in chunks.values():
        start, end = (int(v) for v in chunk["split_time"].split(","))
        if not chunk["slice"] and end > start:
            out.append((start, end))
    return out


def split_long(x: np.ndarray, sr: int, start: int, end: int, max_len: float, min_len: float) -> list[tuple[int, int]]:
    """Split ``x[start:end]`` into pieces ≤ ``max_len`` s, cutting at the quietest 20 ms frame
    of the second half of each window (pieces stay ≥ ``min_len`` s when possible)."""
    max_n, min_n = int(max_len * sr), int(min_len * sr)
    frame, hop = int(0.02 * sr), int(0.01 * sr)
    out = []
    s = start
    while end - s > max_n:
        lo = s + max(min_n, max_n // 2)
        hi = min(s + max_n, end - min_n)
        if hi - lo < frame:
            lo, hi = s + min_n, min(s + max_n, end - min_n)
        if hi - lo < frame:
            cut = s + max_n
        else:
            rms = frame_rms(x[lo:hi], frame, hop)
            cut = lo + int(np.argmin(rms)) * hop + frame // 2
        out.append((s, cut))
        s = cut
    out.append((s, end))
    return out
