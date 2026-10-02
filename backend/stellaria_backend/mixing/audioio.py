"""Reading/writing for the mixer: float32, channel-first ``(channels, n)``, 44.1 kHz."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from ..errors import ApiError

SR = 44100


class MixError(ApiError):
    """A mixer failure with a stable code (reported as the CLI's final ``error`` line)."""

    def __init__(self, code: str, message: str, *, status: int = 422, params: dict | None = None) -> None:
        super().__init__(code, message, status=status, params=params)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resampling along the last axis (``scipy.signal.resample_poly``)."""
    if sr_in == sr_out or x.shape[-1] == 0:
        return x
    from scipy import signal

    g = math.gcd(int(sr_in), int(sr_out))
    return signal.resample_poly(x, int(sr_out) // g, int(sr_in) // g, axis=-1).astype(np.float32)


def read_check(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_file():
        raise MixError("mix.input_missing", f"Input file not found: {p.name}", params={"path": str(p)})
    return p


def read(path: str | Path, sr: int | None = SR) -> tuple[np.ndarray, int]:
    """Decode ``path`` → ``((channels, n) float32, sample_rate)``; resampled to ``sr`` unless None.

    soundfile first (WAV/FLAC/OGG/AIFF); anything else goes through the backend's ffmpeg decoder."""
    p = read_check(path)
    try:
        data, file_sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 - not a soundfile format: decode with ffmpeg
        from ..audio.io import decode

        file_sr = sr or SR
        data = decode(p, sr=file_sr, mono=False, max_channels=8)
        data = data[:, None] if data.ndim == 1 else data
    data = data.T                       # channel-first view; to_mono/to_stereo make it contiguous
    if not np.isfinite(data).all():
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    if sr is not None and file_sr != sr:
        data = resample(data, int(file_sr), int(sr))
        file_sr = sr
    return data.astype(np.float32, copy=False), int(file_sr)


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.ascontiguousarray(x, dtype=np.float32)
    if x.shape[0] == 1:
        return np.ascontiguousarray(x[0], dtype=np.float32)
    return x.mean(axis=0, dtype=np.float32)


def to_stereo(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return np.stack([x, x]).astype(np.float32)
    if x.shape[0] == 1:
        return np.repeat(x, 2, axis=0).astype(np.float32)
    return np.ascontiguousarray(x[:2], dtype=np.float32)


def fit(x: np.ndarray, n: int) -> np.ndarray:
    """Zero-pad (or cut) along the last axis to ``n`` samples."""
    m = x.shape[-1]
    if m == n:
        return x
    if m > n:
        return x[..., :n]
    pad = np.zeros(x.shape[:-1] + (n - m,), dtype=x.dtype)
    return np.concatenate([x, pad], axis=-1)


def write(path: str | Path, x: np.ndarray, sr: int = SR) -> Path:
    """Atomically write ``(channels, n)`` audio: 32-bit float WAV (24-bit PCM for ``.flac``)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    flac = p.suffix.lower() == ".flac"
    tmp = p.with_name(p.name + (".part.flac" if flac else ".part.wav"))
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32).T)
    if flac:
        sf.write(str(tmp), np.clip(data, -1.0, 1.0), sr, subtype="PCM_24", format="FLAC")
    else:
        sf.write(str(tmp), data, sr, subtype="FLOAT", format="WAV")
    os.replace(tmp, p)
    return p


def place(x: np.ndarray, start: int, n: int) -> np.ndarray:
    """``x`` (last axis = time) placed at sample ``start`` of an ``n``-sample zero timeline."""
    out = np.zeros(x.shape[:-1] + (n,), dtype=np.float32)
    if start >= n or start + x.shape[-1] <= 0:
        return out
    src0 = max(0, -start)
    dst0 = max(0, start)
    m = min(x.shape[-1] - src0, n - dst0)
    out[..., dst0: dst0 + m] = x[..., src0: src0 + m]
    return out


def timeline(lengths_offsets: list[tuple[int, float]], sr: int = SR) -> tuple[float, int, list[int]]:
    """``[(samples, offset_s)]`` → ``(base_offset_s, total_samples, start_sample_per_item)``:
    items are placed at ``offset - min(offset)``; the timeline ends with the last item."""
    if not lengths_offsets:
        return 0.0, 0, []
    base = min(o for _, o in lengths_offsets)
    starts = [int(round((o - base) * sr)) for _, o in lengths_offsets]
    total = max(s + m for (m, _), s in zip(lengths_offsets, starts))
    return base, total, starts
