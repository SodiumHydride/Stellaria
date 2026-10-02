"""Shared pieces of the DSP ops: errors, audio I/O, parameter parsing, fades.

Audio inside the ops is float32, channel-first ``(channels, n)``, 44.1 kHz, at most two channels.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..errors import ApiError

SR = 44100
DEFAULT_FADE_S = 0.010            # fade used at a cut whose requested fade is 0 but would click
CLICK_LEVEL = 10 ** (-50 / 20)    # a cut through audio louder than this clicks
CLICK_WINDOW_S = 0.002
RF64_BYTES = 4_000_000_000        # plain WAV cannot hold more

ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]


class DspError(ApiError):
    """A DSP failure with a stable code (reported as the CLI's final ``error`` line)."""

    def __init__(self, code: str, message: str, *, params: dict | None = None, status: int = 422) -> None:
        super().__init__(code, message, status=status, params=params)


def bad_params(reason: str, message: str, **extra: Any) -> DspError:
    """``dsp.bad_params`` with a stable ``reason`` key (see ``docs``/the report for the list)."""
    return DspError("dsp.bad_params", message, params={"reason": reason, **extra})


def bad_job(message: str) -> DspError:
    return DspError("dsp.bad_job", message)


# ----------------------------------------------------------------------------- parameters

def number(params: dict, key: str, default: float | None = None, *, lo: float | None = None,
           hi: float | None = None, reason: str) -> float:
    value = params.get(key, default)
    if value is None:
        if default is None:
            raise bad_params(reason, f"'{key}' is required")
        value = default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise bad_params(reason, f"'{key}' must be a number, got {value!r}")
    value = float(value)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise bad_params(reason, f"'{key}' = {value:g} is outside {lo:g}…{hi:g}")
    return value


def flag(params: dict, key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    return bool(value) if value is not None else default


# ----------------------------------------------------------------------------- audio I/O

def check_input(path: Any) -> Path:
    if not isinstance(path, (str, os.PathLike)) or not str(path):
        raise bad_job("Every input needs a 'path'")
    p = Path(path)
    if not p.is_file():
        raise DspError("dsp.input_missing", f"Input file not found: {p.name}", params={"path": str(p)})
    return p


def downmix(x: np.ndarray) -> np.ndarray:
    """``(channels, n)`` with more than two channels → stereo (ITU-style for 5.1/7.1 layouts)."""
    c = x.shape[0]
    if c <= 2:
        return x
    g = np.float32(1.0 / math.sqrt(2.0))
    left, right = x[0].astype(np.float32), x[1].astype(np.float32)
    if c in (6, 8):                              # FL FR FC LFE BL BR [SL SR]
        left = left + g * x[2] + g * x[4]
        right = right + g * x[2] + g * x[5]
        if c == 8:
            left = left + g * x[6]
            right = right + g * x[7]
        norm = np.float32(1.0 / (1.0 + g * (3 if c == 8 else 2)))
    else:                                        # anything else: the extra channels to both sides
        extra = x[2:].mean(axis=0)
        left, right = left + g * extra, right + g * extra
        norm = np.float32(1.0 / (1.0 + g))
    return np.stack([left * norm, right * norm]).astype(np.float32)


def _as_float(x: np.ndarray) -> np.ndarray:
    """float32 without changing the memory layout (channel-first views of interleaved buffers stay
    views, so writing them back needs no transpose copy)."""
    x = downmix(x)
    return x if x.dtype == np.float32 else x.astype(np.float32)


def read_audio(path: Any, *, start: int = 0, stop: int | None = None, sr: int | None = SR) -> np.ndarray:
    """Decode ``path`` → float32 ``(channels ≤ 2, n)`` at ``sr`` (native rate when None).

    ``start``/``stop`` (samples at ``sr``) read only a region; WAV/FLAC files already at ``sr`` are
    read directly from disk, anything else is decoded fully and sliced. The result is usually a
    channel-first *view* of an interleaved buffer; copy with ``order="K"`` to keep that layout."""
    import soundfile as sf

    from ..mixing.audioio import read

    p = check_input(path)
    try:
        info = sf.info(str(p))
        native = int(info.samplerate)
    except Exception:  # noqa: BLE001 - not a soundfile format (mp3, m4a …): decode below
        native = None
    if native is not None and (sr is None or native == sr):
        try:
            data, _ = sf.read(str(p), start=max(0, start), stop=stop, dtype="float32", always_2d=True)
            x = data.T
            if not np.isfinite(x).all():
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return _as_float(x)
        except Exception:  # noqa: BLE001 - fall back to the full decode
            pass
    x, _ = read(p, sr=sr)
    x = _as_float(x)
    if start > 0 or stop is not None:
        x = x[:, max(0, start): stop]
    return x


class Source:
    """An input read on demand: regions straight from disk when the file is WAV/FLAC at 44.1 kHz,
    otherwise decoded (and resampled) once."""

    def __init__(self, path: Any) -> None:
        import soundfile as sf

        self.path = check_input(path)
        self._full: np.ndarray | None = None
        try:
            info = sf.info(str(self.path))
            self.direct = int(info.samplerate) == SR
            self.frames = int(info.frames) if self.direct else 0
            self.channels = min(2, int(info.channels))
        except Exception:  # noqa: BLE001
            self.direct = False
        if not self.direct:
            self._full = read_audio(self.path)
            self.frames, self.channels = self._full.shape[-1], self._full.shape[0]

    def region(self, a: int, b: int) -> np.ndarray:
        a, b = max(0, a), min(self.frames, b)
        if b <= a:
            return np.zeros((self.channels, 0), dtype=np.float32)
        if self._full is not None:
            return self._full[:, a:b]
        return read_audio(self.path, start=a, stop=b)


def audio_info(path: Any) -> tuple[int, int]:
    """``(frames at 44.1 kHz, channels)`` without decoding when soundfile can read the header."""
    import soundfile as sf

    p = check_input(path)
    try:
        info = sf.info(str(p))
        frames = int(round(info.frames * SR / info.samplerate)) if info.samplerate != SR else int(info.frames)
        return frames, min(2, int(info.channels))
    except Exception:  # noqa: BLE001
        x = read_audio(p)
        return x.shape[-1], x.shape[0]


def write_wav(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    """Atomically write ``(channels, n)`` as 32-bit float WAV (RF64 beyond 4 GB)."""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32).T)
    fmt = "RF64" if data.nbytes > RF64_BYTES else "WAV"
    tmp = path.with_name(path.name + ".part.wav")
    try:
        sf.write(str(tmp), data, sr, subtype="FLOAT", format=fmt)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def match_channels(x: np.ndarray, channels: int) -> np.ndarray:
    if x.shape[0] == channels:
        return x
    if x.shape[0] == 1 and channels == 2:
        return np.repeat(x, 2, axis=0)
    if x.shape[0] == 2 and channels == 1:
        return x.mean(axis=0, keepdims=True, dtype=np.float32)
    return x[:channels]


# ----------------------------------------------------------------------------- fades

def fade_curve(n: int, *, rising: bool) -> np.ndarray:
    """Equal-power fade of ``n`` samples: ``sin`` rising, ``cos`` falling (sampled at bin centres,
    so the first sample of a fade-in is ≈ 0 and a fade-in/out pair sums to unit power)."""
    if n <= 0:
        return np.ones(0, dtype=np.float32)
    t = (np.arange(n, dtype=np.float64) + 0.5) / n
    return (np.sin(0.5 * np.pi * t) if rising else np.cos(0.5 * np.pi * t)).astype(np.float32)


def would_click(x: np.ndarray, at: int, sr: int = SR) -> bool:
    """True when cutting ``x`` at sample ``at`` cuts through audio above -50 dBFS."""
    w = max(1, int(CLICK_WINDOW_S * sr))
    lo, hi = max(0, at - w), min(x.shape[-1], at + w)
    if hi <= lo:
        return False
    return float(np.max(np.abs(x[..., lo:hi]))) > CLICK_LEVEL


def apply_fade_in(x: np.ndarray, n: int) -> None:
    n = min(n, x.shape[-1])
    if n > 0:
        x[..., :n] *= fade_curve(n, rising=True)


def apply_fade_out(x: np.ndarray, n: int) -> None:
    n = min(n, x.shape[-1])
    if n > 0:
        x[..., x.shape[-1] - n:] *= fade_curve(n, rising=False)


def seconds(samples: int | float, sr: int = SR) -> float:
    """Sample count → seconds, rounded for JSON (1 µs, well under a sample)."""
    return round(float(samples) / sr, 6)
