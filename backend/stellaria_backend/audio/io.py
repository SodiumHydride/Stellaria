"""Decoding (soundfile, falling back to the bundled imageio-ffmpeg binary), writing, peaks.

Users do not have ffmpeg installed, so only ``imageio_ffmpeg.get_ffmpeg_exe()`` is used.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from ..errors import ApiError
from ..util import procs

SR = 44100
SOUNDFILE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
AUDIO_EXTS = {
    ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".wma", ".aiff", ".aif",
    ".webm", ".mp4", ".mkv", ".mov", ".ape", ".wv",
}


@dataclass
class AudioInfo:
    duration: float
    sample_rate: int
    channels: int


@lru_cache(maxsize=1)
def ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _decode_failed(path: Path, detail: str) -> ApiError:
    return ApiError("audio.decode_failed", f"Could not decode '{path.name}'", status=422, detail=detail)


def probe(path: Path) -> AudioInfo:
    """Duration / sample rate / channels without decoding the whole file when possible."""
    if path.suffix.lower() in SOUNDFILE_EXTS:
        try:
            info = sf.info(str(path))
            return AudioInfo(info.frames / info.samplerate, info.samplerate, info.channels)
        except Exception:  # noqa: BLE001 - fall back to ffmpeg for odd wav flavours
            pass
    result = procs.run([ffmpeg_exe(), "-hide_banner", "-nostdin", "-i", str(path)], timeout=60)
    text = result.stderr.decode("utf-8", errors="replace")
    stream = re.search(r"Audio: [^\n]*?, (\d+) Hz, ([^,\n]+)", text)
    duration = re.search(r"Duration: (\d+):(\d+):(\d+(?:\.\d+)?)", text)
    if not stream:
        raise _decode_failed(path, text[-2000:])
    layout = stream.group(2).strip()
    channels = {"mono": 1, "stereo": 2}.get(layout)
    if channels is None:
        m = re.match(r"(\d+) channels", layout)
        channels = int(m.group(1)) if m else 2
    seconds = 0.0
    if duration:
        h, m_, s = duration.groups()
        seconds = int(h) * 3600 + int(m_) * 60 + float(s)
    return AudioInfo(seconds, int(stream.group(1)), channels)


def decode(path: Path, sr: int = SR, mono: bool = True, max_channels: int = 2) -> np.ndarray:
    """Decode ``path`` to float32 at ``sr``. Mono → shape (n,); otherwise (n, channels)."""
    if path.suffix.lower() in SOUNDFILE_EXTS:
        try:
            data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
            return _shape(resample(data, file_sr, sr), mono, max_channels)
        except Exception:  # noqa: BLE001 - try ffmpeg below
            pass
    channels = 1 if mono else min(max_channels, max(1, probe(path).channels))
    args = [
        ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path), "-vn",
        "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels), "-ar", str(sr), "pipe:1",
    ]
    proc = procs.popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0 or not out:
        raise _decode_failed(path, err.decode("utf-8", errors="replace")[-2000:])
    data = np.frombuffer(out, dtype=np.float32).reshape(-1, channels)
    return _shape(data, mono, max_channels)


def _shape(data: np.ndarray, mono: bool, max_channels: int) -> np.ndarray:
    if mono:
        return np.ascontiguousarray(data.mean(axis=1) if data.ndim == 2 else data, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] > max_channels:
        data = data[:, :max_channels]
    return np.ascontiguousarray(data, dtype=np.float32)


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    import soxr

    return soxr.resample(data, sr_in, sr_out, quality="HQ").astype(np.float32)


def write_wav(path: Path, data: np.ndarray, sr: int = SR, subtype: str = "PCM_16") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.wav")
    sf.write(str(tmp), np.clip(data, -1.0, 1.0), sr, subtype=subtype, format="WAV")
    tmp.replace(path)


def read_wav(path: Path, mono: bool = True) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return (data.mean(axis=1) if mono else data), sr


def encode(path: Path, data: np.ndarray, sr: int, fmt: str) -> None:
    """Write wav/flac with soundfile, mp3 through the bundled ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.clip(data, -1.0, 1.0).astype(np.float32)
    if fmt == "wav":
        sf.write(str(path), data, sr, subtype="PCM_24", format="WAV")
    elif fmt == "flac":
        sf.write(str(path), data, sr, subtype="PCM_24", format="FLAC")
    elif fmt == "mp3":
        channels = 1 if data.ndim == 1 else data.shape[1]
        args = [
            ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "f32le", "-ar", str(sr), "-ac", str(channels), "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "320k", str(path),
        ]
        proc = procs.popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, err = proc.communicate(np.ascontiguousarray(data).tobytes())
        if proc.returncode != 0:
            raise ApiError("audio.encode_failed", "Could not encode mp3", status=500,
                           detail=err.decode("utf-8", errors="replace")[-2000:])
    else:
        raise ApiError("audio.bad_format", f"Unsupported format '{fmt}'")


def peaks(path: Path, n: int = 800) -> tuple[list[float], float]:
    """Max absolute amplitude (0..1) in ``n`` equal buckets, plus the duration in seconds."""
    n = max(1, min(int(n), 20000))
    with sf.SoundFile(str(path)) as f:
        frames, sr = f.frames, f.samplerate
        duration = frames / sr if sr else 0.0
        if frames == 0:
            return [0.0] * n, 0.0
        out = np.zeros(n, dtype=np.float32)
        edges = np.linspace(0, frames, n + 1).astype(np.int64)
        block = 1 << 18
        pos = 0
        while pos < frames:
            chunk = f.read(min(block, frames - pos), dtype="float32", always_2d=True)
            if len(chunk) == 0:
                break
            amp = np.abs(chunk).max(axis=1)
            lo = np.searchsorted(edges, pos, side="right") - 1
            hi = np.searchsorted(edges, pos + len(amp), side="left")
            for b in range(max(0, lo), min(n, hi)):
                s = max(edges[b], pos) - pos
                e = min(edges[b + 1], pos + len(amp)) - pos
                if e > s:
                    out[b] = max(out[b], float(amp[s:e].max()))
            pos += len(amp)
    return [round(float(v), 4) for v in np.clip(out, 0, 1)], duration
