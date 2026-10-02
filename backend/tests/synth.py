"""Synthetic singing-like audio for tests (harmonics, vibrato, phrases separated by silence)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def note(duration: float, f0: float, sr: int = SR, rng: np.random.Generator | None = None,
         amp: float = 0.3) -> np.ndarray:
    rng = rng or np.random.default_rng(0)
    n = int(duration * sr)
    t = np.arange(n) / sr
    freq = f0 * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(freq) / sr
    out = np.zeros(n)
    for k in range(1, 13):
        if f0 * k > sr / 2.5:
            break
        weight = (1.0 / k) * (1.4 if 500 < f0 * k < 1200 else 1.0) * (1.3 if 2500 < f0 * k < 3500 else 1.0)
        out += weight * np.sin(k * phase + rng.uniform(0, 6))
    env = np.ones(n)
    ramp = min(n // 2, int(0.04 * sr))
    env[:ramp] = np.linspace(0, 1, ramp)
    env[n - ramp:] = np.linspace(1, 0, ramp)
    out = out * env
    out = out / (np.max(np.abs(out)) + 1e-9) * amp
    return (out + rng.normal(0, 0.002, n)).astype(np.float32)


def phrase(seconds: float, base_f0: float, rng: np.random.Generator) -> np.ndarray:
    parts, total = [], 0.0
    while total < seconds:
        d = float(rng.uniform(0.4, 0.9))
        semis = float(rng.integers(-5, 8))
        parts.append(note(d, base_f0 * 2 ** (semis / 12), rng=rng, amp=float(rng.uniform(0.2, 0.35))))
        total += d
    return np.concatenate(parts)[: int(seconds * SR)]


def song(seconds: float, base_f0: float = 220.0, seed: int = 0, phrase_len: float = 4.0,
         gap: float = 0.7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    parts, total = [np.zeros(int(0.3 * SR), dtype=np.float32)], 0.3
    while total < seconds:
        parts.append(phrase(phrase_len, base_f0, rng))
        parts.append(np.zeros(int(gap * SR), dtype=np.float32))
        total += phrase_len + gap
    return np.concatenate(parts)[: int(seconds * SR)]


def write(path: Path, audio: np.ndarray, sr: int = SR) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".mp3":
        import imageio_ffmpeg

        wav = path.with_suffix(".tmp.wav")
        sf.write(wav, audio, sr)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(path)], check=True)
        wav.unlink()
    else:
        sf.write(path, audio, sr)
    return path
