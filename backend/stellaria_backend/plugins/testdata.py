"""The 8-second test song + vocal for ``POST /plugins/{id}/test``.

Synthesized once (numpy, deterministic) into ``<data>/plugins/testdata/`` instead of shipping
binaries: a stereo backing (pad chords, bass, kick and hats at 100 BPM, C-Am-F-G) and a dry
mono "sung" melody (harmonic voice with vibrato and a formant tilt) over it.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100
SECONDS = 8.0
BPM = 100.0
VERSION = "1"
BACKING_NAME = "Stellaria test - backing.wav"
VOCAL_NAME = "Stellaria test - vocal.wav"

_lock = threading.Lock()

CHORDS = [  # MIDI notes, one chord per 2 s
    (48, 60, 64, 67),  # C
    (45, 57, 60, 64),  # Am
    (41, 57, 60, 65),  # F
    (43, 59, 62, 67),  # G
]
MELODY = [  # (start beat, beats, midi)
    (0.0, 1.0, 72), (1.0, 0.5, 74), (1.5, 1.5, 76), (3.5, 0.5, 74),
    (4.0, 1.0, 72), (5.0, 1.0, 69), (6.0, 1.0, 72),
    (7.0, 1.0, 69), (8.0, 1.0, 72), (9.0, 1.5, 77), (10.5, 0.5, 76),
    (11.0, 1.0, 74), (12.0, 1.3, 71),
]


def _hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def _env(n: int, attack: float, release: float) -> np.ndarray:
    env = np.ones(n)
    a, r = min(n // 2, int(attack * SR)), min(n // 2, int(release * SR))
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[n - r:] *= np.linspace(1, 0, r)
    return env


def _tone(freq: float, n: int, harmonics: int, rolloff: float, rng: np.random.Generator,
          detune: float = 0.0, vibrato: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    f = freq * (1 + detune) * (1 + vibrato * np.sin(2 * np.pi * 5.3 * t + rng.uniform(0, 6)))
    phase = 2 * np.pi * np.cumsum(f) / SR
    out = np.zeros(n)
    for k in range(1, harmonics + 1):
        if freq * k > SR / 2.2:
            break
        out += np.sin(k * phase + rng.uniform(0, 6)) / k ** rolloff
    return out


def backing(rng: np.random.Generator) -> np.ndarray:
    n = int(SECONDS * SR)
    left, right = np.zeros(n), np.zeros(n)
    chord_len = int(2.0 * SR)
    for i, chord in enumerate(CHORDS):
        s = i * chord_len
        m = min(chord_len, n - s)
        env = _env(m, 0.15, 0.3)
        bass = _tone(_hz(chord[0] - 12), m, 4, 1.5, rng) * env * 0.45
        left[s:s + m] += bass
        right[s:s + m] += bass
        for j, note in enumerate(chord[1:]):
            pad_l = _tone(_hz(note), m, 8, 1.2, rng, detune=-0.003) * env * 0.12
            pad_r = _tone(_hz(note), m, 8, 1.2, rng, detune=0.003) * env * 0.12
            left[s:s + m] += pad_l * (1.1 if j % 2 else 0.9)
            right[s:s + m] += pad_r * (0.9 if j % 2 else 1.1)
    beat = 60.0 / BPM
    t = np.arange(int(0.25 * SR)) / SR
    kick = np.sin(2 * np.pi * (50 * t + (120 - 50) * (1 - np.exp(-t * 30)) / 30)) * np.exp(-t * 18) * 0.8
    hat_n = int(0.05 * SR)
    for b in range(int(SECONDS / beat) + 1):
        s = int(b * beat * SR)
        if s < n:
            m = min(len(kick), n - s)
            left[s:s + m] += kick[:m]
            right[s:s + m] += kick[:m]
        h = int((b + 0.5) * beat * SR)
        if h < n:
            m = min(hat_n, n - h)
            noise = np.diff(rng.normal(0, 1, m + 1)) * np.exp(-np.arange(m) / SR * 90) * 0.06
            left[h:h + m] += noise * 0.8
            right[h:h + m] += noise
    out = np.stack([left, right], axis=1)
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)  # about -6 dBFS


def vocal(rng: np.random.Generator) -> np.ndarray:
    n = int(SECONDS * SR)
    out = np.zeros(n)
    beat = 60.0 / BPM
    for start, beats, midi in MELODY:
        s = int(start * beat * SR)
        m = min(int(beats * beat * SR * 0.95), n - s)
        if m <= 0:
            continue
        f0 = _hz(midi)
        t = np.arange(m) / SR
        freq = f0 * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t / 0.3, 0, 1))
        phase = 2 * np.pi * np.cumsum(freq) / SR
        voice = np.zeros(m)
        for k in range(1, 30):
            fk = f0 * k
            if fk > SR / 2.5:
                break
            formant = 1.0 + 1.6 * np.exp(-((fk - 700) / 250) ** 2) + 1.0 * np.exp(-((fk - 1200) / 300) ** 2) \
                + 0.6 * np.exp(-((fk - 2800) / 500) ** 2)
            voice += formant / k * np.sin(k * phase + rng.uniform(0, 6))
        voice += rng.normal(0, 0.01, m)  # breath
        out[s:s + m] += voice * _env(m, 0.04, 0.08)
    return (out / (np.abs(out).max() + 1e-9) * 0.5).astype(np.float32)


def ensure_testdata(data_dir: Path) -> tuple[Path, Path]:
    """``(backing, vocal)`` WAV paths, synthesized on first use."""
    folder = data_dir / "plugins" / "testdata"
    back, voc, marker = folder / BACKING_NAME, folder / VOCAL_NAME, folder / "version"
    with _lock:
        current = marker.read_text(encoding="utf-8").strip() if marker.is_file() else ""
        if current != VERSION or not back.is_file() or not voc.is_file():
            folder.mkdir(parents=True, exist_ok=True)
            rng = np.random.default_rng(20261002)
            sf.write(str(back), backing(rng), SR, subtype="PCM_16")
            sf.write(str(voc), vocal(rng), SR, subtype="PCM_16")
            marker.write_text(VERSION, encoding="utf-8")
    return back, voc
