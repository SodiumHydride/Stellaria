"""Validation snapshots: audio extracted from tfevents (cached as wav) and log-mel images.

The mel matches the vocoder's front end (NSF-HiFiGAN 44.1 kHz: n_fft 2048, hop 512, 128 bins,
fmin 40, fmax 16000, natural log with a 1e-5 floor) and is computed with numpy only.
"""

from __future__ import annotations

import base64
import io
import json
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import numpy as np
import soundfile as sf

from ..audio.io import resample
from ..errors import not_found
from ..schemas import ValidationSample, ValidationSnapshot
from .tfevents import TfEventsReader

SR, N_FFT, HOP, N_MELS, FMIN, FMAX = 44100, 2048, 512, 128, 40.0, 16000.0
MEL_MIN, MEL_MAX = -14.0, 3.5


def sample_dir_name(name: str) -> str:
    """Folder-safe form of a sample name (multi-speaker names contain the speaker folder)."""
    return name.replace("/", "__").replace("\\", "__")


def snapshot(mid: str, reader: TfEventsReader, step: int | None) -> ValidationSnapshot | None:
    steps = reader.validation_steps()
    if not steps:
        return None
    if step is None:
        step = steps[-1]
    if step not in steps:
        raise not_found("validation", str(step))
    samples = []
    for name in reader.validation_names(step):
        base = f"/models/{mid}/validation/{step}/{quote(name, safe='')}"
        samples.append(ValidationSample(name=name, gt_url=f"{base}/gt.wav", pred_url=f"{base}/pred.wav",
                                        mel_url=f"{base}/mel"))
    return ValidationSnapshot(step=step, samples=samples)


def audio_file(cache_dir: Path, reader: TfEventsReader, step: int, name: str, kind: str) -> Path:
    """Extract ``<name>/<kind>.wav`` of ``step`` into the cache and return its path."""
    path = cache_dir / "validation" / str(step) / sample_dir_name(name) / f"{kind}.wav"
    if path.is_file():
        return path
    data = reader.audio_bytes(step, name, kind)
    if data is None:
        raise not_found("validation", f"{step}/{name}/{kind}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


@lru_cache(maxsize=1)
def _mel_basis() -> np.ndarray:
    import librosa

    return librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX).astype(np.float64)


def log_mel(x: np.ndarray, sr: int = SR) -> np.ndarray:
    """(frames, 128) natural-log mel, same framing as ``nsf_hifigan.nvSTFT.STFT.get_mel``."""
    x = resample(x.astype(np.float32), sr, SR).astype(np.float64)
    pad_l = (N_FFT - HOP) // 2
    pad_r = max((N_FFT - HOP + 1) // 2, N_FFT - len(x) - pad_l)
    y = np.pad(x, (pad_l, pad_r), mode="reflect" if pad_r < len(x) else "constant")
    n_frames = 1 + (len(y) - N_FFT) // HOP
    if n_frames <= 0:
        return np.zeros((0, N_MELS))
    window = np.hanning(N_FFT + 1)[:-1]  # periodic Hann, like torch.hann_window
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP][:n_frames] * window
    spec = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1))
    mel = spec @ _mel_basis().T
    return np.log(np.maximum(mel, 1e-5))


def quantize(mel: np.ndarray) -> str:
    q = np.clip(np.round((mel - MEL_MIN) / (MEL_MAX - MEL_MIN) * 255.0), 0, 255).astype(np.uint8)
    return base64.b64encode(np.ascontiguousarray(q).tobytes()).decode("ascii")


def mel_payload(cache_dir: Path, reader: TfEventsReader, step: int, name: str) -> dict:
    cached = cache_dir / "validation" / str(step) / sample_dir_name(name) / "mel.json"
    if cached.is_file():
        return json.loads(cached.read_text(encoding="utf-8"))
    mels = []
    for kind in ("pred", "gt"):
        path = audio_file(cache_dir, reader, step, name, kind)
        data, sr = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
        mels.append(log_mel(data.mean(axis=1), sr))
    frames = min(len(mels[0]), len(mels[1]))
    payload = {"frames": frames, "bins": N_MELS, "data": quantize(mels[0][:frames]), "gtData": quantize(mels[1][:frames])}
    cached.write_text(json.dumps(payload), encoding="utf-8")
    return payload
