"""Per-clip measurements (numpy + parselmouth + MFCC). No torch.

Results are cached per clip in ``<character>/cache/analysis/<clip_id>.npz`` keyed by the
file's size and mtime, so re-running a health check only analyzes new or changed clips.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..audio.dsp import EPS, db, frame_rms
from ..audio.io import read_wav, resample
from ..schemas import ClipMetrics

SILENCE_DB = -45.0
PITCH_HOP = 0.01  # seconds per f0 frame
PITCH_FLOOR = 50.0
PITCH_CEILING = 1100.0
ANALYSIS_VERSION = 2


@dataclass
class ClipAnalysis:
    metrics: ClipMetrics
    f0: np.ndarray  # Hz per PITCH_HOP, 0 = unvoiced
    mfcc: np.ndarray  # mean of 20 MFCCs over non-silent frames
    pcm_hash: str


def clipping_ratio(x: np.ndarray) -> float:
    """Share of clipped samples: |x| ≥ 0.999, or flat tops (≥ 4 equal samples near the peak),
    which still reveals clipping after the organize step scaled the clip down."""
    if len(x) == 0:
        return 0.0
    ax = np.abs(x)
    hard = ax >= 0.999
    peak = float(ax.max())
    flat = np.zeros_like(hard)
    if peak > 0.1:
        near = ax >= 0.98 * peak
        same = np.concatenate([[False], np.abs(np.diff(x)) < 1e-6]) & near
        # mark runs of ≥ 3 "same as previous" samples (i.e. ≥ 4 equal samples)
        run = np.zeros(len(x), dtype=np.int32)
        idx = np.flatnonzero(same)
        if len(idx):
            breaks = np.flatnonzero(np.diff(idx) != 1)
            starts = np.concatenate([[0], breaks + 1])
            ends = np.concatenate([breaks, [len(idx) - 1]])
            for s, e in zip(starts, ends):
                if e - s + 1 >= 3:
                    run[idx[s] - 1: idx[e] + 1] = 1
        flat = run.astype(bool)
    return float(np.mean(hard | flat))


def basic_metrics(x: np.ndarray, sr: int, voiced: np.ndarray | None = None) -> dict[str, float | None]:
    """Peak/RMS/silence/clipping and an SNR estimate (90th percentile frame level minus the
    10th percentile of the unvoiced frames; ``None`` when the clip has too few pauses to tell)."""
    peak = float(np.max(np.abs(x))) if len(x) else 0.0
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)) if len(x) else EPS
    frames_db = db(frame_rms(x, int(0.02 * sr), int(0.01 * sr)))
    silence_ratio = float(np.mean(frames_db < SILENCE_DB))
    return {
        "peak_db": round(float(db(peak)), 2),
        "rms_db": round(float(db(rms)), 2),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio(x), 6),
        "snr_db": snr_estimate(frames_db, voiced),
    }


def silence_stats(x: np.ndarray, sr: int) -> tuple[float, float]:
    """(share of 20 ms frames below -45 dBFS, overall RMS in dBFS)."""
    frames_db = db(frame_rms(x, int(0.02 * sr), int(0.01 * sr)))
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)) if len(x) else EPS
    return float(np.mean(frames_db < SILENCE_DB)), float(db(rms))


def snr_estimate(frames_db: np.ndarray, voiced: np.ndarray | None) -> float | None:
    if len(frames_db) < 10:
        return None
    signal = float(np.percentile(frames_db, 90))
    if voiced is None:
        floor_frames = frames_db
    else:
        n = min(len(frames_db), len(voiced))
        floor_frames = frames_db[:n][~voiced[:n]]
        if len(floor_frames) < max(5, 0.05 * n):
            return None  # continuous singing: no pause to measure the noise floor in
    return round(signal - float(np.percentile(floor_frames, 10)), 2)


def pitch_track(x16: np.ndarray, sr: int = 16000) -> np.ndarray:
    import parselmouth

    if len(x16) < int(0.1 * sr):
        return np.zeros(max(1, int(len(x16) / sr / PITCH_HOP)), dtype=np.float32)
    snd = parselmouth.Sound(x16.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch_ac(time_step=PITCH_HOP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    return np.nan_to_num(pitch.selected_array["frequency"]).astype(np.float32)


def mfcc_means(x16: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Mean of 20 MFCCs over the non-silent frames (timbre fingerprint for outlier detection)."""
    import librosa

    if len(x16) < 512:
        return np.zeros(20, dtype=np.float32)
    mfcc = librosa.feature.mfcc(y=x16, sr=sr, n_mfcc=20, n_fft=512, hop_length=160)
    energy = db(frame_rms(np.pad(x16, (256, 256)), 512, 160))[: mfcc.shape[1]]
    voiced = mfcc[:, : len(energy)][:, energy > SILENCE_DB]
    if voiced.shape[1] < 5:
        voiced = mfcc
    return voiced.mean(axis=1).astype(np.float32)


def pcm_hash(x: np.ndarray) -> str:
    pcm = np.round(np.clip(x, -1, 1) * 32767).astype("<i2")
    return hashlib.sha1(pcm.tobytes()).hexdigest()


def f0_summary(f0: np.ndarray) -> dict[str, float | None]:
    voiced = f0[f0 > 0]
    if len(f0) == 0:
        return {"f0_low": None, "f0_high": None, "f0_median": None, "voiced_ratio": None}
    if len(voiced) < 5:
        return {"f0_low": None, "f0_high": None, "f0_median": None, "voiced_ratio": round(len(voiced) / len(f0), 4)}
    return {
        "f0_low": round(float(np.percentile(voiced, 5)), 2),
        "f0_high": round(float(np.percentile(voiced, 95)), 2),
        "f0_median": round(float(np.median(voiced)), 2),
        "voiced_ratio": round(len(voiced) / len(f0), 4),
    }


def analyze_audio(x: np.ndarray, sr: int) -> ClipAnalysis:
    x16 = resample(x.astype(np.float32), sr, 16000)
    f0 = pitch_track(x16)
    metrics = ClipMetrics(**basic_metrics(x, sr, voiced=f0 > 0), **f0_summary(f0))
    return ClipAnalysis(metrics=metrics, f0=f0, mfcc=mfcc_means(x16), pcm_hash=pcm_hash(x))


def _file_key(path: Path) -> str:
    st = path.stat()
    return f"{ANALYSIS_VERSION}-{st.st_size}-{st.st_mtime_ns}"


def analyze_file(path: Path, cache_dir: Path, clip_id: str, force: bool = False) -> ClipAnalysis:
    cache = cache_dir / f"{clip_id}.npz"
    key = _file_key(path)
    if not force and cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                if str(z["key"]) == key:
                    return ClipAnalysis(
                        metrics=ClipMetrics.model_validate_json(str(z["metrics"])),
                        f0=z["f0"], mfcc=z["mfcc"], pcm_hash=str(z["hash"]),
                    )
        except Exception:  # noqa: BLE001 - corrupt cache: recompute
            pass
    x, sr = read_wav(path)
    result = analyze_audio(x, sr)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.stem + ".tmp.npz")
    np.savez(tmp, key=key, metrics=result.metrics.model_dump_json(), f0=result.f0, mfcc=result.mfcc,
             hash=result.pcm_hash)
    tmp.replace(cache)
    return result


def outlier_scores(features: np.ndarray) -> np.ndarray:
    """Robust distance of each row from the dataset median (diagonal Mahalanobis with
    median/MAD scaling). ~1 for typical clips; large values are "outliers"."""
    med = np.median(features, axis=0)
    mad = np.median(np.abs(features - med), axis=0) * 1.4826
    scale = np.maximum(mad, 1e-3 * (np.abs(med) + 1.0))
    z = (features - med) / scale
    z = np.clip(z, -50, 50)
    return np.sqrt(np.mean(z ** 2, axis=1))
