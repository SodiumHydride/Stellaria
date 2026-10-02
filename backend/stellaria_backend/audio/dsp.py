"""Frame-level loudness helpers shared by the organize pipeline and the health check."""

from __future__ import annotations

import numpy as np

EPS = 1e-10


def db(x: float | np.ndarray) -> float | np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(x), EPS))


def frame_rms(x: np.ndarray, frame: int, hop: int) -> np.ndarray:
    """RMS per frame (no padding beyond the last full frame; at least one frame).

    O(n) without a frames × window matrix: squared samples are summed per hop-sized block
    (float32 in, float64 sums) when ``frame`` is a multiple of ``hop``, else via a cumulative sum."""
    if len(x) < frame:
        return np.array([np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)]) if len(x) else np.array([EPS])
    n = 1 + (len(x) - frame) // hop
    if frame % hop == 0:
        k, blocks = frame // hop, len(x) // hop
        sq = np.square(x[: blocks * hop], dtype=np.float32).reshape(blocks, hop)
        cs = np.concatenate([[0.0], np.cumsum(sq.sum(axis=1, dtype=np.float64))])
        energy = cs[k: k + n] - cs[:n]
    else:
        cs = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
        starts = hop * np.arange(n)
        energy = cs[starts + frame] - cs[starts]
    return np.sqrt(np.maximum(energy, 0.0) / frame + EPS)


def active_rms_db(x: np.ndarray, sr: int, floor_db: float = -45.0) -> float:
    """RMS in dBFS over the frames louder than ``floor_db`` (loudness of the sung parts)."""
    hop = int(0.025 * sr)
    rms = frame_rms(x, 2 * hop, hop)
    rms_db = db(rms)
    active = rms[rms_db > floor_db]
    if len(active) == 0:
        return float(db(np.sqrt(np.mean(x.astype(np.float64) ** 2) + EPS)))
    return float(db(np.sqrt(np.mean(active ** 2))))
