"""Vocal activity: where someone is singing, phrases, and a smooth "singing now" key."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import dynamics as dyn

FRAME_S = 0.01


@dataclass
class Activity:
    rate: float                  # frames per second
    level_db: np.ndarray         # RMS level per frame (20 ms window)
    active: np.ndarray           # bool per frame (gaps < min_gap filled, blips < min_len dropped)
    median_db: float             # median level of the sung frames

    def phrases(self) -> list[tuple[float, float]]:
        """``(start_s, end_s)`` of each sung stretch."""
        a = np.concatenate([[False], self.active, [False]])
        edges = np.flatnonzero(np.diff(a.astype(np.int8)))
        return [(s / self.rate, e / self.rate) for s, e in zip(edges[::2], edges[1::2])]

    def key(self, sr: float, n: int, attack_s: float = 0.01, release_s: float = 0.2,
            range_db: float = 12.0) -> np.ndarray:
        """0..1 per sample: how strongly the voice is present (level within ``range_db`` of the
        median sung level → 1), with fast attack and slow release."""
        k = np.clip((self.level_db - (self.median_db - range_db)) / range_db, 0.0, 1.0) if np.isfinite(
            self.median_db) else np.zeros_like(self.level_db)
        k = dyn.ballistics(k, attack_s, release_s, self.rate)
        frame = int(round(sr / self.rate))
        return np.clip(dyn.upsample(k, frame, n), 0.0, 1.0)


def detect(x: np.ndarray, sr: float, *, below_median_db: float = 20.0, min_gap_s: float = 0.25,
           min_len_s: float = 0.3) -> Activity:
    """Frames louder than ``median - below_median_db`` (median of the loud end's frames) are sung."""
    frame = max(1, int(round(FRAME_S * sr)))
    rate = sr / frame
    ms = dyn.one_pole(dyn.block_mean_square(x, frame), dyn.coef(0.02, rate))
    level = 10.0 * np.log10(ms + 1e-12)
    loud = dyn.active_mask(level)
    if not loud.any():
        return Activity(rate, level, np.zeros(level.size, dtype=bool), float("-inf"))
    median = float(np.median(level[loud]))
    active = level > median - below_median_db
    gap = max(1, int(round(min_gap_s * rate)))
    active = ndimage.binary_closing(np.pad(active, gap), structure=np.ones(gap, bool))[gap:-gap]
    shortest = max(1, int(round(min_len_s * rate)))
    active = ndimage.binary_opening(np.pad(active, shortest), structure=np.ones(shortest, bool))[shortest:-shortest]
    return Activity(rate, level, active, median)
