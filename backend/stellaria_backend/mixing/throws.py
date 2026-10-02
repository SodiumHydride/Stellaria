"""Delay throws: the last word of a phrase echoes into the following gap (tempo-synced)."""

from __future__ import annotations

import numpy as np
from scipy import signal

from . import reverb
from .activity import Activity


def throws(vocal: np.ndarray, sr: float, act: Activity, delay_s: float, *, level_db: float = -8.0,
           feedback: float = 0.4, min_gap_s: float = 0.6, tail_s: float = 0.45) -> tuple[np.ndarray, int]:
    """``(2, n)`` stereo echoes and the number of throws.

    For every phrase followed by a gap of at least ``min_gap_s``, its last ``tail_s`` (at most one
    delay time) is sent into a filtered ping-pong delay (HP 400 Hz, LP 3.5 kHz). The echoes are
    ducked by up to 12 dB while the voice sings, so they only bloom in the gaps."""
    n = vocal.size
    out = np.zeros((2, n), dtype=np.float32)
    phrases = act.phrases()
    if not phrases or delay_s <= 0:
        return out, 0
    ir = reverb.delay_ir(reverb.DelaySpec(time_s=delay_s, feedback=feedback, level_db=0.0, repeats=5,
                                          highpass_hz=400.0, lowpass_hz=3500.0), sr)
    fade = int(0.02 * sr)
    count = 0
    for i, (_start, end) in enumerate(phrases):
        nxt = phrases[i + 1][0] if i + 1 < len(phrases) else n / sr
        if nxt - end < min_gap_s:
            continue
        b = min(n, int(end * sr))
        a = max(0, b - int(min(tail_s, delay_s) * sr))
        if b - a < 4 * fade:
            continue
        send = vocal[a:b].astype(np.float64)
        w = np.ones(b - a)
        w[:fade] = np.linspace(0.0, 1.0, fade)
        w[-fade:] = np.linspace(1.0, 0.0, fade)
        wet = signal.fftconvolve(send[None, :] * w, ir.astype(np.float64), axes=-1)
        m = min(wet.shape[-1], n - a)
        out[:, a: a + m] += wet[:, :m].astype(np.float32)
        count += 1
    duck = 1.0 - 0.75 * act.key(sr, n, attack_s=0.03, release_s=0.1)
    out *= (duck * 10.0 ** (level_db / 20.0)).astype(np.float32)
    return out, count
