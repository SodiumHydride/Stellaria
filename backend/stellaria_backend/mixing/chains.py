"""Vocal chains per role and ``vocalChain`` mode.

======================  =========================================================================
lead, ``full``          staging → HP 90 → low cut 230 Hz, −2 dB @ 2.7 kHz → gate → de-esser → resonances →
                        presence EQ → (spectral match) → peak compressor → leveling compressor
lead, ``light``         staging → HP 90 → gate → de-esser (≤ 4 dB) → resonances (60 %) → (spectral match)
                        → gentle leveler
harmony (full/light)    staging → HP 90 → low cut 300 Hz → gate → de-esser → resonances (60 %) → half presence
                        → leveler
any, ``off``            staging only (the user processed the vocal already)
======================  =========================================================================

Every vocal is staged to −20 dBFS active RMS; the role's level (harmonies −4 dB) is applied by
the mixer. The chain is split in two (:func:`prepare`, :func:`finish`) because the leveler's
release is tempo-synced and the tempo is detected while :func:`prepare` runs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import dynamics as dyn
from . import filters, spec, spectral

VOCAL_RMS_DB = -20.0

LogFn = Callable[[str], None]


def _gain(db_value: float) -> float:
    return 10.0 ** (float(db_value) / 20.0)


@dataclass
class Prepared:
    audio: np.ndarray            # after EQ + gate + de-esser, before the compressors
    level_db: float | None       # input active RMS (None = silent)
    role: str
    chain: str
    info: dict = field(default_factory=dict)


def prepare(x: np.ndarray, sr: float, s: spec.MixSettings, role: str = "lead", chain: str = "full") -> Prepared:
    """Staging, EQ, gate, de-esser (nothing tempo-dependent)."""
    level = dyn.active_rms_db(x, sr)
    if level is None:
        return Prepared(np.zeros_like(x, dtype=np.float32), None, role, chain)
    x = (x * _gain(VOCAL_RMS_DB - level)).astype(np.float32)
    if chain == "off":
        return Prepared(x, level, role, chain)
    info: dict = {}
    harmony = role == "harmony"
    if harmony:
        eq = filters.cascade(filters.highpass(s.hp_hz, sr), filters.highpass1(s.harmony_lowcut_hz, sr))
    elif chain == "full":
        eq = filters.cascade(filters.highpass(s.hp_hz, sr), filters.highpass1(s.lowcut_hz, sr),
                             filters.peaking(s.cut_hz, s.cut_db, s.cut_q, sr))
    else:
        eq = filters.highpass(s.hp_hz, sr)
    x = filters.apply(eq, x)
    if s.gate:
        x, info["gate"] = dyn.expand(x, sr)
    x, info["deess"] = dyn.deess(x, sr, freq=s.deess_hz,
                                 max_reduction_db=s.deess_max_db if (chain == "full" and not harmony) else 4.0)
    if s.smooth > 0:
        x, info["smooth"] = spectral.smooth_resonances(
            x, sr, amount=s.smooth * (1.0 if chain == "full" and not harmony else 0.6))
    if chain == "full" or harmony:
        scale = 0.5 if harmony else 1.0
        presence = filters.cascade(
            filters.peaking(s.presence_hz, s.presence_db * scale, s.presence_q, sr),
            filters.peaking(s.air_hz, s.air_db * scale, s.air_q, sr),
            filters.high_shelf(s.air_shelf_hz, s.air_shelf_db, sr) if abs(s.air_shelf_db) > 0.01 and not harmony
            else None,
        )
        x = filters.apply(presence, x)
    return Prepared(x, level, role, chain, info)


def _peak_compress(x: np.ndarray, sr: float, s: spec.MixSettings) -> tuple[np.ndarray, float]:
    det = dyn.detector_db(x, sr, rms_s=0.001, block=16)
    active = dyn.active_mask(det)
    if not active.any():
        return x, 0.0
    threshold = float(np.median(det[active])) + s.peak_above_median_db
    y, gr = dyn.compress(x, sr, threshold_db=threshold, ratio=s.peak_ratio, knee_db=2.0, attack_s=s.peak_attack_s,
                         release_s=s.peak_release_s, rms_s=0.001, block=16)
    return y, float(gr.max(initial=0.0))


def finish(p: Prepared, sr: float, s: spec.MixSettings, t: spec.Timings, log: LogFn = lambda _m: None,
           label: str = "vocal") -> np.ndarray:
    """Peak compressor (lead, full) → leveling compressor → staging back to −20 dBFS."""
    if p.level_db is None:
        log(f"{label}: silent")
        return p.audio
    if p.chain == "off":
        log(f"{label}: level {p.level_db:.1f} dBFS → {VOCAL_RMS_DB:.0f}; chain off")
        return p.audio
    x = p.audio
    peak_gr = 0.0
    lead_full = p.role == "lead" and p.chain == "full"
    if lead_full:
        x, peak_gr = _peak_compress(x, sr, s)
    if p.role == "harmony":
        ratio, offset = 3.0, 2.0
    elif p.chain == "light":
        ratio, offset = 2.5, 1.0
    else:
        ratio, offset = s.comp_ratio, s.comp_offset_db
    det = dyn.detector_db(x, sr)
    active = dyn.active_mask(det)
    median = float(np.median(det[active])) if active.any() else VOCAL_RMS_DB
    threshold = median - offset
    x, gr = dyn.compress(x, sr, threshold_db=threshold, ratio=ratio, knee_db=s.comp_knee_db,
                         attack_s=s.comp_attack_s, release_s=t.comp_release_s)
    after = dyn.active_rms_db(x, sr)
    if after is not None:
        x = (x * _gain(VOCAL_RMS_DB - after)).astype(np.float32)
    gr_active = gr[active[: gr.size]] if active.any() else gr
    de = p.info.get("deess", {})
    gate = p.info.get("gate", {})
    parts = [f"{label} ({p.role}, {p.chain}): level {p.level_db:.1f} dBFS → {VOCAL_RMS_DB:.0f}"]
    if gate.get("threshold") is not None:
        parts.append(f"gate {gate['threshold']:.1f} dB, -{gate['range']:.0f} dB ({gate['closed'] * 100:.0f}% closed)")
    if de.get("threshold") is not None:
        parts.append(f"de-esser max -{de['maxReduction']:.1f} dB")
    sm = p.info.get("smooth")
    if sm:
        parts.append(f"resonances max -{sm['maxReduction']:.1f} dB")
    if lead_full:
        parts.append(f"peak comp max -{peak_gr:.1f} dB")
    parts.append(f"leveler {threshold:.1f} dB {ratio:.1f}:1, release {t.comp_release_s * 1000:.0f} ms, "
                 f"avg -{float(np.mean(gr_active)) if gr_active.size else 0.0:.1f} dB")
    log("; ".join(parts))
    return x


def process(x: np.ndarray, sr: float, s: spec.MixSettings, t: spec.Timings, log: LogFn = lambda _m: None,
            label: str = "vocal", role: str = "lead", chain: str = "full") -> np.ndarray:
    """The whole chain on a mono vocal."""
    return finish(prepare(x, sr, s, role, chain), sr, s, t, log, label)


def pan_gains(pan: float) -> tuple[float, float]:
    """Equal-power pan law for mono sources: -1 = left, 0 = centre (-3 dB each side), 1 = right."""
    theta = (float(np.clip(pan, -1.0, 1.0)) + 1.0) * math.pi / 4.0
    return math.cos(theta), math.sin(theta)


def balance_gains(pan: float) -> tuple[float, float]:
    """Balance law for stereo sources: centre leaves both sides at unity, ±1 silences one side."""
    p = float(np.clip(pan, -1.0, 1.0))
    return min(1.0, 1.0 - p), min(1.0, 1.0 + p)
