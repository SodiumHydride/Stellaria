"""Built-in effect processors. ``x`` is float32 ``(channels, n)``; each returns float32
``(channels', n')`` — mono may become stereo (reverb, delay, doubler, stereo), and reverb/delay
append a tail (the chain caps and trims it)."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..mixing import doubler as doubling
from ..mixing import dynamics as dyn
from ..mixing import filters, reverb, spectral, tempo

MAX_TAIL_S = 6.0
TAIL_FLOOR_DB = -90.0


def _gain(db: float) -> float:
    return 10.0 ** (float(db) / 20.0)


def _stereo(x: np.ndarray) -> np.ndarray:
    return np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x[:2]


def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=0, dtype=np.float32) if x.shape[0] > 1 else x[0]


def _pad(x: np.ndarray, extra: int) -> np.ndarray:
    if extra <= 0:
        return x
    return np.concatenate([x, np.zeros((x.shape[0], extra), dtype=np.float32)], axis=-1)


# ----------------------------------------------------------------------------- eq / dynamics / color

def eq(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    sections = []
    if p["highPass"]:
        sections.append(filters.highpass(p["highPassFreq"], sr))
    if abs(p["lowShelfGain"]) > 0.01:
        sections.append(filters.low_shelf(p["lowShelfFreq"], p["lowShelfGain"], sr))
    for k in (1, 2, 3):
        if abs(p[f"band{k}Gain"]) > 0.01:
            sections.append(filters.peaking(p[f"band{k}Freq"], p[f"band{k}Gain"], p[f"band{k}Q"], sr))
    if abs(p["highShelfGain"]) > 0.01:
        sections.append(filters.high_shelf(p["highShelfFreq"], p["highShelfGain"], sr))
    if p["lowPass"]:
        sections.append(filters.lowpass(p["lowPassFreq"], sr))
    return filters.apply(filters.cascade(*sections), x) if sections else x.copy()


def compressor(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    before = dyn.active_rms_db(x, sr)
    if before is None:
        return x.copy()
    threshold = p["threshold"]
    if p["autoThreshold"]:
        det = dyn.detector_db(x, sr, rms_s=0.005)
        active = dyn.active_mask(det)
        threshold = float(np.median(det[active])) - 4.0 if active.any() else threshold
    y, _ = dyn.compress(x, sr, threshold_db=threshold, ratio=p["ratio"], knee_db=p["knee"],
                        attack_s=p["attack"] / 1000.0, release_s=p["release"] / 1000.0, rms_s=0.005)
    if p["autoMakeup"]:
        after = dyn.active_rms_db(y, sr)
        if after is not None:
            y *= np.float32(_gain(before - after))
    if abs(p["makeup"]) > 0.001:
        y *= np.float32(_gain(p["makeup"]))
    return y


def deesser(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    y, _ = dyn.deess(x, sr, freq=p["frequency"], max_reduction_db=p["range"],
                     above_median_db=12.0 - 12.0 * p["sensitivity"])
    return y


def smooth(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    lo, hi = sorted((p["bandLow"], p["bandHigh"]))
    return spectral.smooth_resonances(x, sr, amount=p["amount"], low_hz=lo, high_hz=hi, max_db=p["maxCut"])[0]


def gate(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    y, _ = dyn.expand(x, sr, threshold_db=None if p["autoThreshold"] else p["threshold"], range_db=p["range"],
                      hold_s=p["hold"] / 1000.0, attack_s=p["attack"] / 1000.0, release_s=p["release"] / 1000.0)
    return y


def limiter(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    y, _ = dyn.limit(x, sr, ceiling_db=p["ceiling"], drive_db=p["gain"], release_s=p["release"] / 1000.0)
    return y


def saturation(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    y = dyn.saturate(x, sr, p["drive"], mode=p["mode"])
    tone = p["tone"]
    if abs(tone) > 0.01:
        y = filters.apply(filters.cascade(filters.low_shelf(300.0, -1.5 * tone, sr),
                                          filters.high_shelf(3000.0, 3.0 * tone, sr)), y)
    if abs(p["output"]) > 0.001:
        y *= np.float32(_gain(p["output"]))
    return y


# ----------------------------------------------------------------------------- space

def _tail_room(ctx: dict, want: int) -> int:
    """Samples of tail this effect may still add (the chain's total tail is capped)."""
    return max(0, min(want, ctx["max_n"] - ctx["n"]))


def reverb_fx(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    ir = reverb.synth_ir(p["type"], p["decay"], sr).astype(np.float64)
    lowpass = float(np.exp(np.interp(p["tone"], [-1.0, 0.0, 1.0], np.log([4000.0, 9000.0, 16000.0]))))
    ir = filters.apply(filters.cascade(filters.highpass(p["lowCut"], sr), filters.lowpass(lowpass, sr)), ir)
    ir = reverb.normalize_energy(reverb.set_width(ir, p["width"]))
    pre = int(round(p["preDelay"] / 1000.0 * sr))
    ir = np.concatenate([np.zeros((2, pre)), ir], axis=-1).astype(np.float32) * np.float32(_gain(p["level"]))
    tail = _tail_room(ctx, ir.shape[-1])
    dry = _pad(_stereo(x), tail)
    return dry + reverb.convolve_modulated(_mono(x), ir, dry.shape[-1], sr=sr, cutoff_hz=lowpass,
                                           modulation=p["modulation"])


def _delay_seconds(p: dict, bpm: float | None) -> float:
    sync = p["sync"]
    if sync != "off" and bpm:
        whole = {"1/16": 1 / 16, "1/8": 1 / 8, "1/8d": 3 / 16, "1/8t": 1 / 12, "1/4": 1 / 4, "1/4d": 3 / 8,
                 "1/2": 1 / 2, "1/1": 1.0}[sync]
        return tempo.note_seconds(float(bpm), whole)          # the song's own tempo (not halved)
    return p["time"] / 1000.0


def delay_fx(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    t = _delay_seconds(p, ctx.get("bpm"))
    fb = p["feedback"]
    # repeats until the echo is under -90 dBFS (relative), at most 6 s of echoes
    per = 20.0 * math.log10(fb) if fb > 0 else -200.0
    repeats = 1 if per <= -199 else int(math.ceil((TAIL_FLOOR_DB - p["level"]) / per)) + 1
    repeats = int(max(1, min(repeats, MAX_TAIL_S / t)))
    ir = reverb.delay_ir(reverb.DelaySpec(time_s=t, feedback=fb, level_db=p["level"], repeats=repeats,
                                          highpass_hz=p["lowCut"], lowpass_hz=p["highCut"],
                                          ping_pong=p["pingPong"]), sr)
    tail = _tail_room(ctx, ir.shape[-1])
    dry = _pad(_stereo(x), tail)
    return dry + reverb.convolve(_mono(x), ir, dry.shape[-1], sr=sr, cutoff_hz=p["highCut"])


def doubler(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    voices = doubling.voices(_mono(x), sr, detune_cents=p["detune"], delay_ms=p["delay"])
    voices = reverb.set_width(voices, p["width"]).astype(np.float32)
    return _stereo(x) + voices * np.float32(0.7 * p["amount"])


# ----------------------------------------------------------------------------- utility

def gain(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    g = _gain(p["gain"]) * (-1.0 if p["invert"] else 1.0)
    return x * np.float32(g)


def stereo(x: np.ndarray, sr: float, p: dict, ctx: dict) -> np.ndarray:
    y = _stereo(x).astype(np.float32, copy=True)
    mid = (y[0] + y[1]) * np.float32(0.5)
    side = (y[0] - y[1]) * np.float32(0.5)
    if p["monoBass"] > 1.0:
        side = side - filters.zero_phase(filters.lowpass4(p["monoBass"], sr), side)
    side *= np.float32(p["width"])
    left, right = mid + side, mid - side
    pan = p["pan"]
    if abs(pan) > 1e-4:
        left *= np.float32(min(1.0, 1.0 - pan))
        right *= np.float32(min(1.0, 1.0 + pan))
    return np.stack([left, right]).astype(np.float32)


RUNNERS: dict[str, Callable[[np.ndarray, float, dict, dict], np.ndarray]] = {
    "eq": eq, "compressor": compressor, "deesser": deesser, "smooth": smooth, "gate": gate, "reverb": reverb_fx,
    "delay": delay_fx, "saturation": saturation, "doubler": doubler, "gain": gain, "stereo": stereo,
    "limiter": limiter,
}
