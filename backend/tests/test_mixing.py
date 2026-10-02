"""Built-in mixer ("一键混音"): DSP blocks, loudness references, the mix and its CLI protocol."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from stellaria_backend.mixing import analyze, builtin_processor
from stellaria_backend.mixing import dynamics as dyn
from stellaria_backend.mixing import filters, loudness, mixer, reverb, spec, tempo
from synth import song

SR = 44100
BACKEND = Path(__file__).resolve().parents[1]
STAGES = {"mix.analyze", "mix.vocal", "mix.reverb", "mix.master"}


# ----------------------------------------------------------------------------- synthetic material

def sibilant_vocal(seconds: float, f0: float = 220.0, seed: int = 0, every: float = 1.1,
                   burst: float = 0.07, in_phrases: bool = False) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Harmonic "singing" plus high-passed noise bursts ("s" sounds); returns the burst spans.
    ``in_phrases``: only bursts that fall inside sung phrases (not in the gaps between them)."""
    x = song(seconds, f0, seed=seed).astype(np.float64)
    sung = x != 0
    rng = np.random.default_rng(seed + 100)
    hp = filters.highpass4(5000.0, SR)
    spans = []
    length = int(burst * SR)
    for start_s in np.arange(0.5, seconds - burst - 0.1, every):
        start = int(start_s * SR)
        if in_phrases and not (sung[start: start + length].all() and sung[min(x.size - 1, start + length + 2205)]):
            continue
        noise = signal.sosfilt(hp, rng.standard_normal(length)) * np.hanning(length) * 0.25
        x[start: start + length] += noise
        spans.append((start, start + length))
    return x.astype(np.float32), spans


def _place(out: np.ndarray, start: int, sound: np.ndarray, gains=(1.0, 1.0)) -> None:
    end = min(out.shape[-1], start + sound.size)
    if end > start:
        out[0, start:end] += sound[: end - start] * gains[0]
        out[1, start:end] += sound[: end - start] * gains[1]


def drum_backing(seconds: float, bpm: float, seed: int = 0) -> np.ndarray:
    """Stereo (2, n) kick/snare/hat groove with bass and a pad at a known tempo."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    out = np.zeros((2, n))
    beat = 60.0 / bpm
    t = np.arange(int(0.3 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(50 + 90 * np.exp(-t / 0.03)) / SR) * np.exp(-t / 0.12)
    snare = signal.sosfilt(filters.highpass(300, SR), rng.standard_normal(t.size)) * np.exp(-t / 0.07) * 0.5
    snare += 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.05)
    hat = signal.sosfilt(filters.highpass4(7000, SR), rng.standard_normal(t.size)) * np.exp(-t / 0.02) * 0.3
    roots = [55.0, 43.65, 49.0, 41.2]
    for k, b in enumerate(np.arange(0.0, seconds, beat)):
        start = int(b * SR)
        _place(out, start, kick * 0.9 if k % 2 == 0 else snare)
        _place(out, start, hat, (0.4, 0.6))
        _place(out, start + int(beat / 2 * SR), hat * 0.6, (0.4, 0.6))
        root = roots[(k // 4) % 4]
        tb = np.arange(int(beat * SR)) / SR
        _place(out, start, 0.35 * np.sin(2 * np.pi * root * tb) * np.exp(-tb / 0.5))
    tt = np.arange(n) / SR
    for ratio in (4.0, 5.0, 6.0):
        out[0] += 0.03 * np.sin(2 * np.pi * 55.0 * ratio * tt + 0.3 * ratio)
        out[1] += 0.03 * np.sin(2 * np.pi * 55.0 * ratio * 1.003 * tt)
    return (out * 0.5).astype(np.float32)


def click_track(seconds: float, bpm: float, sr: int = SR) -> np.ndarray:
    x = np.zeros(int(seconds * sr), dtype=np.float32)
    t = np.arange(int(0.012 * sr)) / sr
    blip = (np.sin(2 * np.pi * 1500 * t) * np.exp(-t / 0.003)).astype(np.float32)
    for b in np.arange(0.0, seconds - 0.02, 60.0 / bpm):
        i = int(b * sr)
        x[i: i + blip.size] += blip
    return x


def write(path: Path, x: np.ndarray, sr: int = SR) -> Path:
    """``x`` channel-first; written as float WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (x.T if x.ndim == 2 else x).astype(np.float32), sr, subtype="FLOAT")
    return path


def band_energy_db(x: np.ndarray, lo: float, hi: float, spans=None) -> float:
    y = signal.sosfilt(signal.butter(6, [lo, hi], "bandpass", fs=SR, output="sos"), x.astype(np.float64))
    if spans is not None:
        y = np.concatenate([y[a:b] for a, b in spans])
    return 10 * math.log10(float(np.mean(y ** 2)) + 1e-20)


def read_out(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr


@pytest.fixture(scope="module")
def material(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("mix material")
    vocal, _ = sibilant_vocal(24.0, 220.0, seed=3)
    duet_b, _ = sibilant_vocal(20.0, 330.0, seed=4)
    return {
        "vocal": write(d / "vocal 歌声.wav", vocal),
        "vocal_b": write(d / "vocal b.wav", duet_b),
        "backing": write(d / "backing 伴奏.wav", drum_backing(26.0, 96.0)),
        "dir": d,
    }


def job_for(material, out: Path, **kw) -> dict:
    job = {
        "vocals": [{"path": str(material["vocal"]), "gainDb": 0.0, "pan": 0.0}],
        "backing": {"path": str(material["backing"]), "gainDb": 0.0},
        "preset": "natural", "params": {}, "seed": 0, "output": str(out),
    }
    job.update(kw)
    return job


# ----------------------------------------------------------------------------- processor description

def test_builtin_processor_wire_dict():
    p = builtin_processor()
    json.dumps(p)
    assert p["id"] == "builtin" and p["kind"] == "builtin" and p["maxVocals"] == 4
    assert p["available"] is True and p["error"] is None
    assert p["name"] == {"code": "mix.builtin", "params": {}}
    assert [x["id"] for x in p["presets"]] == ["natural", "pop", "airy", "dry", "classic"]
    params = {x["id"]: x for x in p["params"]}
    preset_driven = {pid for pid, x in params.items() if x["default"] is None}
    for preset in p["presets"]:
        assert preset["name"] == {"code": f"mix.preset.{preset['id']}", "params": {}}
        # every param the preset decides has its value listed, so the controls can show it
        assert set(preset["params"]) == preset_driven
    main = [pid for pid, x in params.items() if "group" not in x]
    assert main == ["vocalLevel", "reverb", "brightness", "width", "targetLufs", "backingGain", "match",
                    "matchLoudness", "unmask", "gate", "throws", "doubler", "warmth", "smooth", "vocalChain"]
    assert {x["group"] for x in params.values() if "group" in x} == {"eq", "dynamics", "space", "master"}
    assert all(x["default"] is None for x in params.values() if "group" in x)
    for pid, x in params.items():
        assert x["label"] == {"code": f"mix.param.{pid}", "params": {}}
        if x["type"] == "number":
            assert x["min"] < x["max"]
    assert {x["id"] for x in p["params"] if x["type"] == "boolean"} == {"matchLoudness", "gate", "throws"}
    assert params["vocalChain"]["options"] == [{"value": v, "label": {"code": f"mix.vocalChain.{v}", "params": {}}}
                                               for v in ("full", "light", "off")]
    assert (params["vocalLevel"]["min"], params["vocalLevel"]["max"], params["vocalLevel"]["default"]) == (-6, 6, 1)
    assert (params["targetLufs"]["min"], params["targetLufs"]["max"], params["targetLufs"]["default"]) == (-23, -9, -14)
    assert (params["backingGain"]["min"], params["backingGain"]["max"], params["backingGain"]["default"]) == (-12, 6, 0)
    assert (params["brightness"]["min"], params["brightness"]["max"], params["brightness"]["default"]) == (-1, 1, 0)
    assert params["reverb"]["default"] is None and params["width"]["default"] is None


def test_fine_settings_follow_the_preset_unless_set():
    classic = spec.resolve("classic", {})
    assert (classic.comp_ratio, classic.deess_max_db, classic.air_shelf_hz, classic.air_shelf_db) == (5.0, 12.0, 20000.0, -2.0)
    assert classic.weights == {"room": 0.4, "plate": 0.1, "chamber": 0.15, "hall": 0.35}
    natural = spec.resolve("natural", {})
    assert (natural.comp_ratio, natural.deess_max_db, natural.comp_attack_s) == (3.5, 6.0, 0.019)
    s = spec.resolve("natural", {"roomMix": 0, "hallMix": 0.5, "compAttack": 30, "decayScale": 1.5,
                                 "cutGain": -20, "ceiling": None})
    assert s.weights == {"plate": 0.4, "chamber": 0.3, "hall": 0.5}  # a type at 0 leaves the blend
    assert s.comp_attack_s == 0.03 and s.decay_factor == 1.5 and s.cut_db == -12.0  # clamped
    assert s.ceiling_db == -1.0
    # alternative takes vary around the user's scale
    assert 0.1 <= abs(spec.resolve("natural", {"decayScale": 2.0}, seed=7).decay_factor / 2.0 - 1) <= 0.3


def test_settings_resolution_and_clamping():
    s = spec.resolve("pop", {"reverb": 5, "vocalLevel": "2.5", "targetLufs": None, "width": "x", "brightness": True})
    assert s.preset == "pop" and s.reverb == 1.0 and s.vocal_level == 2.5 and s.target_lufs == -14.0
    assert s.width == spec.PRESETS["pop"].width and s.brightness == 0.0
    assert spec.resolve("nope", {}).preset == "natural"
    a, b = spec.resolve("natural", {}, seed=7), spec.resolve("natural", {}, seed=7)
    assert a == b and a != spec.resolve("natural", {}, seed=8)
    base = spec.resolve("natural", {})
    assert abs(a.cut_db - base.cut_db) <= 0.5 and abs(a.presence_hz - base.presence_hz) <= 150
    assert abs(a.comp_attack_s - base.comp_attack_s) <= 0.006 and abs(a.comp_release_offset_s) <= 0.06
    assert 0.1 <= abs(a.decay_factor - 1) <= 0.3 and 0.1 <= abs(a.predelay_factor - 1) <= 0.3


# ----------------------------------------------------------------------------- filters

def test_rbj_biquads():
    sr = SR
    assert filters.response_db(filters.peaking(1400, 3.0, 1.15, sr), [1400], sr)[0] == pytest.approx(3.0, abs=1e-6)
    assert filters.response_db(filters.peaking(2700, -2.0, 1.0, sr), [2700, 50, 20000], sr) == pytest.approx(
        [-2.0, 0.0, 0.0], abs=0.05)
    hs = filters.response_db(filters.high_shelf(10000, 4.0, sr), [20, 10000, 21000], sr)
    assert hs == pytest.approx([0.0, 2.0, 4.0], abs=0.05)
    ls = filters.response_db(filters.low_shelf(200, -6.0, sr), [10, 200, 15000], sr)
    assert ls == pytest.approx([-6.0, -3.0, 0.0], abs=0.05)
    hp = filters.response_db(filters.highpass(90, sr), [90, 45, 2000], sr)
    assert hp[0] == pytest.approx(-3.01, abs=0.01) and hp[1] < -11 and abs(hp[2]) < 0.01
    lp = filters.response_db(filters.lowpass(8000, sr), [8000, 1000], sr)
    assert lp[0] == pytest.approx(-3.01, abs=0.01) and abs(lp[1]) < 0.01
    hp1 = filters.response_db(filters.highpass1(230, sr), [230, 23, 10000], sr)
    assert hp1[0] == pytest.approx(-3.01, abs=0.01) and hp1[1] == pytest.approx(-20.04, abs=0.1) and abs(hp1[2]) < 0.01
    hp4 = filters.response_db(filters.highpass4(6500, sr), [6500, 3250], sr)
    assert hp4[0] == pytest.approx(-3.01, abs=0.01) and hp4[1] < -23
    # apply() filters along the last axis; float32 audio is filtered in float32
    x = np.random.default_rng(0).standard_normal((2, 1000)).astype(np.float32)
    y = filters.apply(filters.highpass(100, sr), x)
    assert y.dtype == np.float32 and y.shape == x.shape
    assert np.allclose(y[1], signal.sosfilt(filters.highpass(100, sr), x[1].astype(np.float64)), atol=1e-4)
    assert filters.apply(filters.highpass(100, sr), x.astype(np.float64)).dtype == np.float64


# ----------------------------------------------------------------------------- loudness

def _sine(freq: float, dbfs: float, seconds: float, sr: int) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return 10 ** (dbfs / 20) * np.sin(2 * np.pi * freq * t)


@pytest.mark.parametrize("sr", [44100, 48000])
def test_loudness_997hz_reference(sr):
    # BS.1770: a 997 Hz sine at -20 dBFS in both channels reads -20.0 LUFS (EBU Tech 3341 case 1
    # is the same at -23 dBFS); K-weighting adds +0.691 dB at 997 Hz, cancelling the -0.691 offset.
    s = _sine(997, -20.0, 10.0, sr)
    assert loudness.integrated_loudness(np.stack([s, s]), sr) == pytest.approx(-20.0, abs=0.05)
    # mono = one channel: 3.01 dB less
    assert loudness.integrated_loudness(s, sr) == pytest.approx(-23.01, abs=0.05)


def test_k_weighting_matches_bs1770_table():
    k = loudness.k_weighting(48000)
    assert k[0] == pytest.approx([1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0,
                                  -1.69065929318241, 0.73248077421585], abs=1e-9)
    assert k[1] == pytest.approx([1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621], abs=1e-9)


def test_loudness_gating_ebu3341():
    sr = 48000
    case3 = np.concatenate([_sine(1000, -36, 10, sr), _sine(1000, -23, 60, sr), _sine(1000, -36, 10, sr)])
    assert loudness.integrated_loudness(np.stack([case3, case3]), sr) == pytest.approx(-23.0, abs=0.1)
    case4 = np.concatenate([_sine(1000, -72, 10, sr), _sine(1000, -36, 10, sr), _sine(1000, -23, 60, sr),
                            _sine(1000, -36, 10, sr), _sine(1000, -72, 10, sr)])
    assert loudness.integrated_loudness(np.stack([case4, case4]), sr) == pytest.approx(-23.0, abs=0.1)
    assert loudness.integrated_loudness(np.zeros((2, sr)), sr) == float("-inf")
    assert loudness.report_lufs(float("-inf")) == -70.0


def test_peak_envelope_is_a_2x_oversampled_peak():
    f, ph = 5000.0, 0.3
    k = np.arange(4000)
    x = np.sin(2 * np.pi * f * k / SR + ph).astype(np.float32)
    mid = np.abs(np.sin(2 * np.pi * f * (k + 0.5) / SR + ph))
    env = loudness.peak_envelope(np.stack([x, 0.5 * x]), oversample=2)
    assert np.max(np.abs(env[50:-50] - np.maximum(np.abs(x), mid)[50:-50])) < 1e-3
    # a full-scale tone at fs/4 with 45° phase has sample peaks of 0.707 but a true peak of 1.0
    q = np.sin(np.pi / 2 * k + np.pi / 4).astype(np.float32)
    assert loudness.sample_peak_db(q) == pytest.approx(-3.01, abs=0.01)
    assert loudness.true_peak_db(q) > -0.1


# ----------------------------------------------------------------------------- dynamics

def test_release_hold_matches_the_recursion():
    rng = np.random.default_rng(1)
    x = np.maximum(rng.standard_normal(50000), 0) * (rng.random(50000) < 0.02) * 10
    for r in (0.0, 0.9, 0.999, 0.99999):
        assert np.allclose(dyn.release_hold(x, r, chunk=1000), dyn.release_hold_reference(x, r), rtol=1e-9, atol=1e-12)


def test_gain_computer_soft_knee():
    gr = dyn.gain_computer(np.array([-40.0, -20.0 - 2.25, -20.0, -20 + 2.25, 0.0]), -20.0, 4.0, 4.5)
    assert gr[0] == 0 and gr[1] == pytest.approx(0, abs=1e-12)
    assert gr[2] == pytest.approx(0.75 * 2.25 ** 2 / 9.0)
    assert gr[3] == pytest.approx(0.75 * 2.25) and gr[4] == pytest.approx(15.0)


def _crest_db(x: np.ndarray) -> float:
    return 20 * math.log10(np.max(np.abs(x)) / np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def test_compressor_reduces_crest_factor_and_range():
    t = np.arange(int(8 * SR)) / SR
    envelope = 0.03 + 0.5 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 2.0)) ** 3      # slow swells
    tone = np.sin(2 * np.pi * 220 * t) + 0.4 * np.sin(2 * np.pi * 440 * t)
    x = (envelope * tone).astype(np.float32)
    level = dyn.detector_db(x, SR)
    threshold = float(np.median(level))
    y, gr = dyn.compress(x, SR, threshold_db=threshold, ratio=4.0, knee_db=4.5, attack_s=0.019, release_s=0.4)
    assert np.all(np.isfinite(y)) and gr.max() > 6
    assert _crest_db(y) < _crest_db(x) - 1.5
    lv_in, lv_out = dyn.detector_db(x, SR), dyn.detector_db(y, SR)
    assert np.ptp(np.percentile(lv_out, [10, 95])) < np.ptp(np.percentile(lv_in, [10, 95])) - 4
    # below the threshold nothing happens
    quiet = (0.001 * tone[: SR]).astype(np.float32)
    assert np.allclose(dyn.compress(quiet, SR, threshold_db=-20, ratio=4)[0], quiet, atol=1e-7)


def test_deesser_reduces_sibilant_band_only():
    x, spans = sibilant_vocal(12.0, 220.0, seed=5)
    y, info = dyn.deess(x, SR, freq=6500.0, max_reduction_db=6.0)
    assert info["threshold"] is not None and 3.0 < info["maxReduction"] <= 6.0 + 1e-9
    reduction = band_energy_db(x, 7000, 10000, spans) - band_energy_db(y, 7000, 10000, spans)
    assert 2.5 < reduction <= 6.5
    # voiced parts away from the bursts: the voice band is untouched
    mask = np.ones(x.size, bool)
    for a, b in spans:
        mask[max(0, a - 4000): b + 8000] = False
    keep = [(int(i), int(i) + 2205) for i in np.flatnonzero(mask[::2205]) * 2205 if mask[int(i): int(i) + 2205].all()]
    assert abs(band_energy_db(x, 150, 4000, keep) - band_energy_db(y, 150, 4000, keep)) < 0.1


def test_soft_clip():
    x = np.linspace(-2, 2, 10001).astype(np.float32)
    y = dyn.soft_clip(x, 0.66, 0.2)
    assert np.max(np.abs(y)) <= 0.66 + 1e-6
    small = np.abs(x) <= 0.66 * 0.8
    assert np.array_equal(y[small], x[small])
    assert np.all(np.diff(y) >= -1e-7)            # monotonic, no fold-back
    assert np.max(np.abs(dyn.soft_clip(x, 0.5, 0.0))) == pytest.approx(0.5)


def test_limiter_holds_the_ceiling():
    rng = np.random.default_rng(2)
    n = 10 * SR
    x = (rng.standard_normal((2, n)) * 0.1).astype(np.float32)
    for pos in rng.integers(1000, n - 1000, 40):
        x[:, pos: pos + 300] *= 8.0                                          # transients
    y, max_gr = dyn.limit(x, SR, ceiling_db=-1.0, drive_db=6.0, release_s=0.08)
    assert max_gr > 6
    assert loudness.true_peak_db(y) <= -1.0 + 0.02
    assert loudness.sample_peak_db(y) <= -1.0
    assert np.all(np.isfinite(y))


# ----------------------------------------------------------------------------- reverb / delay

@pytest.mark.parametrize("kind", reverb.IR_KINDS)
def test_synth_ir_shapes(kind):
    ir = reverb.synth_ir(kind, 1.8, SR)
    assert ir.shape[0] == 2 and ir.dtype == np.float32 and np.all(np.isfinite(ir))
    assert float(np.mean(np.sum(ir.astype(np.float64) ** 2, axis=-1))) == pytest.approx(1.0, rel=1e-3)
    assert abs(np.corrcoef(ir[0], ir[1])[0, 1]) < 0.3                         # decorrelated L/R
    tenth = ir.shape[-1] // 10
    head = np.sum(ir[:, :tenth] ** 2)
    tail = np.sum(ir[:, -tenth:] ** 2)
    assert tail < head * 1e-3                                                # decays


def test_space_ir_predelay_width_and_delay():
    space = reverb.SpaceSpec(weights={"plate": 1.0}, decay_s=1.5, predelay_s=0.03, width=0.0,
                             delay=reverb.DelaySpec(time_s=0.5))
    ir = reverb.space_ir(space, SR)
    assert not np.any(ir[:, : int(0.03 * SR) - 1])
    # width 0 → the reverb is mono; the ping-pong delay taps are not
    rev_only = reverb.space_ir(reverb.SpaceSpec(weights={"plate": 1.0}, decay_s=1.5, predelay_s=0.0, width=0.0), SR)
    assert np.allclose(rev_only[0], rev_only[1], atol=1e-6)
    tap = int(0.5 * SR)
    assert abs(ir[0, tap: tap + 200]).max() > abs(ir[1, tap: tap + 200]).max()
    wet = reverb.convolve(np.r_[1.0, np.zeros(SR - 1)].astype(np.float32), rev_only, SR)
    assert wet.shape == (2, SR) and np.allclose(wet[0, :1000], rev_only[0, :1000], atol=1e-5)


# ----------------------------------------------------------------------------- tempo

@pytest.mark.parametrize("bpm", [90.0, 75.0])
def test_bpm_click_track(bpm):
    detected = tempo.detect_bpm(click_track(30.0, bpm), SR)
    assert detected == pytest.approx(bpm, abs=3.0)
    assert tempo.timing_bpm(detected) == pytest.approx(bpm, abs=3.0)


def test_bpm_halved_above_100_and_fallback():
    detected = tempo.detect_bpm(drum_backing(30.0, 128.0), SR)
    assert detected == pytest.approx(128.0, abs=3.0)
    assert tempo.timing_bpm(detected) == pytest.approx(64.0, abs=1.5)
    assert tempo.detect_bpm(np.zeros(SR * 10, np.float32), SR) is None
    assert tempo.detect_bpm(np.ones(SR, np.float32), SR) is None             # too short
    assert tempo.timing_bpm(None) == 90.0 and tempo.timing_bpm(210.0) == 52.5


def test_bpm_matches_librosa_beat_track():
    import librosa.beat

    x = click_track(20.0, 84.0, sr=tempo.BPM_SR)
    expected = float(np.atleast_1d(librosa.beat.beat_track(y=x, sr=tempo.BPM_SR)[0])[0])
    assert tempo.detect_bpm(x, tempo.BPM_SR) == pytest.approx(expected, abs=1e-6)


def test_synced_note_values():
    assert tempo.synced(90.0, 0.35, 0.5) == pytest.approx(0.5)          # dotted 1/8 at 90 BPM
    assert tempo.synced(60.0, 0.35, 0.5) == pytest.approx(0.375)        # dotted 1/16 at 60 BPM
    assert 0.35 <= tempo.synced(77.0, 0.35, 0.5) <= 0.5
    assert tempo.note_seconds(120.0, 0.25) == pytest.approx(0.5)


# ----------------------------------------------------------------------------- the mix

def _check_output(path: Path, n_expected: int, target: float) -> np.ndarray:
    y, sr = read_out(path)
    assert sr == SR and y.shape == (2, n_expected)
    assert np.all(np.isfinite(y))
    assert loudness.sample_peak_db(y) <= -0.9
    assert loudness.integrated_loudness(y, SR) == pytest.approx(target, abs=0.5)
    return y


@pytest.mark.parametrize("preset,target", [("natural", -14.0), ("pop", -9.0), ("airy", -16.0), ("dry", -23.0)])
def test_mix_quality(material, tmp_path, preset, target):
    out = tmp_path / "out.wav"
    logs: list[str] = []
    stages: list[tuple[float, str]] = []
    result = mixer.mix(job_for(material, out, preset=preset, params={"targetLufs": target}),
                       progress=lambda p, s: stages.append((p, s)), log=logs.append)
    _check_output(out, int(26.0 * SR), target)
    assert result["lufs"] == pytest.approx(target, abs=0.5) and result["peakDb"] <= -0.9
    assert result["bpm"] == pytest.approx(96.0, abs=3.0)
    assert [s for _, s in stages][0] == "mix.analyze" and stages[-1] == (1.0, "mix.master")
    assert {s for _, s in stages} == STAGES
    assert any(m.startswith("master:") for m in logs)


def test_vocal_level_param_moves_the_backing(material, tmp_path):
    def vocal_vs_backing(level: float) -> float:
        out = tmp_path / f"lvl{level}.wav"
        mixer.mix(job_for(material, out, params={"vocalLevel": level, "reverb": 0}))
        y, _ = read_out(out)
        voice = band_energy_db(y.mean(axis=0), 600, 3000)
        drums = band_energy_db(y.mean(axis=0), 30, 120)
        return voice - drums
    assert vocal_vs_backing(6.0) - vocal_vs_backing(-6.0) > 8.0


def test_variants_seeded(material, tmp_path):
    outs = {}
    for name, seed in (("a", 0), ("b", 0), ("c", 1), ("d", 2)):
        mixer.mix(job_for(material, tmp_path / f"{name}.wav", seed=seed))
        outs[name] = read_out(tmp_path / f"{name}.wav")[0]
    assert np.array_equal(outs["a"], outs["b"])                      # seed 0 is deterministic
    assert not np.allclose(outs["a"], outs["c"], atol=1e-4)
    assert not np.allclose(outs["c"], outs["d"], atol=1e-4)
    for k in "cd":                                                   # small variations, same song
        assert np.corrcoef(outs["a"].ravel(), outs[k].ravel())[0, 1] > 0.8
    again = tmp_path / "c2.wav"
    mixer.mix(job_for(material, again, seed=1))
    assert np.array_equal(read_out(again)[0], outs["c"])              # every seed is reproducible


@pytest.mark.parametrize("reverb_amount,min_db", [(0.0, 20.0), (None, 6.0)])
def test_duet_panning(tmp_path, reverb_amount, min_db):
    a, _ = sibilant_vocal(10.0, 220.0, seed=6)
    b, _ = sibilant_vocal(10.0, 330.0, seed=7)
    va = np.concatenate([a, np.zeros(10 * SR, np.float32)])
    vb = np.concatenate([np.zeros(10 * SR, np.float32), b])
    job = {"vocals": [{"path": str(write(tmp_path / "a.wav", va)), "gainDb": 0, "pan": -1.0},
                      {"path": str(write(tmp_path / "b.wav", vb)), "gainDb": -3.0, "pan": 1.0}],
           "backing": None, "preset": "natural",
           "params": {} if reverb_amount is None else {"reverb": reverb_amount},
           "seed": 0, "output": str(tmp_path / "duet.wav")}
    mixer.mix(job)
    y, _ = read_out(tmp_path / "duet.wav")
    first, second = y[:, int(1 * SR): int(9 * SR)], y[:, int(11 * SR): int(19 * SR)]

    def side_db(seg: np.ndarray) -> float:
        e = np.mean(seg.astype(np.float64) ** 2, axis=-1) + 1e-20
        return 10 * math.log10(e[0] / e[1])

    assert side_db(first) > min_db           # vocal A panned hard left
    assert side_db(second) < -min_db         # vocal B panned hard right


def test_pan_law_is_equal_power():
    for pan in (-1.0, -0.3, 0.0, 0.5, 1.0):
        left, right = mixer.pan_gains(pan)
        assert left ** 2 + right ** 2 == pytest.approx(1.0)
    assert mixer.pan_gains(0.0) == pytest.approx((math.sqrt(0.5), math.sqrt(0.5)))
    assert mixer.pan_gains(-1.0) == pytest.approx((1.0, 0.0), abs=1e-12)


def test_lengths_channels_and_resampling(tmp_path):
    vocal, _ = sibilant_vocal(14.0, 247.0, seed=8)
    v48 = signal.resample_poly(vocal, 160, 147).astype(np.float32)          # 48 kHz mono
    backing = drum_backing(9.0, 100.0)                                        # shorter, stereo
    job = {"vocals": [{"path": str(write(tmp_path / "v48.wav", v48, 48000)), "gainDb": 0, "pan": 0.2}],
           "backing": {"path": str(write(tmp_path / "b.wav", backing)), "gainDb": -2.0},
           "preset": "dry", "params": {}, "seed": 3, "output": str(tmp_path / "o.wav")}
    mixer.mix(job)
    _check_output(tmp_path / "o.wav", int(round(14.0 * SR)), -14.0)
    # mono backing + stereo vocal, backing longer
    job["vocals"][0]["path"] = str(write(tmp_path / "vs.wav", np.stack([vocal[: 6 * SR], vocal[: 6 * SR] * 0.8])))
    job["backing"]["path"] = str(write(tmp_path / "bm.wav", backing.mean(axis=0)))
    mixer.mix(job)
    _check_output(tmp_path / "o.wav", backing.shape[-1], -14.0)


def test_vocal_only_and_silent_inputs(tmp_path):
    vocal, _ = sibilant_vocal(8.0, 220.0, seed=9)
    job = {"vocals": [{"path": str(write(tmp_path / "v.wav", vocal)), "gainDb": 0, "pan": 0}],
           "backing": None, "preset": "airy", "params": {}, "seed": 0, "output": str(tmp_path / "o.wav")}
    result = mixer.mix(job)
    assert result["bpm"] is None or result["bpm"] > 0
    _check_output(tmp_path / "o.wav", vocal.size, -14.0)
    job["vocals"][0]["path"] = str(write(tmp_path / "silent.wav", np.zeros(SR * 3, np.float32)))
    result = mixer.mix(job)
    y, _ = read_out(tmp_path / "o.wav")
    assert result["lufs"] == -70.0 and np.all(y == 0)


@pytest.mark.parametrize("vocal_len,backing_len", [(2000, None), (0, 3 * SR), (300, 500)])
def test_tiny_and_empty_inputs(tmp_path, vocal_len, backing_len):
    vocal, _ = sibilant_vocal(2.0, 220.0, seed=10)
    job = {"vocals": [{"path": str(write(tmp_path / "v.wav", vocal[:vocal_len])), "gainDb": 0, "pan": 0}],
           "backing": None if backing_len is None
           else {"path": str(write(tmp_path / "b.wav", drum_backing(3.0, 90.0)[:, :backing_len])), "gainDb": 0},
           "preset": "natural", "params": {}, "seed": 1, "output": str(tmp_path / "o.wav")}
    result = mixer.mix(job)
    y, sr = read_out(tmp_path / "o.wav")
    assert sr == SR and y.shape == (2, max(vocal_len, backing_len or 0))
    assert np.all(np.isfinite(y)) and (y.size == 0 or loudness.sample_peak_db(y) <= -0.9)
    assert math.isfinite(result["lufs"]) and math.isfinite(result["peakDb"])


def test_analyze(material, tmp_path):
    a = analyze(material["backing"])
    assert set(a) == {"bpm", "lufs", "peakDb", "duration"}
    assert a["duration"] == pytest.approx(26.0, abs=1e-3)
    assert a["bpm"] == pytest.approx(96.0, abs=3.0)
    x, _ = read_out(material["backing"])
    assert a["lufs"] == pytest.approx(loudness.integrated_loudness(x, SR), abs=0.01)
    assert a["peakDb"] == pytest.approx(20 * math.log10(np.abs(x).max()), abs=0.01)
    s = analyze(write(tmp_path / "s.wav", np.zeros(SR, np.float32)))
    assert s == {"bpm": None, "lufs": -70.0, "peakDb": -120.0, "duration": 1.0}
    json.dumps(a)


# ----------------------------------------------------------------------------- CLI protocol

def run_cli(job_path: Path, timeout: float = 120.0) -> tuple[int, list[dict], str]:
    env = dict(os.environ, PYTHONPATH=str(BACKEND) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-m", "stellaria_backend.mixing.run", str(job_path)],
                          cwd=str(job_path.parent), env=env, capture_output=True, timeout=timeout)
    lines = proc.stdout.decode("utf-8").splitlines()
    messages = [json.loads(line) for line in lines]            # every stdout line is JSON
    return proc.returncode, messages, proc.stderr.decode("utf-8", errors="replace")


def _check_protocol(messages: list[dict]) -> None:
    last = -1.0
    for m in messages[:-1]:
        assert len(m) in (1, 2)
        if "progress" in m:
            assert set(m) == {"progress", "stage"} and m["stage"] in STAGES
            assert 0.0 <= m["progress"] <= 1.0 and m["progress"] >= last
            last = m["progress"]
        else:
            assert set(m) == {"log"} and isinstance(m["log"], str)


def test_cli_protocol(material, tmp_path):
    out = tmp_path / "输出 mix.wav"
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job_for(material, out, preset="pop", seed=2), ensure_ascii=False), encoding="utf-8")
    code, messages, stderr = run_cli(job_path)
    assert code == 0, stderr
    _check_protocol(messages)
    assert set(messages[-1]) == {"result"}
    result = messages[-1]["result"]
    assert {"lufs", "peakDb", "bpm"} <= set(result)          # v1 keys (v2 adds truePeakDb, vocalToBackingDb, offset)
    assert result["lufs"] == pytest.approx(-14.0, abs=0.5) and result["peakDb"] <= -0.9
    assert {m["stage"] for m in messages if "stage" in m} == STAGES
    assert any("log" in m for m in messages)
    _check_output(out, int(26.0 * SR), -14.0)
    assert not list(tmp_path.glob("*.part.*"))


@pytest.mark.parametrize("case,code", [
    ("missing_input", "mix.input_missing"),
    ("bad_json", "mix.bad_job"),
    ("no_vocals", "mix.no_vocals"),
    ("too_many", "mix.too_many_vocals"),
    ("not_audio", "audio.decode_failed"),
])
def test_cli_errors(material, tmp_path, case, code):
    job = job_for(material, tmp_path / "o.wav")
    job_path = tmp_path / "job.json"
    if case == "missing_input":
        job["vocals"][0]["path"] = str(tmp_path / "nope.wav")
    elif case == "no_vocals":
        job["vocals"] = []
    elif case == "too_many":
        job["vocals"] = job["vocals"] * 5
    elif case == "not_audio":
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"not audio at all" * 10)
        job["vocals"][0]["path"] = str(bogus)
    job_path.write_text("{oops" if case == "bad_json" else json.dumps(job), encoding="utf-8")
    exit_code, messages, _ = run_cli(job_path)
    assert exit_code == 1
    _check_protocol(messages)
    assert set(messages[-1]) == {"error"}
    assert messages[-1]["error"]["code"] == code and messages[-1]["error"]["message"]
    assert not (tmp_path / "o.wav").exists()


def test_cli_three_minute_song_under_15_seconds(tmp_path):
    vocal, _ = sibilant_vocal(180.0, 220.0, seed=11)
    backing = drum_backing(180.0, 92.0, seed=1)
    job = {"vocals": [{"path": str(write(tmp_path / "v.wav", vocal)), "gainDb": 0, "pan": 0}],
           "backing": {"path": str(write(tmp_path / "b.wav", backing)), "gainDb": 0},
           "preset": "airy", "params": {}, "seed": 0, "output": str(tmp_path / "song.wav")}
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    start = time.perf_counter()
    code, messages, stderr = run_cli(job_path)
    elapsed = time.perf_counter() - start
    assert code == 0, stderr
    assert "result" in messages[-1]
    _check_output(tmp_path / "song.wav", 180 * SR, -14.0)
    assert elapsed < 15.0, f"3-minute mix took {elapsed:.1f} s"
