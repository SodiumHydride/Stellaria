"""Mixer v2 (canvas-v3 §4): roles, offsets, reference following, unmasking, gate, throws, doubler,
warmth, two-stage dynamics, true-peak master, the ``bus`` processor."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest
from scipy import signal

from stellaria_backend.mixing import bus_processor, builtin_processor, chains, dynamics, loudness, match, mixer
from stellaria_backend.mixing import reverb, spec
from synth import song
from test_mixing import SR, band_energy_db, drum_backing, read_out, run_cli, sibilant_vocal, write

REPORT_KEYS = {"lufs", "peakDb", "truePeakDb", "bpm", "vocalToBackingDb", "offset"}


def stereo(x: np.ndarray, gain: float = math.sqrt(0.5)) -> np.ndarray:
    return (np.stack([x, x]) * gain).astype(np.float32)


def pad_to(x: np.ndarray, n: int) -> np.ndarray:
    return np.concatenate([x, np.zeros(x.shape[:-1] + (n - x.shape[-1],), np.float32)], axis=-1)


def wet_original(vocal: np.ndarray, amount: float, decay: float = 1.6) -> np.ndarray:
    """A vocal through the mixer's own natural space at ``amount`` (the mixer's reverb scale)."""
    ir = reverb.space_ir(reverb.SpaceSpec(weights={"room": .3, "plate": .4, "chamber": .3}, decay_s=decay,
                                          predelay_s=0.02, width=0.8), SR)
    dry = stereo(vocal)
    if amount <= 0:
        return dry
    wet = reverb.convolve(vocal, ir, vocal.size)
    g = 10 ** ((loudness.integrated_loudness(dry, SR) + 20 * math.log10(amount) - 7
                - loudness.integrated_loudness(wet, SR)) / 20)
    return (dry + wet * np.float32(g)).astype(np.float32)


@pytest.fixture(scope="module")
def song_files(tmp_path_factory) -> dict:
    """An "original song" (vocals with reverb + backing, known balance) and a converted take."""
    d = tmp_path_factory.mktemp("v2 song")
    original, _ = sibilant_vocal(26.0, 220.0, seed=1, in_phrases=True)
    converted, _ = sibilant_vocal(26.0, 262.0, seed=1, in_phrases=True)      # same phrasing, another voice
    backing = drum_backing(26.0, 96.0)
    harmony, _ = sibilant_vocal(12.0, 330.0, seed=1, in_phrases=True)
    orig_vocals = wet_original(original, 0.4)
    files = {"dir": d, "converted": write(d / "converted.wav", converted), "backing": write(d / "inst.wav", backing),
             "harmony": write(d / "harmony.wav", harmony), "orig_vocals": write(d / "orig vocals.wav", orig_vocals)}
    for name, gain in (("loud", 0.25), ("quiet", 1.0)):         # two originals: vocals loud / quiet vs backing
        mix = orig_vocals + backing * np.float32(gain)
        files[f"mix_{name}"] = write(d / f"mix {name}.wav", mix)
        files[f"ratio_{name}"] = (loudness.integrated_loudness(orig_vocals, SR)
                                  - loudness.integrated_loudness(backing * np.float32(gain), SR))
        files[f"backing_{name}"] = write(d / f"inst {name}.wav", backing * np.float32(gain))
    return files


def v2_job(files: dict, out: Path, *, reference: str | None = "loud", params: dict | None = None,
           preset: str = "natural", extra: list | None = None, seed: int = 0) -> dict:
    job = {"inputs": [{"path": str(files["converted"]), "role": "lead", "gainDb": 0, "pan": 0, "offset": 0.0},
                      {"path": str(files[f"backing_{reference or 'loud'}"]), "role": "backing", "gainDb": 0,
                       "pan": 0, "offset": 0.0}, *(extra or [])],
           "reference": None if reference is None else
           {"vocals": str(files["orig_vocals"]), "mix": str(files[f"mix_{reference}"]), "offset": 0.0},
           "processor": "builtin", "preset": preset, "params": params or {}, "seed": seed, "output": str(out)}
    return job


def lu(x: np.ndarray) -> float:
    return loudness.integrated_loudness(x, SR)


# ----------------------------------------------------------------------------- wire

def test_processor_dicts():
    b = builtin_processor()
    params = {p["id"]: p for p in b["params"]}
    assert params["match"]["default"] == 0.7 and params["unmask"]["default"] == 0.5
    assert params["gate"] == {"id": "gate", "label": {"code": "mix.param.gate", "params": {}}, "type": "boolean",
                              "default": True}
    assert params["throws"]["default"] is None and params["doubler"]["default"] is None
    assert params["vocalChain"]["default"] == "full"
    assert {p["id"]: p["params"]["throws"] for p in b["presets"]} == {"natural": False, "pop": True, "airy": True,
                                                                     "dry": False, "classic": False}
    bus = bus_processor()
    json.dumps(bus)
    assert bus["id"] == "bus" and bus["kind"] == "builtin" and bus["presets"] == []
    assert bus["name"] == {"code": "mix.bus", "params": {}}
    assert [(p["id"], p["type"], p["default"]) for p in bus["params"]] == [("normalize", "boolean", False),
                                                                          ("targetLufs", "number", -14.0)]


def test_v2_job_parsing():
    j = mixer.parse_job({"inputs": [{"path": "a", "role": "harmony", "offset": 3, "mute": True},
                                    {"path": "b", "role": "lead", "solo": True},
                                    {"path": "c", "role": "backing", "solo": True}], "output": "o"})
    assert [i.path for i in j.active()] == ["b", "c"] and j.version == 2
    v1 = mixer.parse_job({"vocals": [{"path": "v", "gainDb": 2, "pan": 0.5}], "backing": {"path": "b", "gainDb": -1},
                          "output": "o"})
    assert [(i.role, i.gain_db, i.offset) for i in v1.inputs] == [("lead", 2.0, 0.0), ("backing", -1.0, 0.0)]
    for bad, code in (({"inputs": [{"path": "b", "role": "backing"}], "output": "o"}, "mix.no_vocals"),
                      ({"inputs": [{"path": "a", "role": "lead", "mute": True}], "output": "o"}, "mix.no_vocals"),
                      ({"inputs": [{"path": "a", "role": "singer"}], "output": "o"}, "mix.bad_job"),
                      ({"inputs": [], "output": "o", "processor": "bus"}, "mix.no_inputs"),
                      ({"inputs": [{"path": "a"}] * 5, "output": "o"}, "mix.too_many_vocals"),
                      ({"inputs": [{"path": "a"}], "output": "o", "processor": "nope"}, "mix.bad_job")):
        with pytest.raises(mixer.MixError) as e:
            mixer.parse_job(bad)
        assert e.value.code == code


# ----------------------------------------------------------------------------- building blocks

def test_true_peak_4x():
    k = np.arange(20000)
    fade = np.clip(np.minimum(k, k[::-1]) / 2000.0, 0.0, 1.0)            # no hard edges (they overshoot)
    x = (np.sin(2 * np.pi * 0.31 * k + 0.4) * fade).astype(np.float32)   # analytic peak 1.0
    sample, tp2, tp4 = loudness.sample_peak_db(x), loudness.true_peak_db(x, 2), loudness.true_peak_db(x, 4)
    assert sample <= tp2 <= tp4 + 1e-6 and abs(tp4) < 0.05
    # fs/8 with its peaks a quarter-sample after a sample: only the 4x estimate lands on them
    q = (np.sin(2 * np.pi * k / 8 + np.pi / 2 - np.pi / 16) * fade).astype(np.float32)
    assert loudness.true_peak_db(q, 2) < -0.15 and abs(loudness.true_peak_db(q, 4)) < 0.02


def test_short_term_loudness():
    t = np.arange(20 * SR) / SR
    s = 0.1 * np.sin(2 * np.pi * 997 * t) * (t < 10)
    st = loudness.short_term(np.stack([s, s]), SR)
    assert st[5] == pytest.approx(-20.0, abs=0.1) and st[-3] == float("-inf")


def test_gate_expander_on_hissy_vocal():
    rng = np.random.default_rng(3)
    v = song(20.0, 220.0, seed=4).astype(np.float32)
    hiss = (rng.standard_normal(v.size) * 10 ** (-52 / 20)).astype(np.float32)
    x = v + hiss
    y, info = dynamics.expand(x, SR)
    gaps = [(int((4.3 + 4.7 * k + 0.25) * SR), int((4.3 + 4.7 * k + 0.55) * SR)) for k in range(3)]
    before = np.mean([np.mean(x[a:b] ** 2) for a, b in gaps])
    after = np.mean([np.mean(y[a:b] ** 2) for a, b in gaps])
    assert 10 * math.log10(before / after) >= 6.0
    sung = slice(int(1.0 * SR), int(3.5 * SR))
    assert abs(10 * math.log10(np.mean(y[sung] ** 2) / np.mean(x[sung] ** 2))) < 0.2
    assert 6.0 <= info["range"] <= 12.0


def test_saturation_tube_adds_even_harmonics():
    t = np.arange(SR) / SR
    x = (0.5 * np.sin(2 * np.pi * 441 * t)).astype(np.float32)
    spectrum = lambda y: np.abs(np.fft.rfft(y * np.hanning(y.size)))  # noqa: E731
    tube, tanh = spectrum(dynamics.saturate(x, SR, 9.0, mode="tube")), spectrum(dynamics.saturate(x, SR, 9.0, mode="tanh"))
    h1, h2, h3 = 441, 882, 1323
    assert tube[h2] / tube[h1] > 0.01 and tanh[h2] / tanh[h1] < 1e-3
    assert tanh[h3] / tanh[h1] > 0.01
    quiet = (0.001 * np.sin(2 * np.pi * 441 * t)).astype(np.float32)          # unity small-signal gain
    assert np.max(np.abs(dynamics.saturate(quiet, SR, 9.0, mode="tanh")[1000:-1000] - quiet[1000:-1000])) < 2e-5


def test_peak_compressor_catches_transients():
    t = np.arange(4 * SR) / SR
    x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    for k in range(1, 8):
        i = int(k * 0.5 * SR)
        x[i: i + 400] *= 6.0                                                   # 9 ms spikes
    s = spec.resolve("natural", {})
    y, gr = chains._peak_compress(x, SR, s)

    def transient_db(z: np.ndarray) -> float:          # loudest 5 ms vs the median 5 ms
        rms = np.sqrt(dynamics.block_mean_square(z, int(0.005 * SR)))
        return 20 * math.log10(rms.max() / np.median(rms))
    assert gr > 3.0 and transient_db(y) < transient_db(x) - 3.0
    calm = slice(int(0.1 * SR), int(0.45 * SR))                                 # the rest is untouched
    assert np.allclose(y[calm], x[calm], atol=1e-4)


def test_vocal_chain_off_is_only_gain():
    v, _ = sibilant_vocal(6.0, 220.0, seed=2)
    s = spec.resolve("natural", {"vocalChain": "off"})
    t = spec.timings(s, 90.0)
    y = chains.process(v, SR, s, t, chain="off")
    ratio = y[v != 0] / v[v != 0]
    assert np.allclose(ratio, ratio[0], rtol=1e-5)


def test_reverb_estimate_round_trip():
    v, _ = sibilant_vocal(30.0, 220.0, seed=6)
    for amount, tol in ((0.5, 0.15), (0.15, 0.06)):
        tail = match.tail_ratio_db(wet_original(v, amount), SR)
        assert match.reverb_for_tail(tail, 1.6) == pytest.approx(amount, abs=tol)
    assert match.reverb_for_tail(match.tail_ratio_db(stereo(v), SR), 1.6) <= 0.05


def test_spectral_match_curve_and_fir():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(10 * SR).astype(np.float32)
    dark = signal.sosfilt(signal.butter(2, 3000, fs=SR, output="sos"), noise).astype(np.float32)
    curve = match.match_curve(match.band_levels(dark, SR), match.band_levels(noise, SR), 1.0)
    assert curve[-1] == pytest.approx(4.0) and abs(curve[match.THIRD_OCTAVES < 1000]).max() < 1.0
    y = match.apply_fir(dark, match.match_fir(curve, SR))
    hi = lambda z: band_energy_db(z, 8000, 12000) - band_energy_db(z, 300, 1000)  # noqa: E731
    assert hi(y) - hi(dark) > 3.0
    assert np.argmax(signal.correlate(y[:SR], dark[:SR], mode="full")) == SR - 1      # no delay


# ----------------------------------------------------------------------------- the mix with a reference

@pytest.mark.parametrize("which", ["loud", "quiet"])
def test_follows_the_original_balance(song_files, tmp_path, which):
    buses: dict = {}
    out = tmp_path / "o.wav"
    result = mixer.mix(v2_job(song_files, out, reference=which), buses=buses)
    original = song_files[f"ratio_{which}"]
    ours = lu(buses["vocals"]) - lu(buses["backing"])
    assert ours == pytest.approx(original, abs=1.5)
    assert result["vocalToBackingDb"] == pytest.approx(ours, abs=0.05)
    assert set(result) == REPORT_KEYS
    y, _ = read_out(out)
    assert lu(y) == pytest.approx(-14.0, abs=0.5) and loudness.true_peak_db(y) <= -0.98
    assert result["truePeakDb"] <= -1.0 + 1e-6


def test_without_reference_match_is_ignored(song_files, tmp_path):
    outs = []
    for k, m in enumerate((0.0, 1.0)):
        out = tmp_path / f"{k}.wav"
        mixer.mix(v2_job(song_files, out, reference=None, params={"match": m, "matchLoudness": True}))
        outs.append(read_out(out)[0])
    assert np.array_equal(outs[0], outs[1])


def test_match_loudness_uses_the_original(song_files, tmp_path):
    out = tmp_path / "o.wav"
    result = mixer.mix(v2_job(song_files, out, params={"matchLoudness": True}))
    original = lu(read_out(song_files["mix_loud"])[0])
    assert result["lufs"] == pytest.approx(original, abs=0.3)
    assert lu(read_out(out)[0]) == pytest.approx(original, abs=0.5)


def test_level_riding_follows_the_originals_contour(song_files, tmp_path):
    original, _ = sibilant_vocal(26.0, 220.0, seed=1)
    contour = np.where(np.arange(original.size) < 13 * SR, 0.4, 1.0).astype(np.float32)   # verse quiet, chorus loud
    ref = write(tmp_path / "contour.wav", wet_original(original * contour, 0.2))
    job = v2_job(song_files, tmp_path / "o.wav", reference=None, params={"match": 1.0})
    job["reference"] = {"vocals": str(ref), "mix": None, "offset": 0.0}
    buses: dict = {}
    mixer.mix(job, buses=buses)
    v, b = buses["vocals"], buses["backing"]
    diff = lambda a, z: lu(a[:, z]) - lu(b[:, z])  # noqa: E731
    verse, chorus = slice(int(2 * SR), int(11 * SR)), slice(int(15 * SR), int(24 * SR))
    assert diff(v, chorus) - diff(v, verse) > 2.5          # the original's +8 dB contour, clamped to ±4


def test_offsets_roles_and_report(song_files, tmp_path):
    extra = [{"path": str(song_files["harmony"]), "role": "harmony", "gainDb": 0, "pan": -0.6, "offset": 16.0}]
    job = v2_job(song_files, tmp_path / "o.wav", reference=None, extra=extra)
    for i in job["inputs"][:2]:
        i["offset"] = 10.0
    buses: dict = {}
    result = mixer.mix(job, buses=buses)
    assert result["offset"] == 10.0 and set(result) == REPORT_KEYS
    y, _ = read_out(tmp_path / "o.wav")
    assert y.shape[-1] == 26 * SR                                     # harmony ends at 16 + 12 = 28 ≤ 36
    v = buses["vocals"]
    # the harmony (panned left) sits from 6 s on: before that the vocal bus is centred
    side = lambda z: 10 * math.log10(np.mean(v[0, z] ** 2) / np.mean(v[1, z] ** 2))  # noqa: E731
    assert abs(side(slice(int(1 * SR), int(5.5 * SR)))) < 0.5
    assert side(slice(int(7 * SR), int(16 * SR))) > 0.5


def test_doubler_widens_harmonies(song_files, tmp_path):
    extra = [{"path": str(song_files["harmony"]), "role": "harmony", "gainDb": 0, "pan": 0.0, "offset": 0.0}]
    widths = []
    for d in (0.0, 1.0):
        buses: dict = {}
        mixer.mix(v2_job(song_files, tmp_path / f"d{d}.wav", reference=None, extra=extra,
                         params={"doubler": d, "reverb": 0, "throws": False}), buses=buses)
        v = buses["vocals"]
        mid, side = (v[0] + v[1]) / 2, (v[0] - v[1]) / 2
        widths.append(10 * math.log10(np.mean(side ** 2) / np.mean(mid ** 2) + 1e-12))
    assert widths[0] < -60 and widths[1] > -25


def test_unmask_dips_the_backing_only_while_singing(song_files, tmp_path):
    bands = []
    for amount in (0.0, 1.0):
        buses: dict = {}
        mixer.mix(v2_job(song_files, tmp_path / f"u{amount}.wav", reference=None, params={"unmask": amount}),
                  buses=buses)
        b = buses["backing"].mean(axis=0)
        sung = [(int((0.3 + 4.7 * k + 0.5) * SR), int((0.3 + 4.7 * k + 3.5) * SR)) for k in range(5)]
        gaps = [(int((4.3 + 4.7 * k + 0.25) * SR), int((4.3 + 4.7 * k + 0.6) * SR)) for k in range(4)]
        rel = lambda spans: band_energy_db(b, 1200, 3500, spans) - band_energy_db(b, 40, 120, spans)  # noqa: E731
        bands.append((rel(sung), rel(gaps)))
    assert bands[0][0] - bands[1][0] > 1.5                 # dipped while singing
    assert abs(bands[0][1] - bands[1][1]) < 0.6           # (almost) untouched in the gaps


def test_throws_fill_the_gaps(song_files, tmp_path):
    gaps = []
    for on in (False, True):
        buses: dict = {}
        mixer.mix(v2_job(song_files, tmp_path / f"t{on}.wav", reference=None,
                         params={"throws": on, "reverb": 0, "gate": True}), buses=buses)
        v = buses["vocals"].mean(axis=0)
        spans = [(int((4.3 + 4.7 * k + 0.15) * SR), int((4.3 + 4.7 * k + 0.6) * SR)) for k in range(4)]
        gaps.append(10 * math.log10(np.mean(np.concatenate([v[a:b] for a, b in spans]) ** 2) + 1e-20))
    assert gaps[1] - gaps[0] > 10.0


def test_gate_in_the_mix(tmp_path):
    rng = np.random.default_rng(5)
    v = song(20.0, 220.0, seed=4).astype(np.float32)
    hissy = write(tmp_path / "hissy.wav", v + (rng.standard_normal(v.size) * 10 ** (-50 / 20)).astype(np.float32))
    floors = []
    for gate in (False, True):
        out = tmp_path / f"g{gate}.wav"
        mixer.mix({"inputs": [{"path": str(hissy), "role": "lead"}], "output": str(out),
                   "params": {"gate": gate, "reverb": 0, "throws": False, "warmth": 0}})
        y = read_out(out)[0].mean(axis=0)
        spans = [(int((4.3 + 4.7 * k + 0.25) * SR), int((4.3 + 4.7 * k + 0.55) * SR)) for k in range(3)]
        sung = np.mean(y[int(1 * SR): int(3.5 * SR)] ** 2)
        floors.append(10 * math.log10(np.mean(np.concatenate([y[a:b] for a, b in spans]) ** 2) / sung))
    assert floors[0] - floors[1] >= 6.0


def test_warmth_and_light_chain_run(song_files, tmp_path):
    for params in ({"warmth": 1.0}, {"vocalChain": "light"}, {"vocalChain": "off", "gate": False}):
        out = tmp_path / "o.wav"
        result = mixer.mix(v2_job(song_files, out, reference=None, params=params))
        y, _ = read_out(out)
        assert np.all(np.isfinite(y)) and lu(y) == pytest.approx(-14.0, abs=0.5)
        assert result["truePeakDb"] <= -1.0 + 1e-6


def test_seed_zero_is_deterministic_v2(song_files, tmp_path):
    outs = []
    for k in range(2):
        out = tmp_path / f"{k}.wav"
        mixer.mix(v2_job(song_files, out, preset="pop"))
        outs.append(read_out(out)[0])
    assert np.array_equal(outs[0], outs[1])


# ----------------------------------------------------------------------------- bus processor

def test_bus_is_a_plain_sum(song_files, tmp_path):
    a = read_out(song_files["backing"])[0]
    h = read_out(song_files["harmony"])[0]
    job = {"processor": "bus", "output": str(tmp_path / "bus.wav"), "params": {},
           "inputs": [{"path": str(song_files["backing"]), "role": "backing", "gainDb": 2.0, "pan": 0, "offset": 5.0},
                      {"path": str(song_files["harmony"]), "role": "other", "gainDb": -1.0, "pan": 0.5, "offset": 7.0},
                      {"path": str(song_files["converted"]), "role": "lead", "gainDb": 0.0, "mute": True,
                       "offset": 5.0}]}
    result = mixer.mix(job)
    y, _ = read_out(tmp_path / "bus.wav")
    n = max(a.shape[-1], int(2 * SR) + h.shape[-1])
    expect = np.zeros((2, n), np.float32)
    expect[:, : a.shape[-1]] += a * np.float32(10 ** (2 / 20))
    expect[0, 2 * SR: 2 * SR + h.shape[-1]] += h[0] * np.float32(0.5 * 10 ** (-1 / 20))
    expect[1, 2 * SR: 2 * SR + h.shape[-1]] += h[0] * np.float32(10 ** (-1 / 20))
    assert y.shape == expect.shape and np.allclose(y, expect, atol=1e-6)
    assert result["offset"] == 5.0 and result["vocalToBackingDb"] is None and set(result) == REPORT_KEYS
    job["inputs"][1]["solo"] = True
    mixer.mix(job)
    y, _ = read_out(tmp_path / "bus.wav")
    assert np.allclose(y[:, : 2 * SR], 0) and np.abs(y).max() > 0.01


def test_bus_normalize(song_files, tmp_path):
    job = {"processor": "bus", "output": str(tmp_path / "bus.wav"), "params": {"normalize": True, "targetLufs": -10},
           "inputs": [{"path": str(song_files["backing"]), "role": "backing", "gainDb": 6.0}]}
    result = mixer.mix(job)
    y, _ = read_out(tmp_path / "bus.wav")
    assert lu(y) == pytest.approx(-10.0, abs=0.5) and loudness.true_peak_db(y) <= -0.98
    assert result["truePeakDb"] <= -1.0 + 1e-6


# ----------------------------------------------------------------------------- CLI + performance

def test_cli_v2_three_inputs_with_reference_under_20_seconds(tmp_path):
    lead, _ = sibilant_vocal(180.0, 262.0, seed=11)
    harmony, _ = sibilant_vocal(170.0, 330.0, seed=11)
    backing = drum_backing(180.0, 92.0, seed=1)
    original, _ = sibilant_vocal(180.0, 220.0, seed=11)
    orig = stereo(original)
    files = {k: write(tmp_path / f"{k}.wav", v) for k, v in
             dict(lead=lead, harmony=harmony, backing=backing, orig=orig, mix=orig + backing * 0.5).items()}
    job = {"inputs": [{"path": str(files["lead"]), "role": "lead", "gainDb": 0, "pan": 0, "offset": 0},
                      {"path": str(files["harmony"]), "role": "harmony", "gainDb": 0, "pan": 0.5, "offset": 10},
                      {"path": str(files["backing"]), "role": "backing", "gainDb": 0, "pan": 0, "offset": 0}],
           "reference": {"vocals": str(files["orig"]), "mix": str(files["mix"]), "offset": 0},
           "processor": "builtin", "preset": "pop", "params": {"doubler": 0.8, "warmth": 0.5}, "seed": 3,
           "output": str(tmp_path / "out.wav")}
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    start = time.perf_counter()
    code, messages, stderr = run_cli(job_path)
    elapsed = time.perf_counter() - start
    assert code == 0, stderr
    assert set(messages[-1]["result"]) == REPORT_KEYS
    y, _ = read_out(tmp_path / "out.wav")
    assert y.shape == (2, 180 * SR) and np.all(np.isfinite(y))
    assert lu(y) == pytest.approx(-14.0, abs=0.5) and loudness.sample_peak_db(y) <= -0.9
    assert elapsed < 20.0, f"3-minute mix with 3 inputs took {elapsed:.1f} s"
