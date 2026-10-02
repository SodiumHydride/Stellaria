"""Built-in effects library (``stellaria_backend.fx``): specs, every effect, chain mechanics, VST slots."""

from __future__ import annotations

import json
import math
import sys
import time
import types

import numpy as np
import pytest

from stellaria_backend import fx
from stellaria_backend.errors import ApiError
from stellaria_backend.mixing import filters, loudness
from test_mixing import SR, band_energy_db, drum_backing, sibilant_vocal


def slot(effect: str, params: dict | None = None, mix: float = 1.0, bypass: bool = False) -> dict:
    return {"id": f"s-{effect}", "kind": "builtin", "effect": effect, "params": params or {}, "mix": mix,
            "bypass": bypass}


def run(x: np.ndarray, *slots: dict, bpm: float | None = 120.0, vst: dict | None = None) -> np.ndarray:
    return fx.apply_chain(x, SR, list(slots), bpm=bpm, vst_plugins=vst or {})


@pytest.fixture(scope="module")
def vocal() -> np.ndarray:
    v, _ = sibilant_vocal(8.0, 220.0, seed=2)
    return v[None, :]


def impulse(seconds: float = 2.0, channels: int = 1) -> np.ndarray:
    x = np.zeros((channels, int(seconds * SR)), np.float32)
    x[:, 1000] = 0.5
    return x


# ----------------------------------------------------------------------------- specs

def test_effect_specs_wire():
    specs = fx.effect_specs()
    json.dumps(specs)
    assert [s["id"] for s in specs] == ["eq", "compressor", "deesser", "smooth", "gate", "reverb", "delay", "saturation",
                                        "doubler", "gain", "stereo", "limiter"]
    categories = {s["id"]: s["category"] for s in specs}
    assert categories == {"eq": "eq", "compressor": "dynamics", "deesser": "dynamics", "smooth": "dynamics",
                          "gate": "dynamics",
                          "limiter": "dynamics", "reverb": "space", "delay": "space", "doubler": "space",
                          "saturation": "color", "gain": "utility", "stereo": "utility"}
    for s in specs:
        assert s["name"] == {"code": f"fx.effect.{s['id']}", "params": {}}
        for p in s["params"]:
            assert p["label"] == {"code": f"fx.param.{p['id']}", "params": {}}
            assert p["type"] in ("number", "boolean", "choice") and p["default"] is not None
            if p["type"] == "number":
                assert p["min"] <= p["default"] <= p["max"]
            if p["type"] == "choice":
                assert p["default"] in [o["value"] for o in p["options"]]
                assert all(o["label"]["code"].startswith("fx.") for o in p["options"])
    reverb = {p["id"]: p for p in next(s for s in specs if s["id"] == "reverb")["params"]}
    assert [o["label"]["code"] for o in reverb["type"]["options"]] == [f"fx.reverbType.{t}" for t in
                                                                       ("room", "plate", "chamber", "hall")]


@pytest.mark.parametrize("effect", list(fx.EFFECTS))
def test_every_effect_at_defaults_sounds_sane(vocal, effect):
    y = run(vocal, slot(effect))
    assert y.dtype == np.float32 and y.ndim == 2 and y.shape[0] in (1, 2) and np.all(np.isfinite(y))
    assert y.shape[-1] >= vocal.shape[-1]
    if not fx.EFFECTS[effect].tail:
        assert y.shape[-1] == vocal.shape[-1]
    # dual mono counts +3 LU in BS.1770; otherwise defaults stay within a few LU of the input
    stereo_gain = 3.01 if y.shape[0] == 2 else 0.0
    change = loudness.integrated_loudness(y, SR) - loudness.integrated_loudness(vocal, SR) - stereo_gain
    assert -4.0 < change < 2.5, change
    assert loudness.sample_peak_db(y) < 3.0


# ----------------------------------------------------------------------------- chain mechanics

def test_bypass_mix_zero_and_empty_chain(vocal):
    assert np.array_equal(run(vocal), vocal)
    assert np.array_equal(run(vocal, slot("eq", {"band2Gain": 9}, bypass=True)), vocal)
    assert np.array_equal(run(vocal, slot("eq", {"band2Gain": 9}, mix=0.0)), vocal)


def test_mix_blends_dry_and_wet(vocal):
    wet = run(vocal, slot("gain", {"gain": -12}))
    half = run(vocal, slot("gain", {"gain": -12}, mix=0.5))
    assert np.allclose(half, 0.5 * vocal + 0.5 * wet, atol=1e-6)
    stereo_wet = run(vocal, slot("stereo", {"pan": 1.0}, mix=0.5))      # mono dry broadcast to stereo
    assert stereo_wet.shape[0] == 2 and np.allclose(stereo_wet[0], 0.5 * vocal[0], atol=1e-6)


def test_reverb_tail_is_kept_trimmed_and_capped():
    x = impulse(1.0)
    y = run(x, slot("reverb", {"decay": 2.0, "level": -6}))
    assert y.shape[0] == 2 and SR < y.shape[-1] <= SR + 6 * SR
    floor = 10 ** (-90 / 20)
    assert np.abs(y[:, -1]).max() > floor * 0.1 or y.shape[-1] == SR + 6 * SR
    assert np.abs(y[:, SR:]).max() > 1e-4                                # there is a tail
    long = run(x, slot("reverb", {"decay": 8.0, "level": 0}), slot("delay", {"feedback": 0.95, "sync": "1/2"}))
    assert long.shape[-1] <= SR + 6 * SR
    # nothing to ring: the output keeps the input length
    assert run(np.zeros((1, SR), np.float32), slot("reverb")).shape[-1] == SR


def test_delay_sync_and_ping_pong():
    y = run(impulse(3.0), slot("delay", {"sync": "1/4", "feedback": 0.5, "level": -6, "pingPong": True}),
            bpm=120.0)
    first = 1000 + int(0.5 * SR)                                          # a quarter note at 120 BPM
    region = slice(first - 50, first + 400)
    assert np.abs(y[0, region]).max() > 1.9 * np.abs(y[1, region]).max()  # first echo leans left (0.8 / 0.2 power)
    second = slice(first + int(0.5 * SR) - 50, first + int(0.5 * SR) + 400)
    assert np.abs(y[1, second]).max() > 1.9 * np.abs(y[0, second]).max()  # then right
    ms = run(impulse(3.0), slot("delay", {"sync": "off", "time": 250, "pingPong": False}), bpm=120.0)
    echo = 1000 + int(0.25 * SR)
    assert np.abs(ms[0, echo - 50: echo + 400]).max() > 0.01
    assert np.allclose(ms[0], ms[1], atol=1e-6)
    no_bpm = run(impulse(3.0), slot("delay", {"sync": "1/4", "time": 300}), bpm=None)  # falls back to time
    assert np.abs(no_bpm[:, 1000 + int(0.3 * SR) - 50: 1000 + int(0.3 * SR) + 400]).max() > 0.01


def test_eq_bands():
    t = np.arange(2 * SR) / SR
    for freq, params, expect in ((1000.0, {"band2Gain": 6.0}, 6.0), (50.0, {"highPassFreq": 120.0}, -9.0),
                                 (3500.0, {}, 1.5)):
        x = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)[None]
        y = run(x, slot("eq", params))
        gain = 20 * math.log10(np.sqrt(np.mean(y[0, SR:] ** 2)) / np.sqrt(np.mean(x[0, SR:] ** 2)))
        assert gain == pytest.approx(expect, abs=1.0) if expect != -9.0 else gain < expect


def test_compressor_and_limiter():
    t = np.arange(6 * SR) / SR
    env = 0.03 + 0.6 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 2.0)) ** 3
    x = (env * np.sin(2 * np.pi * 220 * t)).astype(np.float32)[None]
    y = run(x, slot("compressor", {"ratio": 6}))
    rng = lambda z: np.ptp(np.percentile(20 * np.log10(np.abs(z[0]) + 1e-9)[::64], [50, 99]))  # noqa: E731
    assert rng(y) < rng(x) - 3.0
    from stellaria_backend.mixing import dynamics
    assert dynamics.active_rms_db(y, SR) == pytest.approx(dynamics.active_rms_db(x, SR), abs=0.3)   # auto makeup
    loud = run(x * 3, slot("limiter", {"ceiling": -2.0, "gain": 6.0}))
    assert loudness.true_peak_db(loud) <= -2.0 + 0.02


def test_deesser_and_gate(vocal):
    v, spans = sibilant_vocal(8.0, 220.0, seed=2)
    y = run(v[None], slot("deesser", {"range": 8}))
    assert band_energy_db(v, 7000, 10000, spans) - band_energy_db(y[0], 7000, 10000, spans) > 2.5
    hiss = (np.random.default_rng(1).standard_normal(v.size) * 1e-3).astype(np.float32)
    g = run((v + hiss)[None], slot("gate", {"range": 20}))
    gap = slice(int(4.5 * SR), int(4.85 * SR))                         # after hold (80 ms) and release
    assert 10 * math.log10(np.mean(hiss[gap] ** 2) / np.mean(g[0, gap] ** 2)) > 10


def test_saturation_doubler_stereo_gain():
    t = np.arange(SR) / SR
    x = (0.5 * np.sin(2 * np.pi * 441 * t)).astype(np.float32)[None]
    y = run(x, slot("saturation", {"drive": 12, "mode": "tube"}))
    spec = np.abs(np.fft.rfft(y[0] * np.hanning(SR)))
    assert spec[882] / spec[441] > 0.01                                  # even harmonics
    d = run(x, slot("doubler", {"amount": 1.0}))
    assert d.shape[0] == 2 and not np.allclose(d[0], d[1])
    s = run(np.stack([x[0], -x[0]]), slot("stereo", {"width": 0.0}))
    assert np.allclose(s[0], s[1]) and np.abs(s).max() < 1e-6            # pure side → silence at width 0
    bass = np.stack([np.sin(2 * np.pi * 60 * t), -np.sin(2 * np.pi * 60 * t)]).astype(np.float32) * 0.3
    mono = run(bass, slot("stereo", {"monoBass": 150}))
    assert np.abs(mono[:, 4000:-4000]).max() < 0.05                      # side below 150 Hz removed
    assert np.allclose(run(x, slot("gain", {"gain": 6, "invert": True})), -x * 10 ** (6 / 20), atol=1e-6)


# ----------------------------------------------------------------------------- errors / VST

def test_unknown_effect_and_bad_slot(vocal):
    with pytest.raises(ApiError) as e:
        run(vocal, slot("flanger"))
    assert e.value.code == "fx.unknown_effect" and e.value.params == {"effect": "flanger"}
    with pytest.raises(ApiError) as e:
        run(vocal, {"kind": "midi"})
    assert e.value.code == "fx.bad_slot"


def test_vst_slots(vocal, monkeypatch):
    vst_slot = {"id": "v", "kind": "vst", "pluginId": "abc", "state": None, "params": {"Mix": 0.5}, "mix": 1.0,
                "bypass": False}
    with pytest.raises(ApiError) as e:
        run(vocal, vst_slot)                                              # plugin not in vst_plugins
    assert e.value.code == "vst.not_found"
    plugin = {"id": "abc", "name": "Pro-Q", "path": "/x.vst3"}
    if "stellaria_backend.vst.processing" not in sys.modules:
        try:
            import stellaria_backend.vst.processing  # noqa: F401
        except ModuleNotFoundError:
            with pytest.raises(ApiError) as e:
                run(vocal, vst_slot, vst={"abc": plugin})
            assert e.value.code == "vst.host_missing"
    calls = []

    def fake_apply(x, sr, s, p):
        calls.append((x.shape, sr, s["pluginId"], p["name"]))
        return np.concatenate([x * 0.5, np.zeros((x.shape[0], SR), np.float32)], axis=-1)   # with a tail

    package = types.ModuleType("stellaria_backend.vst")
    module = types.ModuleType("stellaria_backend.vst.processing")
    module.apply = fake_apply
    monkeypatch.setitem(sys.modules, "stellaria_backend.vst", package)
    monkeypatch.setitem(sys.modules, "stellaria_backend.vst.processing", module)
    y = run(vocal, slot("gain", {"gain": 0}), {**vst_slot, "mix": 0.5}, vst={"abc": plugin})
    assert calls == [(vocal.shape, SR, "abc", "Pro-Q")]
    assert y.shape[-1] == vocal.shape[-1]                                 # silent tail trimmed
    assert np.allclose(y[:, : vocal.shape[-1]], 0.75 * vocal, atol=1e-6)
    calls.clear()
    run(vocal, {**vst_slot, "bypass": True}, vst={"abc": plugin})
    assert calls == []


def test_vst_module_not_imported_without_vst_slots(vocal, monkeypatch):
    monkeypatch.delitem(sys.modules, "stellaria_backend.vst.processing", raising=False)
    run(vocal, slot("eq"))
    assert "stellaria_backend.vst.processing" not in sys.modules


# ----------------------------------------------------------------------------- performance

def test_six_slot_chain_on_a_four_minute_song_under_10_seconds():
    x = drum_backing(240.0, 96.0)
    chain = [slot("eq"), slot("compressor"), slot("deesser"), slot("saturation", mix=0.5), slot("reverb"),
             slot("limiter")]
    start = time.perf_counter()
    y = run(x, *chain, bpm=96.0)
    elapsed = time.perf_counter() - start
    assert y.shape[0] == 2 and y.shape[-1] >= x.shape[-1] and np.all(np.isfinite(y))
    assert elapsed < 10.0, f"6-slot chain took {elapsed:.1f} s"


def test_filters_zero_phase_matches_sosfiltfilt():
    from scipy import signal

    x = np.random.default_rng(0).standard_normal((2, 20000))
    sos = filters.highpass4(1000.0, SR)
    assert np.allclose(filters.zero_phase(sos, x)[:, 3000:-3000], signal.sosfiltfilt(sos, x, axis=-1)[:, 3000:-3000],
                       atol=1e-9)
