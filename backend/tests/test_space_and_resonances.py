"""The resonance suppressor ("去刺耳") and the reverb tail's modulation."""

from __future__ import annotations

import numpy as np
from scipy import signal

from stellaria_backend import fx
from stellaria_backend.mixing import filters, reverb, spectral

SR = 44100


def band_db(x: np.ndarray, lo: float, hi: float) -> float:
    f, p = signal.welch(x, SR, nperseg=4096)
    return float(10.0 * np.log10(p[(f >= lo) & (f <= hi)].mean()))


def noise_with_resonance(seconds: float = 4.0) -> np.ndarray:
    noise = np.random.default_rng(3).standard_normal(int(SR * seconds)).astype(np.float32) * 0.05
    return filters.apply(filters.peaking(4000.0, 12.0, 6.0, SR), noise).astype(np.float32)


def sawtooth(seconds: float = 3.0, f0: float = 220.0, vibrato: float = 0.0) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    phase = t + vibrato * np.sin(2 * np.pi * 5.5 * t)
    return (sum(np.sin(2 * np.pi * f0 * k * phase) / k for k in range(1, 60)) * 0.1).astype(np.float32)


# ----------------------------------------------------------------------------- resonances

def test_resonance_is_pulled_down_and_its_neighbours_are_not():
    x = noise_with_resonance()
    y, info = spectral.smooth_resonances(x, SR, amount=1.0)
    assert y.shape == x.shape and y.dtype == np.float32
    assert band_db(y, 3800, 4200) - band_db(x, 3800, 4200) < -3.0
    for lo, hi in ((800, 1200), (2300, 2700), (6500, 7500)):
        assert abs(band_db(y, lo, hi) - band_db(x, lo, hi)) < 0.3
    assert 3.0 < info["maxReduction"] <= 6.0


def test_amount_scales_the_cut():
    x = noise_with_resonance()
    half, _ = spectral.smooth_resonances(x, SR, amount=0.5)
    full, _ = spectral.smooth_resonances(x, SR, amount=1.0)
    cut_half = band_db(half, 3800, 4200) - band_db(x, 3800, 4200)
    cut_full = band_db(full, 3800, 4200) - band_db(x, 3800, 4200)
    assert cut_full < cut_half < -1.0


def test_harmonics_are_not_mistaken_for_resonances():
    for x in (sawtooth(), sawtooth(vibrato=0.002)):
        y, info = spectral.smooth_resonances(x, SR, amount=1.0)
        assert info["maxReduction"] < 1.0
        for lo, hi in ((2000, 3000), (3000, 5000), (5000, 8000)):
            assert abs(band_db(y, lo, hi) - band_db(x, lo, hi)) < 0.3


def test_resonances_off_and_stereo():
    x = noise_with_resonance(1.0)
    y, info = spectral.smooth_resonances(x, SR, amount=0.0)
    assert np.array_equal(y, x) and info["maxReduction"] == 0.0
    # nothing to cut: the analysis/synthesis round trip is transparent
    y, _ = spectral.smooth_resonances(x, SR, amount=1e-9)
    assert np.abs(y - x).max() < 1e-5
    st = np.stack([x, x[::-1].copy()])
    y, _ = spectral.smooth_resonances(st, SR, amount=1.0)
    assert y.shape == st.shape and np.all(np.isfinite(y))
    assert spectral.smooth_resonances(np.zeros(0, np.float32), SR)[0].shape == (0,)


def test_stereo_is_linked():
    x = noise_with_resonance(2.0)
    other = np.random.default_rng(9).standard_normal(x.size).astype(np.float32) * 0.05     # no resonance
    y, _ = spectral.smooth_resonances(np.stack([x, 0.5 * x]), SR, amount=1.0)
    assert np.allclose(y[1], 0.5 * y[0], atol=1e-6)                       # the same cut on both sides
    y, _ = spectral.smooth_resonances(np.stack([x, other]), SR, amount=1.0)
    assert band_db(y[1], 3800, 4200) < band_db(other, 3800, 4200) - 0.5     # the clean side follows


def test_smooth_effect_in_a_chain():
    x = np.stack([noise_with_resonance(2.0)] * 2)
    slot = {"id": "s", "kind": "builtin", "effect": "smooth", "params": {"amount": 1.0}, "mix": 1.0, "bypass": False}
    y = fx.apply_chain(x, SR, [slot], bpm=None, vst_plugins={})
    assert y.shape == x.shape
    assert band_db(y[0], 3800, 4200) < band_db(x[0], 3800, 4200) - 3.0
    slot["params"] = {"amount": 1.0, "bandLow": 7000.0}     # the resonance is outside the band (and its fade) now
    y = fx.apply_chain(x, SR, [slot], bpm=None, vst_plugins={})
    assert band_db(y[0], 3800, 4200) > band_db(x[0], 3800, 4200) - 0.5


# ----------------------------------------------------------------------------- reverb modulation

def instantaneous(y: np.ndarray, f: float) -> tuple[np.ndarray, np.ndarray]:
    z = signal.hilbert(y[4000:-4000].astype(np.float64))
    inst = np.diff(np.unwrap(np.angle(z))) * SR / (2 * np.pi)
    inst = np.convolve(inst, np.ones(441) / 441, mode="valid")[1000:-1000]
    return np.abs(z)[1000:-1000], 1200.0 * np.log2(inst / f)


def test_modulation_drifts_a_few_cents_without_dulling():
    for f in (1000.0, 8000.0):
        tone = np.sin(2 * np.pi * f * np.arange(SR * 4) / SR).astype(np.float32)
        y = reverb.modulate(np.stack([tone, tone]), SR, reverb.MOD_MAX_MS)
        assert y.shape == (2, tone.size) and y.dtype == np.float32
        env, cents = instantaneous(y[0], f)
        assert 3.0 < np.abs(cents).max() < 9.0
        assert env.min() > 0.95 and env.max() < 1.01
        # the two channels drift differently (that is what widens the tail)
        assert np.abs(y[0] - y[1]).max() > 0.1


def test_modulation_zero_is_the_plain_convolution():
    send = np.random.default_rng(1).standard_normal(SR * 2).astype(np.float32) * 0.1
    ir = reverb.space_ir(reverb.SpaceSpec(weights={"plate": 0.5, "hall": 0.5}, decay_s=1.5, predelay_s=0.02, width=1.0), SR)
    plain = reverb.convolve(send, ir, send.size, sr=SR, cutoff_hz=8000.0)
    assert np.array_equal(reverb.convolve_modulated(send, ir, send.size, sr=SR, cutoff_hz=8000.0, modulation=0.0), plain)
    assert np.array_equal(reverb.modulate(plain, SR, 0.0), plain)


def test_modulation_leaves_the_early_reflections_alone():
    ir = reverb.space_ir(reverb.SpaceSpec(weights={"room": 1.0}, decay_s=1.2, predelay_s=0.0, width=1.0), SR)
    send = np.zeros(SR * 2, np.float32)
    send[0] = 1.0
    plain = reverb.convolve(send, ir, send.size, sr=SR, cutoff_hz=12000.0)
    moving = reverb.convolve_modulated(send, ir, send.size, sr=SR, cutoff_hz=12000.0, modulation=1.0)
    first = int(np.argmax(np.abs(plain).max(axis=0) > np.abs(plain).max() * 1e-3))
    early = slice(first, first + int(0.06 * SR))
    assert np.abs(moving[:, early] - plain[:, early]).max() < 1e-4
    late = slice(first + int(0.3 * SR), first + int(1.0 * SR))
    assert np.abs(moving[:, late] - plain[:, late]).max() > 1e-3
    # about the same energy either way
    ratio = float((moving ** 2).sum() / (plain ** 2).sum())
    assert 0.85 < ratio < 1.1
