"""Process-node DSP ops (``stellaria_backend.dsp``): canvas-v3 §2 behaviour and quality bars.

Synthetic audio only. Timing bars are for a 4-minute stereo song: trim/split/comp < 1 s, built-in
pitch < 20 s, 6-slot fx chain < 10 s. Optional engines (pedalboard / python-stretch / lameenc /
mutagen) are exercised when installed and skipped otherwise.
"""

from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from conftest import BACKEND

from stellaria_backend.dsp import analysis, encode, naming, pitch as pitch_mod, vocoder
from stellaria_backend.dsp.common import SR, DspError, fade_curve
from stellaria_backend.dsp.ops import run_job

STAGES = {"dsp.read", "dsp.write", "dsp.trim", "dsp.split", "dsp.pitch", "dsp.fx", "dsp.fx.vst", "dsp.comp",
          "dsp.concat", "dsp.export", "dsp.export.loudness", "dsp.export.encode"}
CLICK_BAR = 0.05


# ----------------------------------------------------------------------------- helpers

def sine(seconds: float, freq: float = 1000.0, amp: float = 0.5, channels: int = 2, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(seconds * SR))) / SR
    x = (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)
    return np.repeat(x[None], channels, axis=0)


def write(path: Path, x: np.ndarray, sr: int = SR) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32).T, sr, subtype="FLOAT")
    return str(path)


def read(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr


class Recorder:
    def __init__(self) -> None:
        self.lines: list[dict] = []

    def progress(self, fraction: float, stage: str, **extra) -> None:
        self.lines.append({"progress": fraction, "stage": stage, **extra})

    def log(self, message: str) -> None:
        self.lines.append({"log": message})

    def stages(self) -> set[str]:
        return {m["stage"] for m in self.lines if "stage" in m}


def run(job: dict, rec: Recorder | None = None) -> dict:
    rec = rec or Recorder()
    return run_job(job, rec.progress, rec.log)


def job_for(op: str, inputs: list[tuple[str, float]], params: dict, out: Path, **extra) -> dict:
    return {"op": op, "inputs": [{"path": p, "offset": o} for p, o in inputs], "params": params,
            "outputDir": str(out), "bpm": None, "vstPlugins": {}, "names": {}, **extra}


def edge_steps(y: np.ndarray) -> float:
    """The steps from silence into the first sample and from the last sample into silence."""
    return float(max(np.max(np.abs(y[:, 0])), np.max(np.abs(y[:, -1]))))


def click_at(y: np.ndarray, at: int, width: int = 220) -> float:
    """Discontinuity around sample ``at``: max |second difference| (a sine's own is tiny: 0.0101
    for 1 kHz at -6 dBFS; a jump of d shows as ~d)."""
    seg = y[:, max(0, at - width): at + width]
    return float(np.max(np.abs(np.diff(seg, n=2, axis=-1)))) if seg.shape[-1] > 2 else 0.0


def run_cli(job: dict, tmp_path: Path, env_extra: dict | None = None, prelude: str | None = None):
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": str(BACKEND), **(env_extra or {})}
    if prelude:
        args = [sys.executable, "-c", prelude + "\nfrom stellaria_backend.dsp.run import main\nimport sys\n"
                f"sys.exit(main([{str(job_path)!r}]))"]
    else:
        args = [sys.executable, "-m", "stellaria_backend.dsp.run", str(job_path)]
    proc = subprocess.run(args, cwd=str(BACKEND), env=env, capture_output=True, timeout=300)
    lines = [json.loads(line) for line in proc.stdout.decode("utf-8").splitlines() if line.strip()]
    return proc.returncode, lines, proc.stderr.decode("utf-8", errors="replace")


def check_protocol(lines: list[dict]) -> None:
    last = -1.0
    for m in lines[:-1]:
        if "progress" in m:
            assert m["stage"] in STAGES, m
            assert 0.0 <= m["progress"] <= 1.0 and m["progress"] >= last
            last = m["progress"]
        else:
            assert set(m) == {"log"} and isinstance(m["log"], str)


@pytest.fixture(scope="module")
def song4(tmp_path_factory) -> str:
    """A 4-minute stereo 'song': tones + noise bursts, 44.1 kHz float WAV."""
    rng = np.random.default_rng(1)
    n = 240 * SR
    t = np.arange(n, dtype=np.float32) / SR
    x = np.empty((2, n), dtype=np.float32)
    x[0] = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(n).astype(np.float32)
    x[1] = 0.2 * np.sin(2 * np.pi * 330 * t) + 0.05 * rng.standard_normal(n).astype(np.float32)
    return write(tmp_path_factory.mktemp("song4") / "song.wav", x)


# ----------------------------------------------------------------------------- package

def test_import_is_cheap():
    code = ("import sys, stellaria_backend.dsp as d; "
            "heavy = [m for m in ('scipy', 'librosa', 'torch', 'soundfile') if m in sys.modules]; "
            "print(heavy); assert not heavy, heavy; assert callable(d.sections) and callable(d.key)")
    proc = subprocess.run([sys.executable, "-c", code], cwd=str(BACKEND), capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_fade_curves_are_equal_power():
    fi, fo = fade_curve(441, rising=True), fade_curve(441, rising=False)
    assert np.allclose(fi ** 2 + fo ** 2, 1.0, atol=1e-6)
    assert fi[0] < 0.005 and fo[-1] < 0.005 and np.all(np.diff(fi) > 0)


# ----------------------------------------------------------------------------- trim

def test_trim_is_sample_accurate_with_offset(tmp_path):
    x = sine(5.0, 440, 0.5) * np.linspace(0.5, 1.0, 5 * SR, dtype=np.float32)
    src = write(tmp_path / "in.wav", x)
    res = run(job_for("trim", [(src, 12.5)], {"start": 1.25, "end": 3.75, "fadeIn": 0.05, "fadeOut": 0.1},
                      tmp_path / "o"))
    out = res["outputs"]["out"]
    y, sr = read(out["path"])
    s, e = int(1.25 * SR), int(3.75 * SR)
    assert sr == SR and y.shape == (2, e - s)
    assert out["offset"] == pytest.approx(12.5 + s / SR, abs=1e-6)
    assert out["duration"] == pytest.approx((e - s) / SR, abs=1e-6)
    fi, fo = int(0.05 * SR), int(0.1 * SR)
    assert np.array_equal(y[:, fi: -fo], x[:, s + fi: e - fo])                 # untouched middle
    assert np.allclose(y[:, :fi], x[:, s: s + fi] * fade_curve(fi, rising=True), atol=1e-7)
    assert np.allclose(y[:, -fo:], x[:, e - fo: e] * fade_curve(fo, rising=False), atol=1e-7)
    assert res["result"] == {"start": pytest.approx(s / SR), "end": pytest.approx(e / SR),
                             "fadeIn": pytest.approx(0.05), "fadeOut": pytest.approx(0.1)}


def test_trim_cuts_do_not_click(tmp_path):
    src = write(tmp_path / "in.wav", sine(4.0, 1000.0, 0.5))           # 1 kHz at -6 dBFS
    start = 1.0 + 0.25 / 1000                                          # cut right at a peak
    res = run(job_for("trim", [(src, 0.0)], {"start": start, "end": 3.0 + 0.25 / 1000, "fadeIn": 0, "fadeOut": 0},
                      tmp_path / "o"))
    y, _ = read(res["outputs"]["out"]["path"])
    assert res["result"]["fadeIn"] == pytest.approx(0.01) and res["result"]["fadeOut"] == pytest.approx(0.01)
    assert edge_steps(y) < CLICK_BAR
    assert click_at(y, 0) < CLICK_BAR and click_at(y, y.shape[-1] - 1) < CLICK_BAR
    raw = sine(4.0, 1000.0, 0.5)[:, int(round(start * SR)):]
    assert edge_steps(raw[:, :1000]) > 0.4                             # the metric does see a hard cut


def test_trim_keeps_cuts_in_silence_and_file_edges_unfaded(tmp_path):
    x = np.zeros((1, 3 * SR), dtype=np.float32)
    x[:, SR: 2 * SR] = sine(1.0, 300.0, 0.3, channels=1)
    src = write(tmp_path / "in.wav", x)
    res = run(job_for("trim", [(src, 0.0)], {"start": 0.5, "end": 3.0, "fadeIn": 0, "fadeOut": 0}, tmp_path / "o"))
    y, _ = read(res["outputs"]["out"]["path"])
    assert res["result"]["fadeIn"] == 0 and res["result"]["fadeOut"] == 0     # silence / the file's own end
    assert y.shape == (1, int(2.5 * SR)) and np.array_equal(y, x[:, int(0.5 * SR):])


@pytest.mark.parametrize("params,reason", [
    ({"start": 2.0, "end": 1.0}, "trim.range"),
    ({"start": 9.0, "end": 10.0}, "trim.range"),
    ({"start": "a", "end": 1.0}, "trim.range"),
    ({"start": 0.0, "end": 1.0, "fadeIn": -1}, "trim.fade"),
])
def test_trim_bad_params(tmp_path, params, reason):
    src = write(tmp_path / "in.wav", sine(2.0))
    with pytest.raises(DspError) as e:
        run(job_for("trim", [(src, 0.0)], params, tmp_path / "o"))
    assert e.value.code == "dsp.bad_params" and e.value.params["reason"] == reason


# ----------------------------------------------------------------------------- split

def test_split_parts_offsets_and_centred_fades(tmp_path):
    x = sine(10.0, 1000.0, 0.5)
    src = write(tmp_path / "in.wav", x)
    rec = Recorder()
    res = run(job_for("split", [(src, 3.0)], {"cuts": [6.0, 2.5], "fade": 0.02, "labels": ["主歌", "副歌"]},
                      tmp_path / "o"), rec)
    outs = res["outputs"]
    assert list(outs) == ["part1", "part2", "part3"]
    half = int(0.02 * SR) // 2
    bounds = [(0, int(2.5 * SR) + half), (int(2.5 * SR) - half, int(6.0 * SR) + half), (int(6.0 * SR) - half, 10 * SR)]
    for out, (a, b) in zip(outs.values(), bounds):
        y, _ = read(out["path"])
        assert y.shape[-1] == b - a
        assert out["offset"] == pytest.approx(3.0 + a / SR, abs=1e-6)
        if a > 0:                                                    # cut edges only (not the file's own)
            assert np.max(np.abs(y[:, 0])) < CLICK_BAR
        if b < 10 * SR:
            assert np.max(np.abs(y[:, -1])) < CLICK_BAR
        inner = slice(2 * half if a else 0, (b - a) - 2 * half if b < 10 * SR else b - a)
        assert np.array_equal(y[:, inner], x[:, a:b][:, inner])
    assert [p["label"] for p in res["result"]["parts"]] == ["主歌", "副歌", ""]
    # parts placed at their offsets cross-fade with equal power (no gap at the cut)
    y1, _ = read(outs["part1"]["path"])
    y2, _ = read(outs["part2"]["path"])
    c = int(2.5 * SR)
    g1 = y1[0, c - half: c + half] / np.where(np.abs(x[0, c - half: c + half]) > 1e-3, x[0, c - half: c + half], 1)
    g2 = y2[0, :2 * half] / np.where(np.abs(x[0, c - half: c + half]) > 1e-3, x[0, c - half: c + half], 1)
    ok = np.abs(x[0, c - half: c + half]) > 0.05
    assert np.allclose(g1[ok] ** 2 + g2[ok] ** 2, 1.0, atol=1e-3)
    assert {"dsp.read", "dsp.split", "dsp.write"} <= rec.stages()


def test_split_default_fade_and_errors(tmp_path):
    src = write(tmp_path / "in.wav", sine(4.0, 1000.0, 0.5))
    res = run(job_for("split", [(src, 0.0)], {"cuts": [2.00025], "fade": 0, "labels": []}, tmp_path / "o"))
    assert res["result"]["fades"] == [pytest.approx(0.01)]
    y1, _ = read(res["outputs"]["part1"]["path"])
    y2, _ = read(res["outputs"]["part2"]["path"])
    assert np.max(np.abs(y1[:, -1])) < CLICK_BAR and np.max(np.abs(y2[:, 0])) < CLICK_BAR
    with pytest.raises(DspError) as e:
        run(job_for("split", [(src, 0.0)], {"cuts": [5.0], "fade": 0}, tmp_path / "o2"))
    assert e.value.params["reason"] == "split.cuts"
    res = run(job_for("split", [(src, 0.0)], {"cuts": [], "fade": 0}, tmp_path / "o3"))
    assert list(res["outputs"]) == ["part1"] and res["outputs"]["part1"]["duration"] == pytest.approx(4.0)


# ----------------------------------------------------------------------------- comp

def test_comp_aligns_by_offset_and_crossfades(tmp_path):
    a = sine(12.0, 440.0, 0.4)
    b = sine(10.0, 660.0, 0.4)
    pa = write(tmp_path / "a.wav", a)
    pb = write(tmp_path / "b.wav", b)
    segs = [{"start": 5.0, "end": 10.0, "input": 0}, {"start": 10.0, "end": 13.0, "input": 1},
            {"start": 13.0, "end": 14.0, "input": -1}, {"start": 14.0, "end": 16.0, "input": 0}]
    res = run(job_for("comp", [(pa, 5.0), (pb, 7.0)], {"segments": segs, "crossfade": 0.04}, tmp_path / "o"))
    out = res["outputs"]["out"]
    y, _ = read(out["path"])
    assert out["offset"] == pytest.approx(5.0)                       # min input offset
    assert y.shape == (2, int(11.0 * SR))                            # until the last segment's end (16 s)

    def at(t: float) -> int:                                         # song time → output sample
        return int(round((t - 5.0) * SR))

    def src_a(t: float) -> int:
        return int(round((t - 5.0) * SR))

    def src_b(t: float) -> int:
        return int(round((t - 7.0) * SR))

    assert np.array_equal(y[:, at(6): at(9)], a[:, src_a(6): src_a(9)])
    assert np.array_equal(y[:, at(11): at(12.9)], b[:, src_b(11): src_b(12.9)])
    assert np.max(np.abs(y[:, at(13.1): at(13.9)])) == 0.0
    assert np.array_equal(y[:, at(14.1): at(15.9)], a[:, src_a(14.1): src_a(15.9)])
    k = at(10.0)                                                     # centre of the 40 ms cross-fade
    expect = a[0, src_a(10.0)] * math.cos(math.pi / 4) + b[0, src_b(10.0)] * math.sin(math.pi / 4)
    assert y[0, k] == pytest.approx(expect, abs=2e-3)
    assert [s["input"] for s in res["result"]["segments"]] == [0, 1, -1, 0]


def test_comp_boundaries_do_not_click(tmp_path):
    a = sine(6.0, 1000.0, 0.5)
    b = sine(6.0, 1000.0, 0.5, phase=2.0)                            # same tone, different phase
    pa, pb = write(tmp_path / "a.wav", a), write(tmp_path / "b.wav", b)
    segs = [{"start": 0.0, "end": 2.0, "input": 0}, {"start": 2.0, "end": 4.0, "input": 1},
            {"start": 4.0, "end": 6.0, "input": 0}]
    for cf in (0.0, 0.02):
        res = run(job_for("comp", [(pa, 0.0), (pb, 0.0)], {"segments": segs, "crossfade": cf}, tmp_path / f"o{cf}"))
        y, _ = read(res["outputs"]["out"]["path"])
        for t in (2.0, 4.0):
            assert click_at(y, int(t * SR)) < CLICK_BAR, (cf, t)
        assert res["result"]["crossfades"][1] >= 0.01                 # 0 would click → 10 ms
        assert np.max(np.abs(y[:, -1])) < CLICK_BAR                    # the end fade lies inside the file
    segs[-1]["end"] = 5.5                                             # ends mid-input: a cut at the end
    res = run(job_for("comp", [(pa, 0.0), (pb, 0.0)], {"segments": segs, "crossfade": 0.0}, tmp_path / "end"))
    y, _ = read(res["outputs"]["out"]["path"])
    assert y.shape[-1] == int(5.5 * SR) and np.max(np.abs(y[:, -1])) < CLICK_BAR


def test_comp_overlaps_later_wins_and_errors(tmp_path):
    pa = write(tmp_path / "a.wav", np.full((1, 4 * SR), 0.1, dtype=np.float32))
    pb = write(tmp_path / "b.wav", np.full((1, 4 * SR), 0.2, dtype=np.float32))
    segs = [{"start": 0.0, "end": 3.0, "input": 0}, {"start": 2.0, "end": 4.0, "input": 1}]
    res = run(job_for("comp", [(pa, 0.0), (pb, 0.0)], {"segments": segs, "crossfade": 0.0}, tmp_path / "o"))
    assert [(s["start"], s["end"], s["input"]) for s in res["result"]["segments"]] == [(0.0, 2.0, 0), (2.0, 4.0, 1)]
    for bad, reason in [([{"start": 0, "end": 1, "input": 5}], "comp.segments"),
                        ([{"start": 1, "end": 0, "input": 0}], "comp.segments"), ([], "comp.segments")]:
        with pytest.raises(DspError) as e:
            run(job_for("comp", [(pa, 0.0), (pb, 0.0)], {"segments": bad, "crossfade": 0}, tmp_path / "o2"))
        assert e.value.params["reason"] == reason
    with pytest.raises(DspError) as e:
        run(job_for("comp", [(pa, 0.0)], {"segments": segs, "crossfade": 0}, tmp_path / "o3"))
    assert e.value.params["reason"] == "inputs.count"


# ----------------------------------------------------------------------------- concat

def test_concat_gap_crossfade_and_channels(tmp_path):
    a = sine(2.0, 440.0, 0.3, channels=1)
    b = sine(3.0, 550.0, 0.3, channels=2)
    pa, pb = write(tmp_path / "a.wav", a), write(tmp_path / "b.wav", b)
    res = run(job_for("concat", [(pa, 4.0), (pb, 9.0)], {"gap": 0.5, "crossfade": 0}, tmp_path / "o"))
    out = res["outputs"]["out"]
    y, _ = read(out["path"])
    assert out["offset"] == 0.0 and y.shape == (2, int(5.5 * SR))
    assert res["result"]["starts"] == [0.0, 2.5]
    assert np.max(np.abs(y[:, int(2.01 * SR): int(2.49 * SR)])) == 0.0
    assert np.array_equal(y[0, int(0.1 * SR): int(1.9 * SR)], a[0, int(0.1 * SR): int(1.9 * SR)])
    res = run(job_for("concat", [(pa, 0.0), (pb, 0.0)], {"gap": 0.5, "crossfade": 0.25}, tmp_path / "o2"))
    y, _ = read(res["outputs"]["out"]["path"])
    assert y.shape[-1] == int(4.75 * SR) and res["result"]["starts"] == [0.0, 1.75]
    k = int(1.875 * SR)                                               # mid cross-fade: cos/sin(π/4)
    exp = a[0, k] * math.cos(math.pi / 4) + b[0, k - int(1.75 * SR)] * math.sin(math.pi / 4)
    assert y[0, k] == pytest.approx(exp, abs=2e-3)


def test_concat_butt_splices_do_not_click(tmp_path):
    a = sine(1.0, 1000.0, 0.5)[:, : SR - 11]                          # ends mid-wave
    b = sine(1.0, 1000.0, 0.5, phase=1.3)
    pa, pb = write(tmp_path / "a.wav", a), write(tmp_path / "b.wav", b)
    res = run(job_for("concat", [(pa, 0.0), (pb, 0.0)], {"gap": 0, "crossfade": 0}, tmp_path / "o"))
    y, _ = read(res["outputs"]["out"]["path"])
    assert y.shape[-1] == a.shape[-1] + b.shape[-1]
    assert click_at(y, a.shape[-1]) < CLICK_BAR


# ----------------------------------------------------------------------------- pitch

def _peak_hz(x: np.ndarray) -> float:
    m = x.mean(axis=0) if x.ndim == 2 else x
    m = m[SR // 2: SR // 2 + 4 * SR]
    spec = np.abs(np.fft.rfft(m * np.hanning(m.size), n=1 << 20))
    k = int(np.argmax(spec))
    a, b, c = spec[k - 1], spec[k], spec[k + 1]
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * SR / (1 << 20)


@pytest.mark.parametrize("semis", [-12, -3, 3, 7, 12])
def test_builtin_pitch_keeps_duration_and_hits_the_pitch(semis):
    x = sine(6.0, 440.0, 0.4) + sine(6.0, 660.0, 0.1)
    y = vocoder.pitch_shift(x, semis, 1.0)
    assert y.shape == x.shape                                          # exact: well within 1 ms
    want = 440.0 * 2 ** (semis / 12)
    assert abs(1200 * math.log2(_peak_hz(y) / want)) < 3.0             # cents
    level = 20 * np.log10(np.sqrt(np.mean(y[:, SR: -SR] ** 2)) / np.sqrt(np.mean(x[:, SR: -SR] ** 2)))
    assert abs(level) < 1.0


@pytest.mark.parametrize("tempo", [0.5, 0.8, 1.25, 2.0])
def test_builtin_tempo_changes_length_not_pitch(tempo):
    x = sine(4.0, 500.0, 0.4, channels=1)
    y = vocoder.pitch_shift(x, 0.0, tempo)
    assert y.shape == (1, int(round(x.shape[-1] / tempo)))
    assert abs(1200 * math.log2(_peak_hz(y) / 500.0)) < 3.0


def _backing(seconds: float, alpha: float = 1.0, tonal: bool = True, drums: bool = True, seed: int = 0):
    """Chords + bass (all frequencies × alpha) and/or kick/snare/hats; returns (audio, hit samples)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    beat, bar = 0.6, 2.4
    prog = [[0, 4, 7], [9, 12, 16], [5, 9, 12], [7, 11, 14]]
    if tonal:
        for k in range(int(np.ceil(seconds / bar))):
            s, e = int(k * bar * SR), min(n, int((k + 1) * bar * SR))
            if s >= n:
                break
            tt = t[s:e] - t[s]
            env = np.minimum(1, tt / 0.01) * np.exp(-tt * 0.4)
            for vi, semi in enumerate(prog[k % 4]):
                f = 261.63 * 2 ** (semi / 12) * alpha
                sig = sum((0.5 ** h) * np.sin(2 * np.pi * f * (h + 1) * tt + vi) for h in range(4))
                pan = (vi - 1) * 0.5
                out[0, s:e] += 0.12 * env * sig * np.sqrt(0.5 * (1 - pan))
                out[1, s:e] += 0.12 * env * sig * np.sqrt(0.5 * (1 + pan))
            fb = 261.63 * 2 ** ((prog[k % 4][0] % 12 - 24) / 12) * alpha
            out[:, s:e] += 0.18 * env * sum((0.6 ** h) * np.sin(2 * np.pi * fb * (h + 1) * tt) for h in range(5))
    hits = []
    if drums:
        for k in range(int(seconds / beat)):
            s, L = int(k * beat * SR), int(0.15 * SR)
            if s + L > n:
                break
            tt = np.arange(L) / SR
            if k % 2 == 0:
                sig = np.sin(2 * np.pi * np.cumsum((50 + 70 * np.exp(-tt * 30)) * alpha) / SR) * np.exp(-tt * 18) * 0.6
            else:
                sig = (rng.standard_normal(L) * 0.25 + 0.2 * np.sin(2 * np.pi * 190 * alpha * tt)) * np.exp(-tt * 25)
            out[:, s:s + L] += sig
            hits.append(s)
    return out.astype(np.float32), hits


def _lsd(y: np.ndarray, ref: np.ndarray) -> float:
    """Log-spectral distance (dB) in 1/6-octave bands, 2048-sample frames."""
    def bands(x):
        m = x.mean(axis=0)
        fr = np.lib.stride_tricks.sliding_window_view(m, 2048)[::512]
        s = np.abs(np.fft.rfft(fr * np.hanning(2048), axis=-1)) ** 2
        f = np.fft.rfftfreq(2048, 1 / SR)
        edges = 60 * 2 ** (np.arange(0, 8.0, 1 / 6))
        return 10 * np.log10(np.stack([s[:, (f >= lo) & (f < hi)].sum(axis=1)
                                       for lo, hi in zip(edges[:-1], edges[1:])], axis=1) + 1e-10)
    a, b = bands(y), bands(ref)
    k = min(len(a), len(b))
    a, b = a[:k], b[:k]
    mask = b > b.max() - 60
    return float(np.mean(np.abs(a - b)[mask]))


def _pre_echo(y: np.ndarray, hits: list[int]) -> float:
    m = np.abs(y).max(axis=0)
    vals = []
    for h in hits[2:-2]:
        pre = m[h - int(0.03 * SR): h - int(0.004 * SR)]
        post = m[h: h + int(0.03 * SR)]
        rms_pre, rms_post = np.sqrt(np.mean(pre ** 2)), np.sqrt(np.mean(post ** 2))
        vals.append(20 * np.log10(rms_pre + 1e-9) - 20 * np.log10(rms_post + 1e-9))
    return float(np.mean(vals))


@pytest.mark.parametrize("semis", [-3, 3])
def test_builtin_pitch_quality_on_a_backing_track(semis):
    """±3 semitones on a backing track: partials land where an ideal shifter puts them (log-spectral
    distance to the analytically shifted rendering ≤ 1.2 dB; Signalsmith Stretch scores ~0.9–1.0 dB
    on this signal) and drum attacks are not smeared backwards (pre-echo ≤ -35 dB; Signalsmith
    ~-28 dB)."""
    alpha = 2 ** (semis / 12)
    tonal, _ = _backing(12.0, drums=False)
    truth, _ = _backing(12.0, alpha, drums=False)
    drums, hits = _backing(12.0, tonal=False)
    assert _lsd(tonal, truth) > 8.0                                   # the metric sees a wrong pitch
    assert _lsd(vocoder.pitch_shift(tonal, semis), truth) < 1.2
    assert _pre_echo(vocoder.pitch_shift(drums, semis), hits) < -35.0


def test_builtin_pitch_keeps_the_stereo_image():
    x = np.zeros((2, 6 * SR), dtype=np.float32)
    x[0] = sine(6.0, 330.0, 0.4, channels=1)[0]                       # hard left
    x[1] = 0.25 * sine(6.0, 495.0, 0.4, channels=1)[0]                # quieter, right only
    y = vocoder.pitch_shift(x, 3.0)
    yl = np.abs(np.fft.rfft(y[0, SR: 5 * SR]))
    f = np.fft.rfftfreq(4 * SR, 1 / SR)
    target_r = 495.0 * 2 ** 0.25
    left_leak = yl[np.abs(f - target_r) < 3].max() / yl.max()
    assert 20 * np.log10(left_leak + 1e-12) < -40.0                   # the right source stays right
    assert np.sqrt(np.mean(y[1] ** 2)) / np.sqrt(np.mean(y[0] ** 2)) == pytest.approx(0.25, rel=0.1)


def _vowel(seconds: float, f0: float) -> np.ndarray:
    """Harmonic source shaped by fixed formants at 700 / 1220 / 2600 Hz."""
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    for h in range(1, int(8000 / f0)):
        f = h * f0
        formants = ((700, 110, 1.0), (1220, 120, 0.6), (2600, 160, 0.3))
        gain = sum(a / (1 + ((f - fc) / bw) ** 2) for fc, bw, a in formants)
        out += gain * np.sin(2 * np.pi * f * t + h)
    return (0.3 * out / np.max(np.abs(out)))[None].astype(np.float32)


def _first_formant(y: np.ndarray) -> float:
    m = y[0, SR // 2: SR // 2 + 2 * SR]
    spec = np.abs(np.fft.rfft(m * np.hanning(m.size)))
    f = np.fft.rfftfreq(m.size, 1 / SR)
    band = (f > 300) & (f < 1000)
    env = np.convolve(spec, np.ones(301) / 301, mode="same")          # ~150 Hz smoothing
    return float(f[band][np.argmax(env[band])])


def test_builtin_formant_preservation():
    x = _vowel(3.0, 150.0)
    keep = vocoder.pitch_shift(x, 5.0, preserve_formant=True)
    move = vocoder.pitch_shift(x, 5.0, preserve_formant=False)
    assert _first_formant(keep) == pytest.approx(700.0, rel=0.12)
    assert _first_formant(move) == pytest.approx(700.0 * 2 ** (5 / 12), rel=0.12)


def test_builtin_pitch_four_minute_song_under_20_seconds(song4):
    x, _ = read(song4)
    start = time.perf_counter()
    y = vocoder.pitch_shift(x, 3.0, quality="high")
    elapsed = time.perf_counter() - start
    assert y.shape == x.shape
    assert elapsed < 20.0, f"built-in pitch of 4 min took {elapsed:.1f} s"


def test_choose_engine_preference_and_formants():
    assert pitch_mod.choose_engine(None, False, ["rubberband", "signalsmith", "builtin"]) == "rubberband"
    assert pitch_mod.choose_engine(None, False, ["signalsmith", "builtin"]) == "signalsmith"
    assert pitch_mod.choose_engine(None, True, ["signalsmith", "builtin"]) == "builtin"
    assert pitch_mod.choose_engine(None, True, ["rubberband", "signalsmith", "builtin"]) == "rubberband"
    assert pitch_mod.choose_engine("builtin", False, ["rubberband", "signalsmith", "builtin"]) == "builtin"
    assert pitch_mod.choose_engine("rubberband", False, ["builtin"]) == "builtin"


def test_pitch_op_offsets_engines_and_identity(tmp_path):
    src = write(tmp_path / "in.wav", sine(3.0, 440.0, 0.4))
    rec = Recorder()
    res = run(job_for("pitch", [(src, 10.0)], {"semitones": 2, "cents": 50, "tempo": 1.25, "preserveFormant": False,
                                               "quality": "high", "engine": "builtin"}, tmp_path / "o"), rec)
    out = res["outputs"]["out"]
    assert out["offset"] == pytest.approx(10.0 / 1.25)
    assert out["duration"] == pytest.approx(round(3.0 * SR / 1.25) / SR, abs=1e-6)
    assert res["result"] == {"engine": "builtin", "formantPreserved": False, "semitones": 2.5, "tempo": 1.25}
    assert "dsp.pitch" in rec.stages()
    res = run(job_for("pitch", [(src, 4.0)], {"semitones": 0, "cents": 0, "tempo": 1.0}, tmp_path / "o2"))
    assert res["result"]["engine"] == "none" and res["outputs"]["out"]["offset"] == 4.0
    y, _ = read(res["outputs"]["out"]["path"])
    x, _ = read(src)
    assert np.array_equal(x, y)
    res = run(job_for("pitch", [(src, 0.0)], {"semitones": 3, "preserveFormant": True}, tmp_path / "o3"))
    assert res["result"]["engine"] in ("rubberband", "builtin") and res["result"]["formantPreserved"] is True
    for bad in ({"semitones": 30}, {"tempo": 3.0}, {"cents": 150}, {"semitones": 24, "cents": 50}):
        with pytest.raises(DspError) as e:
            run(job_for("pitch", [(src, 0.0)], bad, tmp_path / "o4"))
        assert e.value.params["reason"] == "pitch.range"


@pytest.mark.parametrize("engine", ["signalsmith", "rubberband"])
def test_optional_pitch_engines(tmp_path, engine):
    available = pitch_mod.available_engines()
    if engine not in available:
        pytest.skip(f"{engine} is not installed")
    x = sine(4.0, 440.0, 0.4)
    for semis, tempo in ((12, 1.0), (-12, 1.0), (3, 1.25)):
        y, used, _ = pitch_mod.shift(x, semis, tempo, engine=engine)
        assert used == engine and y.shape == (2, int(round(x.shape[-1] / tempo)))
        assert abs(1200 * math.log2(_peak_hz(y) / (440 * 2 ** (semis / 12)))) < 5.0


def test_pitch_engine_failure_falls_back(monkeypatch):
    monkeypatch.setattr(pitch_mod, "available_engines", lambda: ["signalsmith", "builtin"])
    monkeypatch.setattr(pitch_mod, "_signalsmith", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    logs: list[str] = []
    y, used, _ = pitch_mod.shift(sine(1.0, 440.0), 2.0, 1.0, log=logs.append)
    assert used == "builtin" and y.shape == (2, SR) and any("boom" in m for m in logs)


# ----------------------------------------------------------------------------- fx

@pytest.fixture
def fake_fx(monkeypatch):
    """Stand-in for ``stellaria_backend.fx``: each slot gains, a 'reverb' adds a 1 s tail, a mono
    input comes back stereo (as some plugins do)."""
    calls: list[dict] = []
    module = types.ModuleType("stellaria_backend.fx")

    def apply_chain(x, sr, chain, *, bpm, vst_plugins):
        assert x.dtype == np.float32 and x.ndim == 2 and sr == SR and chain
        calls.append({"slots": [s["id"] for s in chain], "bpm": bpm, "plugins": vst_plugins, "channels": x.shape[0]})
        y = np.repeat(x, 2, axis=0) if x.shape[0] == 1 else x.copy()
        for slot in chain:
            y = y * np.float32(10 ** (float(slot.get("params", {}).get("gain", 0.0)) / 20))
            if slot.get("effect") == "reverb":
                tail = np.exp(-np.arange(SR) / (0.08 * SR)).astype(np.float32) * 0.2
                y = np.concatenate([y, np.repeat(tail[None], y.shape[0], axis=0)], axis=1)
        return y

    module.apply_chain = apply_chain
    module.effect_specs = lambda: []
    monkeypatch.setitem(sys.modules, "stellaria_backend.fx", module)
    import stellaria_backend

    monkeypatch.setattr(stellaria_backend, "fx", module, raising=False)
    return calls


def test_fx_runs_slots_reports_plugins_and_trims_tail(tmp_path, fake_fx):
    src = write(tmp_path / "in.wav", sine(2.0, 300.0, 0.3, channels=1))
    chain = [{"id": "a", "kind": "builtin", "effect": "eq", "bypass": False, "mix": 1.0, "params": {"gain": -6.0}},
             {"id": "b", "kind": "builtin", "effect": "gate", "bypass": True, "mix": 1.0, "params": {}},
             {"id": "c", "kind": "vst", "pluginId": "p1", "state": None, "bypass": False, "mix": 0.5, "params": {}},
             {"id": "d", "kind": "builtin", "effect": "reverb", "bypass": False, "mix": 0.3, "params": {}}]
    plugins = {"p1": {"id": "p1", "name": "Pro-Q 3", "path": "/x.vst3", "pluginName": None, "category": "effect"}}
    rec = Recorder()
    res = run(job_for("fx", [(src, 7.5)], {"chain": chain}, tmp_path / "o", bpm=96.0, vstPlugins=plugins), rec)
    assert [c["slots"] for c in fake_fx] == [["a"], ["c"], ["d"]]       # bypassed slot skipped; VST alone
    assert all(c["bpm"] == 96.0 and c["plugins"] == plugins for c in fake_fx)
    vst_lines = [m for m in rec.lines if m.get("stage") == "dsp.fx.vst"]
    assert vst_lines == [{"progress": pytest.approx(vst_lines[0]["progress"]), "stage": "dsp.fx.vst", "slot": 2,
                          "plugin": "Pro-Q 3"}]
    out = res["outputs"]["out"]
    y, _ = read(out["path"])
    assert out["offset"] == 7.5 and y.shape[0] == 2
    tail = y.shape[-1] - 2 * SR
    assert 0.3 * SR < tail < 1.0 * SR + 1                              # trimmed where it falls under -90 dBFS
    assert np.max(np.abs(y[:, -100:])) < 10 ** (-80 / 20)
    assert res["result"] == {"slots": 3, "tail": pytest.approx(tail / SR, abs=1e-6)}


def test_fx_groups_builtin_runs_and_isolates_vst_slots(tmp_path, fake_fx):
    src = write(tmp_path / "in.wav", sine(1.0, 300.0, 0.3))
    slot = lambda i, kind="builtin": {"id": i, "kind": kind, "effect": "eq", "pluginId": "p", "bypass": False,  # noqa: E731
                                      "mix": 1.0, "params": {}}
    chain = [slot("a"), slot("b"), slot("v", "vst"), slot("c"), slot("w", "vst"), slot("x", "vst"), slot("d")]
    rec = Recorder()
    run(job_for("fx", [(src, 0.0)], {"chain": chain}, tmp_path / "o", vstPlugins={"p": {"name": "Comp"}}), rec)
    assert [c["slots"] for c in fake_fx] == [["a", "b"], ["v"], ["c"], ["w"], ["x"], ["d"]]
    fx_lines = [m for m in rec.lines if m.get("stage") in ("dsp.fx", "dsp.fx.vst") and "slot" in m]
    assert [(m["stage"], m["slot"]) for m in fx_lines] == [("dsp.fx", 0), ("dsp.fx.vst", 2), ("dsp.fx", 3),
                                                          ("dsp.fx.vst", 4), ("dsp.fx.vst", 5), ("dsp.fx", 6)]


def test_fx_tail_is_capped_at_six_seconds():
    from stellaria_backend.dsp.fxop import trim_tail

    y = np.full((2, 20 * SR), 0.1, dtype=np.float32)
    out = trim_tail(y, 5 * SR)
    assert out.shape[-1] == 11 * SR and abs(out[0, -1]) < 1e-3          # faded at the cap
    assert trim_tail(np.ones((1, 10), np.float32), 20).shape == (1, 20)  # shorter → padded


def test_fx_six_slot_chain_four_minutes(song4, tmp_path, monkeypatch):
    """A 6-slot chain on a 4-minute song. The op's own share (read, write, tail, checks — the time
    outside ``apply_chain``) stays under 1.5 s; with the stand-in effects the whole op is under the
    10 s bar. The built-in effects' speed is the fx package's bar (``test_fx.py`` measures the
    same 4-minute 6-slot case)."""
    import stellaria_backend.dsp.fxop  # noqa: F401 - imported before the effects are wrapped

    effects = ["eq", "compressor", "deesser", "saturation", "reverb", "limiter"]
    try:
        import stellaria_backend.fx as fx_pkg
        from stellaria_backend.fx.chain import apply_chain as inner   # the real one, not the lazy wrapper

        real = set(effects) <= {s["id"] for s in fx_pkg.effect_specs()}
    except ImportError:
        real = False
    if not real:
        effects = ["gain"] * 6
        inner = lambda x, sr, chain, *, bpm, vst_plugins: x * np.float32(0.9)  # noqa: E731
    spent = [0.0]

    def timed(x, sr, chain, *, bpm, vst_plugins):
        start = time.perf_counter()
        try:
            return inner(x, sr, chain, bpm=bpm, vst_plugins=vst_plugins)
        finally:
            spent[0] += time.perf_counter() - start

    module = types.ModuleType("stellaria_backend.fx")
    module.apply_chain = timed
    monkeypatch.setitem(sys.modules, "stellaria_backend.fx", module)
    chain = [{"id": str(i), "kind": "builtin", "effect": e, "bypass": False, "mix": 0.5 if e == "saturation" else 1.0,
              "params": {}} for i, e in enumerate(effects)]
    start = time.perf_counter()
    res = run(job_for("fx", [(song4, 0.0)], {"chain": chain}, tmp_path / "o", bpm=96.0))
    total = time.perf_counter() - start
    assert res["result"]["slots"] == 6
    assert total - spent[0] < 1.5, f"fx op overhead {total - spent[0]:.2f} s"
    if not real:
        assert total < 10.0
    print(f"6-slot chain on 4 min: {total:.1f} s total, {spent[0]:.1f} s in apply_chain (real effects: {real})")


def test_fx_cli_names_the_plugin_before_it_runs(tmp_path):
    """A plugin that kills the process: the last stdout line names it (→ vst.crashed {name})."""
    src = write(tmp_path / "in.wav", sine(1.0))
    prelude = ("import sys, types, os\n"
               "m = types.ModuleType('stellaria_backend.fx')\n"
               "def apply_chain(x, sr, chain, *, bpm, vst_plugins):\n"
               "    if chain[0].get('kind') == 'vst': os._exit(139)\n"
               "    return x\n"
               "m.apply_chain = apply_chain\n"
               "import stellaria_backend\n"
               "sys.modules['stellaria_backend.fx'] = m\n")
    chain = [{"id": "a", "kind": "builtin", "effect": "eq", "bypass": False, "mix": 1, "params": {}},
             {"id": "b", "kind": "vst", "pluginId": "v", "state": None, "bypass": False, "mix": 1, "params": {}}]
    job = job_for("fx", [(src, 0.0)], {"chain": chain}, tmp_path / "o",
                  vstPlugins={"v": {"id": "v", "name": "Valhalla VintageVerb", "path": "/v.vst3"}})
    code, lines, _ = run_cli(job, tmp_path, prelude=prelude)
    assert code == 139
    assert lines[-1]["stage"] == "dsp.fx.vst" and lines[-1]["plugin"] == "Valhalla VintageVerb"


# ----------------------------------------------------------------------------- export

def _export_job(src: str, folder: Path, offset: float = 0.0, names: dict | None = None, **params) -> dict:
    p = {"format": "wav", "sampleRate": 44100, "bitDepth": 24, "mp3Bitrate": 320, "loudness": None,
         "folder": str(folder), "fileName": "{project} - {label}", "keepPosition": False, "tags": None}
    p.update(params)
    return {"op": "export", "inputs": [{"path": src, "offset": offset}], "params": p, "outputDir": None,
            "names": names if names is not None else {"project": "星籁 Demo", "label": "Mix"}}


def _music(seconds: float = 8.0) -> np.ndarray:
    x, _ = _backing(seconds)
    return (x * 0.8).astype(np.float32)


def _decode(path: str) -> tuple[np.ndarray, int]:
    if path.endswith(".mp3"):
        from stellaria_backend.audio.io import ffmpeg_exe

        info = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True).stderr.decode()
        sr = 48000 if "48000 Hz" in info else 44100
        raw = subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "2",
                              "pipe:1"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, np.float32).reshape(-1, 2).T, sr
    return read(path)


@pytest.mark.parametrize("fmt,rate,bits", [("wav", 44100, 16), ("wav", 48000, 24), ("wav", 44100, 32),
                                           ("flac", 48000, 16), ("flac", 44100, 24), ("mp3", 44100, 16),
                                           ("mp3", 48000, 16)])
def test_export_round_trip(tmp_path, fmt, rate, bits):
    x = _music()
    src = write(tmp_path / "src.wav", x)
    res = run(_export_job(src, tmp_path, fmt=fmt, sampleRate=rate, bitDepth=bits, loudness=-16.0, format=fmt))
    r = res["result"]
    assert res["outputs"] == {}
    assert Path(r["path"]).name == f"星籁 Demo - Mix.{fmt}" and Path(r["path"]).is_file()
    assert r["format"] == fmt and r["sampleRate"] == rate
    y, sr = _decode(r["path"])
    assert sr == rate
    assert abs(y.shape[-1] - x.shape[-1] * rate / SR) <= 2                 # exact length (mp3: gapless)
    if fmt != "mp3":
        info = sf.info(r["path"])
        assert info.subtype == {16: "PCM_16", 24: "PCM_24", 32: "FLOAT" if fmt == "wav" else "PCM_24"}[bits]
    from stellaria_backend.mixing import loudness as ld

    assert ld.integrated_loudness(y, sr) == pytest.approx(-16.0, abs=0.5)
    assert r["lufs"] == pytest.approx(-16.0, abs=0.3)
    assert r["truePeakDb"] <= -0.95
    assert not list(tmp_path.glob("*.part"))


def test_export_loudness_with_true_peak_ceiling(tmp_path):
    x = _music(10.0)
    src = write(tmp_path / "src.wav", x)
    res = run(_export_job(src, tmp_path, loudness=-9.0, bitDepth=32))  # loud target: needs limiting
    r = res["result"]
    y, sr = read(r["path"])
    from stellaria_backend.dsp.export import true_peak_db
    from stellaria_backend.mixing import loudness as ld

    assert r["limited"] is True
    assert ld.integrated_loudness(y, sr) == pytest.approx(-9.0, abs=0.5)
    assert true_peak_db(y) <= -1.0 + 0.05


def test_export_keep_position_resample_and_no_overwrite(tmp_path):
    x = sine(2.0, 440.0, 0.3)
    src = write(tmp_path / "src.wav", x)
    first = run(_export_job(src, tmp_path, offset=1.5, keepPosition=True, sampleRate=48000))["result"]
    y, sr = read(first["path"])
    assert sr == 48000 and abs(y.shape[-1] - 3.5 * 48000) <= 2
    assert np.max(np.abs(y[:, : int(1.49 * 48000)])) < 1e-4
    assert first["duration"] == pytest.approx(3.5, abs=1e-4)
    before = Path(first["path"]).read_bytes()
    second = run(_export_job(src, tmp_path, offset=1.5))["result"]
    third = run(_export_job(src, tmp_path, offset=1.5))["result"]
    assert Path(second["path"]).name == "星籁 Demo - Mix (2).wav"
    assert Path(third["path"]).name == "星籁 Demo - Mix (3).wav"
    assert Path(first["path"]).read_bytes() == before                 # never overwritten


def test_export_16_bit_is_tpdf_dithered(tmp_path):
    x = np.full((1, SR), 0.3 / 32768, dtype=np.float32)                # 0.3 LSB of DC
    src = write(tmp_path / "src.wav", x)
    r = run(_export_job(src, tmp_path, bitDepth=16))["result"]
    q, _ = sf.read(r["path"], dtype="int16")
    assert q.std() > 0.3 and abs(q.mean() - 0.3) < 0.05                 # noise around the true level
    q = encode.quantize(np.zeros((1, 1000)), 24, dither=False)
    assert q.dtype == np.int32 and not q.any()


def test_export_tags(tmp_path):
    src = write(tmp_path / "src.wav", sine(1.0, 440.0, 0.3))
    tags = {"title": "夜空中最亮的星", "artist": "Stellaria 歌手"}
    for fmt in ("flac", "wav", "mp3"):
        r = run(_export_job(src, tmp_path, format=fmt, tags=tags))["result"]
        assert r["tagged"] is True
        if fmt in ("flac", "wav"):
            with sf.SoundFile(r["path"]) as f:
                assert f.title == tags["title"] and f.artist == tags["artist"]
        else:
            head = Path(r["path"]).read_bytes()[:4096]
            assert head.startswith(b"ID3")
            assert tags["title"].encode("utf-16-le") in head and tags["artist"].encode("utf-16-le") in head


def test_id3_writer_round_trips_with_mutagen():
    pytest.importorskip("mutagen")
    from io import BytesIO

    from mutagen.id3 import ID3

    tag = encode.id3v2_tag("标题 Title", "歌手")
    tags = ID3(BytesIO(tag + b"\x00" * 16))
    assert str(tags["TIT2"]) == "标题 Title" and str(tags["TPE1"]) == "歌手" and tags.version == (2, 3, 0)


def test_lame_info_frame_fields():
    header = bytes([0xFF, 0xFB, 0xE0, 0x44])                           # MPEG-1 L3, 320 kbps, 44.1 kHz, joint
    frame = encode.lame_info_frame(header, frames=100, stream_bytes=104400, samples=113000, sr=44100, bitrate=320)
    assert len(frame) == 1044 and frame[:2] == b"\xff\xfb"
    pos = 4 + 32
    assert frame[pos: pos + 4] == b"Info" and struct.unpack(">I", frame[pos + 8: pos + 12])[0] == 100
    lame = pos + 120
    assert frame[lame: lame + 9] == b"LAME3.100"
    d = frame[lame + 21: lame + 24]
    assert (d[0] << 4) | (d[1] >> 4) == 576 and ((d[1] & 0x0F) << 8) | d[2] == 100 * 1152 - 576 - 113000
    assert struct.unpack(">H", frame[lame + 34: lame + 36])[0] == encode._crc16(frame[:190])
    assert encode._crc16(b"123456789") == 0xBB3D                        # CRC-16/ARC check value


def test_mp3_with_lameenc_is_gapless(tmp_path):
    pytest.importorskip("lameenc")
    x = np.zeros((2, 3 * SR + 77), dtype=np.float32)
    x[:, SR + 5] = 0.9
    path = tmp_path / "g.mp3"
    assert encode.write_mp3(path, x, SR, 320, title=None, artist=None) == "lameenc"
    y, _ = _decode(str(path))
    assert y.shape[-1] == x.shape[-1] and abs(int(np.argmax(np.abs(y[0]))) - (SR + 5)) <= 1


def test_mp3_falls_back_to_ffmpeg_and_reports_missing_encoder(tmp_path, monkeypatch):
    x = sine(1.0, 440.0, 0.3)
    monkeypatch.setattr(encode, "lameenc_available", lambda: False)
    if encode.ffmpeg_candidates():
        assert encode.write_mp3(tmp_path / "f.mp3", x, SR, 192, title="t", artist=None) == "ffmpeg"
        y, _ = _decode(str(tmp_path / "f.mp3"))
        assert abs(y.shape[-1] - SR) <= 1
    monkeypatch.setattr(encode, "ffmpeg_candidates", lambda: [])
    src = write(tmp_path / "src.wav", x)
    with pytest.raises(DspError) as e:
        run(_export_job(src, tmp_path, format="mp3"))
    assert e.value.code == "export.encoder_missing" and e.value.params == {"format": "mp3"}
    assert not list(tmp_path.glob("*.mp3*")) or {p.name for p in tmp_path.glob("*.mp3*")} == {"f.mp3"}


def test_export_folders(tmp_path):
    src = write(tmp_path / "src.wav", sine(0.5))
    with pytest.raises(DspError) as e:
        run(_export_job(src, tmp_path / "missing"))
    assert e.value.code == "export.folder_missing" and e.value.params == {"folder": str(tmp_path / "missing")}
    job = _export_job(src, tmp_path)
    job["params"]["folder"] = None
    job["defaultFolder"] = str(tmp_path / "lib" / "exports")
    r = run(job)["result"]
    assert Path(r["path"]).parent == tmp_path / "lib" / "exports"
    job.pop("defaultFolder")
    with pytest.raises(DspError) as e:
        run(job)
    assert e.value.code == "dsp.bad_job"
    with pytest.raises(DspError) as e:
        run(_export_job(src, tmp_path, format="ogg"))
    assert e.value.params["reason"] == "export.format"
    with pytest.raises(DspError) as e:
        run(_export_job(src, tmp_path, loudness=3))
    assert e.value.params["reason"] == "export.loudness"


# ----------------------------------------------------------------------------- names

def test_fill_template_tokens():
    import datetime as dt

    now = dt.datetime(2026, 10, 2, 9, 30)
    names = {"project": "翻唱计划", "song": "晴天", "voice": "星籁A", "label": "副歌", "index": 7}
    assert naming.fill("{project} - {song} ({voice}) {label} {date} #{index:03}", names, now=now) == \
        "翻唱计划 - 晴天 (星籁A) 副歌 2026-10-02 #007"
    assert naming.fill("{song} - {voice} - {label}", {"song": "晴天", "voice": "", "label": None}) == "晴天"
    assert naming.fill("{voice} - {song}", {"song": "晴天"}) == "晴天"
    assert naming.fill("{index}", {}) == "1"
    assert naming.fill("{song} {unknown}", {"song": "a"}) == "a {unknown}"


@pytest.mark.parametrize("raw,expected", [
    ("CON", "_CON"), ("con.backup", "_con.backup"), ("COM1", "_COM1"), ("lpt9 ", "_lpt9"), ("NUL.", "_NUL"),
    ('a<b>c:d"e/f\\g|h?i*j', "a_b_c_d_e_f_g_h_i_j"), ("song. . ", "song"), ("...hidden", "hidden"),
    ("tab\there\x07", "tab here_"), ("", "export"), ("  ", "export"), ("???", "export"),
    ("zero​width‮", "zerowidth"), ("é", "é"), ("星籁 · 晴天", "星籁 · 晴天"),
])
def test_sanitize(raw, expected):
    assert naming.sanitize(raw) == expected


def test_sanitize_length_is_counted_in_utf8_bytes():
    s = naming.sanitize("星" * 200)
    assert len(s.encode("utf-8")) <= naming.MAX_NAME_BYTES and s == "星" * (naming.MAX_NAME_BYTES // 3)
    assert len(naming.sanitize("a" * 300)) == naming.MAX_NAME_BYTES


def test_claim_never_overwrites(tmp_path):
    (tmp_path / "x.wav").write_bytes(b"keep")
    p2 = naming.claim(tmp_path, "x", "wav")
    p3 = naming.claim(tmp_path, "x", "wav")
    assert (p2.name, p3.name) == ("x (2).wav", "x (3).wav")
    assert (tmp_path / "x.wav").read_bytes() == b"keep"


def test_file_stem_falls_back_to_names():
    assert naming.file_stem("{voice}", {"label": "Chorus"}) == "Chorus"
    assert naming.file_stem("", {"project": "P", "label": "L"}) == "P - L"
    assert naming.file_stem("{song}", {}) == "export"


# ----------------------------------------------------------------------------- CLI protocol

def test_cli_protocol(tmp_path):
    src = write(tmp_path / "in 输入.wav", sine(3.0))
    job = job_for("split", [(src, 1.0)], {"cuts": [1.0, 2.0], "fade": 0.01, "labels": []}, tmp_path / "out 输出")
    code, lines, stderr = run_cli(job, tmp_path)
    assert code == 0, stderr
    check_protocol(lines)
    assert set(lines[-1]) == {"result"}
    result = lines[-1]["result"]
    assert set(result) == {"outputs", "result"} and list(result["outputs"]) == ["part1", "part2", "part3"]
    for out in result["outputs"].values():
        assert set(out) == {"path", "offset", "duration"} and Path(out["path"]).is_file()
    assert {"dsp.read", "dsp.split", "dsp.write"} <= {m["stage"] for m in lines if "stage" in m}
    assert any("log" in m for m in lines)
    assert not list((tmp_path / "out 输出").glob("*.part*"))


@pytest.mark.parametrize("case,code", [
    ("missing_input", "dsp.input_missing"), ("bad_json", "dsp.bad_job"), ("unknown_op", "dsp.bad_job"),
    ("no_output_dir", "dsp.bad_job"), ("input_count", "dsp.bad_params"), ("bad_params", "dsp.bad_params"),
    ("not_audio", "audio.decode_failed"), ("folder_missing", "export.folder_missing"),
])
def test_cli_errors(tmp_path, case, code):
    src = write(tmp_path / "in.wav", sine(1.0))
    job = job_for("trim", [(src, 0.0)], {"start": 0.1, "end": 0.5}, tmp_path / "o")
    if case == "missing_input":
        job["inputs"][0]["path"] = str(tmp_path / "nope.wav")
    elif case == "unknown_op":
        job["op"] = "reverse"
    elif case == "no_output_dir":
        job.pop("outputDir")
    elif case == "input_count":
        job["inputs"] *= 2
    elif case == "bad_params":
        job["params"] = {"start": 0.5, "end": 0.1}
    elif case == "not_audio":
        bogus = tmp_path / "bogus.wav"
        bogus.write_bytes(b"not audio at all" * 10)
        job["inputs"][0]["path"] = str(bogus)
    elif case == "folder_missing":
        job = _export_job(src, tmp_path / "gone")
    if case == "bad_json":
        (tmp_path / "job.json").write_text("{oops", encoding="utf-8")
        proc = subprocess.run([sys.executable, "-m", "stellaria_backend.dsp.run", str(tmp_path / "job.json")],
                              cwd=str(BACKEND), capture_output=True, timeout=60)
        exit_code = proc.returncode
        lines = [json.loads(x) for x in proc.stdout.decode().splitlines() if x.strip()]
    else:
        exit_code, lines, _ = run_cli(job, tmp_path)
    assert exit_code == 1
    check_protocol(lines)
    assert set(lines[-1]) == {"error"}
    err = lines[-1]["error"]
    assert err["code"] == code and err["message"] and isinstance(err["params"], dict)
    if case == "missing_input":
        assert err["params"] == {"path": str(tmp_path / "nope.wav")}
    if case == "input_count":
        assert err["params"]["reason"] == "inputs.count"
    if case == "folder_missing":
        assert err["params"] == {"folder": str(tmp_path / "gone")}
    assert not (tmp_path / "o").exists() or not list((tmp_path / "o").iterdir())


# ----------------------------------------------------------------------------- timing bars (4-minute song)

@pytest.mark.parametrize("op", ["trim", "split", "comp"])
def test_edit_ops_four_minutes_under_a_second(song4, tmp_path, op):
    if op == "trim":
        job = job_for("trim", [(song4, 0.0)], {"start": 10.0, "end": 230.0, "fadeIn": 0.5, "fadeOut": 2.0}, tmp_path)
    elif op == "split":
        job = job_for("split", [(song4, 0.0)], {"cuts": [30.0, 75.5, 120.0, 180.25], "fade": 0.02, "labels": []},
                      tmp_path)
    else:
        segs = [{"start": t, "end": t + 30.0, "input": (k % 2)} for k, t in enumerate(np.arange(0.0, 240.0, 30.0))]
        job = job_for("comp", [(song4, 0.0), (song4, 0.0)], {"segments": segs, "crossfade": 0.05}, tmp_path)
    run(job_for("trim", [(song4, 0.0)], {"start": 0.0, "end": 1.0}, tmp_path / "warm"))   # imports, file cache
    times = []
    for _ in range(2):                    # best of two: a cold page cache / disk write-back is not the op
        start = time.perf_counter()
        res = run(job)
        times.append(time.perf_counter() - start)
    assert res["outputs"]
    assert min(times) < 1.0, f"{op} of a 4-minute song took {min(times):.2f} s"


# ----------------------------------------------------------------------------- analysis

NOTE = {"C": 0, "C#": 1, "Db": 1, "D": 2, "Eb": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "Ab": 8, "A": 9, "Bb": 10, "B": 11}


def _chord(name: str) -> tuple[int, list[int]]:
    minor = name.endswith("m")
    root = NOTE[name[:-1] if minor else name]
    return root, [root, root + (3 if minor else 4), root + 7]


def _tone(f: float, n: int, harmonics: int = 4, decay: float = 0.0, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    s = sum((0.55 ** h) * np.sin(2 * np.pi * f * (h + 1) * t + phase * (h + 1)) for h in range(harmonics))
    return s * np.minimum(1.0, t / 0.01) * np.exp(-decay * t) * np.minimum(1.0, (n - np.arange(n)) / (0.01 * SR))


SONG_PLAN = [("intro", 4, ["C", "G", "Am", "F"]), ("verse", 8, ["Am", "Em", "F", "C"]),
             ("chorus", 8, ["F", "G", "C", "Am"]), ("verse", 8, ["Am", "Em", "F", "C"]),
             ("chorus", 8, ["F", "G", "C", "Am"]), ("bridge", 4, ["Dm", "G", "Em", "A"]),
             ("chorus", 8, ["F", "G", "C", "Am"]), ("outro", 4, ["C", "G", "Am", "F"])]


def make_song(bpm: float = 128.0, plan=SONG_PLAN, seed: int = 0, detune: float = 0.0, drums_everywhere: bool = False):
    """Pads, bass, drums (verse/chorus/bridge), a crash on chorus downbeats, and a vocal stem in
    2-bar phrases. Returns (mix (2, n), vocals (n,), truth dict)."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = 4 * beat
    bars = sum(b for _, b, _ in plan)
    n = int(bars * bar * SR) + SR
    mix = np.zeros((2, n))
    vocals = np.zeros(n)
    truth: dict = {"beats": [], "downbeats": [], "sections": [], "phrases": []}
    b0 = 0
    for label, nb, prog in plan:
        truth["sections"].append((b0 * bar, (b0 + nb) * bar, label))
        level = {"intro": 0.35, "outro": 0.3, "verse": 0.6, "chorus": 1.0, "bridge": 0.5}[label]
        drums = drums_everywhere or label in ("verse", "chorus", "bridge")
        for k in range(nb):
            bstart = (b0 + k) * bar
            s, L = int(bstart * SR), int(bar * SR)
            root, notes = _chord(prog[k % len(prog)])
            for i, semi in enumerate(notes):
                sig = _tone(261.63 * 2 ** ((semi + detune) / 12), L, decay=0.3, phase=i)
                pan = (i - 1) * 0.4
                mix[0, s:s + L] += 0.07 * level * sig * (1 - pan)
                mix[1, s:s + L] += 0.07 * level * sig * (1 + pan)
            fb = 261.63 * 2 ** ((root % 12 - 24 + detune) / 12)
            for q in range(4):
                bs, bl = s + int(q * beat * SR), int(beat * SR * 0.9)
                mix[:, bs:bs + bl] += 0.12 * level * _tone(fb, bl, harmonics=5, decay=3.0)
            for q in range(4):
                t = bstart + q * beat
                truth["beats"].append(t)
                if q == 0:
                    truth["downbeats"].append(t)
                if not drums:
                    continue
                ps, ld = int(t * SR), int(0.18 * SR)
                tt = np.arange(ld) / SR
                if q in (0, 2):
                    f = 50 + 80 * np.exp(-tt * 35)
                    hit = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt * 14) * (0.9 if q == 0 else 0.7)
                else:
                    hit = (rng.standard_normal(ld) * 0.3 + 0.25 * np.sin(2 * np.pi * 185 * tt)) * np.exp(-tt * 22)
                mix[:, ps:ps + ld] += level * hit
                for off in (0.0, beat / 2):
                    hs, hl = int((t + off) * SR), int(0.04 * SR)
                    hat = np.diff(np.concatenate([[0], rng.standard_normal(hl)]))
                    hat *= np.exp(-np.arange(hl) / SR * 90) * 0.05
                    mix[:, hs:hs + hl] += level * hat
            if k == 0 and label == "chorus":
                lc = int(1.2 * SR)
                crash = np.diff(np.concatenate([[0], rng.standard_normal(lc)])) * np.exp(-np.arange(lc) / SR * 3)
                mix[:, s:s + lc] += 0.08 * crash
        if label in ("verse", "chorus", "bridge"):
            for ph in range(0, nb, 2):
                a, e = (b0 + ph) * bar + 0.1, (b0 + ph + 2) * bar - 0.5
                truth["phrases"].append((a, e))
                pos = a
                while pos < e - 0.05:
                    d = min(e - pos, float(rng.uniform(0.3, 0.6)))
                    _, notes = _chord(prog[(ph + int((pos - a) / bar)) % len(prog)])
                    f0 = 261.63 * 2 ** ((notes[int(rng.integers(0, 3))] + 12 + detune) / 12)
                    m = int(d * SR)
                    t = np.arange(m) / SR
                    ph_ = 2 * np.pi * np.cumsum(f0 * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * t))) / SR
                    v = sum((1 / (h + 1)) * np.sin((h + 1) * ph_) for h in range(8))
                    env = np.minimum(1, t / 0.02) * np.minimum(1, (m - np.arange(m)) / (0.02 * SR))
                    vs = int(pos * SR)
                    vocals[vs:vs + m] += 0.12 * (0.8 + 0.4 * level) * v * env
                    pos += d
        b0 += nb
    n_out = int(bars * bar * SR)
    mix = mix[:, :n_out] + vocals[None, :n_out]
    return mix.astype(np.float32), vocals[:n_out].astype(np.float32), truth


@pytest.fixture(scope="module")
def song_files(tmp_path_factory):
    mix, vocals, truth = make_song()
    d = tmp_path_factory.mktemp("analysis")
    return write(d / "song.wav", mix), write(d / "vocals.wav", vocals[None]), truth


def test_sections_beats_downbeats_and_labels(song_files):
    song, vocals, truth = song_files
    start = time.perf_counter()
    r = analysis.sections(song, vocals_path=vocals)
    elapsed = time.perf_counter() - start
    assert set(r) == {"bpm", "beats", "downbeats", "phrases", "sections"}
    assert r["bpm"] == pytest.approx(128.0, rel=0.02)
    beats, true_beats = np.array(r["beats"]), np.array(truth["beats"])
    err = np.array([np.min(np.abs(beats - t)) for t in true_beats])
    assert np.median(err) < 0.015 and np.mean(err < 0.05) > 0.95
    downs = np.array(r["downbeats"])
    hit = np.mean([np.min(np.abs(downs - t)) < 0.06 for t in truth["downbeats"]])
    assert hit > 0.9 and len(downs) <= len(truth["downbeats"]) + 2
    got = [(s["start"], s["end"], s["label"]) for s in r["sections"]]
    assert len(got) == len(truth["sections"])
    for (a, b, label), (ta, tb, tl) in zip(got, truth["sections"]):
        assert abs(a - ta) < 0.1 and abs(b - tb) < 0.1
        assert label == tl
    phrases = r["phrases"]
    matched = sum(any(abs(p["start"] - a) < 0.15 and abs(p["end"] - b) < 0.15 for p in phrases)
                  for a, b in truth["phrases"])
    assert matched == len(truth["phrases"]) and len(phrases) == len(truth["phrases"])
    assert elapsed < 15.0


def test_sections_without_stem_uses_bar_phrases(song_files):
    song, _, truth = song_files
    r = analysis.sections(song)
    bar = 4 * 60.0 / 128.0
    assert r["phrases"] and r["phrases"][0]["start"] == 0.0
    lengths = [p["end"] - p["start"] for p in r["phrases"]]
    assert all(abs(x - 4 * bar) < 0.2 or x <= 4 * bar + 0.2 for x in lengths)   # 4-bar groups in sections
    starts = {round(s["start"], 1) for s in r["sections"]}
    assert starts <= {round(p["start"], 1) for p in r["phrases"]}


def test_sections_other_tempo_and_silence(tmp_path):
    plan = [("intro", 4, ["D", "A", "Bm", "G"]), ("verse", 8, ["Bm", "G", "D", "A"]),
            ("chorus", 8, ["G", "A", "D", "Bm"]), ("verse", 8, ["Bm", "G", "D", "A"]),
            ("chorus", 8, ["G", "A", "D", "Bm"]), ("outro", 4, ["D", "A", "Bm", "G"])]
    mix, _, truth = make_song(bpm=96.0, plan=plan, seed=3)
    r = analysis.sections(write(tmp_path / "s.wav", mix))
    assert r["bpm"] == pytest.approx(96.0, rel=0.03) or r["bpm"] == pytest.approx(192.0, rel=0.03)
    starts = [s["start"] for s in r["sections"]]
    found = sum(any(abs(s - t) < 0.3 for s in starts) for t, _, _ in truth["sections"])
    assert found >= len(truth["sections"]) - 1
    assert {s["label"] for s in r["sections"]} <= {*analysis.LABELS, None}
    assert "chorus" in {s["label"] for s in r["sections"]}
    silent = analysis.sections(write(tmp_path / "z.wav", np.zeros((1, 3 * SR), np.float32)))
    assert silent == {"bpm": None, "beats": [], "downbeats": [], "phrases": [],
                      "sections": [{"start": 0.0, "end": 3.0, "label": None}]}


@pytest.mark.parametrize("prog,expected,detune", [
    (["D", "G", "A", "D"], "D", 0.0),
    (["Em", "Am", "B", "Em"], "Em", 0.0),
    (["F#", "B", "C#", "F#"], "F#", 0.0),
    (["Bb", "Eb", "F", "Bb"], "Bb", 0.35),           # 35 cents sharp: tuning is estimated
    (["Cm", "Fm", "G", "Cm"], "Cm", -0.3),
])
def test_key(tmp_path, prog, expected, detune):
    plan = [("verse", 8, prog), ("chorus", 8, prog)]
    mix, _, _ = make_song(bpm=110.0, plan=plan, seed=5, detune=detune)
    assert analysis.key(write(tmp_path / "k.wav", mix)) == expected


def test_key_none_without_tonality(tmp_path):
    rng = np.random.default_rng(0)
    noise = (rng.standard_normal((1, 5 * SR)) * 0.1).astype(np.float32)
    assert analysis.key(write(tmp_path / "n.wav", noise)) is None
    assert analysis.key(write(tmp_path / "z.wav", np.zeros((1, 3 * SR), np.float32))) is None
    assert analysis.key_from_chroma(np.ones(12))[0] is None


def test_key_names_follow_the_contract():
    major = np.roll(analysis.MAJOR, 1)
    minor = np.roll(analysis.MINOR, 1)
    assert analysis.key_from_chroma(major)[0] == "Db" and analysis.key_from_chroma(minor)[0] == "C#m"
    assert analysis.key_from_chroma(np.roll(analysis.MAJOR, 5))[0] == "F"
