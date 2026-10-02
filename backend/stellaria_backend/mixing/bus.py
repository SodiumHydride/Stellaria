"""The ``bus`` processor: inputs placed by offset, gain/pan/mute/solo, summed; optional loudness
normalization with a true-peak ceiling. Nothing else (for pre-mixing stems into a backing)."""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import loudness, master, spec
from .audioio import SR, place, read, read_check, timeline, to_stereo, write
from .chains import balance_gains
from .job import MixJob

ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]


def mix_bus(j: MixJob, progress: ProgressFn, log: LogFn) -> dict:
    sr = SR
    normalize = bool(spec.param(j.params, "normalize", table=spec.BUS_PARAM_BY_ID))
    target = float(spec.param(j.params, "targetLufs", table=spec.BUS_PARAM_BY_ID))
    progress(0.0, "mix.analyze")
    for i in j.inputs:
        read_check(i.path)
    active = {id(i) for i in j.active()}
    audio = [to_stereo(read(i.path, sr)[0]) for i in j.inputs]
    base, n, starts = timeline([(a.shape[-1], i.offset) for a, i in zip(audio, j.inputs)], sr)
    log(f"bus: {len(j.inputs)} input(s), {len(active)} playing, {n / sr:.1f} s from {base:.2f} s")
    out = np.zeros((2, n), dtype=np.float32)
    for k, (i, a, start) in enumerate(zip(j.inputs, audio, starts)):
        progress(0.1 + 0.6 * k / len(j.inputs), "mix.master")
        if id(i) not in active:
            continue
        left, right = balance_gains(i.pan)
        g = 10.0 ** (i.gain_db / 20.0)
        placed = place(a, start, n)
        out[0] += placed[0] * np.float32(left * g)
        out[1] += placed[1] * np.float32(right * g)
    del audio
    progress(0.7, "mix.master")
    if normalize:
        r = master.normalize(out, sr, target, log=log)
        out, lufs, peak, true_peak = r.audio, r.lufs, r.peak_db, r.true_peak_db
    else:
        lufs = loudness.integrated_loudness(out, sr)
        peak = loudness.sample_peak_db(out)
        true_peak = loudness.true_peak_db(out)
    progress(0.95, "mix.master")
    write(j.output, out, sr)
    progress(1.0, "mix.master")
    return {"lufs": loudness.report_lufs(lufs), "peakDb": round(peak, 2), "truePeakDb": round(true_peak, 2),
            "bpm": None, "vocalToBackingDb": None, "offset": round(base, 6)}
