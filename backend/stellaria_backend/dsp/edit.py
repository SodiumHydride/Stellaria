"""Editing ops: trim, split, comp, concat — sample-accurate, equal-power fades, pure numpy."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .common import (DEFAULT_FADE_S, SR, Source, apply_fade_in, apply_fade_out, audio_info, bad_params,
                     fade_curve, match_channels, number, read_audio, seconds, would_click)

MAX_SPLIT_PARTS = 64
MAX_CROSSFADE_S = 10.0
MAX_GAP_S = 60.0


@dataclass
class Output:
    port: str
    audio: np.ndarray
    offset: float
    extra: dict[str, Any] = field(default_factory=dict)


def _fade_samples(requested: float, x: np.ndarray, at: int, is_cut: bool) -> int:
    """Requested fade, or the 10 ms default when it is 0 at a real cut through audible audio."""
    n = int(round(requested * SR))
    if n == 0 and is_cut and would_click(x, at):
        n = int(round(DEFAULT_FADE_S * SR))
    return n


# ----------------------------------------------------------------------------- trim

def trim(job, ctx) -> tuple[list[Output], dict]:
    """``{start, end, fadeIn, fadeOut}`` (seconds within the input) → port ``out``.

    Fades lie inside the region; a fade of 0 at a cut through audible audio becomes 10 ms."""
    p, src = job.params, job.inputs[0]
    total, _ = audio_info(src.path)
    start = number(p, "start", 0.0, lo=-1e-3, reason="trim.range")
    end = number(p, "end", total / SR, lo=-1e-3, reason="trim.range")
    s = max(0, int(round(start * SR)))
    e = min(total, int(round(end * SR)))
    if s >= total:
        raise bad_params("trim.range", f"The region starts at {start:.3f} s, after the end of the input "
                                       f"({total / SR:.3f} s)")
    if e - s < 1:
        raise bad_params("trim.range", f"The region {start:.3f}–{end:.3f} s is empty")
    fade_in = number(p, "fadeIn", 0.0, lo=0.0, hi=600.0, reason="trim.fade")
    fade_out = number(p, "fadeOut", 0.0, lo=0.0, hi=600.0, reason="trim.fade")
    ctx.progress(0.05, "dsp.read")
    # read a little around the region so "would it click" sees the samples at the cut
    lo, hi = max(0, s - 1), min(total, e + 1)
    raw = read_audio(src.path, start=lo, stop=hi)
    x = raw[:, s - lo: s - lo + (e - s)].copy(order="K")          # keeps the interleaved layout
    ctx.progress(0.5, "dsp.trim")
    fi = _fade_samples(fade_in, raw, s - lo, s > 0)
    fo = _fade_samples(fade_out, raw, e - lo, e < total)
    length = x.shape[-1]
    if fi + fo > length:                          # both fit: scale them down together
        k = length / float(fi + fo)
        fi, fo = int(math.floor(fi * k)), int(math.floor(fo * k))
    apply_fade_in(x, fi)
    apply_fade_out(x, fo)
    result = {"start": seconds(s), "end": seconds(e), "fadeIn": seconds(fi), "fadeOut": seconds(fo)}
    return [Output("out", x, src.offset + s / SR)], result


# ----------------------------------------------------------------------------- split

def split(job, ctx) -> tuple[list[Output], dict]:
    """``{cuts, fade, labels}`` → ports ``part1`` … ``partN`` (N = cuts + 1).

    The fade at each cut is centred on it: part *i* ends ``fade/2`` after the cut with an
    equal-power fade-out and part *i+1* starts ``fade/2`` before it with the matching fade-in, so
    parts placed at their offsets cross-fade instead of leaving a gap. A fade of 0 at a cut through
    audible audio becomes 10 ms. Offsets are the input offset + each part's first sample."""
    p, src = job.params, job.inputs[0]
    cuts_raw = p.get("cuts", [])
    if not isinstance(cuts_raw, list):
        raise bad_params("split.cuts", "'cuts' must be a list of seconds")
    labels = p.get("labels") if isinstance(p.get("labels"), list) else []
    fade = number(p, "fade", 0.0, lo=0.0, hi=60.0, reason="split.fade")
    ctx.progress(0.02, "dsp.read")
    x = read_audio(src.path)
    n = x.shape[-1]
    if n == 0:
        raise bad_params("split.cuts", "The input is empty")
    cuts: list[int] = []
    for c in cuts_raw:
        if isinstance(c, bool) or not isinstance(c, (int, float)) or not math.isfinite(float(c)):
            raise bad_params("split.cuts", f"Cut {c!r} is not a number")
        k = int(round(float(c) * SR))
        if k <= 0 or k >= n:
            raise bad_params("split.cuts", f"Cut at {float(c):.3f} s is outside the input (0–{n / SR:.3f} s)",
                             cut=round(float(c), 3))
        cuts.append(k)
    cuts = sorted(set(cuts))
    if len(cuts) + 1 > MAX_SPLIT_PARTS:
        raise bad_params("split.cuts", f"At most {MAX_SPLIT_PARTS - 1} cuts")
    ctx.progress(0.4, "dsp.split")
    bounds = [0, *cuts, n]
    fades: list[int] = []
    for i, c in enumerate(cuts):                  # per-cut fade, never wider than its neighbours
        f = _fade_samples(fade, x, c, True)
        room = min(c - bounds[i], bounds[i + 2] - c)
        fades.append(max(0, min(f, room)))
    outputs: list[Output] = []
    parts: list[dict] = []
    for i in range(len(cuts) + 1):
        fi = fades[i - 1] if i > 0 else 0
        fo = fades[i] if i < len(cuts) else 0
        a = bounds[i] - fi // 2 if i > 0 else 0
        b = bounds[i + 1] + (fo - fo // 2) if i < len(cuts) else n
        part = x[:, a:b].copy(order="K")
        apply_fade_in(part, fi)
        apply_fade_out(part, fo)
        port = f"part{i + 1}"
        label = labels[i] if i < len(labels) and isinstance(labels[i], str) else ""
        outputs.append(Output(port, part, src.offset + a / SR))
        parts.append({"port": port, "start": seconds(a), "end": seconds(b), "label": label})
    return outputs, {"parts": parts, "fades": [seconds(f) for f in fades]}


# ----------------------------------------------------------------------------- comp

@dataclass
class _Seg:
    start: int
    end: int
    input: int


def comp(job, ctx) -> tuple[list[Output], dict]:
    """``{segments: [{start, end, input}], crossfade}`` → port ``out``.

    Inputs are placed on the song timeline by their offsets (output offset = the smallest);
    segment times are song-timeline seconds; ``input`` -1 (or a gap between segments) is silence.
    Overlapping segments: the later-starting one wins from its start. Every boundary gets an
    equal-power cross-fade of ``crossfade`` seconds centred on it (a fade to/from silence at the
    outer edges and around silent stretches; 10 ms when 0 would click)."""
    p = job.params
    count = len(job.inputs)
    raw = p.get("segments")
    if not isinstance(raw, list) or not raw:
        raise bad_params("comp.segments", "'segments' must be a non-empty list")
    crossfade = number(p, "crossfade", 0.0, lo=0.0, hi=MAX_CROSSFADE_S, reason="comp.crossfade")
    base = min(i.offset for i in job.inputs)
    segs: list[_Seg] = []
    for item in raw:
        if not isinstance(item, dict):
            raise bad_params("comp.segments", "Each segment is {start, end, input}")
        a = number(item, "start", reason="comp.segments")
        b = number(item, "end", reason="comp.segments")
        idx = item.get("input")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < -1 or idx >= count:
            raise bad_params("comp.segments", f"Segment input {idx!r} is not -1 or an input index (0–{count - 1})")
        if b <= a:
            raise bad_params("comp.segments", f"Segment {a:.3f}–{b:.3f} s ends before it starts")
        s, e = max(0, int(round((a - base) * SR))), int(round((b - base) * SR))
        if e > s:
            segs.append(_Seg(s, e, idx))
    if not segs:
        raise bad_params("comp.segments", "Every segment lies before the first input")
    segs.sort(key=lambda g: g.start)
    for prev, cur in zip(segs, segs[1:]):         # later start wins
        if cur.start < prev.end:
            prev.end = cur.start
    segs = [g for g in segs if g.end > g.start]
    timeline: list[_Seg] = []                     # contiguous: explicit silence in the gaps
    for g in segs:
        if timeline and g.start > timeline[-1].end:
            timeline.append(_Seg(timeline[-1].end, g.start, -1))
        elif not timeline and g.start > 0:
            timeline.append(_Seg(0, g.start, -1))
        if timeline and timeline[-1].input == g.input and timeline[-1].end == g.start:
            timeline[-1].end = g.end
        else:
            timeline.append(_Seg(g.start, g.end, g.input))

    ctx.progress(0.02, "dsp.read")
    used = {g.input for g in timeline if g.input >= 0}
    sources: dict[int, Source] = {}
    starts = [int(round((src.offset - base) * SR)) for src in job.inputs]
    for i in sorted(used):
        sources[i] = Source(job.inputs[i].path)
        ctx.progress(0.02 + 0.3 * (len(sources)) / max(1, len(used)), "dsp.read")
    channels = max([src.channels for src in sources.values()] or [1])

    def covered_end(g: _Seg) -> int:
        return min(g.end, starts[g.input] + sources[g.input].frames) if g.input >= 0 else g.start

    length = max([covered_end(g) for g in timeline if g.input >= 0] or [timeline[-1].end])
    timeline = [g for g in timeline if g.start < length]
    timeline[-1].end = min(timeline[-1].end, length)

    def audible(i: int, at: int) -> bool:
        """Would a hard switch to/from input ``i`` at timeline sample ``at`` click?"""
        if i < 0:
            return False
        k = at - starts[i]
        w = int(0.002 * SR)
        if not 0 <= k <= sources[i].frames:
            return False
        return would_click(sources[i].region(k - w, k + w), min(k, w))

    # cross-fade length per boundary (index b = boundary before timeline[b]; b = len → the end)
    cf = int(round(crossfade * SR))
    fades: list[int] = []
    for b in range(len(timeline) + 1):
        left = timeline[b - 1] if b > 0 else None
        right = timeline[b] if b < len(timeline) else None
        if (left is None or left.input < 0) and (right is None or right.input < 0):
            fades.append(0)
            continue
        f = cf
        if f == 0:
            at = right.start if right is not None else left.end
            clicks = any(audible(g.input, at) for g in (left, right) if g is not None)
            f = int(round(DEFAULT_FADE_S * SR)) if clicks else 0
        room = min([g.end - g.start for g in (left, right) if g is not None])
        fades.append(max(0, min(f, room)))

    ctx.progress(0.4, "dsp.comp")
    out = np.zeros((channels, length), dtype=np.float32, order="F")   # interleaved: written without a copy
    for b, g in enumerate(timeline):
        if g.input < 0:
            continue
        fi, fo = fades[b], fades[b + 1]
        # fades are centred on their boundary, except at the output's own edges where they lie inside
        fi_at = g.start if g.start == 0 else g.start - fi // 2
        fo_at = g.end - fo if g.end >= length else g.end - fo // 2
        a = max(min(g.start, fi_at), 0)
        e = min(max(g.end, fo_at + fo), length)
        src, s0 = sources[g.input], starts[g.input]
        lo, hi = max(a, s0), min(e, s0 + src.frames)                 # where this input has audio
        if hi <= lo:
            continue
        gain = np.ones(e - a, dtype=np.float32)
        if fi:
            k0 = fi_at - a                                           # fade-in occupies [k0, k0 + fi)
            seg = fade_curve(fi, rising=True)
            lo_k, hi_k = max(0, k0), min(e - a, k0 + fi)
            gain[lo_k:hi_k] = seg[lo_k - k0: hi_k - k0]
        if fo:
            k0 = fo_at - a
            seg = fade_curve(fo, rising=False)
            lo_k, hi_k = max(0, k0), min(e - a, k0 + fo)
            gain[lo_k:hi_k] *= seg[lo_k - k0: hi_k - k0]
        audio = match_channels(src.region(lo - s0, hi - s0), channels)
        out[:, lo:hi] += audio * gain[lo - a: hi - a]
        ctx.progress(0.4 + 0.5 * (b + 1) / len(timeline), "dsp.comp")
    applied = [{"start": round(base + g.start / SR, 6), "end": round(base + g.end / SR, 6), "input": g.input}
               for g in timeline]
    return [Output("out", out, base)], {"segments": applied, "crossfades": [seconds(f) for f in fades]}


# ----------------------------------------------------------------------------- concat

def concat(job, ctx) -> tuple[list[Output], dict]:
    """``{gap, crossfade}`` → port ``out`` at offset 0: inputs one after another, separated by
    ``gap`` seconds of silence, or overlapping by an equal-power ``crossfade`` (which then wins;
    clamped to half the shorter neighbour). Junctions that would click get 10 ms edge fades."""
    p = job.params
    gap = number(p, "gap", 0.0, lo=0.0, hi=MAX_GAP_S, reason="concat.gap")
    crossfade = number(p, "crossfade", 0.0, lo=0.0, hi=MAX_CROSSFADE_S, reason="concat.crossfade")
    if crossfade > 0:
        gap = 0.0
    xs: list[np.ndarray] = []
    for i, src in enumerate(job.inputs):
        xs.append(read_audio(src.path))
        ctx.progress(0.05 + 0.5 * (i + 1) / len(job.inputs), "dsp.read")
    channels = max(x.shape[0] for x in xs)
    xs = [match_channels(x, channels).copy(order="K") for x in xs]
    ctx.progress(0.6, "dsp.concat")
    gap_n = int(round(gap * SR))
    starts: list[int] = []
    overlaps: list[int] = []
    pos = 0
    for i, x in enumerate(xs):
        if i == 0:
            starts.append(0)
            pos = x.shape[-1]
            continue
        prev = xs[i - 1]
        ov = 0
        if crossfade > 0:
            ov = min(int(round(crossfade * SR)), prev.shape[-1] // 2, x.shape[-1] // 2)
        if ov > 0:
            prev[:, prev.shape[-1] - ov:] *= fade_curve(ov, rising=False)
            x[:, :ov] *= fade_curve(ov, rising=True)
        else:                                      # butt splice or gap: soften edges that would click
            edge = int(round(DEFAULT_FADE_S * SR / 2))
            if prev.shape[-1] and would_click(prev, prev.shape[-1] - 1):
                apply_fade_out(prev, min(edge, prev.shape[-1] // 2))
            if x.shape[-1] and would_click(x, 0):
                apply_fade_in(x, min(edge, x.shape[-1] // 2))
        overlaps.append(ov)
        start = pos + gap_n - ov
        starts.append(start)
        pos = start + x.shape[-1]
    out = np.zeros((channels, pos), dtype=np.float32, order="F")
    for s, x in zip(starts, xs):
        out[:, s: s + x.shape[-1]] += x
    return [Output("out", out, 0.0)], {"starts": [seconds(s) for s in starts],
                                       "crossfades": [seconds(o) for o in overlaps], "gap": seconds(gap_n)}
