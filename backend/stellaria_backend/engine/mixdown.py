"""Timeline mixdown (numpy): clip gain/fades/offsets/mute, track gain/pan/mute/solo, master gain,
a time range and per-track stems; plus the export file (resample, level, encode).

The arrangement's playback in the app follows the same rules (linear fades, a 2 ms ramp where a
clip starts or ends in the middle of its audio, none where a split clip continues), so what is
heard is what is exported."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

SR = 44100
SILENT_DB = -60.0        # a fader at or below this is off (−∞), as in the app
DECLICK = 0.002          # seconds of ramp where a clip starts or ends in the middle of its audio
JOIN = 0.0005            # clips this close that continue the same audio play as one
PEAK_DB = -1.0           # "normalize": the peak goes here

MediaPath = Callable[[str], Path]


def _db(gain_db: Any) -> float:
    db = float(gain_db or 0)
    return 0.0 if db <= SILENT_DB else float(10 ** (db / 20))


def _num(clip: dict[str, Any], key: str) -> float:
    return float(clip.get(key) or 0)


def _read_segment(path: Path, offset: float, duration: float) -> np.ndarray:
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        start = max(0, int(round(offset * sr)))
        frames = max(0, int(round(duration * sr)))
        f.seek(min(start, f.frames))
        data = f.read(frames, dtype="float32", always_2d=True)
    if sr != SR and len(data):
        import soxr

        data = soxr.resample(data, sr, SR).astype(np.float32)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data[:, :2]


def _joins(clips: list[dict[str, Any]]) -> tuple[set[str], set[str]]:
    """Ids of clips whose start / end continues another clip of the track back to back with the
    same audio (a split): no declick ramp there, so the two play as one."""
    starts: set[str] = set()
    ends: set[str] = set()
    for a in clips:
        a_end = _num(a, "start") + _num(a, "duration")
        for b in clips:
            if a is b or a.get("mediaId") != b.get("mediaId"):
                continue
            if abs(a_end - _num(b, "start")) < JOIN and abs(_num(a, "offset") + _num(a, "duration") - _num(b, "offset")) < JOIN:
                ends.add(a["id"])
                starts.add(b["id"])
    return starts, ends


def envelope(n: int, t0: float, duration: float, fade_in: float, fade_out: float, ramp_in: float = 0.0,
             ramp_out: float = 0.0, sr: int = SR) -> np.ndarray:
    """Gain of ``n`` samples from ``t0`` seconds into a clip of ``duration`` seconds: linear fades
    (at least ``ramp_in`` / ``ramp_out`` long). Starting inside a fade continues it."""
    t = t0 + np.arange(n, dtype=np.float64) / sr
    g = np.ones(n, dtype=np.float64)
    fi, fo = max(fade_in, ramp_in), max(fade_out, ramp_out)
    if fi > 0:
        g *= np.clip(t / fi, 0.0, 1.0)
    if fo > 0:
        g *= np.clip((duration - t) / fo, 0.0, 1.0)
    return g.astype(np.float32)


def audible_tracks(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    """Tracks that are heard: not muted, and soloed when any track is."""
    tracks = timeline.get("tracks", [])
    soloed = any(t.get("solo") for t in tracks)
    return [t for t in tracks if not t.get("mute") and (not soloed or t.get("solo"))]


def _span(timeline: dict[str, Any], start: float | None, end: float | None) -> tuple[float, int]:
    """(first second, samples) of the part to render: the whole song unless a range is given."""
    clip_ends = [_num(c, "start") + _num(c, "duration") for t in timeline.get("tracks", []) for c in t.get("clips", [])]
    length = max([float(timeline.get("length") or 0), *clip_ends, 0.0])
    t0 = max(0.0, float(start or 0))
    t1 = length if end is None else max(0.0, float(end))
    return t0, max(0, int(round((t1 - t0) * SR)))


def _render_track(track: dict[str, Any], media_path: MediaPath, t0: float, n: int) -> np.ndarray | None:
    """One track over ``n`` samples from ``t0``, gain and pan applied; None when no clip sounds there."""
    clips = [c for c in track.get("clips", []) if not c.get("mute")]
    joined_start, joined_end = _joins(clips)
    t1 = t0 + n / SR
    out: np.ndarray | None = None
    for clip in clips:
        cs, cd = _num(clip, "start"), _num(clip, "duration")
        a, b = max(cs, t0), min(cs + cd, t1)
        if b <= a:
            continue
        data = _read_segment(media_path(clip["mediaId"]), _num(clip, "offset") + (a - cs), b - a)
        if not len(data):
            continue
        env = envelope(len(data), a - cs, cd, _num(clip, "fadeIn"), _num(clip, "fadeOut"),
                       0.0 if clip["id"] in joined_start else DECLICK, 0.0 if clip["id"] in joined_end else DECLICK)
        data = data * (env * _db(clip.get("gainDb", 0)))[:, None]
        i = int(round((a - t0) * SR))
        j = min(n, i + len(data))
        if j <= i:
            continue
        if out is None:
            out = np.zeros((n, 2), dtype=np.float32)
        out[i:j] += data[: j - i]
    if out is None:
        return None
    pan = max(-1.0, min(1.0, float(track.get("pan", 0))))
    balance = np.array([min(1.0, 1.0 - pan), min(1.0, 1.0 + pan)], dtype=np.float32) * _db(track.get("gainDb", 0))
    return out * balance


def render_stems(timeline: dict[str, Any], media_path: MediaPath, *, start: float | None = None,
                 end: float | None = None) -> list[tuple[dict[str, Any], np.ndarray]]:
    """Each audible track that sounds in the range on its own, (samples, 2) at ``SR`` with its gain,
    pan and the master gain: the stems add up to :func:`render_timeline`."""
    t0, n = _span(timeline, start, end)
    if n == 0:
        return []
    master = _db(timeline.get("masterGainDb", 0))
    out = []
    for track in audible_tracks(timeline):
        data = _render_track(track, media_path, t0, n)
        if data is not None:
            out.append((track, data * master))
    return out


def render_timeline(timeline: dict[str, Any], media_path: MediaPath, normalize: bool = False, *,
                    start: float | None = None, end: float | None = None) -> np.ndarray:
    """Mix ``timeline`` (or ``start``…``end`` seconds of it) to a (samples, 2) float32 array at
    ``SR``. ``media_path(media_id)`` resolves files; ``normalize`` sets the peak to −1 dBFS."""
    t0, n = _span(timeline, start, end)
    out = np.zeros((n, 2), dtype=np.float32)
    for _, data in render_stems(timeline, media_path, start=start, end=end):
        out += data
    if normalize and len(out):
        peak = float(np.max(np.abs(out)))
        if peak > 0:
            out *= 10 ** (PEAK_DB / 20) / peak
    return out


# ----------------------------------------------------------------------------- export file

def resample(x: np.ndarray, sr: int, sr_in: int = SR) -> np.ndarray:
    """(samples, 2) at ``sr_in`` → at ``sr``."""
    if sr == sr_in or not len(x):
        return x
    from ..dsp.export import resample as rs

    return np.ascontiguousarray(rs(np.ascontiguousarray(x.T), sr_in, sr).T)


def level_gain(mix: np.ndarray, sr: int, *, peak: bool, loudness: float | None) -> float:
    """The plain gain that brings ``mix`` to ``loudness`` LUFS, or its peak to −1 dBFS (1 when
    neither, or when it is silent). Stems share it, so they still add up to the leveled mix."""
    if loudness is not None:
        from ..mixing import loudness as ld

        lufs = ld.integrated_loudness(np.ascontiguousarray(mix.T), sr)
        return float(10 ** ((loudness - lufs) / 20)) if np.isfinite(lufs) else 1.0
    if peak:
        top = float(np.max(np.abs(mix))) if len(mix) else 0.0
        return float(10 ** (PEAK_DB / 20) / top) if top > 0 else 1.0
    return 1.0


def level(x: np.ndarray, sr: int, *, peak: bool, loudness: float | None, float_out: bool,
          log: Callable[[str], None] = lambda _m: None) -> np.ndarray:
    """The finished mix: ``loudness`` LUFS with the true peak ≤ −1 dBTP (the export op's
    loudness code), or the peak at −1 dBFS; otherwise as mixed, limited only where an integer or
    MP3 file would clip."""
    if loudness is not None:
        from ..dsp.export import normalize

        y, _, _ = normalize(np.ascontiguousarray(x.T), sr, loudness, log=log)
        return np.ascontiguousarray(y.T)
    if peak:
        return (x * np.float32(level_gain(x, sr, peak=True, loudness=None))).astype(np.float32)
    return safe(x, sr, float_out=float_out, log=log)


def safe(x: np.ndarray, sr: int, *, float_out: bool, log: Callable[[str], None] = lambda _m: None) -> np.ndarray:
    """Limit just under 0 dBFS when the audio would clip in an integer or MP3 file."""
    if float_out or not len(x) or float(np.max(np.abs(x))) <= 1.0:
        return x
    from ..mixing import dynamics as dyn

    y, gr = dyn.limit(np.ascontiguousarray(x.T), sr, ceiling_db=-0.1)
    log(f"mixdown: the audio peaks above 0 dBFS; limited by up to {gr:.1f} dB so it does not clip")
    return np.ascontiguousarray(y.T)


def write_audio(dest: Path, x: np.ndarray, sr: int, fmt: str, *, bits: int = 24, bitrate: int = 320,
                title: str | None = None, log: Callable[[str], None] = lambda _m: None) -> None:
    """Encode (samples, 2) into ``dest`` (written next to it first, then moved into place)."""
    from ..dsp.encode import write_mp3, write_pcm

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    data = np.ascontiguousarray(x.T, dtype=np.float32)
    try:
        if fmt == "mp3":
            write_mp3(tmp, data, sr, int(bitrate), title=title, artist=None, log=log)
        else:
            write_pcm(tmp, data, sr, fmt, int(bits), title=title, artist=None)
        os.replace(tmp, dest)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
