"""Export op: the final file — resample, position, loudness, dither, encode, tag, name, never overwrite."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Callable

import numpy as np

from .common import SR, DspError, bad_job, bad_params, check_input, downmix, number, seconds
from .edit import Output

FORMATS = ("wav", "flac", "mp3")
RATES = (44100, 48000)
DEPTHS = (16, 24, 32)
BITRATES = (192, 256, 320)
CEILING_DBTP = -1.0
LOUDNESS_RANGE = (-40.0, -5.0)
SAFETY_CEILING_DB = -0.1        # only when "keep loudness" would clip an integer/MP3 file


def _db(v: float) -> float:
    return 20.0 * math.log10(max(float(v), 1e-12))


def true_peak_db(x: np.ndarray) -> float:
    """True peak in dBTP: the mixer's oversampled (BS.1770-style, 4×) peak estimate."""
    from ..mixing import loudness as ld

    x = np.asarray(x, dtype=np.float32)
    return float(ld.true_peak_db(x)) if x.size else -120.0


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out or x.shape[-1] == 0:
        return x
    try:
        import soxr

        return np.ascontiguousarray(soxr.resample(np.ascontiguousarray(x.T), sr_in, sr_out, quality="VHQ").T,
                                    dtype=np.float32)
    except ImportError:  # pragma: no cover - soxr is a runtime requirement
        from ..mixing.audioio import resample as poly

        return poly(x, sr_in, sr_out)


def normalize(x: np.ndarray, sr: int, target: float, *, ceiling_db: float = CEILING_DBTP,
              log: Callable[[str], None] = lambda _m: None) -> tuple[np.ndarray, bool, float]:
    """Integrated loudness → ``target`` LUFS with the true peak ≤ ``ceiling_db`` dBTP.

    Plain gain when it fits under the ceiling; otherwise the mixer's look-ahead limiter on the
    oversampled peak envelope, its drive found with the secant method on a K-weighted copy (the
    limiter's gain varies slowly enough to commute with K-weighting), then measured exactly and
    trimmed so the true peak stays under the ceiling. Returns ``(audio, limited, true_peak_db)``."""
    from ..mixing import dynamics as dyn
    from ..mixing import loudness as ld

    l0 = ld.integrated_loudness(x, sr)
    env = ld.peak_envelope(x)                           # true-peak envelope; scales with any gain
    tp0 = _db(float(env.max())) if env.size else -120.0
    if not math.isfinite(l0):
        log("export: silent input, loudness left as is")
        return x, False, max(tp0, -120.0)
    gain_db = target - l0
    if tp0 + gain_db <= ceiling_db:
        log(f"export: {l0:.2f} → {target:.2f} LUFS with {gain_db:+.2f} dB")
        return (x * np.float32(10 ** (gain_db / 20))).astype(np.float32), False, tp0 + gain_db

    xk = ld.k_weight(x, sr)
    block, n = 8, x.shape[-1]
    lim = dict(ceiling_db=ceiling_db - 0.1, lookahead_s=0.0015, release_s=0.08, block=block)

    def gains(drive: float) -> np.ndarray:
        gr = dyn.limiter_gain(env, sr, drive_db=drive, **lim)
        return dyn.upsample(dyn.db_to_gain(drive - gr), block, n)

    def loud(drive: float) -> float:
        return ld.gated_loudness(ld.block_energies(xk * gains(drive), sr)) - target

    d0, f0 = gain_db, loud(gain_db)
    d1 = d0 - f0
    f1 = loud(d1)
    for _ in range(8):
        if abs(f1) < 0.05 or d1 > gain_db + 24.0:
            break
        slope = (f1 - f0) / (d1 - d0) if d1 != d0 else 1.0
        slope = min(max(slope, 0.05), 1.5)
        d0, f0 = d1, f1
        d1 = min(d1 - f1 / slope, gain_db + 24.0)
        f1 = loud(d1)
    xk = None
    y = (x * gains(d1)).astype(np.float32)
    tp = true_peak_db(y)
    if tp > ceiling_db:
        trim = ceiling_db - tp - 0.01
        y *= np.float32(10 ** (trim / 20))
        tp += trim
    lufs = ld.integrated_loudness(y, sr)
    log(f"export: {l0:.2f} → {lufs:.2f} LUFS (target {target:.2f}), limiter drive {d1:+.2f} dB, "
        f"true peak {tp:.2f} dBTP")
    return y, True, tp


def _read_native(path) -> tuple[np.ndarray, int]:
    from ..mixing.audioio import read

    p = check_input(path)
    x, sr = read(p, sr=None)
    return np.ascontiguousarray(downmix(x), dtype=np.float32), int(sr)


def _text(v) -> str | None:
    if not isinstance(v, str):
        return None
    v = "".join(ch for ch in v if ch >= " " or ch == "\t").strip()
    return v or None


def export(job, ctx) -> tuple[list[Output], dict]:
    """``ExportParams`` → the final file; ``outputs`` is empty and ``result.path`` names the file."""
    from .encode import write_mp3, write_pcm
    from .naming import claim, file_stem

    p = job.params
    fmt = p.get("format", "wav")
    if fmt not in FORMATS:
        raise bad_params("export.format", f"Unknown format {fmt!r}")
    sr = p.get("sampleRate", 44100)
    if sr not in RATES:
        raise bad_params("export.format", f"Sample rate {sr!r} is not one of {RATES}")
    bits = p.get("bitDepth", 24)
    if bits not in DEPTHS:
        raise bad_params("export.format", f"Bit depth {bits!r} is not one of {DEPTHS}")
    bitrate = p.get("mp3Bitrate", 320)
    if fmt == "mp3" and bitrate not in BITRATES:
        raise bad_params("export.format", f"MP3 bitrate {bitrate!r} is not one of {BITRATES}")
    target = None
    if p.get("loudness") is not None:
        target = number(p, "loudness", lo=LOUDNESS_RANGE[0], hi=LOUDNESS_RANGE[1], reason="export.loudness")
    tags = p.get("tags") if isinstance(p.get("tags"), dict) else {}
    title, artist = _text(tags.get("title")), _text(tags.get("artist"))

    folder_value = p.get("folder") or job.raw.get("defaultFolder")
    if not isinstance(folder_value, str) or not folder_value:
        raise bad_job("No export folder: set params.folder or the job's defaultFolder")
    folder = Path(folder_value)
    if not p.get("folder"):
        folder.mkdir(parents=True, exist_ok=True)       # the app's own default folder
    if not folder.is_dir():
        raise DspError("export.folder_missing", f"The export folder does not exist: {folder}",
                       params={"folder": str(folder)})

    src = job.inputs[0]
    ctx.progress(0.02, "dsp.read")
    x, native = _read_native(src.path)
    ctx.progress(0.15, "dsp.export")
    x = resample(x, native, sr)
    if p.get("keepPosition") and src.offset > 0:
        pad = int(round(src.offset * sr))
        x = np.concatenate([np.zeros((x.shape[0], pad), dtype=np.float32), x], axis=1)
        ctx.log(f"export: {src.offset:.3f} s of silence in front (keepPosition)")
    limited = False
    tp: float | None = None
    if target is not None:
        ctx.progress(0.3, "dsp.export.loudness")
        x, limited, tp = normalize(x, sr, target, log=ctx.log)
    elif (fmt != "wav" or bits != 32) and x.size and float(np.max(np.abs(x))) > 1.0:
        from ..mixing import dynamics as dyn

        x, gr = dyn.limit(x, sr, ceiling_db=SAFETY_CEILING_DB)
        limited = True
        ctx.log(f"export: the audio peaks above 0 dBFS; limited by up to {gr:.1f} dB so it does not clip")

    ctx.progress(0.6, "dsp.export.encode")
    stem = file_stem(p.get("fileName"), job.names)
    ext = fmt
    try:
        dest = claim(folder, stem, ext)
    except OSError as e:
        raise DspError("export.write_failed", f"Cannot create a file in {folder}: {e}",
                       params={"folder": str(folder), "reason": str(e)}) from None
    tmp = dest.with_name(dest.name + ".part")
    try:
        tagged = bool(title or artist)
        if fmt == "mp3":
            encoder = write_mp3(tmp, x, sr, int(bitrate), title=title, artist=artist, log=ctx.log)
            written_bits = 16
        else:
            written_bits = write_pcm(tmp, x, sr, fmt, int(bits), title=title, artist=artist)
            encoder = "soundfile"
            if fmt == "wav" and tagged:
                from .encode import mutagen_available, tag_with_mutagen

                if mutagen_available():
                    tag_with_mutagen(tmp, "wav", title, artist)
        os.replace(tmp, dest)
    except BaseException as e:
        for f in (tmp, dest):
            try:
                f.unlink()
            except OSError:
                pass
        if isinstance(e, OSError):
            raise DspError("export.write_failed", f"Cannot write {dest.name}: {e}",
                           params={"folder": str(folder), "reason": str(e)}) from None
        raise
    if fmt == "flac" and bits == 32:
        ctx.log("export: FLAC holds at most 24 bits; wrote 24-bit")
    from ..mixing import loudness as ld

    lufs = ld.report_lufs(ld.integrated_loudness(x, sr))
    tp = round(true_peak_db(x) if tp is None else tp, 2)
    ctx.log(f"export: wrote {dest.name} ({fmt}, {sr} Hz, {written_bits} bit, {lufs:.2f} LUFS, {tp:.2f} dBTP)")
    result = {"path": str(dest), "format": fmt, "sampleRate": sr, "bitDepth": written_bits,
              "mp3Bitrate": int(bitrate) if fmt == "mp3" else None, "lufs": lufs, "truePeakDb": tp,
              "duration": seconds(x.shape[-1], sr), "encoder": encoder, "tagged": tagged, "limited": limited}
    return [], result


__all__ = ["export", "normalize", "true_peak_db", "SR"]
