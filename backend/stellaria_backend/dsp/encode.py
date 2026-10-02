"""Final-file encoders for the export op: WAV / FLAC (soundfile) and MP3 (lameenc or ffmpeg).

* Integer formats are quantised here (TPDF dither at 16 bit, plain rounding at 24 bit) so levels
  round-trip exactly and nothing wraps around.
* MP3 through ``lameenc``: the raw LAME frames carry no gapless information, so decoders would play
  576 + 529 samples of encoder/decoder delay and the end padding. A LAME "Info" frame (CBR Xing
  header with the LAME extension: encoder delay, end padding, frame/byte counts, CRC) is written
  in front, exactly as the ``lame`` CLI does, so ffmpeg, mpg123, foobar2000, iTunes … trim it and
  the file decodes to the exact input length and position.
* MP3 through ffmpeg (the bundled imageio-ffmpeg binary, else ``ffmpeg`` on PATH) when ``lameenc``
  is not installed; its libmp3lame muxer writes the same Info frame itself.
* Tags (title/artist): Vorbis comments for FLAC and a RIFF ``LIST/INFO`` chunk for WAV via
  soundfile; ID3v2.3 for MP3 (and an ``id3`` chunk in WAV) via ``mutagen`` when it is installed,
  otherwise by the small ID3v2.3 writer below.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path
from typing import Callable

import numpy as np

from .common import DspError

LAME_DELAY = 576                    # LAME's encoder delay (the decoder adds its own 529)
SAMPLES_PER_FRAME = 1152            # MPEG-1 Layer III
_BITRATES = (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320)
_RATES = {44100: 0, 48000: 1, 32000: 2}


# ----------------------------------------------------------------------------- PCM

def quantize(x: np.ndarray, bits: int, *, dither: bool, seed: int = 0) -> np.ndarray:
    """float ``(C, n)`` → integer samples ``(n, C)``: int16 for 16 bit, int32 holding 24-bit values
    shifted into the top bytes for 24 bit (what libsndfile expects for ``PCM_24``).

    Runs in float32, which is exact here: scaling by 2^(bits-1) is exact and float32 holds every
    24-bit integer. TPDF dither (±1 LSB, deterministic seed) when ``dither``."""
    scale = np.float32(1 << (bits - 1))
    y = np.empty((x.shape[-1], x.shape[0]), dtype=np.float32)
    np.multiply(np.asarray(x, dtype=np.float32).T, scale, out=y)
    if dither:
        rng = np.random.default_rng(seed)
        y += rng.random(y.shape, dtype=np.float32)
        y -= rng.random(y.shape, dtype=np.float32)
    np.rint(y, out=y)
    np.clip(y, -scale, scale - 1, out=y)
    if bits == 16:
        return y.astype(np.int16)
    q = y.astype(np.int32)
    q <<= 32 - bits
    return q


def write_pcm(path: Path, x: np.ndarray, sr: int, fmt: str, bits: int, *, title: str | None,
              artist: str | None) -> int:
    """Write WAV/FLAC; returns the bit depth actually written (FLAC has no 32-bit float)."""
    import soundfile as sf

    if fmt == "flac" and bits == 32:
        bits = 24
    container = "FLAC" if fmt == "flac" else "WAV"
    if bits == 32:
        data, subtype = np.ascontiguousarray(np.asarray(x, dtype=np.float32).T), "FLOAT"
    else:
        data, subtype = quantize(x, bits, dither=bits == 16), f"PCM_{bits}"
    if container == "WAV" and data.nbytes > 4_000_000_000:
        container = "RF64"
    with sf.SoundFile(str(path), "w", sr, data.shape[1], subtype=subtype, format=container) as f:
        if title:
            f.title = title
        if artist:
            f.artist = artist
        f.write(data)
    return bits


# ----------------------------------------------------------------------------- MP3: lameenc + Info frame

def lameenc_available() -> bool:
    try:
        import lameenc  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _crc16(data: bytes) -> int:
    """CRC-16/ARC (poly 0x8005 reflected, init 0) — LAME's tag CRC."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _frames(data: bytes) -> tuple[int, bytes]:
    """(audio frame count, first frame header) of a raw MPEG-1 Layer III CBR stream."""
    pos, count, first = 0, 0, b""
    n = len(data)
    while pos + 4 <= n:
        h = data[pos: pos + 4]
        if h[0] != 0xFF or (h[1] & 0xE0) != 0xE0:
            pos += 1                                   # resync (lameenc never needs it)
            continue
        version, layer = (h[1] >> 3) & 3, (h[1] >> 1) & 3
        br_i, sr_i, padding = h[2] >> 4, (h[2] >> 2) & 3, (h[2] >> 1) & 1
        if version != 3 or layer != 1 or br_i in (0, 15) or sr_i == 3:
            pos += 1
            continue
        rate = (44100, 48000, 32000)[sr_i]
        size = 144000 * _BITRATES[br_i] // rate + padding
        if not first:
            first = h
        count += 1
        pos += size
    return count, first


def lame_info_frame(first_header: bytes, frames: int, stream_bytes: int, samples: int, sr: int,
                    bitrate: int) -> bytes:
    """A LAME/Info (CBR Xing) frame describing ``frames`` audio frames of ``samples`` samples."""
    br_i = _BITRATES.index(bitrate)
    h1 = (first_header[1] & 0xFE) | 0x01               # no CRC
    h2 = (br_i << 4) | (_RATES[sr] << 2)              # no padding, private 0
    h3 = first_header[3]
    size = 144000 * bitrate // sr
    frame = bytearray(size)
    frame[0:4] = bytes([0xFF, h1, h2, h3])
    mono = (h3 >> 6) == 3
    pos = 4 + (17 if mono else 32)
    total = stream_bytes + size
    padding = frames * SAMPLES_PER_FRAME - LAME_DELAY - samples
    tag = bytearray()
    tag += b"Info" + struct.pack(">I", 0x0F) + struct.pack(">I", frames) + struct.pack(">I", total)
    tag += bytes(int(i * 256 / 100) for i in range(100))                # linear TOC (CBR)
    tag += struct.pack(">I", 0)                                          # quality
    lame = bytearray()
    lame += b"LAME3.100"
    lame += bytes([0x01])                                                # tag rev 0, CBR
    lame += bytes([min(255, int(round(min(20000, sr / 2) / 100)))])      # lowpass / 100 Hz
    lame += struct.pack(">f", 0.0) + b"\x00\x00" + b"\x00\x00"           # peak, radio/audiophile gain
    lame += bytes([0x00, min(255, bitrate)])                             # flags/ATH, bitrate
    lame += bytes([(LAME_DELAY >> 4) & 0xFF, ((LAME_DELAY & 0x0F) << 4) | ((padding >> 8) & 0x0F), padding & 0xFF])
    lame += bytes([{44100: 1, 48000: 2}.get(sr, 0) << 6])               # misc: source rate
    lame += b"\x00" + b"\x00\x00"                                        # mp3 gain, preset/surround
    lame += struct.pack(">I", total) + b"\x00\x00"                       # music length, music CRC
    frame[pos: pos + len(tag)] = tag
    frame[pos + len(tag): pos + len(tag) + len(lame)] = lame
    crc = _crc16(bytes(frame[:190]))
    end = pos + len(tag) + len(lame)
    frame[end: end + 2] = struct.pack(">H", crc)
    return bytes(frame)


def encode_mp3_lameenc(x: np.ndarray, sr: int, bitrate: int) -> bytes:
    import lameenc

    pcm = quantize(x, 16, dither=True)                                   # (n, C) int16
    channels = pcm.shape[1]
    enc = lameenc.Encoder()
    enc.set_bit_rate(int(bitrate))
    enc.set_in_sample_rate(int(sr))
    enc.set_channels(int(channels))
    enc.set_quality(3)                                                   # LAME's own default
    data = bytes(enc.encode(np.ascontiguousarray(pcm).tobytes())) + bytes(enc.flush())
    frames, first = _frames(data)
    if not frames:
        raise DspError("export.encode_failed", "The MP3 encoder produced no audio", params={"format": "mp3"})
    if frames * SAMPLES_PER_FRAME - LAME_DELAY - pcm.shape[0] > 4095:   # padding must fit 12 bits
        return data
    return lame_info_frame(first, frames, len(data), pcm.shape[0], sr, bitrate) + data


# ----------------------------------------------------------------------------- MP3: ffmpeg

def ffmpeg_candidates() -> list[str]:
    out: list[str] = []
    try:
        from ..audio.io import ffmpeg_exe

        out.append(ffmpeg_exe())
    except Exception:  # noqa: BLE001 - imageio-ffmpeg missing or its binary unavailable
        pass
    found = shutil.which("ffmpeg")
    if found and found not in out:
        out.append(found)
    return out


def encode_mp3_ffmpeg(exe: str, x: np.ndarray, sr: int, bitrate: int, dest: Path, *, title: str | None,
                      artist: str | None) -> None:
    from ..util import procs

    pcm = quantize(x, 16, dither=True)
    args = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "s16le", "-ar", str(sr),
            "-ac", str(pcm.shape[1]), "-i", "pipe:0", "-codec:a", "libmp3lame", "-b:a", f"{int(bitrate)}k",
            "-id3v2_version", "3"]
    if title:
        args += ["-metadata", f"title={title}"]
    if artist:
        args += ["-metadata", f"artist={artist}"]
    args += ["-f", "mp3", str(dest)]
    proc = procs.popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    _, err = proc.communicate(np.ascontiguousarray(pcm).tobytes())
    if proc.returncode != 0:
        raise RuntimeError(err.decode("utf-8", errors="replace")[-1500:] or f"exit code {proc.returncode}")


# ----------------------------------------------------------------------------- tags

def mutagen_available() -> bool:
    try:
        import mutagen  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def id3v2_tag(title: str | None, artist: str | None) -> bytes:
    """A minimal ID3v2.3 tag (UTF-16 text frames TIT2/TPE1) — what Windows Explorer and every
    player read; v2.3 rather than v2.4 for the older readers."""
    body = b""
    for fid, text in (("TIT2", title), ("TPE1", artist)):
        if not text:
            continue
        payload = b"\x01" + b"\xff\xfe" + text.encode("utf-16-le")
        body += fid.encode("ascii") + struct.pack(">I", len(payload)) + b"\x00\x00" + payload
    if not body:
        return b""
    size = len(body)
    synchsafe = bytes([(size >> 21) & 0x7F, (size >> 14) & 0x7F, (size >> 7) & 0x7F, size & 0x7F])
    return b"ID3\x03\x00\x00" + synchsafe + body


def tag_with_mutagen(path: Path, fmt: str, title: str | None, artist: str | None) -> None:
    from mutagen.id3 import ID3, TIT2, TPE1

    if fmt == "mp3":
        tags = ID3()
    elif fmt == "wav":
        from mutagen.wave import WAVE

        wav = WAVE(str(path))
        if wav.tags is None:
            wav.add_tags()
        tags = wav.tags
    else:
        return
    if title:
        tags.add(TIT2(encoding=1, text=[title]))
    if artist:
        tags.add(TPE1(encoding=1, text=[artist]))
    tags.save(str(path), v2_version=3)


def write_mp3(path: Path, x: np.ndarray, sr: int, bitrate: int, *, title: str | None, artist: str | None,
              log: Callable[[str], None] = lambda _m: None, prefer: str | None = None) -> str:
    """Encode MP3 into ``path``. Returns the encoder used (``lameenc`` | ``ffmpeg``)."""
    use_lameenc = lameenc_available() and prefer != "ffmpeg"
    tagged_by_mutagen = mutagen_available() and bool(title or artist)
    if use_lameenc:
        data = encode_mp3_lameenc(x, sr, bitrate)
        head = b"" if tagged_by_mutagen else id3v2_tag(title, artist)
        with open(path, "wb") as f:
            f.write(head)
            f.write(data)
        if tagged_by_mutagen:
            tag_with_mutagen(path, "mp3", title, artist)
        return "lameenc"
    errors = []
    for exe in ffmpeg_candidates():
        try:
            encode_mp3_ffmpeg(exe, x, sr, bitrate, path, title=title, artist=artist)
            return "ffmpeg"
        except Exception as e:  # noqa: BLE001 - try the next ffmpeg, then give up
            errors.append(f"{exe}: {e}")
            log(f"export: ffmpeg at {exe} cannot encode MP3 ({str(e).strip()[:200]})")
    raise DspError("export.encoder_missing", "No MP3 encoder is available (install lameenc, or ffmpeg with "
                   "libmp3lame)", params={"format": "mp3"})
