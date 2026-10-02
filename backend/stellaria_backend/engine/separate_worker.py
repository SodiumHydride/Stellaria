"""Vocal separation worker (separator add-on). Runs in its own process; imports torch.

Modes (JSON lines on stdout: {"type": "progress", "value": x} … {"type": "result", "data": {...}}
or {"type": "error", "code", "message", "detail", "params"?}):
  --input in.wav --vocals v.wav --instrumental i.wav     separate one file
      [--drums d.wav --bass b.wav --other o.wav]         4 stems: also keep Demucs' other stems
                                                         (instrumental = their sum, as before)
  --measure list.json                                    music residue ratio per clip
  --download                                             fetch the Demucs weights only
  --input in.wav --dry d.wav [--wet w.wav]               clean up one file: the dry voice (and
      --cleanup model.ckpt --cleanup-config config.yaml  what was removed: input − dry)
``--model`` names the Demucs model. With ``--roformer model.ckpt --roformer-config config.yaml``
the vocals come from that Roformer model and instrumental = mix − vocals; for 4 stems Demucs
then splits the instrumental (drums, bass; other = the rest). Cleanup keeps the input's channel
count (mono in, mono out; more than two channels: the first two).
``--device cpu`` forces the CPU (next to a training that holds the GPU).
``--fake`` (fake engine, any model): vocals = the input band-passed to 150–5000 Hz,
instrumental = the rest (bass = its part below 150 Hz, drums = above 5 kHz, other = the
remainder), small fixed music ratios; dry = the input high-passed at 120 Hz.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from typing import Any, Callable

SR = 44100
RUNTIME_MODULES = ("einops", "rotary_embedding_torch")  # what Roformer models need from the add-on


def _emit_factory():
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def emit(obj: dict[str, Any]) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    return emit


def _device(forced: str | None) -> str:
    import torch

    if forced:
        return forced
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def _read(path: str):
    import numpy as np
    import soundfile as sf

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != SR:
        import soxr

        data = soxr.resample(data, sr, SR).astype(np.float32)
    return data


class Demucs:
    def __init__(self, model: str, progress: Callable[[float], None], device: str | None = None) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        from demucs.api import Separator

        self._progress = progress
        self._base = 0.0
        self._span = 1.0
        self.sep = Separator(model=model, device=_device(device), progress=False, callback=self._cb)

    def _cb(self, info: dict[str, Any]) -> None:
        length = max(1, int(info.get("audio_length", 1)))
        models = max(1, int(info.get("models", 1)))
        frac = (int(info.get("model_idx_in_bag", 0)) + min(1.0, int(info.get("segment_offset", 0)) / length)) / models
        self._progress(self._base + self._span * frac)

    def stems(self, data, base: float = 0.0, span: float = 1.0) -> dict[str, Any]:
        """Every Demucs stem as (samples, channels) float32 arrays (vocals, drums, bass, other, …)."""
        import torch

        self._base, self._span = base, span
        wav = torch.from_numpy(data.T.copy())
        if wav.shape[0] == 1:
            wav = wav.repeat(2, 1)
        _, stems = self.sep.separate_tensor(wav, SR)
        return {k: v.T.cpu().numpy() for k, v in stems.items()}

    def split(self, data, base: float = 0.0, span: float = 1.0):
        """(vocals, instrumental) as (samples, channels) float32 arrays."""
        stems = self.stems(data, base, span)
        return stems["vocals"], sum(v for k, v in stems.items() if k != "vocals")


def _stereo(data):
    """(samples, 2): mono duplicated, extra channels dropped."""
    import numpy as np

    if data.shape[1] == 1:
        return np.repeat(data, 2, axis=1)
    return np.ascontiguousarray(data[:, :2])


class Roformer:
    def __init__(self, checkpoint: str, config: str, progress: Callable[[float], None],
                 device: str | None = None) -> None:
        from .roformer.inference import RoformerSeparator

        self._progress = progress
        self.sep = RoformerSeparator(checkpoint, config, _device(device))

    def extract(self, data, base: float = 0.0, span: float = 1.0):
        """What the model keeps (the vocals; the dry voice) as a (samples, 2) float32 array."""
        return self.sep.extract(_stereo(data), SR, lambda v: self._progress(base + span * v))

    def split(self, data, base: float = 0.0, span: float = 1.0):
        """(vocals, instrumental) as (samples, 2) float32 arrays."""
        mix = _stereo(data)
        vocals = self.extract(mix, base, span)
        return vocals, mix - vocals


def _clean(opts: argparse.Namespace, progress: Callable[[float], None]) -> dict[str, Any]:
    """``--input`` → ``--dry`` (without reverb / echo / noise) and ``--wet`` (input − dry)."""
    import numpy as np
    import soundfile as sf

    data = _read(opts.input)
    x = data if data.shape[1] <= 2 else np.ascontiguousarray(data[:, :2])
    if opts.fake:
        from scipy.signal import butter, sosfiltfilt

        dry = sosfiltfilt(butter(2, 120, btype="highpass", fs=SR, output="sos"), x, axis=0).astype(np.float32)
        progress(1.0)
    else:
        dry = Roformer(opts.cleanup, opts.cleanup_config, progress, opts.device).extract(x)
        if x.shape[1] == 1:
            dry = dry.mean(axis=1, keepdims=True)
    sf.write(opts.dry, np.clip(dry, -1, 1), SR, subtype="PCM_16")
    out = {"dry": opts.dry}
    if opts.wet:
        sf.write(opts.wet, np.clip(x - dry, -1, 1), SR, subtype="PCM_16")
        out["wet"] = opts.wet
    return out


def _ratio(vocals, inst) -> float:
    import numpy as np

    ev, ei = float(np.mean(vocals ** 2)), float(np.mean(inst ** 2))
    return round(ei / (ev + ei + 1e-12), 4)


def run(opts: argparse.Namespace, emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    def progress(value: float) -> None:
        emit({"type": "progress", "value": round(max(0.0, min(1.0, value)), 4)})

    def splitter():
        if opts.roformer:
            return Roformer(opts.roformer, opts.roformer_config, progress, opts.device)
        return Demucs(opts.model, progress, opts.device)

    if opts.download:
        if not opts.fake:
            Demucs(opts.model, progress)
        return {}
    if opts.measure:
        with open(opts.measure, "r", encoding="utf-8") as f:
            files = json.load(f)
        ratios: dict[str, float] = {}
        model = None if opts.fake else splitter()
        for i, path in enumerate(files):
            if model is None:
                ratios[path] = round(0.01 + int(hashlib.md5(path.encode()).hexdigest()[:4], 16) / 65535 * 0.05, 4)
            else:
                ratios[path] = _ratio(*model.split(_read(path), i / len(files), 1 / len(files)))
            progress((i + 1) / len(files))
        return {"ratios": ratios}
    if opts.dry:
        return _clean(opts, progress)
    data = _read(opts.input)
    four = bool(opts.drums and opts.bass and opts.other)
    if opts.fake:
        from scipy.signal import butter, sosfiltfilt

        band = butter(4, [150, 5000], btype="bandpass", fs=SR, output="sos")
        vocals = sosfiltfilt(band, data, axis=0).astype(np.float32)
        inst = data - vocals
        bass = sosfiltfilt(butter(4, 150, btype="lowpass", fs=SR, output="sos"), inst, axis=0).astype(np.float32)
        drums = sosfiltfilt(butter(4, 5000, btype="highpass", fs=SR, output="sos"), inst, axis=0).astype(np.float32)
        parts = {"drums": drums, "bass": bass, "other": inst - drums - bass}
        progress(1.0)
    elif opts.roformer:
        vocals, inst = Roformer(opts.roformer, opts.roformer_config, progress, opts.device).split(
            data, 0.0, 0.6 if four else 1.0)
        parts = {"drums": None, "bass": None, "other": inst}
        if four:  # Demucs splits the accompaniment; other = whatever drums and bass leave
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # the Roformer model is gone; give its memory back first
            stems = Demucs(opts.model, progress, opts.device).stems(inst, 0.6, 0.4)
            zero = np.zeros_like(inst)
            drums, bass = stems.get("drums", zero), stems.get("bass", zero)
            parts = {"drums": drums, "bass": bass, "other": inst - drums - bass}
    else:
        stems = Demucs(opts.model, progress, opts.device).stems(data)
        vocals = stems.pop("vocals")
        inst = sum(stems.values())
        parts = {"drums": stems.get("drums"), "bass": stems.get("bass"),
                 "other": sum(v for k, v in stems.items() if k not in ("drums", "bass"))}  # 6-stem models: guitar/piano → other
    out = {"vocals": opts.vocals, "instrumental": opts.instrumental}
    sf.write(opts.vocals, np.clip(vocals, -1, 1), SR, subtype="PCM_16")
    sf.write(opts.instrumental, np.clip(inst, -1, 1), SR, subtype="PCM_16")
    if four:
        for name in ("drums", "bass", "other"):
            x = parts[name] if parts[name] is not None else np.zeros_like(inst)
            sf.write(getattr(opts, name), np.clip(x, -1, 1), SR, subtype="PCM_16")
            out[name] = getattr(opts, name)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--vocals")
    parser.add_argument("--instrumental")
    parser.add_argument("--drums")
    parser.add_argument("--bass")
    parser.add_argument("--other")
    parser.add_argument("--measure")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--model", default="htdemucs")
    parser.add_argument("--roformer")
    parser.add_argument("--roformer-config")
    parser.add_argument("--dry")
    parser.add_argument("--wet")
    parser.add_argument("--cleanup")
    parser.add_argument("--cleanup-config")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--fake", action="store_true")
    opts = parser.parse_args(argv)
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    emit = _emit_factory()
    try:
        emit({"type": "result", "data": run(opts, emit)})
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        text = str(exc)
        task = "cleanup" if opts.dry else "separate"
        code, params = f"{task}.failed", {}
        if "out of memory" in text.lower():
            code = "engine.out_of_memory"
        elif type(exc).__name__ == "CheckpointMismatch":  # the full mismatch stays in the detail
            code, text = f"{task}.model_mismatch", "The checkpoint does not match its config"
        elif isinstance(exc, ImportError) and exc.name in RUNTIME_MODULES:  # add-on installed before Roformer support
            code, params = "addon.outdated", {"resource": "separator"}
        emit({"type": "error", "code": code, "message": text or type(exc).__name__, "detail": traceback.format_exc(),
              "params": params})
        sys.exit(1)


if __name__ == "__main__":
    main()
