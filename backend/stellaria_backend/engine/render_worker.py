"""Persistent render worker. Runs in its own process (this one imports torch).

Protocol (JSON lines):
  stdin  ← {"id": 1, "cmd": "render" | "pitch", "args": {...}} | {"id": 2, "cmd": "ping"} | {"cmd": "unload"} | {"cmd": "shutdown"}
  stdout → {"type": "ready", "device": "cuda"} once, then per request
           {"id": 1, "type": "progress", "value": 0.4} … {"id": 1, "type": "result", "data": {...}}
           or {"id": 1, "type": "error", "code": "...", "message": "...", "detail": "..."}
Everything the engine prints goes to stderr (stdout is reserved for the protocol).

The inference path mirrors ``main_reflow.py`` (slicing with ``split()``, f0/volume/units, mask,
cross-fade, key/formant/vocal-register shift, spk_mix) and keeps the units encoder and the last
model loaded between requests, per device: a request may ask for ``"device": "cpu"`` (rendering
next to a training that holds the GPU) without evicting the GPU models. Such CPU-side models
are released after ``CPU_IDLE_RELEASE`` seconds without use. f0 is cached per input content +
extractor.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Callable

from stellaria_backend.engine import pitch_edit

Progress = Callable[[float], None]
CPU_IDLE_RELEASE = 120.0  # seconds


class BadCheckpoint(Exception):
    """The checkpoint file cannot be read (e.g. still being written by a training)."""


class DeviceCache:
    """One cached object per (device, slot), plus when each device was last used."""

    def __init__(self) -> None:
        self._items: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.last_used: dict[str, float] = {}

    def get(self, device: str, slot: str, key: Any, build: Callable[[], Any]) -> Any:
        self.last_used[device] = time.monotonic()
        item = self._items.get((device, slot))
        if item is not None and item[0] == key:
            return item[1]
        self._items.pop((device, slot), None)  # free the old one before building the new one
        value = build()
        self._items[(device, slot)] = (key, value)
        return value

    def devices(self) -> set[str]:
        return {d for d, _ in self._items}

    def drop(self, device: str | None = None) -> None:
        for k in [k for k in self._items if device is None or k[0] == device]:
            del self._items[k]

    def release_idle(self, device: str, idle: float, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        if device in self.devices() and now - self.last_used.get(device, now) >= idle:
            self.drop(device)
            return True
        return False


def pick_device() -> str:
    """cuda → mps (verified with a tiny forward pass) → cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        try:
            conv = torch.nn.Conv1d(1, 2, 3).to("mps")
            float(conv(torch.zeros(1, 1, 16, device="mps")).sum().item())
            return "mps"
        except Exception:  # noqa: BLE001 - broken MPS: CPU it is
            pass
    return "cpu"


class EngineWorker:
    def __init__(self, engine_root: str) -> None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        sys.path.insert(0, engine_root)
        os.chdir(engine_root)
        self.device = pick_device()  # best device; requests may still ask for "cpu"
        self._cache = DeviceCache()
        self._lock = threading.Lock()
        threading.Thread(target=self._janitor, name="cpu-release", daemon=True).start()

    def _janitor(self) -> None:
        """Release models loaded for CPU-side renders once they sit idle (GPU machines only)."""
        while True:
            time.sleep(10)
            with self._lock:
                if self.device != "cpu" and self._cache.release_idle("cpu", CPU_IDLE_RELEASE):
                    gc.collect()
                    print(" [cache] released idle CPU models")

    def unload(self) -> None:
        with self._lock:
            self._cache.drop()
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def _load_model(self, model_path: str, overrides: dict[str, str], device: str):
        key = (model_path, os.path.getmtime(model_path), json.dumps(overrides, sort_keys=True))
        return self._cache.get(device, "model", key, lambda: self._build_model(model_path, overrides, device))

    # mirrors reflow.vocoder.load_model_vocoder, but replaces pretrained paths that do not
    # exist on this machine (e.g. a release imported from another computer)
    def _build_model(self, model_path: str, overrides: dict[str, str], device: str):
        import torch
        import yaml
        from reflow.vocoder import DotDict, Unit2Wav, Vocoder

        config_file = os.path.join(os.path.split(model_path)[0], "config.yaml")
        with open(config_file, "r", encoding="utf-8") as f:
            args = DotDict(yaml.safe_load(f))
        if overrides.get("vocoderCkpt") and not os.path.exists(args["vocoder"]["ckpt"]):
            args["vocoder"]["ckpt"] = overrides["vocoderCkpt"]
        if overrides.get("encoderCkpt") and not os.path.exists(args["data"]["encoder_ckpt"]):
            args["data"]["encoder_ckpt"] = overrides["encoderCkpt"]
        vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
        if args.model.type != "RectifiedFlow":
            raise ValueError(f" [x] Unknown Model: {args.model.type}")
        model = Unit2Wav(
            args.data.sampling_rate, args.data.block_size, args.model.win_length, args.data.encoder_out_channels,
            args.model.n_spk, args.model.use_pitch_aug, vocoder.dimension, args.model.n_aux_layers,
            args.model.n_aux_chans, args.model.n_layers, args.model.n_chans,
            glu_type=args.model.glu_type if args.model.glu_type is not None else "atanglu",
        )
        print(" [Loading] " + model_path)
        try:
            ckpt = torch.load(model_path, map_location=torch.device(device))
        except Exception as exc:  # truncated / half-written file
            raise BadCheckpoint(model_path) from exc
        model.to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return model, vocoder, args

    def _units_encoder(self, args, device: str):
        from ddsp.vocoder import Units_Encoder

        gate = args.data.cnhubertsoft_gate if args.data.encoder == "cnhubertsoftfish" else 10
        key = (args.data.encoder, args.data.encoder_ckpt, args.data.encoder_sample_rate, args.data.encoder_hop_size)
        return self._cache.get(device, "encoder", key, lambda: Units_Encoder(
            args.data.encoder, args.data.encoder_ckpt, args.data.encoder_sample_rate,
            args.data.encoder_hop_size, cnhubertsoft_gate=gate, device=device,
        ))

    def pitch(self, a: dict[str, Any], progress: Progress) -> dict[str, Any]:
        """The input's raw pitch line on the analysis grid, for the pitch editor."""
        import librosa

        device = "cpu" if a.get("device") == "cpu" else self.device
        with self._lock:
            audio, sample_rate = librosa.load(a["input"], sr=None)
            if len(audio.shape) > 1:
                audio = librosa.to_mono(audio)
            progress(0.1)
            hop = pitch_edit.ANALYSIS_HOP_S * sample_rate
            f0 = raw_f0(audio, sample_rate, hop, a, device)
            progress(1.0)
            return pitch_line(f0, hop / sample_rate, a["output"])

    def render(self, a: dict[str, Any], progress: Progress) -> dict[str, Any]:
        device = "cpu" if a.get("device") == "cpu" else self.device
        with self._lock:
            try:
                return self._render(a, progress, device)
            except RuntimeError:
                if device != "mps":
                    raise
                traceback.print_exc()
                print(" [MPS] render failed, falling back to CPU")
                self.device = "cpu"
                self._cache.drop("mps")
                return self._render(a, progress, "cpu")

    def _render(self, a: dict[str, Any], progress: Progress, device: str) -> dict[str, Any]:
        import librosa
        import numpy as np
        import soundfile as sf
        import torch
        from ddsp.core import upsample
        from ddsp.vocoder import Volume_Extractor
        from main_reflow import cross_fade, split

        started = time.time()
        model, vocoder, args = self._load_model(a["modelPath"], a.get("overrides") or {}, device)

        # load input
        full, sample_rate = librosa.load(a["input"], sr=None)
        if len(full.shape) > 1:
            full = librosa.to_mono(full)
        region = a.get("region")
        start_s = float(region[0]) if region else 0.0
        audio = full[int(region[0] * sample_rate): int(region[1] * sample_rate)] if region else full
        hop_size = args.data.block_size * sample_rate / args.data.sampling_rate
        win_size = args.data.volume_smooth_size * sample_rate / args.data.sampling_rate

        # f0 of the whole input (cached; a region takes its frames from it, so the pitch line is
        # the one the pitch editor shows), the user's pitch edits, then the key change
        f0_min = float(a.get("f0Min", 50))
        raw = raw_f0(full, sample_rate, hop_size, a, device)
        f0 = pitch_edit.interp_unvoiced(raw, f0_min)
        n_frames = int(len(audio) // hop_size) + 1
        first = int(round(start_s * sample_rate / hop_size)) if region else 0
        f0 = f0[first: first + n_frames]
        if len(f0) < n_frames:
            f0 = np.pad(f0, (0, n_frames - len(f0)), mode="edge")
        f0 = pitch_edit.apply(f0, pitch_edit.parse(a.get("pitchEdit")), hop_size / sample_rate,
                              first * hop_size / sample_rate)
        progress(0.15)
        f0 = torch.from_numpy(f0).float().to(device).unsqueeze(-1).unsqueeze(0)

        # key change
        f0 = f0 * 2 ** (float(a.get("key", 0)) / 12)

        # formant change
        formant_shift_key = torch.from_numpy(np.array([[float(a.get("formant", 0))]])).float().to(device)

        # vocal register change
        if vocoder.vocoder.h.pc_aug:
            vocal_register_factor = 2 ** (float(a.get("register", 0)) / 12)
        else:
            print("Vocal register shift is not supported for current vocoder!")
            vocal_register_factor = 1

        # volume and mask
        volume_extractor = Volume_Extractor(hop_size, win_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(a.get("threshold", -60)) / 20)).astype("float")
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(device).unsqueeze(-1).unsqueeze(0)

        units_encoder = self._units_encoder(args, device)

        # speaker id or mix-speaker dictionary
        spk_mix = a.get("spkMix")
        spk_mix_dict = {int(k): float(v) for k, v in spk_mix.items()} if spk_mix else None
        spk_id = torch.LongTensor(np.array([[int(a.get("spkId", 1))]])).to(device)

        # sampling settings ('auto' semantics of main_reflow when not given)
        method = a.get("method") or args.infer.method
        infer_step = int(a["steps"]) if a.get("steps") is not None else args.infer.infer_step
        if a.get("tStart") is None:
            t_start = float(args.model.t_start) if args.model.t_start is not None else 0.0
        else:
            t_start = float(a["tStart"])
            if args.model.t_start is not None and t_start < args.model.t_start:
                t_start = args.model.t_start
        if infer_step < 0:
            raise ValueError("infer step cannot be negative!")

        # forward and save the output
        result = np.zeros(0)
        current_length = 0
        segments = split(audio, sample_rate, hop_size)
        print("Cut the input audio into " + str(len(segments)) + " slices")
        with torch.no_grad():
            for i, segment in enumerate(segments):
                start_frame = segment[0]
                seg_input = torch.from_numpy(segment[1]).float().unsqueeze(0).to(device)
                seg_units = units_encoder.encode(seg_input, sample_rate, hop_size)
                seg_f0 = f0[:, start_frame: start_frame + seg_units.size(1), :]
                seg_volume = volume[:, start_frame: start_frame + seg_units.size(1), :]
                seg_mel = model(
                    seg_units, seg_f0 / vocal_register_factor, seg_volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
                    aug_shift=formant_shift_key, vocoder=vocoder, infer_step=infer_step, method=method,
                    t_start=t_start, use_tqdm=False,
                )
                seg_output = vocoder.infer(seg_mel, seg_f0)
                seg_output *= mask[:, start_frame * args.data.block_size: (start_frame + seg_units.size(1)) * args.data.block_size]
                seg_output = seg_output.squeeze().cpu().numpy()

                silent_length = round(start_frame * args.data.block_size) - current_length
                if silent_length >= 0:
                    result = np.append(result, np.zeros(silent_length))
                    result = np.append(result, seg_output)
                else:
                    result = cross_fade(result, seg_output, current_length + silent_length)
                current_length = current_length + silent_length + len(seg_output)
                progress(0.15 + 0.85 * (i + 1) / max(1, len(segments)))
        os.makedirs(os.path.dirname(os.path.abspath(a["output"])), exist_ok=True)
        sf.write(a["output"], result, args.data.sampling_rate)
        return {"output": a["output"], "duration": len(result) / args.data.sampling_rate, "device": device,
                "seconds": round(time.time() - started, 3), "segments": len(segments)}


def raw_f0(audio: Any, sample_rate: float, hop_size: float, a: dict[str, Any], device: str) -> Any:
    """Raw f0 (Hz per frame, 0 = unvoiced) of ``audio``, cached per content + extractor settings."""
    import numpy as np
    from ddsp.vocoder import F0_Extractor

    extractor = a.get("f0Extractor", "rmvpe")
    f0_min, f0_max = float(a.get("f0Min", 50)), float(a.get("f0Max", 1100))
    md5 = hashlib.md5(np.ascontiguousarray(audio).tobytes()).hexdigest()
    cache_dir = a.get("f0CacheDir")
    cache_file = os.path.join(cache_dir, f"raw_{extractor}_{hop_size}_{f0_min}_{f0_max}_{md5}.npy") if cache_dir else None
    if cache_file and os.path.exists(cache_file):
        return np.load(cache_file, allow_pickle=False)
    f0 = F0_Extractor(extractor, sample_rate, hop_size, f0_min, f0_max).extract(audio, uv_interp=False, device=device)
    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_file, f0, allow_pickle=False)
    return f0


def pitch_line(f0: Any, frame_s: float, output: str) -> dict[str, Any]:
    """Saves the raw f0 for the backend (which builds the editor's notes and key from it)."""
    import numpy as np

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    np.save(output, np.asarray(f0, dtype=np.float32), allow_pickle=False)
    return {"output": output, "frameS": frame_s, "frames": int(len(f0))}


class FakeWorker:
    """``--fake-engine``: output = input with a simple pitch shift and gain change."""

    device = "cuda"  # pretends to have the fake GPU the backend reports in fake mode

    def unload(self) -> None:
        pass

    def pitch(self, a: dict[str, Any], progress: Progress) -> dict[str, Any]:
        """WORLD's pitch tracker (DIO + StoneMask) instead of the engine's extractors."""
        import numpy as np
        import pyworld
        import soundfile as sf

        data, sr = sf.read(a["input"], dtype="float64", always_2d=True)
        audio = np.ascontiguousarray(data.mean(axis=1))
        frame_s = pitch_edit.ANALYSIS_HOP_S
        f0_min, f0_max = float(a.get("f0Min", 50)), float(a.get("f0Max", 1100))
        f0, t = pyworld.dio(audio, sr, f0_floor=f0_min, f0_ceil=f0_max, frame_period=frame_s * 1000)
        f0 = pyworld.stonemask(audio, f0, t, sr)
        progress(1.0)
        return pitch_line(f0, frame_s, a["output"])

    def render(self, a: dict[str, Any], progress: Progress) -> dict[str, Any]:
        import numpy as np
        import soundfile as sf

        data, sr = sf.read(a["input"], dtype="float32", always_2d=True)
        audio = data.mean(axis=1)
        region = a.get("region")
        if region:
            audio = audio[int(region[0] * sr): int(region[1] * sr)]
        if sr != 44100:
            import soxr

            audio = soxr.resample(audio, sr, 44100).astype(np.float32)
            sr = 44100
        key = float(a.get("key", 0))
        steps = 10
        seconds = len(audio) / sr
        for i in range(steps):
            time.sleep(min(0.3, 0.01 + seconds * 0.002))
            progress(0.9 * (i + 1) / steps)
        if key:
            import librosa

            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=key)
        audio = audio * 10 ** (-2 / 20)  # a little quieter so the difference is audible
        sf.write(a["output"], np.clip(audio, -1, 1), sr)
        progress(1.0)
        device = "cpu" if a.get("device") == "cpu" else self.device
        return {"output": a["output"], "duration": len(audio) / sr, "device": device, "seconds": 0.0, "segments": 1}


def classify(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, BadCheckpoint):
        return "render.bad_checkpoint", "The checkpoint could not be read"
    text = f"{type(exc).__name__}: {exc}"
    if "out of memory" in text.lower():
        return "engine.out_of_memory", "The GPU ran out of memory"
    if isinstance(exc, FileNotFoundError):
        name = str(getattr(exc, "filename", "") or "")
        if name.endswith((".pt", ".ckpt", ".bin", ".pth")):
            return "engine.missing_model", "A model file is missing"
        return "render.input_missing", "A file needed for rendering is missing"
    return "render.failed", str(exc) or type(exc).__name__


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--fake", action="store_true")
    opts = parser.parse_args(argv)

    proto = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    os.dup2(2, 1)  # the engine's prints must not corrupt the protocol
    sys.stdout = sys.stderr

    def emit(obj: dict[str, Any]) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    worker = FakeWorker() if opts.fake else EngineWorker(opts.engine)
    emit({"type": "ready", "device": worker.device})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        rid, cmd = req.get("id"), req.get("cmd")
        if cmd == "shutdown":
            emit({"id": rid, "type": "result", "data": {}})
            break
        if cmd == "ping":
            emit({"id": rid, "type": "result", "data": {"device": worker.device}})
        elif cmd == "unload":
            worker.unload()
            emit({"id": rid, "type": "result", "data": {}})
        elif cmd in ("render", "pitch"):
            last = [0.0]

            def progress(value: float, rid: Any = rid) -> None:
                now = time.monotonic()
                if now - last[0] >= 0.2 or value >= 1.0:
                    last[0] = now
                    emit({"id": rid, "type": "progress", "value": round(value, 4)})

            try:
                handler = worker.render if cmd == "render" else worker.pitch
                emit({"id": rid, "type": "result", "data": handler(req["args"], progress)})
            except Exception as exc:  # noqa: BLE001 - reported to the backend, worker stays alive
                code, message = classify(exc)
                traceback.print_exc()
                emit({"id": rid, "type": "error", "code": code, "message": message, "detail": traceback.format_exc()})
        else:
            emit({"id": rid, "type": "error", "code": "render.bad_request", "message": f"unknown cmd {cmd}", "detail": ""})


if __name__ == "__main__":
    main()
