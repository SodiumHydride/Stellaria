"""Checkpoint inspection and base-model seeding. Runs in its own process (imports torch).

  inspect --config cfg.yaml --ckpt a.pt [--ckpt b.pt ...]
      → {"result": {"config": {...summary...}, "checkpoints": [{"path", "step", "ok", "error"}]}}
        or {"result": {"incompatible": "<reason>", ...}}
  seed --base base.pt --config model.yaml --out exp/model_0.pt
      → {"result": {"kept": n, "dropped": [names], "speakerDropped": bool}}

Stdout carries exactly one JSON line; the engine's prints go to stderr. ``--fake`` (fake engine)
checks configs only and reads steps from file names / fake checkpoint headers, without torch.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from typing import Any

SUPPORTED_ENCODERS = {"hubertsoft": 256, "contentvec768l12": 768, "contentvec768l12tta2x": 768}
VOCODERS = ("nsf-hifigan", "nsf-hifigan-log10")
MELS = 128
STEP_RE = re.compile(r"model_(\d+)\.pt$")


def load_yaml(path: str) -> dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def summarize(conf: dict[str, Any]) -> dict[str, Any]:
    data, model = conf.get("data") or {}, conf.get("model") or {}
    return {
        "type": model.get("type"), "encoder": data.get("encoder"), "encoderOutChannels": data.get("encoder_out_channels"),
        "samplingRate": data.get("sampling_rate"), "blockSize": data.get("block_size"),
        "vocoder": (conf.get("vocoder") or {}).get("type"), "f0Extractor": data.get("f0_extractor"),
        "nLayers": model.get("n_layers"), "nChans": model.get("n_chans"), "nAuxLayers": model.get("n_aux_layers"),
        "nAuxChans": model.get("n_aux_chans"), "usePitchAug": bool(model.get("use_pitch_aug")),
        "nSpk": int(model.get("n_spk") or 1), "gluType": model.get("glu_type"),
    }


def incompatibility(s: dict[str, Any]) -> str | None:
    """Why a config cannot be used with Stellaria's engine and pretrained models, or None."""
    if s["type"] != "RectifiedFlow":
        return "not_reflow"
    if s["samplingRate"] != 44100 or s["blockSize"] != 512:
        return "sample_rate"
    if s["vocoder"] not in VOCODERS:
        return "vocoder"
    if s["encoder"] not in SUPPORTED_ENCODERS or s["encoderOutChannels"] != SUPPORTED_ENCODERS[s["encoder"]]:
        return "encoder"
    if None in (s["nLayers"], s["nChans"]) or s["nSpk"] < 1:
        return "model"
    return None


def build_model(conf: dict[str, Any]):
    from reflow.vocoder import DotDict, Unit2Wav

    args = DotDict(conf)
    # same defaults as reflow.vocoder.load_model_vocoder
    glu = args.model.glu_type if args.model.glu_type is not None else "atanglu"
    return Unit2Wav(args.data.sampling_rate, args.data.block_size, args.model.win_length, args.data.encoder_out_channels,
                    args.model.n_spk, args.model.use_pitch_aug, MELS, args.model.n_aux_layers, args.model.n_aux_chans,
                    args.model.n_layers, args.model.n_chans, glu_type=glu)


def file_step(path: str, fake: bool) -> int | None:
    if fake:
        try:
            with open(path, "rb") as f:
                header = json.loads(f.readline())
            if isinstance(header, dict) and "global_step" in header:
                return int(header["global_step"])
        except (OSError, ValueError, UnicodeDecodeError):
            pass
    m = STEP_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else None


def inspect(opts: argparse.Namespace) -> dict[str, Any]:
    summary = summarize(load_yaml(opts.config))
    reason = incompatibility(summary)
    if reason:
        return {"incompatible": reason, "config": summary}
    model = None if opts.fake else build_model(load_yaml(opts.config))
    checkpoints = []
    for path in opts.ckpt:
        entry: dict[str, Any] = {"path": path, "step": file_step(path, opts.fake), "ok": True, "error": None}
        try:
            if opts.fake:
                if os.path.getsize(path) == 0:
                    raise ValueError("empty file")
            else:
                import torch

                ckpt = torch.load(path, map_location="cpu")
                if not isinstance(ckpt, dict) or "model" not in ckpt:
                    raise ValueError("not a DDSP-SVC checkpoint (no 'model' state dict)")
                model.load_state_dict(ckpt["model"])  # strict, like rendering does
                if ckpt.get("global_step") is not None:
                    entry["step"] = int(ckpt["global_step"])
        except Exception as exc:  # noqa: BLE001 - reported per checkpoint
            entry.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:500])
        checkpoints.append(entry)
    return {"config": summary, "checkpoints": checkpoints}


def seed(opts: argparse.Namespace) -> dict[str, Any]:
    """Write the base as the first checkpoint of a new model: step 0, no optimizer state, only
    tensors whose name and shape exist in the new model (speaker embeddings go when n_spk differs)."""
    if opts.fake:  # a fake-engine checkpoint at step 0 (manifest bases are never downloaded there)
        with open(opts.out, "wb") as f:
            f.write(json.dumps({"global_step": 0, "fake": True, "base": os.path.basename(opts.base)}).encode() + b"\n")
        return {"kept": 0, "dropped": [], "speakerDropped": False, "missing": [], "fake": True}
    import torch

    base = torch.load(opts.base, map_location="cpu")["model"]
    target = build_model(load_yaml(opts.config)).state_dict()
    kept, dropped = {}, []
    for name, tensor in base.items():
        if name in target and tuple(target[name].shape) == tuple(tensor.shape):
            kept[name] = tensor
        else:
            dropped.append(name)
    tmp = opts.out + ".tmp"
    torch.save({"global_step": 0, "model": kept}, tmp)
    os.replace(tmp, opts.out)
    return {"kept": len(kept), "dropped": dropped, "speakerDropped": any("spk_embed" in n for n in dropped),
            "missing": sorted(set(target) - set(kept))}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["inspect", "seed"])
    parser.add_argument("--engine", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", action="append", default=[])
    parser.add_argument("--base")
    parser.add_argument("--out")
    parser.add_argument("--fake", action="store_true")
    opts = parser.parse_args(argv)
    proto = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        if not opts.fake:
            sys.path.insert(0, opts.engine)
            os.chdir(opts.engine)
        result = inspect(opts) if opts.command == "inspect" else seed(opts)
        proto.write(json.dumps({"result": result}) + "\n")
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        proto.write(json.dumps({"error": {"code": "model.inspect_failed", "message": str(exc) or type(exc).__name__,
                                          "detail": traceback.format_exc()}}) + "\n")
        proto.flush()
        sys.exit(1)
    proto.flush()


if __name__ == "__main__":
    main()
