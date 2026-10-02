"""A stand-in for ``python -m stellaria_backend.dsp.run <job.json>`` (docs/canvas-v3.md §6) used by
the graph tests: same JSON-lines protocol and offsets rules, trivial DSP.

Extra knobs for tests: ``params.fail`` → an error line; an fx slot with ``params.crash`` →
names the plugin in a stage line, then dies like a segfaulting plugin."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def read(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, sr
    return x


def write(out_dir, port, x, offset):
    path = Path(out_dir) / f"{port}.wav"
    sf.write(str(path), x, SR, subtype="FLOAT")
    return {"path": str(path), "offset": round(offset, 6), "duration": round(len(x) / SR, 6)}


def unique(path: Path) -> Path:
    if not path.exists():
        return path
    n = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        if not candidate.exists():
            return candidate
        n += 1


def main():
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    op, params, out_dir = job["op"], job["params"], job["outputDir"]
    inputs = [(read(i["path"]), float(i["offset"])) for i in job["inputs"]]
    emit({"progress": 0.1, "stage": f"dsp.{op}"})
    emit({"log": f"op {op}, bpm {job.get('bpm')}, names {json.dumps(job.get('names'), ensure_ascii=False)}"})
    if params.get("fail"):
        emit({"error": {"code": "dsp.bad_params", "message": "nope", "params": {"field": "fail"}}})
        return 1
    outputs, result = {}, {}
    x, off = inputs[0]
    if op == "trim":
        a, b = int(params["start"] * SR), int(params["end"] * SR)
        outputs["out"] = write(out_dir, "out", x[a:b], off + params["start"])
    elif op == "split":
        bounds = [0.0, *params["cuts"], len(x) / SR]
        for i in range(len(bounds) - 1):
            outputs[f"part{i + 1}"] = write(out_dir, f"part{i + 1}", x[int(bounds[i] * SR):int(bounds[i + 1] * SR)], off + bounds[i])
    elif op == "pitch":
        tempo = float(params.get("tempo") or 1.0)
        outputs["out"] = write(out_dir, "out", x, off / tempo)
        result = {"engine": "builtin"}
    elif op == "fx":
        for slot in params.get("chain") or []:
            if slot.get("kind") == "vst" and not slot.get("bypass"):
                plugin = job["vstPlugins"][slot["pluginId"]]
                emit({"progress": 0.5, "stage": "dsp.fx", "plugin": plugin["name"]})
                if slot.get("params", {}).get("crash"):
                    os._exit(139)
        outputs["out"] = write(out_dir, "out", x * 0.5, off)
    elif op == "comp":
        base = min(o for _, o in inputs)
        end = max(s["end"] for s in params["segments"])
        out = np.zeros((int(round((end - base) * SR)), x.shape[1]), dtype=np.float32)
        for seg in params["segments"]:
            if seg["input"] < 0:
                continue
            src, src_off = inputs[seg["input"]]
            a, b = int(round((seg["start"] - base) * SR)), int(round((seg["end"] - base) * SR))
            sa = int(round((seg["start"] - src_off) * SR))
            piece = src[max(0, sa): max(0, sa) + (b - a)]
            out[a:a + len(piece), : piece.shape[1]] = piece
        outputs["out"] = write(out_dir, "out", out, base)
    elif op == "concat":
        outputs["out"] = write(out_dir, "out", np.concatenate([i[0] for i in inputs]), 0.0)
    elif op == "export":
        names = job["names"]
        stem = params["fileName"].format(**{k: names.get(k, "") for k in ("project", "song", "voice", "label", "date", "index")})
        folder = Path(params.get("folder") or job["defaultFolder"])
        if not params.get("folder"):
            folder.mkdir(parents=True, exist_ok=True)
        dest = unique(folder / f"{stem}.{params['format']}")
        data = np.concatenate([np.zeros((int(off * SR), x.shape[1]), dtype=np.float32), x]) if params.get("keepPosition") else x
        sf.write(str(dest), data, SR, format=params["format"].upper())
        result = {"path": str(dest)}
    else:
        emit({"error": {"code": "dsp.unknown_op", "message": op}})
        return 1
    emit({"progress": 1.0, "stage": f"dsp.{op}"})
    emit({"result": {"outputs": outputs, "result": result}})
    return 0


if __name__ == "__main__":
    sys.exit(main())
