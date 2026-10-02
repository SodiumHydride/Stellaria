"""Fake ``preprocess.py``: tqdm progress over a few seconds and small feature files."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import yaml
from tqdm import tqdm


def _files(audio_dir: str) -> list[str]:
    out = []
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.endswith(".wav"):
                out.append(os.path.relpath(os.path.join(root, f), audio_dir))
    return sorted(out)


def run(path: str, seconds_per_file: float) -> None:
    files = _files(os.path.join(path, "audio"))
    aug = {}
    for name in tqdm(files, total=len(files)):
        time.sleep(seconds_per_file)
        for sub, shape in (("units", (50, 8)), ("f0", (50,)), ("volume", (50,)), ("mel", (50, 8)),
                           ("aug_mel", (50, 8)), ("aug_vol", (50,))):
            out = os.path.join(path, sub, name + ".npy")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            np.save(out, np.zeros(shape, dtype=np.float32))
        aug[name] = 0.0
    np.save(os.path.join(path, "pitch_aug_dict.npy"), aug)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("-j", "--workers", type=int, default=1)
    cmd = parser.parse_args()
    with open(cmd.config, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    print(f"Using {cmd.workers} worker processes (fake engine)")
    n = len(_files(os.path.join(args["data"]["train_path"], "audio"))) + len(
        _files(os.path.join(args["data"]["valid_path"], "audio")))
    per_file = min(0.2, 3.0 / max(1, n))
    run(args["data"]["train_path"], per_file)
    run(args["data"]["valid_path"], per_file)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
