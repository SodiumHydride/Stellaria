"""Fake ``train_reflow.py``: plausible decaying losses, validation audio, dummy checkpoints.

Speed (steps/s) comes from ``STELLARIA_FAKE_SPEED``. Writes real tfevents with the same tags
as the engine (``train/*``, ``validation/*``, ``<name>/gt.wav`` and ``<name>/pred.wav``).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import time
import wave

import numpy as np
import soundfile as sf
import yaml
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

CKPT = re.compile(r"^model_(\d+)\.pt$")
VAL_SECONDS = 8.0


def wav_bytes(x: np.ndarray, sr: int) -> bytes:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class Writer:
    def __init__(self, logdir: str) -> None:
        self.w = EventFileWriter(logdir)
        self.last_flush = time.monotonic()

    def scalars(self, step: int, values: dict[str, float]) -> None:
        summary = Summary(value=[Summary.Value(tag=k, simple_value=float(v)) for k, v in values.items()])
        self.w.add_event(Event(wall_time=time.time(), step=step, summary=summary))

    def audio(self, step: int, tag: str, x: np.ndarray, sr: int) -> None:
        audio = Summary.Audio(sample_rate=sr, num_channels=1, length_frames=len(x),
                              encoded_audio_string=wav_bytes(x, sr), content_type="audio/wav")
        self.w.add_event(Event(wall_time=time.time(), step=step, summary=Summary(value=[Summary.Value(tag=tag, audio=audio)])))

    def maybe_flush(self, force: bool = False) -> None:
        if force or time.monotonic() - self.last_flush > 0.5:
            self.w.flush()
            self.last_flush = time.monotonic()


def latest_step(expdir: str) -> int:
    steps = [int(m.group(1)) for f in os.listdir(expdir) if (m := CKPT.match(f))]
    return max(steps) if steps else 0


def save_ckpt(expdir: str, step: int) -> None:
    path = os.path.join(expdir, f"model_{step}.pt")
    header = json.dumps({"global_step": step, "fake": True}).encode() + b"\n"
    with open(path, "wb") as f:
        f.write(header + os.urandom(64 * 1024))
    print(f" [*] model checkpoint saved: {path}")


def val_items(valid_path: str, sr: int) -> list[tuple[str, np.ndarray]]:
    audio_dir = os.path.join(valid_path, "audio")
    items = []
    for root, _, files in os.walk(audio_dir):
        for f in sorted(files):
            if f.endswith(".wav"):
                name_ext = os.path.join(root, f)[len(audio_dir) + 1:]
                data, file_sr = sf.read(os.path.join(root, f), dtype="float32", always_2d=True)
                x = data.mean(axis=1)[: int(VAL_SECONDS * file_sr)]
                if file_sr != sr:
                    import soxr

                    x = soxr.resample(x, file_sr, sr).astype(np.float32)
                items.append((os.path.splitext(name_ext)[0], x))
    return sorted(items)[:10]


def predicted(x: np.ndarray, step: int, rng: random.Random) -> np.ndarray:
    """gt made slightly worse: smoothing + noise that shrink as training goes on."""
    quality = 1 - math.exp(-step / 3000)
    k = max(1, int(6 * (1 - quality)) + 1)
    smooth = np.convolve(x, np.ones(k) / k, mode="same") if k > 1 else x
    noise = np.random.default_rng(rng.randint(0, 1 << 30)).normal(0, 0.02 * (1 - quality) + 0.002, len(x))
    return (0.92 * smooth + noise).astype(np.float32)


def losses(step: int, rng: random.Random) -> tuple[float, float]:
    ddsp = 0.18 + 1.6 * math.exp(-step / 1200) + rng.gauss(0, 0.015 + 0.06 * math.exp(-step / 4000))
    reflow = 0.05 + 0.45 * math.exp(-step / 5000) + rng.gauss(0, 0.01)
    return max(0.01, ddsp), max(0.005, reflow)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    cmd = parser.parse_args()
    with open(cmd.config, "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)
    expdir = args["env"]["expdir"]
    os.makedirs(expdir, exist_ok=True)
    with open(os.path.join(expdir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(args, f)
    train, sr = args["train"], args["data"]["sampling_rate"]
    interval_val, force_save = int(train["interval_val"]), int(train["interval_force_save"])
    max_steps = train.get("max_steps")
    speed = float(os.environ.get("STELLARIA_FAKE_SPEED", "25"))
    stop_flag = os.path.join(expdir, "STOP")
    log_path = os.path.join(expdir, "log_info.txt")

    def log_info(msg: str) -> None:
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    print(" > config:", cmd.config)
    print(" >    exp:", expdir)
    step = latest_step(expdir)
    if step or os.path.exists(os.path.join(expdir, "model_0.pt")):  # model_0.pt: a base-model seed
        print(" [*] restoring model from", os.path.join(expdir, f"model_{step}.pt"))
    items = val_items(args["data"]["valid_path"], sr)
    writer = Writer(os.path.join(expdir, "logs"))
    rng = random.Random(step)
    time.sleep(min(1.0, 5.0 / speed))  # "loading data"
    log_info("======= start training (fake engine) =======")
    lr0 = float(train["lr"])
    started = time.time()
    while True:
        step += 1
        time.sleep(1.0 / speed)
        ddsp, reflow = losses(step, rng)
        lr = lr0 * 0.9 ** max((step - 2) // 4000, 0)
        writer.scalars(step, {"train/loss": ddsp + reflow, "train/ddsp_loss": ddsp, "train/reflow_loss": reflow,
                              "train/lr": lr})
        if step % 10 == 0:
            log_info(f"epoch: {step // 40} | {step % 40:3d}/ 40 | {expdir} | batch/s: {speed:.2f} | lr: {lr:.6} | "
                     f"loss: {ddsp + reflow:.3f} | time: {time.time() - started:.1f}s | step: {step}")
        if step % interval_val == 0:
            save_ckpt(expdir, step)
            last_val = step - interval_val
            if last_val % force_save != 0 and os.path.exists(os.path.join(expdir, f"model_{last_val}.pt")):
                os.remove(os.path.join(expdir, f"model_{last_val}.pt"))
            print(" [*] testing...")
            for name, x in items:
                writer.audio(step, f"{name}/gt.wav", x, sr)
                writer.audio(step, f"{name}/pred.wav", predicted(x, step, rng), sr)
            vd, vr = losses(step, rng)
            writer.scalars(step, {
                "validation/mel_val_mse": 0.3 + 2.0 * math.exp(-step / 2500), "validation/mel_val_snr": 12 + 8 * (1 - math.exp(-step / 4000)),
                "validation/mel_val_psnr": 20 + 9 * (1 - math.exp(-step / 4000)), "validation/mel_val_sisnr": 10 + 7 * (1 - math.exp(-step / 4000)),
                "validation/loss": vd + vr + 0.05, "validation/ddsp_loss": vd + 0.04, "validation/reflow_loss": vr + 0.01,
            })
            log_info(f" --- <validation> --- \nloss: {vd + vr + 0.05:.3f}. ")
            writer.maybe_flush(force=True)
        writer.maybe_flush()
        reached = max_steps is not None and step >= int(max_steps)
        if reached or (step % 10 == 0 and os.path.exists(stop_flag)):
            if step % interval_val != 0:
                save_ckpt(expdir, step)
            if os.path.exists(stop_flag):
                os.remove(stop_flag)
            log_info(f" [*] training stopped at step {step}")
            writer.maybe_flush(force=True)
            writer.w.close()
            return


if __name__ == "__main__":
    main()
