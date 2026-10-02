"""tfevents reading: incremental polls, partial records, resume overlap, downsampling, audio, mel."""

from __future__ import annotations

import base64
import io
import time

import numpy as np
import pytest
import soundfile as sf
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from stellaria_backend.engine.tfevents import TfEventsReader, downsample
from stellaria_backend.engine.validation import N_MELS, audio_file, log_mel, mel_payload, snapshot
from stellaria_backend.fake.trainer import wav_bytes
from synth import note


def scalar_event(step: int, values: dict[str, float], wall: float) -> Event:
    return Event(wall_time=wall, step=step,
                 summary=Summary(value=[Summary.Value(tag=k, simple_value=v) for k, v in values.items()]))


def audio_event(step: int, tag: str, x: np.ndarray, wall: float) -> Event:
    audio = Summary.Audio(sample_rate=44100, num_channels=1, length_frames=len(x), encoded_audio_string=wav_bytes(x, 44100),
                          content_type="audio/wav")
    return Event(wall_time=wall, step=step, summary=Summary(value=[Summary.Value(tag=tag, audio=audio)]))


def test_incremental_reading_and_queries(tmp_path):
    logdir = tmp_path / "logs"
    w = EventFileWriter(str(logdir))
    t0 = 1_700_000_000.0
    for s in range(1, 101):
        w.add_event(scalar_event(s, {"train/loss": 1.0 / s, "train/lr": 5e-4}, t0 + s * 0.5))
    w.flush()
    reader = TfEventsReader(logdir)
    r = reader.poll()
    assert r.last_step == 100 and r.scalars["train/loss"] == (100, pytest.approx(1.0 / 100)) and r.new_validation_steps == []
    assert reader.tags() == ["train/loss", "train/lr"]
    assert abs(reader.speed() - 2.0) < 1e-6  # 1 step per 0.5 s
    assert abs(reader.training_time() - 49.5) < 1e-6

    gt, pred = note(1.0, 220), note(1.0, 230)
    for s in range(101, 151):
        w.add_event(scalar_event(s, {"train/loss": 1.0 / s}, t0 + s * 0.5))
    w.add_event(audio_event(150, "1/clip a/gt.wav", gt, t0 + 76))
    w.add_event(audio_event(150, "1/clip a/pred.wav", pred, t0 + 76))
    w.add_event(scalar_event(150, {"validation/loss": 0.3}, t0 + 76))
    w.flush()
    r = reader.poll()
    assert r.last_step == 150 and set(r.scalars) == {"train/loss", "validation/loss"}
    assert r.new_validation_steps == [150] and reader.validation_names(150) == ["1/clip a"]
    data, sr = sf.read(io.BytesIO(reader.audio_bytes(150, "1/clip a", "gt")))
    assert sr == 44100 and len(data) == len(gt) and np.max(np.abs(data - gt)) < 1e-3
    series = {s.tag: s for s in reader.scalars()}
    assert len(series["train/loss"].points) == 150 and series["validation/loss"].points[0][:2] == (150, pytest.approx(0.3))
    since = reader.scalars(["train/loss"], since=140)
    assert [p[0] for p in since[0].points] == list(range(141, 151))
    w.close()


def test_partial_record_is_retried(tmp_path):
    logdir = tmp_path / "logs"
    w = EventFileWriter(str(logdir))
    w.add_event(scalar_event(1, {"a": 1.0}, 1.0))
    w.add_event(scalar_event(2, {"a": 2.0}, 2.0))
    w.flush()
    w.close()
    path = next(logdir.glob("events.out.tfevents.*"))
    full = path.read_bytes()
    path.write_bytes(full[:-7])  # cut the last record
    reader = TfEventsReader(logdir)
    assert reader.poll().last_step == 1
    path.write_bytes(full)
    assert reader.poll().last_step == 2


def test_resumed_run_supersedes_overlap(tmp_path):
    logdir = tmp_path / "logs"
    first = EventFileWriter(str(logdir), filename_suffix=".run1")
    for s in range(1, 21):
        first.add_event(scalar_event(s, {"train/loss": 1.0}, 1000.0 + s))
    first.close()
    time.sleep(1.1)  # tfevents file names carry the creation second
    second = EventFileWriter(str(logdir), filename_suffix=".run2")
    for s in range(11, 31):
        second.add_event(scalar_event(s, {"train/loss": 2.0}, 2000.0 + s))
    second.close()
    reader = TfEventsReader(logdir)
    reader.poll()
    points = reader.scalars()[0].points
    assert [p[0] for p in points] == list(range(1, 31))
    assert all(v == 1.0 for s, v, _ in points if s <= 10) and all(v == 2.0 for s, v, _ in points if s > 10)
    assert abs(reader.training_time() - 38.0) < 1e-6


def test_downsample_keeps_extremes():
    points = [(i, float(np.sin(i / 50)), float(i)) for i in range(10000)]
    points[4321] = (4321, 99.0, 4321.0)
    out = downsample(points, 500)
    assert len(out) <= 502 and (4321, 99.0, 4321.0) in out and out[-1] == points[-1]
    assert [p[0] for p in out] == sorted(p[0] for p in out)


def test_validation_snapshot_and_mel(tmp_path):
    logdir = tmp_path / "exp" / "logs"
    w = EventFileWriter(str(logdir))
    gt = note(2.0, 220, amp=0.5)
    w.add_event(audio_event(500, "歌 1/gt.wav", gt, 10.0))
    w.add_event(audio_event(500, "歌 1/pred.wav", gt * 0.5, 10.0))
    w.close()
    reader = TfEventsReader(logdir)
    reader.poll()
    snap = snapshot("mdl_x", reader, None)
    assert snap.step == 500 and snap.samples[0].name == "歌 1"
    assert snap.samples[0].gt_url == "/models/mdl_x/validation/500/%E6%AD%8C%201/gt.wav"
    cache = tmp_path / "cache"
    assert audio_file(cache, reader, 500, "歌 1", "pred").is_file()
    payload = mel_payload(cache, reader, 500, "歌 1")
    frames = 1 + (len(gt) + 1536 - 2048) // 512  # vocoder framing (reflect pad 768 + 768)
    assert payload["bins"] == N_MELS and payload["frames"] == frames
    data = np.frombuffer(base64.b64decode(payload["data"]), dtype=np.uint8)
    gt_data = np.frombuffer(base64.b64decode(payload["gtData"]), dtype=np.uint8)
    assert data.size == gt_data.size == frames * N_MELS
    assert gt_data.mean() > data.mean()  # pred is 6 dB quieter
    mel = log_mel(gt)
    assert mel.shape == (frames, N_MELS) and -12 < mel.max() < 3.5 and mel.min() >= np.log(1e-5) - 1e-6
