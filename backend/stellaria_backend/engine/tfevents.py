"""Incremental tfevents reader (TFRecord framing + tensorboard protos; no TensorFlow, no torch).

* Each event file is read from the last complete record onward; a half-written record at
  the end is retried on the next poll.
* Files are processed in creation order. When a resumed run logs a step that is ≤ the last
  known step of a series, the older tail is dropped (the new run supersedes it).
* Audio summaries (``<name>/gt.wav`` / ``<name>/pred.wav``) are indexed by file offset and
  only decoded on request.
"""

from __future__ import annotations

import bisect
import re
import struct
import threading
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..schemas import ScalarSeries

if TYPE_CHECKING:
    from ..context import AppContext

AUDIO_TAG = re.compile(r"^(?P<name>.+)/(?P<kind>gt|pred)\.wav$")


def _event_pb2():
    from tensorboard.compat.proto import event_pb2

    return event_pb2


@dataclass
class _Series:
    steps: array = field(default_factory=lambda: array("q"))
    values: array = field(default_factory=lambda: array("d"))
    walls: array = field(default_factory=lambda: array("d"))

    def add(self, step: int, value: float, wall: float) -> None:
        if self.steps and step <= self.steps[-1]:
            cut = bisect.bisect_left(self.steps, step)
            del self.steps[cut:], self.values[cut:], self.walls[cut:]
        self.steps.append(step)
        self.values.append(value)
        self.walls.append(wall)


@dataclass
class _File:
    path: Path
    offset: int = 0
    first_wall: float | None = None
    last_wall: float | None = None


@dataclass
class PollResult:
    scalars: dict[str, tuple[int, float]]  # tag → latest (step, value) among new points
    last_step: int | None
    new_validation_steps: list[int]


def _file_order(path: Path) -> tuple[float, str]:
    parts = path.name.split(".")
    try:
        return float(parts[3]), path.name  # events.out.tfevents.<unix time>.<host>...
    except (IndexError, ValueError):
        return path.stat().st_mtime, path.name


class TfEventsReader:
    def __init__(self, logdir: Path) -> None:
        self.logdir = logdir
        self._lock = threading.Lock()
        self._files: dict[str, _File] = {}
        self._order: list[str] = []
        self._series: dict[str, _Series] = {}
        self._audio: dict[int, dict[str, dict[str, tuple[str, int]]]] = {}

    def reset(self) -> None:
        with self._lock:
            self._files.clear()
            self._order.clear()
            self._series.clear()
            self._audio.clear()

    # ---- reading ---------------------------------------------------------------------
    def poll(self) -> PollResult:
        with self._lock:
            before = set(self.validation_steps_unlocked())
            new_scalars: dict[str, tuple[int, float]] = {}
            if self.logdir.is_dir():
                paths = sorted((p for p in self.logdir.glob("events.out.tfevents.*") if p.is_file()), key=_file_order)
                for path in paths:
                    key = str(path)
                    if key not in self._files:
                        self._files[key] = _File(path)
                        self._order.append(key)
                    self._read_file(self._files[key], new_scalars)
            after = self.validation_steps_unlocked()
            last = self._last_step_unlocked()
        return PollResult(new_scalars, last, [s for s in after if s not in before])

    def _read_file(self, f: _File, new_scalars: dict[str, tuple[int, float]]) -> None:
        event_pb2 = _event_pb2()
        try:
            size = f.path.stat().st_size
        except FileNotFoundError:
            return
        if size < f.offset:  # truncated/replaced: start over
            f.offset = 0
        if size == f.offset:
            return
        with open(f.path, "rb") as fh:
            fh.seek(f.offset)
            while True:
                start = fh.tell()
                header = fh.read(12)
                if len(header) < 12:
                    break
                (length,) = struct.unpack("<Q", header[:8])
                payload = fh.read(length)
                crc = fh.read(4)
                if len(payload) < length or len(crc) < 4:
                    break
                f.offset = fh.tell()
                event = event_pb2.Event()
                try:
                    event.ParseFromString(payload)
                except Exception:  # noqa: BLE001 - skip a corrupt record
                    continue
                if not event.HasField("summary"):
                    continue  # e.g. the file_version record written when the writer opens
                wall = float(event.wall_time)
                if f.first_wall is None:
                    f.first_wall = wall
                f.last_wall = wall
                for value in event.summary.value:
                    self._index_value(value, int(event.step), wall, str(f.path), start, new_scalars)
            fh.seek(f.offset)

    def _index_value(self, value, step: int, wall: float, file: str, offset: int,
                     new_scalars: dict[str, tuple[int, float]]) -> None:
        kind = value.WhichOneof("value")
        if kind == "simple_value":
            scalar = float(value.simple_value)
        elif kind == "tensor" and (value.tensor.float_val or value.tensor.double_val):
            scalar = float((value.tensor.float_val or value.tensor.double_val)[0])
        elif kind == "audio":
            m = AUDIO_TAG.match(value.tag)
            if m:
                self._audio.setdefault(step, {}).setdefault(m.group("name"), {})[m.group("kind")] = (file, offset)
            return
        else:
            return
        self._series.setdefault(value.tag, _Series()).add(step, scalar, wall)
        prev = new_scalars.get(value.tag)
        if prev is None or step >= prev[0]:
            new_scalars[value.tag] = (step, scalar)

    # ---- queries ---------------------------------------------------------------------
    def tags(self) -> list[str]:
        with self._lock:
            return sorted(self._series)

    def _last_step_unlocked(self) -> int | None:
        steps = [s.steps[-1] for s in self._series.values() if s.steps]
        return max(steps) if steps else None

    def last_step(self) -> int | None:
        with self._lock:
            return self._last_step_unlocked()

    def scalars(self, tags: list[str] | None = None, since: int | None = None, max_points: int = 1500) -> list[ScalarSeries]:
        with self._lock:
            out = []
            for tag in sorted(self._series):
                if tags and tag not in tags:
                    continue
                s = self._series[tag]
                lo = bisect.bisect_right(s.steps, since) if since is not None else 0
                points = list(zip(s.steps[lo:], s.values[lo:], s.walls[lo:]))
                out.append(ScalarSeries(tag=tag, points=downsample(points, max_points)))
            return out

    def speed(self, window: float = 60.0) -> float | None:
        """Steps per second over the last ``window`` seconds of the most frequent series."""
        with self._lock:
            series = max(self._series.values(), key=lambda s: len(s.steps), default=None)
            if series is None or len(series.steps) < 2:
                return None
            w1, s1 = series.walls[-1], series.steps[-1]
            i = bisect.bisect_left(series.walls, w1 - window, hi=len(series.walls) - 1)
            w0, s0 = series.walls[i], series.steps[i]
            if w1 - w0 < 1.0 or s1 <= s0:
                return None
            return (s1 - s0) / (w1 - w0)

    def last_wall(self) -> float | None:
        with self._lock:
            walls = [f.last_wall for f in self._files.values() if f.last_wall is not None]
            return max(walls) if walls else None

    def training_time(self) -> float:
        """Sum over runs (event files) of the time between their first and last event."""
        with self._lock:
            return sum((f.last_wall - f.first_wall) for f in self._files.values()
                       if f.first_wall is not None and f.last_wall is not None)

    def validation_steps_unlocked(self) -> list[int]:
        return sorted(s for s, names in self._audio.items()
                      if any("gt" in kinds and "pred" in kinds for kinds in names.values()))

    def validation_steps(self) -> list[int]:
        with self._lock:
            return self.validation_steps_unlocked()

    def validation_names(self, step: int) -> list[str]:
        with self._lock:
            names = self._audio.get(step, {})
            return sorted(n for n, kinds in names.items() if "gt" in kinds and "pred" in kinds)

    def audio_bytes(self, step: int, name: str, kind: str) -> bytes | None:
        with self._lock:
            loc = self._audio.get(step, {}).get(name, {}).get(kind)
        if loc is None:
            return None
        file, offset = loc
        event_pb2 = _event_pb2()
        with open(file, "rb") as fh:
            fh.seek(offset)
            (length,) = struct.unpack("<Q", fh.read(12)[:8])
            event = event_pb2.Event()
            event.ParseFromString(fh.read(length))
        for value in event.summary.value:
            if value.tag == f"{name}/{kind}.wav" and value.HasField("audio"):
                return bytes(value.audio.encoded_audio_string)
        return None


def downsample(points: list[tuple[int, float, float]], max_points: int) -> list[tuple[int, float, float]]:
    """Keep at most ~``max_points`` points; each bucket keeps its min and max (in step order)
    plus the very last point, so spikes and the current value survive."""
    if max_points <= 0 or len(points) <= max_points:
        return points
    buckets = max(1, max_points // 2)
    size = len(points) / buckets
    out: list[tuple[int, float, float]] = []
    for b in range(buckets):
        chunk = points[int(b * size): int((b + 1) * size)]
        if not chunk:
            continue
        lo = min(chunk, key=lambda p: p[1])
        hi = max(chunk, key=lambda p: p[1])
        out.extend(sorted({lo, hi}, key=lambda p: p[0]))
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


class MetricsStore:
    """One reader per model, shared by the training monitor and the API endpoints."""

    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self._readers: dict[str, TfEventsReader] = {}
        self._lock = threading.Lock()

    def reader(self, mid: str) -> TfEventsReader:
        with self._lock:
            reader = self._readers.get(mid)
            if reader is None:
                reader = self._readers[mid] = TfEventsReader(self.ctx.models.exp_dir(mid) / "logs")
            return reader

    def polled(self, mid: str) -> TfEventsReader:
        reader = self.reader(mid)
        reader.poll()
        return reader

    def forget(self, mid: str) -> None:
        with self._lock:
            self._readers.pop(mid, None)
