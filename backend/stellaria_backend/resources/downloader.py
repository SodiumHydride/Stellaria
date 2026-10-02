"""Resumable downloads with ordered mirrors.

* Sources are tried in order; a failure (network, HTTP error, size/sha256 mismatch) falls
  over to the next one.
* Partial data lives in ``<dest>.part``; a later attempt continues with an HTTP Range
  request (``206``) or restarts when the server ignores ranges (``200``).
* Region ``auto`` orders sources by measured latency (reachable first).
"""

from __future__ import annotations

import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit

import httpx

from ..errors import ApiError, JobCancelled
from .manifest import Source

CHUNK = 1 << 16
PROBE_TTL = 600.0


@dataclass
class Progress:
    bytes_done: int
    total: int | None
    speed: float | None
    source: str


ProgressFn = Callable[[Progress], None]


def order_sources(sources: list[Source], region: str, latency: dict[str, float | None] | None = None) -> list[Source]:
    """``cn`` → cn mirrors first; ``global`` → global first; ``auto`` → by measured latency."""
    usable = [s for s in sources if s.url]
    if region == "auto" and latency is not None:
        def key(s: Source) -> tuple[int, float]:
            ms = latency.get(s.id)
            return (1, 0.0) if ms is None else (0, ms)
        return sorted(usable, key=key)
    preferred = "cn" if region == "cn" else "global"
    return sorted(usable, key=lambda s: 0 if s.region in (preferred, "any") else 1)


class LatencyProbe:
    """Measures round-trip time to each source's host once per ``PROBE_TTL``."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._results: dict[str, float | None] = {}
        self._measured_at = 0.0

    def measure(self, sources: list[Source], timeout: float = 3.0) -> dict[str, float | None]:
        with self._lock:
            if self._results and time.monotonic() - self._measured_at < PROBE_TTL and all(
                s.id in self._results for s in sources
            ):
                return dict(self._results)
        by_id: dict[str, str] = {}
        for s in sources:
            if s.url and s.id not in by_id:
                parts = urlsplit(s.url)
                by_id[s.id] = f"{parts.scheme}://{parts.netloc}/"

        def probe(item: tuple[str, str]) -> tuple[str, float | None]:
            sid, url = item
            start = time.perf_counter()
            try:
                with httpx.Client(timeout=timeout, follow_redirects=False) as client:
                    client.head(url)
                return sid, round((time.perf_counter() - start) * 1000, 1)
            except Exception:  # noqa: BLE001 - unreachable
                return sid, None

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = dict(pool.map(probe, by_id.items()))
        with self._lock:
            self._results.update(results)
            self._measured_at = time.monotonic()
            return dict(self._results)

    @property
    def last(self) -> dict[str, float | None]:
        with self._lock:
            return dict(self._results)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(
    sources: list[Source],
    dest: Path,
    *,
    size: int | None = None,
    sha256: str | None = None,
    on_progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    log: Callable[[str], None] = lambda _l: None,
    timeout: float = 30.0,
) -> Source:
    """Download to ``dest`` (atomically renamed from ``dest.part``); returns the source used."""
    if not sources:
        raise ApiError("download.no_source", "No download source available", status=409)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    errors: list[str] = []
    for source in sources:
        try:
            _fetch(source, part, size, on_progress, cancel, timeout)
            if size is not None and part.stat().st_size != size:
                raise _Mismatch(f"size {part.stat().st_size} != expected {size}")
            if sha256:
                digest = sha256_of(part)
                if digest.lower() != sha256.lower():
                    raise _Mismatch(f"sha256 {digest} != expected {sha256}")
            part.replace(dest)
            return source
        except JobCancelled:
            raise
        except _Mismatch as exc:
            part.unlink(missing_ok=True)  # corrupt data must not be resumed
            errors.append(f"{source.id}: {exc}")
            log(f"[{source.id}] verification failed: {exc}")
        except Exception as exc:  # noqa: BLE001 - try the next mirror
            errors.append(f"{source.id}: {exc}")
            log(f"[{source.id}] failed: {exc}")
    raise ApiError("download.failed", "All download sources failed", status=502, detail="\n".join(errors))


class _Mismatch(Exception):
    pass


def _fetch(source: Source, part: Path, size: int | None, on_progress: ProgressFn | None,
           cancel: threading.Event | None, timeout: float) -> None:
    have = part.stat().st_size if part.exists() else 0
    if size is not None and have == size:
        return
    if size is not None and have > size:
        part.unlink()
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    assert source.url
    with httpx.Client(timeout=httpx.Timeout(timeout, connect=15.0), follow_redirects=True) as client:
        with client.stream("GET", source.url, headers=headers) as resp:
            if resp.status_code == 416 and have:
                return  # nothing left to fetch; verification decides
            if resp.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {resp.status_code}")
            mode = "ab" if resp.status_code == 206 and have else "wb"
            if mode == "wb":
                have = 0
            length = resp.headers.get("content-length")
            total = size if size is not None else (have + int(length) if length else None)
            done = have
            window_start, window_bytes = time.monotonic(), 0
            speed: float | None = None
            with open(part, mode) as f:
                for chunk in resp.iter_bytes(CHUNK):
                    if cancel is not None and cancel.is_set():
                        raise JobCancelled()
                    f.write(chunk)
                    done += len(chunk)
                    window_bytes += len(chunk)
                    elapsed = time.monotonic() - window_start
                    if elapsed >= 0.5:
                        current = window_bytes / elapsed
                        speed = current if speed is None else 0.7 * speed + 0.3 * current
                        window_start, window_bytes = time.monotonic(), 0
                        if on_progress:
                            on_progress(Progress(done, total, speed, source.id))
            if on_progress:
                on_progress(Progress(done, total, speed, source.id))
