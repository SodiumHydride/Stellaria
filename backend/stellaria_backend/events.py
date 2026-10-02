"""WebSocket event hub.

Events are published from any thread (job runners, monitors) and delivered to every
connected client's asyncio queue on the server loop. Most event types go to every client;
``gpu`` and ``job.log`` only reach clients that subscribed to the matching topic
(``gpu``, ``job.log:<jobId>``).

Throttling: ``publish(..., throttle_key=k, min_interval=s)`` sends at most one event per
``s`` seconds for key ``k``; the latest payload wins and is flushed by a background thread
when the interval has passed, so the final state is never lost.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

SUBSCRIPTION_TYPES = {"gpu", "job.log"}
MAX_QUEUE = 2000


@dataclass(eq=False)
class Client:
    queue: asyncio.Queue
    topics: set[str] = field(default_factory=set)
    overflow: bool = False


@dataclass
class _Pending:
    type: str
    data: Any
    topic: str | None
    due: float


class EventHub:
    def __init__(self) -> None:
        self._clients: set[Client] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_sent: dict[str, float] = {}
        self._pending: dict[str, _Pending] = {}
        self._cond = threading.Condition()
        self._stop = False
        self._flusher = threading.Thread(target=self._flush_loop, name="events-flusher", daemon=True)
        self._flusher.start()

    # ---- lifecycle -------------------------------------------------------------------
    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()

    # ---- clients ---------------------------------------------------------------------
    def register(self) -> Client:
        client = Client(queue=asyncio.Queue())
        with self._lock:
            self._clients.add(client)
        return client

    def unregister(self, client: Client) -> None:
        with self._lock:
            self._clients.discard(client)

    def subscribe(self, client: Client, topics: list[str]) -> None:
        with self._lock:
            client.topics.update(t for t in topics if isinstance(t, str))

    def unsubscribe(self, client: Client, topics: list[str]) -> None:
        with self._lock:
            client.topics.difference_update(topics)

    def has_subscribers(self, topic: str) -> bool:
        with self._lock:
            return any(topic in c.topics for c in self._clients)

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    # ---- publishing ------------------------------------------------------------------
    def publish(
        self,
        type_: str,
        data: Any,
        *,
        topic: str | None = None,
        throttle_key: str | None = None,
        min_interval: float = 0.0,
    ) -> None:
        """Send ``{type, data}``. ``topic`` restricts delivery to subscribers of that topic."""
        if throttle_key is None or min_interval <= 0:
            if throttle_key is not None:
                with self._cond:
                    self._pending.pop(throttle_key, None)
                    self._last_sent[throttle_key] = time.monotonic()
            self._deliver(type_, data, topic)
            return
        now = time.monotonic()
        with self._cond:
            last = self._last_sent.get(throttle_key, 0.0)
            if now - last >= min_interval and throttle_key not in self._pending:
                self._last_sent[throttle_key] = now
                send_now = True
            else:
                self._pending[throttle_key] = _Pending(type_, data, topic, last + min_interval)
                self._cond.notify()
                send_now = False
        if send_now:
            self._deliver(type_, data, topic)

    def _flush_loop(self) -> None:
        while True:
            due_items: list[_Pending] = []
            with self._cond:
                if self._stop:
                    return
                now = time.monotonic()
                for key, item in list(self._pending.items()):
                    if item.due <= now:
                        due_items.append(item)
                        del self._pending[key]
                        self._last_sent[key] = now
                next_due = min((p.due for p in self._pending.values()), default=None)
                if not due_items:
                    timeout = None if next_due is None else max(0.0, next_due - now)
                    self._cond.wait(timeout=timeout if timeout is not None else 1.0)
                    continue
            for item in due_items:
                self._deliver(item.type, item.data, item.topic)

    def _deliver(self, type_: str, data: Any, topic: str | None) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        message = {"type": type_, "data": data}
        with self._lock:
            targets = [c for c in self._clients if topic is None or topic in c.topics]
        for client in targets:
            try:
                loop.call_soon_threadsafe(self._enqueue, client, message)
            except RuntimeError:  # loop closed during shutdown
                return

    @staticmethod
    def _enqueue(client: Client, message: dict[str, Any]) -> None:
        if client.queue.qsize() >= MAX_QUEUE:
            # a client that does not read is dropped rather than growing memory forever
            client.overflow = True
            return
        client.queue.put_nowait(message)
