"""``/api/events`` WebSocket: server → client events, client → server (un)subscribe."""

from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import __version__
from ..context import AppContext

router = APIRouter()


@router.websocket("/events")
async def events(ws: WebSocket) -> None:
    ctx: AppContext = ws.app.state.ctx
    await ws.accept()
    client = ctx.hub.register()
    await ws.send_json({"type": "hello", "data": {"version": __version__}})

    async def sender() -> None:
        while True:
            message = await client.queue.get()
            await ws.send_json(message)
            if client.overflow:
                await ws.close(code=1013)  # try again later: the client fell too far behind
                return

    send_task = asyncio.create_task(sender())
    try:
        while True:
            data = await ws.receive_json()
            topics = data.get("topics") if isinstance(data, dict) else None
            if not isinstance(topics, list):
                continue
            if data.get("type") == "subscribe":
                ctx.hub.subscribe(client, topics)
            elif data.get("type") == "unsubscribe":
                ctx.hub.unsubscribe(client, topics)
    except (WebSocketDisconnect, RuntimeError, ValueError):
        pass
    finally:
        ctx.hub.unregister(client)
        send_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await send_task
