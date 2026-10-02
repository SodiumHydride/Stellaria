"""FastAPI application factory: routers, token auth, CORS, error shape, lifespan."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import traceback
from contextlib import asynccontextmanager
from typing import Any, Callable
from urllib.parse import parse_qs

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Receive, Scope, Send

from . import __version__
from .api import characters, events, files, jobs, models, projects, resources, system
from .context import AppContext, Options
from .errors import ApiError

log = logging.getLogger(__name__)

OPEN_PATHS = {"/api/health"}
TAURI_ORIGINS = r"^(tauri://localhost|https?://tauri\.localhost)$"


def error_body(code: str, message: str, detail: str = "", params: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "detail": detail, "params": params or {}}}


class TokenAuth:
    """``Authorization: Bearer <token>`` or ``?token=`` on every ``/api`` path except health."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or not scope["path"].startswith("/api") \
                or scope["path"] in OPEN_PATHS or scope.get("method") == "OPTIONS" or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401})
            return
        response = JSONResponse(error_body("auth.invalid_token", "Missing or invalid token"), status_code=401)
        await response(scope, receive, send)

    def _authorized(self, scope: Scope) -> bool:
        for name, value in scope.get("headers", []):
            if name == b"authorization":
                text = value.decode("latin-1")
                if text.startswith("Bearer ") and text[7:].strip() == self.token:
                    return True
        query = parse_qs(scope.get("query_string", b"").decode("latin-1"))
        return query.get("token", [None])[0] == self.token


async def _gpu_ticker(ctx: AppContext) -> None:
    """Push ``gpu`` events every 2 s while at least one client subscribed."""
    while True:
        await asyncio.sleep(2.0)
        if ctx.hub.has_subscribers("gpu"):
            gpus = await asyncio.to_thread(ctx.system.gpus)
            ctx.hub.publish("gpu", [g.wire() for g in gpus], topic="gpu")


def create_app(options: Options, on_ready: Callable[[], None] | None = None,
               ctx: AppContext | None = None) -> FastAPI:
    ctx = ctx or AppContext(options)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        ctx.hub.attach_loop(asyncio.get_running_loop())
        await asyncio.to_thread(ctx.start)
        ticker = asyncio.create_task(_gpu_ticker(ctx))
        if on_ready is not None:
            on_ready()
        try:
            yield
        finally:
            ticker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ticker
            await asyncio.to_thread(ctx.stop)

    app = FastAPI(title="Stellaria backend", version=__version__, lifespan=lifespan,
                  docs_url="/docs" if options.dev else None, redoc_url=None,
                  openapi_url="/openapi.json" if options.dev else None)
    app.state.ctx = ctx
    for module in (system, characters, models, projects, jobs, resources, files, events):
        app.include_router(module.router, prefix="/api")
    from .api import plugins as plugins_api
    app.include_router(plugins_api.router, prefix="/api")
    from .api import vst as vst_api
    app.include_router(vst_api.router, prefix="/api")
    from .api import presets as presets_api
    app.include_router(presets_api.router, prefix="/api")

    app.add_middleware(TokenAuth, token=options.token)
    cors: dict[str, Any] = {"allow_origins": ["*"]} if options.dev else {"allow_origin_regex": TAURI_ORIGINS}
    app.add_middleware(CORSMiddleware, allow_methods=["*"], allow_headers=["*"], **cors)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(error_body(exc.code, exc.message, exc.detail, exc.params), status_code=exc.status)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if exc.status_code == 404 else f"http.{exc.status_code}"
        return JSONResponse(error_body(code, str(exc.detail)), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        detail = json.dumps(exc.errors(), default=str, ensure_ascii=False)
        return JSONResponse(error_body("request.invalid", "Invalid request", detail), status_code=422)

    @app.exception_handler(Exception)
    async def _internal_error(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error")
        return JSONResponse(error_body("internal", str(exc) or type(exc).__name__, traceback.format_exc()),
                            status_code=500)

    return app
