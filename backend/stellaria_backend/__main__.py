"""``python -m stellaria_backend`` — see docs/api.md (Process)."""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import secrets
import socket
import sys
from pathlib import Path

from . import __version__


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="stellaria_backend")
    p.add_argument("--library", required=True, type=Path, help="library root (created if missing)")
    p.add_argument("--data", required=True, type=Path, help="app data dir (models/, logs/, preferences.json)")
    p.add_argument("--engine", required=True, type=Path, help="DDSP-SVC root (contains train_reflow.py)")
    p.add_argument("--resources", required=True, type=Path, help="manifest.json")
    p.add_argument("--uv", type=Path, default=None, help="uv binary (used to install add-on packages)")
    p.add_argument("--port", type=int, default=None, help="0 = pick a free port")
    p.add_argument("--token", default=None)
    p.add_argument("--dev", action="store_true", help="port 8765 unless --port, token 'dev', CORS *")
    p.add_argument("--fake-engine", action="store_true", help="simulate preprocess/train/render")
    return p.parse_args(argv)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    stderr = logging.StreamHandler(sys.stderr)
    stderr.setFormatter(fmt)
    file = logging.handlers.RotatingFileHandler(log_dir / "backend.log", maxBytes=5 << 20, backupCount=3, encoding="utf-8")
    file.setFormatter(fmt)
    root.handlers[:] = [stderr, file]


def bind(port: int) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if sys.platform != "win32":  # on Windows SO_REUSEADDR would allow two servers on one port
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    sock.listen(128)
    sock.set_inheritable(False)
    return sock


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    import uvicorn

    from .app import create_app
    from .context import Options

    for d in (args.library, args.data):
        d.mkdir(parents=True, exist_ok=True)
    setup_logging(args.data / "logs")
    port = args.port if args.port is not None else (8765 if args.dev else 0)
    token = args.token or ("dev" if args.dev else secrets.token_urlsafe(24))
    sock = bind(port)
    options = Options(library=args.library.resolve(), data=args.data.resolve(), engine=args.engine.resolve(),
                      resources=args.resources.resolve(), uv=args.uv, port=sock.getsockname()[1], token=token,
                      dev=args.dev, fake_engine=args.fake_engine)
    real_stdout = sys.stdout

    def ready() -> None:
        real_stdout.write("STELLARIA_READY " + json.dumps({"port": options.port, "token": token, "version": __version__}) + "\n")
        real_stdout.flush()
        sys.stdout = sys.stderr  # stdout carries exactly one line

    app = create_app(options, on_ready=ready)
    server = uvicorn.Server(uvicorn.Config(app, log_config=None, access_log=False, lifespan="on",
                                           timeout_graceful_shutdown=30))

    def request_exit() -> None:
        server.should_exit = True

    app.state.request_exit = request_exit
    logging.getLogger(__name__).info("starting on 127.0.0.1:%s (dev=%s, fake=%s)", options.port, args.dev, args.fake_engine)
    server.run(sockets=[sock])


if __name__ == "__main__":
    main()
