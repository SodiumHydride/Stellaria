"""``AppContext``: process options plus every service, wired once per backend process."""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .events import EventHub
from .jobs import JobManager
from .library.characters import CharacterStore
from .library.models import ModelStore
from .library.projects import ProjectStore
from .library.store import Library
from .preferences import PreferencesStore
from .util.jsonio import read_json, write_json

log = logging.getLogger(__name__)


@dataclass
class Options:
    library: Path
    data: Path
    engine: Path
    resources: Path
    uv: Path | None = None
    port: int = 0
    token: str = ""
    dev: bool = False
    fake_engine: bool = False
    # not part of the documented CLI: tests and developers may train on CPU
    allow_cpu_train: bool = field(default_factory=lambda: os.environ.get("STELLARIA_ALLOW_CPU_TRAIN") == "1")
    # fake engine training speed in steps/s
    fake_speed: float = field(default_factory=lambda: float(os.environ.get("STELLARIA_FAKE_SPEED", "25")))
    python: str | None = None  # interpreter for engine subprocesses (default: sys.executable)


class FileTokens:
    """``/files/{token}`` → file path, for downloads in browser mode (exports, quick renders)."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    def register(self, file: Path) -> str:
        token = secrets.token_urlsafe(16)
        with self._lock:
            data = read_json(self._path, {}) or {}
            data[token] = str(file)
            write_json(self._path, data)
        return token

    def resolve(self, token: str) -> Path | None:
        data = read_json(self._path, {}) or {}
        value = data.get(token)
        return Path(value) if value else None


def _warm_up_audio(engine_root: Path) -> None:
    """librosa's first call takes ~1.5 s (imports, numba); pay it, and compile the engine's
    Slicer, before the first slicing preview."""
    import librosa
    import numpy as np

    from .pipeline.slicing import engine_slicer_class

    engine_slicer_class(str(engine_root))
    librosa.feature.rms(y=np.zeros(4096, dtype=np.float32), frame_length=2048, hop_length=512)


class AppContext:
    def __init__(self, options: Options) -> None:
        from .engine.basemodels import BaseModelStore
        from .engine.render_client import RenderClient
        from .engine.rendering import NodeUpdates
        from .engine.separation import SeparationClient
        from .engine.tfevents import MetricsStore
        from .engine.training import TrainingService
        from .resources.manager import ResourceManager
        from .system import SystemService

        self.options = options
        for sub in ("models", "logs", "cache", "exports", "tmp"):
            (options.data / sub).mkdir(parents=True, exist_ok=True)
        self.hub = EventHub()
        self.prefs = PreferencesStore(options.data / "preferences.json", self.hub)
        self.lib = Library(options.library)
        self.characters = CharacterStore(self.lib)
        self.models = ModelStore(self.lib)
        self.projects = ProjectStore(self.lib)
        self.system = SystemService(self)
        self.jobs = JobManager(self.hub, gpu_policy=lambda: self.prefs.get().gpu_policy,
                               free_vram=self.system.free_vram)
        self.resources = ResourceManager(self)
        self.metrics = MetricsStore(self)
        self.base_models = BaseModelStore(self)
        self.training = TrainingService(self)
        self.render = RenderClient(self)
        self.separation = SeparationClient(self)
        self.node_updates = NodeUpdates()
        self.files = FileTokens(options.data / "exports" / "tokens.json")
        from .plugins.registry import PluginRegistry
        self.plugins = PluginRegistry(self)
        from .vst.registry import VstRegistry
        self.vst = VstRegistry(self)
        from .creation.analysis import SongAnalysis
        self.analysis = SongAnalysis(self)
        self.shutdown_requested = threading.Event()

    @property
    def python(self) -> str:
        return self.options.python or sys.executable

    @property
    def tmp_dir(self) -> Path:
        return self.options.data / "tmp"

    # ---- lifecycle -------------------------------------------------------------------
    def start(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)  # uploads left behind by a previous run
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.system.start_probe()
        self.training.reattach_all()
        if (count := self.projects.reset_interrupted()):
            log.info("reset %d canvas node(s) left running by the previous session", count)
        threading.Thread(target=_warm_up_audio, args=(self.options.engine,), name="warm-up", daemon=True).start()

    def stop(self, timeout: float = 20.0) -> None:
        """Graceful stop: trainings get STOP (checkpoint first), other jobs are cancelled."""
        log.info("stopping: pausing trainings and cancelling jobs")
        self.training.stop_all(timeout)
        self.jobs.shutdown(timeout=5.0)
        self.render.close()
        self.analysis.shutdown()
        self.hub.close()

    # ---- events ----------------------------------------------------------------------
    def emit_character(self, cid: str) -> None:
        from .library.views import character_view

        if self.characters.exists(cid):
            self.hub.publish("character", character_view(self, cid).wire(), throttle_key=f"chr:{cid}",
                             min_interval=0.2)

    def emit_model(self, mid: str, throttled: bool = False) -> None:
        from .library.views import model_view

        if self.models.exists(mid):
            self.hub.publish("model", model_view(self, mid).wire(), throttle_key=f"mdl:{mid}",
                             min_interval=1.0 if throttled else 0.0)
            for mc in self.models.get(mid).characters:
                self.emit_character(mc.character_id)

    def emit_clips(self, cid: str) -> None:
        self.hub.publish("clips", {"characterId": cid})
        self.emit_character(cid)
