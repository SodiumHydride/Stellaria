"""Machine information: OS/CPU/RAM, NVIDIA GPUs (NVML, nvidia-smi fallback), torch probe.

The torch probe runs ONCE, in a subprocess with a timeout, so the API process never imports
torch. GPU readings are cached for ≤ 1 s.
"""

from __future__ import annotations

import json
import logging
import math
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import psutil

from . import __version__
from .schemas import GpuInfo, Msg, SystemInfo, TorchInfo
from .util import procs

if TYPE_CHECKING:
    from .context import AppContext

log = logging.getLogger(__name__)

PROBE_TIMEOUT = 120.0
PROBE_SCRIPT = r"""
import json, sys
r = {"available": False, "device": "cpu", "triton": False}
try:
    import torch
    r.update(available=True, version=torch.__version__, cuda=torch.version.cuda)
    if torch.cuda.is_available():
        r["device"] = "cuda"
        try:
            import triton  # noqa: F401
            r["triton"] = True
        except Exception:
            pass
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        r["device"] = "mps"
except Exception as e:
    r["error"] = repr(e)
sys.stdout.write("STELLARIA_PROBE " + json.dumps(r) + "\n")
"""


def os_name() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    return "linux"


def os_version() -> str:
    if sys.platform == "darwin":
        return platform.mac_ver()[0] or platform.release()
    if sys.platform == "win32":
        return platform.version()
    return platform.release()


def cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            out = procs.run(["sysctl", "-n", "machdep.cpu.brand_string"], timeout=5).stdout.decode().strip()
            if out:
                return out
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def query_gpus() -> list[GpuInfo]:
    """NVIDIA GPUs with live values; empty list when there is none (or no driver)."""
    try:
        return _nvml_gpus()
    except Exception:  # noqa: BLE001 - NVML missing/broken: try nvidia-smi
        pass
    try:
        return _smi_gpus()
    except Exception:  # noqa: BLE001
        return []


def _nvml_gpus() -> list[GpuInfo]:
    import pynvml

    pynvml.nvmlInit()
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = temp = cc = None
            try:
                util = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
            except pynvml.NVMLError:
                pass
            try:
                temp = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            except pynvml.NVMLError:
                pass
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                cc = f"{major}.{minor}"
            except pynvml.NVMLError:
                pass
            gpus.append(GpuInfo(
                index=i, name=name.decode() if isinstance(name, bytes) else name, vram_total=int(mem.total),
                vram_used=int(mem.used), utilization=util, temperature=temp, compute_capability=cc, driver=driver,
            ))
        return gpus
    finally:
        pynvml.nvmlShutdown()


def _smi_gpus() -> list[GpuInfo]:
    fields = "index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,compute_cap,driver_version"
    out = procs.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout=10)
    if out.returncode != 0:
        return []

    def num(v: str) -> float | None:
        try:
            return float(v)
        except ValueError:
            return None

    gpus = []
    for line in out.stdout.decode("utf-8", errors="replace").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        total, used = num(parts[2]), num(parts[3])
        gpus.append(GpuInfo(
            index=int(parts[0]), name=parts[1], vram_total=int((total or 0) * 1024**2),
            vram_used=int(used * 1024**2) if used is not None else None, utilization=num(parts[4]),
            temperature=num(parts[5]), compute_capability=parts[6] or None, driver=parts[7] or None,
        ))
    return gpus


def probe_torch(python: str = sys.executable, timeout: float = PROBE_TIMEOUT) -> TorchInfo:
    try:
        result = procs.run([python, "-c", PROBE_SCRIPT], timeout=timeout)
        text = result.stdout.decode("utf-8", errors="replace")
        m = re.search(r"STELLARIA_PROBE (\{.*\})", text)
        if not m:
            log.warning("torch probe gave no result: %s", result.stderr.decode(errors="replace")[-500:])
            return TorchInfo()
        data = json.loads(m.group(1))
        if data.get("error"):
            log.warning("torch probe: %s", data["error"])
        return TorchInfo.model_validate(data)
    except subprocess.TimeoutExpired:
        log.warning("torch probe timed out after %ss", timeout)
        return TorchInfo()


class SystemService:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self._torch: TorchInfo | None = None
        self._probe_done = threading.Event()
        self._gpu_cache: tuple[float, list[GpuInfo]] = (0.0, [])
        self._gpu_lock = threading.Lock()

    def start_probe(self) -> None:
        if self._probe_done.is_set():  # already known (set_torch)
            return

        def run() -> None:
            try:
                self._torch = probe_torch()
                log.info("torch probe: %s", self._torch.wire())
            finally:
                self._probe_done.set()

        threading.Thread(target=run, name="torch-probe", daemon=True).start()

    def torch(self, wait: float = PROBE_TIMEOUT) -> TorchInfo:
        self._probe_done.wait(wait)
        return self._torch or TorchInfo()

    def set_torch(self, info: TorchInfo) -> None:
        """Test hook: skip the subprocess probe."""
        self._torch = info
        self._probe_done.set()

    def gpus(self) -> list[GpuInfo]:
        with self._gpu_lock:
            ts, cached = self._gpu_cache
            if time.monotonic() - ts <= 1.0:
                return cached
            gpus = query_gpus()
            if not gpus and self.ctx.options.fake_engine:
                gpus = [self._fake_gpu()]
            self._gpu_cache = (time.monotonic(), gpus)
            return gpus

    def _fake_gpu(self) -> GpuInfo:
        """A pretend 8 GB GPU so the UI's GPU panels have something to show in fake mode."""
        training = self.ctx.jobs.find_active("train") is not None
        wave = 0.5 + 0.5 * math.sin(time.time() / 3.0)
        total = 8 * 1024**3
        return GpuInfo(
            index=0, name="Stellaria Fake GPU", vram_total=total,
            vram_used=int(total * (0.62 + 0.05 * wave if training else 0.08)),
            utilization=round(85 + 10 * wave, 1) if training else round(3 * wave, 1),
            temperature=round(68 + 6 * wave, 1) if training else 41.0, compute_capability="8.9", driver="fake",
        )

    def free_vram(self) -> int | None:
        gpus = self.gpus()
        if not gpus or gpus[0].vram_used is None:
            return None
        return gpus[0].vram_total - gpus[0].vram_used

    def best_vram_gb(self) -> float | None:
        gpus = self.gpus()
        return max(g.vram_total for g in gpus) / 1024**3 if gpus else None

    def train_blockers(self) -> list[Msg]:
        if self.ctx.options.fake_engine or self.ctx.options.allow_cpu_train:
            return []
        torch = self.torch()
        if not [g for g in self.gpus()]:
            return [Msg(code="train.need_nvidia")]
        if not torch.available:
            return [Msg(code="train.torch_missing")]
        if torch.device != "cuda":
            return [Msg(code="train.cuda_unavailable")]
        return []

    def train_device(self) -> str:
        """Device written into the engine config: cuda when available, else cpu."""
        return "cuda" if self.torch().device == "cuda" else "cpu"

    def info(self) -> SystemInfo:
        torch = self.torch()
        blockers = self.train_blockers()
        opts = self.ctx.options
        return SystemInfo(
            version=__version__, os=os_name(), os_version=os_version(), arch=platform.machine(),
            cpu=cpu_name(), cpu_count=psutil.cpu_count(logical=True) or 1,
            ram_total=int(psutil.virtual_memory().total), gpus=self.gpus(), torch=torch,
            can_train=not blockers, train_blockers=blockers, library_path=str(opts.library),
            data_path=str(opts.data), engine_version=engine_version(opts.engine), fake_engine=opts.fake_engine,
        )


def engine_version(engine: Path) -> str:
    """``engine_version.txt`` when the shell staged one, else the git commit when available."""
    staged = engine / "engine_version.txt"
    if staged.is_file():
        return staged.read_text(encoding="utf-8").strip()
    head = engine / ".git" / "HEAD"
    try:
        ref = head.read_text(encoding="utf-8").strip()
        if ref.startswith("ref: "):
            ref_path = engine / ".git" / ref[5:]
            commit = ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else ""
            branch = ref.rsplit("/", 1)[-1]
            return f"{branch}+{commit[:7]}" if commit else branch
        return ref[:7]
    except OSError:
        return "unknown"
