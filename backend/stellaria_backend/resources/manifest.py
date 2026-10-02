"""Typed view of ``manifest.json`` and the on-disk location of each resource."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Source:
    id: str
    region: str  # 'global' | 'cn' | 'any'
    url: str | None


@dataclass
class ResourceDef:
    id: str
    kind: str  # 'model' | 'addon'
    required: bool
    name: dict[str, str]
    description: dict[str, str]
    size: int | None
    sha256: str | None
    install: dict[str, Any]
    sources: list[Source] = field(default_factory=list)
    base: dict[str, Any] | None = None  # kind 'base': {encoder, nLayers, nChans, nAuxLayers, nAuxChans, nSpk, step, gluType?, config?}
    # a vocal separation checkpoint: {arch, config (a file in engine/roformer/configs), sdr?, aliases?}
    separation: dict[str, Any] | None = None
    # a vocal cleanup checkpoint (dry voice out): {arch, config, kind ('dereverb' | 'dereverb_echo'
    # | 'denoise'), target (the config's instrument that is the dry voice), sdr?, aliases?}
    cleanup: dict[str, Any] | None = None
    license: str | None = None
    homepage: str | None = None

    @property
    def install_type(self) -> str:
        return self.install.get("type", "file")

    def dest_files(self, models_dir: Path) -> list[Path]:
        """Files that must exist for the resource to count as installed."""
        if self.install_type == "file":
            return [models_dir / self.install["dest"]]
        if self.install_type == "zip":
            base = models_dir / self.install["dest"]
            return [base / name for name in self.install.get("files", {}).values()]
        return []

    def required_files(self, models_dir: Path) -> list[Path]:
        """Destination files that must exist (licence/notice texts are optional)."""
        return [p for p in self.dest_files(models_dir) if p.suffix.lower() not in (".txt", ".md")]

    def primary_file(self, models_dir: Path) -> Path | None:
        """The checkpoint file the engine loads (first .pt/.ckpt/.bin of the destination)."""
        for path in self.dest_files(models_dir):
            if path.suffix in (".pt", ".ckpt", ".bin", ".pth", ".safetensors"):
                return path
        files = self.dest_files(models_dir)
        return files[0] if files else None

    @property
    def packages(self) -> list[str]:
        return list(self.install.get("packages", []))

    @property
    def module_name(self) -> str | None:
        """Importable module of an add-on's first package (``demucs==4.1.0`` → ``demucs``)."""
        if not self.packages:
            return None
        name = self.packages[0]
        for sep in ("==", ">=", "<=", "~=", "[", "<", ">", ";"):
            name = name.split(sep, 1)[0]
        return name.strip().replace("-", "_")


@dataclass
class Manifest:
    resources: list[ResourceDef]
    pypi: list[Source]
    raw: dict[str, Any]

    def get(self, rid: str) -> ResourceDef | None:
        return next((r for r in self.resources if r.id == rid), None)


def _sources(items: list[dict[str, Any]]) -> list[Source]:
    return [Source(id=s["id"], region=s.get("region", "any"), url=s.get("url")) for s in items or []]


def load_manifest(path: Path) -> Manifest:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    resources = [
        ResourceDef(
            id=r["id"], kind=r.get("kind", "model"), required=bool(r.get("required", False)),
            name=r.get("name", {}), description=r.get("description", {}), size=r.get("size"),
            sha256=r.get("sha256"), install=r.get("install", {}), sources=_sources(r.get("sources", [])),
            base=r.get("base"), separation=r.get("separation"), cleanup=r.get("cleanup"), license=r.get("license"),
            homepage=r.get("homepage"),
        )
        for r in raw.get("resources", [])
    ]
    pypi = _sources(raw.get("runtime", {}).get("pypi", []))
    return Manifest(resources=resources, pypi=pypi, raw=raw)
