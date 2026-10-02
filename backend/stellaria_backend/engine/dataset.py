"""Training dataset assembly: ``<model>/dataset/{train,val}/audio/[<spk>/]<clip>.wav``.

Files are hard links to the character's clips (copied when linking is impossible). A
fingerprint of the planned files decides whether assembling (and preprocessing) can be
skipped; ``manifest.json`` maps dataset files back to clips.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ApiError
from ..library.characters import CharacterStore
from ..schemas import ModelRecord
from ..util.fsutil import link_or_copy, remove_tree
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso


@dataclass
class DatasetEntry:
    clip_id: str
    character_id: str
    spk_id: int
    split: str
    rel: str  # path relative to the dataset dir, '/'-separated
    src: Path
    duration: float


def plan(rec: ModelRecord, characters: CharacterStore) -> list[DatasetEntry]:
    multi = len(rec.characters) > 1
    entries: list[DatasetEntry] = []
    for mc in rec.characters:
        if not characters.exists(mc.character_id):
            raise ApiError("model.character_missing", "A character of this model no longer exists", status=409,
                           detail=mc.character_id)
        for clip in characters.clips(mc.character_id):
            if clip.status != "active":
                continue
            folder = f"{clip.split}/audio/{mc.spk_id}" if multi else f"{clip.split}/audio"
            entries.append(DatasetEntry(
                clip_id=clip.id, character_id=mc.character_id, spk_id=mc.spk_id, split=clip.split,
                rel=f"{folder}/{Path(clip.file).name}", src=characters.clip_path(mc.character_id, clip),
                duration=clip.duration,
            ))
    return entries


def fingerprint(entries: list[DatasetEntry]) -> str:
    items = []
    for e in sorted(entries, key=lambda e: e.rel):
        try:
            st = e.src.stat()
            items.append((e.rel, st.st_size, st.st_mtime_ns))
        except FileNotFoundError:
            items.append((e.rel, -1, 0))
    return hashlib.sha1(json.dumps(items).encode("utf-8")).hexdigest()


def preprocess_fingerprint(dataset_fp: str, rec: ModelRecord) -> str:
    c = rec.config
    key = [dataset_fp, c.encoder, c.f0_extractor, c.use_pitch_aug]
    return hashlib.sha1(json.dumps(key).encode("utf-8")).hexdigest()


def manifest_path(dataset_dir: Path) -> Path:
    return dataset_dir / "manifest.json"


def read_manifest(dataset_dir: Path) -> dict[str, Any] | None:
    return read_json(manifest_path(dataset_dir))


def validate(entries: list[DatasetEntry]) -> None:
    if not any(e.split == "train" for e in entries):
        raise ApiError("dataset.empty", "There are no active training clips", status=409)
    if not any(e.split == "val" for e in entries):
        raise ApiError("dataset.no_val", "There are no validation clips", status=409)
    missing = [str(e.src) for e in entries if not e.src.is_file()]
    if missing:
        raise ApiError("dataset.missing_files", "Some clip files are missing", status=409, detail="\n".join(missing[:20]))


def assemble(dataset_dir: Path, entries: list[DatasetEntry]) -> dict[str, Any]:
    """(Re)build the dataset folder from scratch and write its manifest."""
    validate(entries)
    remove_tree(dataset_dir)
    links = 0
    for e in entries:
        links += link_or_copy(e.src, dataset_dir / e.rel)
    manifest = {
        "fingerprint": fingerprint(entries),
        "preparedAt": now_iso(),
        "clipCount": len(entries),
        "duration": round(sum(e.duration for e in entries), 3),
        "hardLinks": links,
        "entries": [{"clipId": e.clip_id, "characterId": e.character_id, "path": e.rel} for e in entries],
        "preprocess": None,
    }
    write_json(manifest_path(dataset_dir), manifest)
    return manifest


def mark_preprocessed(dataset_dir: Path, fp: str) -> None:
    manifest = read_manifest(dataset_dir) or {}
    manifest["preprocess"] = {"fingerprint": fp, "doneAt": now_iso()}
    write_json(manifest_path(dataset_dir), manifest)


def status(rec: ModelRecord, characters: CharacterStore, dataset_dir: Path) -> tuple[str, dict[str, Any] | None]:
    """'none' | 'stale' | 'ready' plus the manifest."""
    manifest = read_manifest(dataset_dir)
    if not manifest:
        return "none", None
    try:
        current = fingerprint(plan(rec, characters))
    except ApiError:
        return "stale", manifest
    return ("ready" if current == manifest.get("fingerprint") else "stale"), manifest
