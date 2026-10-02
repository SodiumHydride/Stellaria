"""Characters: metadata, raw imports, clips and the last health report."""

from __future__ import annotations

import random
from pathlib import Path

from ..errors import not_found
from ..schemas import CharacterRecord, Clip, HealthReport, RawImport
from ..util.fsutil import remove_tree
from ..util.ids import new_id
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso
from .store import Library

PALETTE = ["#8b7cf6", "#5ec8f2", "#f27eb5", "#f5b94f", "#6ee7b7", "#f87171", "#a3e635", "#c084fc", "#38bdf8", "#fb923c"]


class CharacterStore:
    def __init__(self, lib: Library) -> None:
        self.lib = lib

    # ---- paths -----------------------------------------------------------------------
    def dir(self, cid: str) -> Path:
        return self.lib.characters_dir / cid

    def clips_dir(self, cid: str) -> Path:
        return self.dir(cid) / "clips"

    def raw_dir(self, cid: str) -> Path:
        return self.dir(cid) / "raw"

    def cache_dir(self, cid: str) -> Path:
        return self.dir(cid) / "cache"

    def avatar_path(self, cid: str) -> Path:
        return self.dir(cid) / "avatar.png"

    def clip_path(self, cid: str, clip: Clip) -> Path:
        return self.dir(cid) / clip.file

    def lock(self, cid: str):
        return self.lib.lock(f"chr:{cid}")

    # ---- records ---------------------------------------------------------------------
    def exists(self, cid: str) -> bool:
        return (self.dir(cid) / "character.json").is_file()

    def get(self, cid: str) -> CharacterRecord:
        data = read_json(self.dir(cid) / "character.json")
        if data is None:
            raise not_found("character", cid)
        return CharacterRecord.model_validate(data)

    def list_records(self) -> list[CharacterRecord]:
        records = []
        for d in sorted(self.lib.characters_dir.iterdir()):
            data = read_json(d / "character.json") if d.is_dir() else None
            if data is not None:
                records.append(CharacterRecord.model_validate(data))
        return records

    def save(self, rec: CharacterRecord, touch: bool = True) -> CharacterRecord:
        if touch:
            rec.updated_at = now_iso()
        write_json(self.dir(rec.id) / "character.json", rec.wire())
        return rec

    def create(self, name: str, color: str | None = None, tags: list[str] | None = None) -> CharacterRecord:
        now = now_iso()
        rec = CharacterRecord(
            id=new_id("chr"), name=name.strip() or "Unnamed", color=color or random.choice(PALETTE),
            tags=tags or [], created_at=now, updated_at=now,
        )
        for sub in ("raw", "clips"):
            (self.dir(rec.id) / sub).mkdir(parents=True, exist_ok=True)
        write_json(self.dir(rec.id) / "raw.json", [])
        write_json(self.dir(rec.id) / "clips.json", [])
        return self.save(rec, touch=False)

    def delete(self, cid: str) -> None:
        self.get(cid)
        remove_tree(self.dir(cid))

    # ---- raw imports -----------------------------------------------------------------
    def raws(self, cid: str) -> list[RawImport]:
        return [RawImport.model_validate(r) for r in read_json(self.dir(cid) / "raw.json", [])]

    def save_raws(self, cid: str, raws: list[RawImport]) -> None:
        write_json(self.dir(cid) / "raw.json", [r.wire() for r in raws])

    # ---- clips -----------------------------------------------------------------------
    def clips(self, cid: str) -> list[Clip]:
        return [Clip.model_validate(c) for c in read_json(self.dir(cid) / "clips.json", [])]

    def save_clips(self, cid: str, clips: list[Clip], bump: bool = True) -> None:
        """Persist clips. ``bump`` marks the material as changed (health becomes stale)."""
        with self.lock(cid):
            write_json(self.dir(cid) / "clips.json", [c.wire() for c in clips])
            if bump:
                rec = self.get(cid)
                rec.clips_revision += 1
                self.save(rec)

    # ---- health ----------------------------------------------------------------------
    def health(self, cid: str) -> tuple[HealthReport | None, int]:
        data = read_json(self.dir(cid) / "health.json")
        if not data:
            return None, -1
        return HealthReport.model_validate(data), int(data.get("clipsRevision", -1))

    def save_health(self, cid: str, report: HealthReport, revision: int) -> None:
        write_json(self.dir(cid) / "health.json", {**report.wire(), "clipsRevision": revision})
