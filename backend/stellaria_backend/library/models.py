"""Model records (``models/<mdl_id>/model.json``) and their folder layout."""

from __future__ import annotations

from pathlib import Path

from ..errors import not_found
from ..schemas import ModelCharacter, ModelRecord, TrainConfig
from ..util.fsutil import remove_tree
from ..util.ids import new_id
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso
from .store import Library


class ModelStore:
    def __init__(self, lib: Library) -> None:
        self.lib = lib

    def dir(self, mid: str) -> Path:
        return self.lib.models_dir / mid

    def exp_dir(self, mid: str) -> Path:
        return self.dir(mid) / "exp"

    def dataset_dir(self, mid: str) -> Path:
        return self.dir(mid) / "dataset"

    def config_path(self, mid: str) -> Path:
        return self.dir(mid) / "config.yaml"

    def releases_dir(self, mid: str) -> Path:
        return self.dir(mid) / "releases"

    def cache_dir(self, mid: str) -> Path:
        return self.dir(mid) / "cache"

    def log_path(self, mid: str) -> Path:
        """Combined engine output (preprocess + training stdout/stderr)."""
        return self.dir(mid) / "model.log"

    def lock(self, mid: str):
        return self.lib.lock(f"mdl:{mid}")

    def exists(self, mid: str) -> bool:
        return (self.dir(mid) / "model.json").is_file()

    def get(self, mid: str) -> ModelRecord:
        data = read_json(self.dir(mid) / "model.json")
        if data is None:
            raise not_found("model", mid)
        return ModelRecord.model_validate(data)

    def list_records(self) -> list[ModelRecord]:
        out = []
        for d in sorted(self.lib.models_dir.iterdir()):
            data = read_json(d / "model.json") if d.is_dir() else None
            if data is not None:
                out.append(ModelRecord.model_validate(data))
        return out

    def save(self, rec: ModelRecord, touch: bool = True) -> ModelRecord:
        if touch:
            rec.updated_at = now_iso()
        write_json(self.dir(rec.id) / "model.json", rec.wire())
        return rec

    def create(self, name: str, character_ids: list[str], config: TrainConfig) -> ModelRecord:
        now = now_iso()
        rec = ModelRecord(
            id=new_id("mdl"), name=name,
            kind="single" if len(character_ids) == 1 else "constellation",
            characters=[ModelCharacter(character_id=c, spk_id=i + 1) for i, c in enumerate(character_ids)],
            config=config, target_steps=config.target_steps, created_at=now, updated_at=now,
        )
        self.exp_dir(rec.id).mkdir(parents=True, exist_ok=True)
        return self.save(rec, touch=False)

    def delete(self, mid: str) -> None:
        self.get(mid)
        remove_tree(self.dir(mid))
