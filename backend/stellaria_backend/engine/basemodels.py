"""Base models (底模, ``docs/creation-v2.md`` §7): checkpoints a new voice's training starts from.

Sources: manifest resources of kind ``base`` (downloaded like other models), imported files,
and copies of the user's own releases (``<data>/base-models/<id>/{model.pt, config.yaml,
base.json}``). Seeding writes the base as ``exp/model_0.pt`` before a model's first training run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..errors import ApiError, not_found
from ..jobs import Job, JobContext
from ..schemas import BaseModelInfo, BaseModelRecord, msg
from ..util.fsutil import remove_tree
from ..util.ids import new_id
from ..util.jsonio import read_json, write_json
from ..util.timeutil import now_iso
from . import checkpoints
from .config import read_yaml
from .inspect_worker import incompatibility, summarize
from .modelfiles import copy_file, glu_type_of, inspect_files, run_inspector

if TYPE_CHECKING:
    from ..context import AppContext

STRUCTURE = ("encoder", "n_layers", "n_chans", "n_aux_layers", "n_aux_chans")
_STATUS = {"ready": "ready", "missing": "missing", "downloading": "downloading", "installing": "downloading", "failed": "failed"}


class BaseModelStore:
    def __init__(self, ctx: "AppContext") -> None:
        self.ctx = ctx
        self.root = ctx.options.data / "base-models"

    # ---- records ---------------------------------------------------------------------------
    def _dir(self, bid: str) -> Path:
        return self.root / bid

    def _record(self, bid: str) -> BaseModelRecord | None:
        data = read_json(self._dir(bid) / "base.json")
        return BaseModelRecord.model_validate(data) if data else None

    def _manifest_bases(self) -> list:
        return [r for r in self.ctx.resources.manifest.resources if r.kind == "base" and r.base]

    def _manifest_view(self, rdef) -> BaseModelInfo:
        meta = rdef.base or {}
        lang = self.ctx.prefs.get().language
        name = rdef.name.get(lang) or rdef.name.get("en") or rdef.id
        resource = self.ctx.resources.view(rdef.id)
        return BaseModelInfo(
            id=rdef.id, name=name, notes=(rdef.description.get(lang) or ""), created_at=str(meta.get("createdAt", "")),
            source="download", resource_id=rdef.id, status=_STATUS.get(resource.status, "missing"),
            encoder=meta["encoder"], n_layers=meta["nLayers"], n_chans=meta["nChans"], n_aux_layers=meta["nAuxLayers"],
            n_aux_chans=meta["nAuxChans"], n_spk=meta.get("nSpk", 1), step=meta.get("step", 0), size=rdef.size or 0,
        )

    @staticmethod
    def _local_view(rec: BaseModelRecord) -> BaseModelInfo:
        return BaseModelInfo(**rec.model_dump(exclude={"glu_type"}), status="ready", resource_id=None)

    def list(self) -> list[BaseModelInfo]:
        local = []
        if self.root.is_dir():
            for d in sorted(self.root.iterdir()):
                rec = self._record(d.name) if d.is_dir() else None
                if rec is not None:
                    local.append(self._local_view(rec))
        return [*sorted(local, key=lambda b: b.created_at), *(self._manifest_view(r) for r in self._manifest_bases())]

    def get(self, bid: str) -> BaseModelInfo:
        rec = self._record(bid)
        if rec is not None:
            return self._local_view(rec)
        rdef = self.ctx.resources.manifest.get(bid)
        if rdef is not None and rdef.kind == "base" and rdef.base:
            return self._manifest_view(rdef)
        raise not_found("base", bid)

    def structure(self, bid: str) -> dict[str, Any]:
        """Structural TrainConfig fields (snake_case) plus ``glu_type`` of a base model."""
        rec = self._record(bid)
        if rec is not None:
            return {**{f: getattr(rec, f) for f in STRUCTURE}, "glu_type": rec.glu_type}
        view = self.get(bid)
        meta = self.ctx.resources.manifest.get(bid).base or {}
        return {**{f: getattr(view, f) for f in STRUCTURE}, "glu_type": meta.get("gluType") or "atanglu"}

    def files(self, bid: str) -> tuple[Path, Path]:
        """(checkpoint, config.yaml) of a ready base model."""
        if self._record(bid) is not None:
            return self._dir(bid) / "model.pt", self._dir(bid) / "config.yaml"
        self.get(bid)
        self.ctx.resources.require([bid])
        rdef = self.ctx.resources.manifest.get(bid)
        ckpt = rdef.primary_file(self.ctx.resources.models_dir)
        assert ckpt is not None
        return ckpt, ckpt.parent / (rdef.base or {}).get("config", "config.yaml")

    # ---- creating --------------------------------------------------------------------------
    def _save(self, bid: str, source: str, name: str, summary: dict[str, Any], step: int) -> BaseModelInfo:
        rec = BaseModelRecord(
            id=bid, name=name.strip() or "Base model", created_at=now_iso(), source=source, encoder=summary["encoder"],
            n_layers=int(summary["nLayers"]), n_chans=int(summary["nChans"]),
            n_aux_layers=int(summary.get("nAuxLayers") or 0), n_aux_chans=int(summary.get("nAuxChans") or 0),
            n_spk=int(summary.get("nSpk") or 1), step=step, size=(self._dir(bid) / "model.pt").stat().st_size,
            glu_type=glu_type_of(summary),
        )
        write_json(self._dir(bid) / "base.json", rec.wire())
        return self._local_view(rec)

    def submit_import(self, ckpt: Path, config: Path | None, name: str) -> Job:
        config = config or ckpt.parent / "config.yaml"
        for path in (ckpt, config):
            if not path.is_file():
                raise ApiError("import.not_found", f"'{path}' does not exist", status=422)

        def run(jctx: JobContext) -> dict[str, Any]:
            jctx.stage("stage.inspect", 0.0)
            result = inspect_files(self.ctx, jctx, config, [ckpt])
            bid = new_id("base")
            try:
                jctx.stage("stage.import_local", 0.2)
                copy_file(jctx, ckpt, self._dir(bid) / "model.pt", (0.2, 0.95))
                shutil.copyfile(config, self._dir(bid) / "config.yaml")
                step = result["checkpoints"][0].get("step") or 0
                self._save(bid, "local", name or ckpt.stem, result["config"], int(step))
            except BaseException:
                remove_tree(self._dir(bid))
                raise
            return {"baseModelId": bid}

        return self.ctx.jobs.submit("import_base", msg("job.import_base", name=name or ckpt.stem), run)

    def from_release(self, mid: str, rel_id: str, name: str) -> BaseModelInfo:
        rec = self.ctx.models.get(mid)
        release = next((r for r in rec.releases if r.id == rel_id), None)
        if release is None:
            raise not_found("release", rel_id)
        src = checkpoints.release_dir(self.ctx.models.dir(mid), rel_id)
        summary = summarize(read_yaml(src / "config.yaml") or {})
        reason = incompatibility(summary)
        if reason:
            raise ApiError("model.import_incompatible", "This release cannot be a base model", status=422,
                           params={"reason": reason})
        bid = new_id("base")
        try:
            copy_file(None, src / "model.pt", self._dir(bid) / "model.pt")
            shutil.copyfile(src / "config.yaml", self._dir(bid) / "config.yaml")
            default = release.name if release.name.startswith(rec.name) else f"{rec.name} · {release.name}"
            return self._save(bid, "release", name or default, summary, release.step)
        except BaseException:
            remove_tree(self._dir(bid))
            raise

    def patch(self, bid: str, name: str | None, notes: str | None) -> BaseModelInfo:
        rec = self._record(bid)
        if rec is None:
            self.get(bid)
            raise ApiError("base.read_only", "Downloadable base models cannot be edited", status=409)
        if name is not None and name.strip():
            rec.name = name.strip()
        if notes is not None:
            rec.notes = notes
        write_json(self._dir(bid) / "base.json", rec.wire())
        return self._local_view(rec)

    def delete(self, bid: str) -> None:
        if self._record(bid) is None:
            self.get(bid)
            raise ApiError("base.read_only", "Remove downloadable base models under Resources", status=409)
        users = [m.name for m in self.ctx.models.list_records() if m.config.base_model_id == bid
                 and checkpoints.latest_checkpoint(self.ctx.models.exp_dir(m.id)) is None]
        if users:
            raise ApiError("base.in_use", "A model that has not started training uses this base model", status=409,
                           params={"models": ", ".join(users)})
        remove_tree(self._dir(bid))

    # ---- seeding ---------------------------------------------------------------------------
    def seed(self, jctx: JobContext, mid: str) -> dict[str, Any]:
        """Write the model's base as ``exp/model_0.pt`` (first training run only)."""
        bid = self.ctx.models.get(mid).config.base_model_id
        assert bid is not None
        ckpt, _ = self.files(bid)
        out = self.ctx.models.exp_dir(mid) / "model_0.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        result = run_inspector(self.ctx, jctx, ["seed", "--base", str(ckpt), "--config",
                                                str(self.ctx.models.config_path(mid)), "--out", str(out)])
        dropped = result.get("dropped") or []
        jctx.log(f"seeded from base model {bid}: kept {result.get('kept')} tensors, dropped {len(dropped)}"
                 + (f" ({', '.join(dropped[:20])})" if dropped else "")
                 + (" — speaker embedding reset (different number of voices)" if result.get("speakerDropped") else ""))
        return result
