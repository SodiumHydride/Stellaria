"""Importing DDSP-SVC reflow models trained elsewhere (``.pt`` checkpoints + ``config.yaml``) as a
voice without material: validated in a subprocess, copied into a new model's ``exp/`` and
published as one release per checkpoint.

Characters: ``characterId`` attaches the (first) voice, ``newCharacter`` creates it; for a
multi-speaker model (``n_spk > 1``) ``characterIds`` may map speakers 1..n to characters, the
rest are created as "<name> spk<i>".
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..errors import ApiError
from ..jobs import Job, JobContext
from ..schemas import Release, Wire, msg
from ..util.fsutil import link_or_copy, remove_tree
from ..util.ids import new_id
from ..util.jsonio import write_json
from ..util.timeutil import now_iso
from .config import default_config, recommended_preset
from .modelfiles import assign_steps, copy_file, glu_type_of, inspect_files, train_config_from

if TYPE_CHECKING:
    from ..context import AppContext


class NewCharacter(Wire):
    name: str
    color: str | None = None


class ExternalImport(Wire):
    ckpt_paths: list[str] = Field(min_length=1)
    config_path: str | None = None
    character_id: str | None = None
    character_ids: list[str] | None = None  # multi-speaker mapping: speaker i+1 → character_ids[i]
    new_character: NewCharacter | None = None
    name: str | None = None


def submit_import_external(ctx: "AppContext", req: ExternalImport) -> Job:
    ckpts = [Path(p) for p in req.ckpt_paths]
    config = Path(req.config_path) if req.config_path else ckpts[0].parent / "config.yaml"
    for path in [*ckpts, config]:
        if not path.is_file():
            raise ApiError("import.not_found", f"'{path}' does not exist", status=422)
    mapped = [c for c in (req.character_ids or []) if c]
    if len(set(mapped)) != len(mapped):
        raise ApiError("model.bad_characters", "Map each speaker to a different voice", status=422)
    for cid in [req.character_id, *mapped]:
        if cid:
            ctx.characters.get(cid)
    title = req.name or (req.new_character.name if req.new_character else config.parent.name)
    return ctx.jobs.submit("import_model", msg("job.import_model", name=title),
                           lambda jctx: _import(ctx, jctx, req, ckpts, config))


def _characters(ctx: "AppContext", req: ExternalImport, n_spk: int, name: str) -> tuple[list[str], list[str]]:
    """(character id per speaker, ids created by this import)."""
    mapped = list(req.character_ids or [])
    if req.character_id and not mapped:
        mapped = [req.character_id]
    ids, created = [], []
    for i in range(n_spk):
        if i < len(mapped) and mapped[i]:
            ids.append(mapped[i])
            continue
        if i == 0 and req.new_character is not None:
            rec = ctx.characters.create(req.new_character.name, req.new_character.color)
        else:
            rec = ctx.characters.create(name if n_spk == 1 else f"{name} spk{i + 1}")
        ids.append(rec.id)
        created.append(rec.id)
    return ids, created


def _import(ctx: "AppContext", jctx: JobContext, req: ExternalImport, ckpts: list[Path], config: Path) -> dict[str, Any]:
    from ..packs import relocate_config

    jctx.stage("stage.inspect", 0.0)
    result = inspect_files(ctx, jctx, config, ckpts)
    summary = result["config"]
    steps = assign_steps(result["checkpoints"])
    name = (req.name or (req.new_character.name if req.new_character else "") or config.parent.name).strip() or "Imported"
    char_ids, created = _characters(ctx, req, int(summary["nSpk"]), name)
    cfg = train_config_from(summary, default_config(recommended_preset(ctx.system.best_vram_gb())))
    rec = ctx.models.create(name, char_ids, cfg.model_copy(update={"target_steps": max(steps)}))
    mid = rec.id
    try:
        exp = ctx.models.exp_dir(mid)
        exp_config = exp / "config.yaml"
        shutil.copyfile(config, exp_config)
        relocate_config(ctx, exp_config, ctx.models.dir(mid))
        releases: list[Release] = []
        for i, (src, step) in enumerate(zip(ckpts, steps)):
            jctx.stage("stage.import_local", i / len(ckpts), file=src.name, index=i + 1, total=len(ckpts))
            target = exp / f"model_{step}.pt"
            copy_file(jctx, src, target, (i / len(ckpts), (i + 1) / len(ckpts)))
            release = Release(id=new_id("rel"), name=f"{name} · {step}", step=step, created_at=now_iso(),
                              size=target.stat().st_size)
            folder = ctx.models.releases_dir(mid) / release.id
            link_or_copy(target, folder / "model.pt")  # no second copy on disk
            shutil.copyfile(exp_config, folder / "config.yaml")
            write_json(folder / "release.json", release.wire())
            releases.append(release)
        with ctx.models.lock(mid):
            rec = ctx.models.get(mid)
            rec.releases, rec.status, rec.glu_type = releases, "done", glu_type_of(summary)
            ctx.models.save(rec)
    except BaseException:
        remove_tree(ctx.models.dir(mid))
        for cid in created:
            ctx.characters.delete(cid)
        raise
    ctx.emit_model(mid)
    return {"characterId": char_ids[0], "characterIds": char_ids, "modelId": mid,
            "releaseIds": [r.id for r in releases]}
