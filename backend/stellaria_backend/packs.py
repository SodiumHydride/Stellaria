"""``.stella`` packages: characters (optionally with material) and model releases.

Layout of the zip::

    pack.json                                 {format, version, kind, createdAt, appVersion, characters, models}
    characters/<chr_id>/character.json        (+ avatar.png, clips.json, clips/*.wav, health.json with material)
    models/<mdl_id>/model.json
    models/<mdl_id>/releases/<rel_id>/{model.pt, config.yaml, release.json}

On import, a character that already exists (same id) is reused; models always get a new id.
Engine configs are rewritten to this machine's pretrained-model paths.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from . import __version__
from .engine.config import ENCODER_RESOURCE, read_yaml, write_yaml
from .errors import ApiError, not_found
from .exports import destination, finish
from .jobs import JobContext
from .schemas import CharacterRecord, Clip, ModelRecord, Release
from .util.ids import new_id
from .util.jsonio import write_json
from .util.timeutil import now_iso

if TYPE_CHECKING:
    from .context import AppContext

FORMAT = "stellaria-pack"
VERSION = 1
STORED = (".wav", ".pt", ".png", ".flac", ".mp3")


def _add(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    kind = zipfile.ZIP_STORED if src.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
    zf.write(src, arcname, compress_type=kind)


def _add_json(zf: zipfile.ZipFile, arcname: str, data: Any) -> None:
    zf.writestr(arcname, json.dumps(data, ensure_ascii=False, indent=2), compress_type=zipfile.ZIP_DEFLATED)


def _write_character(ctx: "AppContext", zf: zipfile.ZipFile, cid: str, include_material: bool) -> None:
    chars = ctx.characters
    rec = chars.get(cid)
    base = f"characters/{cid}"
    _add_json(zf, f"{base}/character.json", rec.wire())
    if chars.avatar_path(cid).is_file():
        _add(zf, chars.avatar_path(cid), f"{base}/avatar.png")
    if include_material:
        clips = [c for c in chars.clips(cid) if c.status == "active"]
        _add_json(zf, f"{base}/clips.json", [c.wire() for c in clips])
        for c in clips:
            _add(zf, chars.clip_path(cid, c), f"{base}/{c.file}")
        health = chars.dir(cid) / "health.json"
        if health.is_file():
            _add(zf, health, f"{base}/health.json")


def export_character(ctx: "AppContext", jctx: JobContext, cid: str, include_material: bool,
                     path: str | None) -> dict[str, Any]:
    rec = ctx.characters.get(cid)
    dest, browser = destination(ctx, path, rec.name, "stella")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with zipfile.ZipFile(tmp, "w") as zf:
        jctx.stage("stage.export", 0.1)
        _add_json(zf, "pack.json", {"format": FORMAT, "version": VERSION, "kind": "character", "createdAt": now_iso(),
                                    "appVersion": __version__, "characters": [cid], "models": []})
        _write_character(ctx, zf, cid, include_material)
    tmp.replace(dest)
    return finish(ctx, dest, browser)


def export_model(ctx: "AppContext", jctx: JobContext, mid: str, release_id: str, path: str | None) -> dict[str, Any]:
    rec = ctx.models.get(mid)
    release = next((r for r in rec.releases if r.id == release_id), None)
    if release is None:
        raise not_found("release", release_id)
    rel_dir = ctx.models.releases_dir(mid) / release_id
    dest, browser = destination(ctx, path, f"{rec.name}-{release.name}", "stella")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    char_ids = [mc.character_id for mc in rec.characters if ctx.characters.exists(mc.character_id)]
    model_json = rec.model_copy(update={"status": "idle", "error": None, "releases": [release]}).wire()
    with zipfile.ZipFile(tmp, "w") as zf:
        jctx.stage("stage.export", 0.1)
        _add_json(zf, "pack.json", {"format": FORMAT, "version": VERSION, "kind": "model", "createdAt": now_iso(),
                                    "appVersion": __version__, "characters": char_ids, "models": [mid]})
        for cid in char_ids:
            _write_character(ctx, zf, cid, include_material=False)
        base = f"models/{mid}"
        _add_json(zf, f"{base}/model.json", model_json)
        for name in ("model.pt", "config.yaml", "release.json"):
            _add(zf, rel_dir / name, f"{base}/releases/{release_id}/{name}")
    tmp.replace(dest)
    return finish(ctx, dest, browser)


def _safe_members(zf: zipfile.ZipFile) -> list[str]:
    names = zf.namelist()
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            raise ApiError("pack.invalid", f"Unsafe path in pack: {name}", status=422)
    return names


def import_pack(ctx: "AppContext", jctx: JobContext, path: Path, cleanup: list[Path]) -> dict[str, Any]:
    try:
        return _import(ctx, jctx, path)
    finally:
        for p in cleanup:
            shutil.rmtree(p, ignore_errors=True)


def _import(ctx: "AppContext", jctx: JobContext, path: Path) -> dict[str, Any]:
    try:
        zf = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ApiError("pack.invalid", "This is not a Stellaria package", status=422, detail=str(exc)) from exc
    with zf:
        names = _safe_members(zf)
        try:
            meta = json.loads(zf.read("pack.json"))
        except KeyError as exc:
            raise ApiError("pack.invalid", "pack.json missing", status=422) from exc
        if meta.get("format") != FORMAT or int(meta.get("version", 0)) > VERSION:
            raise ApiError("pack.unsupported", "Unsupported package version", status=422)
        jctx.stage("stage.import_pack", 0.1)
        character_ids = [_import_character(ctx, zf, names, cid) for cid in meta.get("characters", [])]
        model_ids = [_import_model(ctx, zf, names, mid) for mid in meta.get("models", [])]
    for cid in character_ids:
        ctx.emit_character(cid)
    for mid in model_ids:
        ctx.emit_model(mid)
    return {"characterIds": character_ids, "modelIds": model_ids}


def _extract(zf: zipfile.ZipFile, member: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as src, open(dest, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)


def _import_character(ctx: "AppContext", zf: zipfile.ZipFile, names: list[str], cid: str) -> str:
    chars = ctx.characters
    if chars.exists(cid):
        return cid  # same character already on this machine
    base = f"characters/{cid}/"
    rec = CharacterRecord.model_validate(json.loads(zf.read(base + "character.json")))
    rec.clips_revision = 0
    folder = chars.dir(cid)
    for sub in ("raw", "clips"):
        (folder / sub).mkdir(parents=True, exist_ok=True)
    clips: list[Clip] = []
    if base + "clips.json" in names:
        clips = [Clip.model_validate(c) for c in json.loads(zf.read(base + "clips.json"))]
        for c in clips:
            _extract(zf, base + c.file, folder / c.file)
            c.raw_id = None  # originals are not part of a pack
    if base + "avatar.png" in names:
        _extract(zf, base + "avatar.png", chars.avatar_path(cid))
    if base + "health.json" in names:
        health = json.loads(zf.read(base + "health.json"))
        health["clipsRevision"] = 0
        write_json(folder / "health.json", health)
    write_json(folder / "raw.json", [])
    write_json(folder / "clips.json", [c.wire() for c in clips])
    chars.save(rec, touch=False)
    return cid


def _import_model(ctx: "AppContext", zf: zipfile.ZipFile, names: list[str], src_mid: str) -> str:
    base = f"models/{src_mid}/"
    rec = ModelRecord.model_validate(json.loads(zf.read(base + "model.json")))
    mid = src_mid if not ctx.models.exists(src_mid) else new_id("mdl")
    rec = rec.model_copy(update={"id": mid, "status": "idle", "error": None})
    folder = ctx.models.dir(mid)
    (folder / "exp").mkdir(parents=True, exist_ok=True)
    releases: list[Release] = []
    for release in rec.releases:
        prefix = f"{base}releases/{release.id}/"
        if prefix + "model.pt" not in names:
            continue
        dest = folder / "releases" / release.id
        for name in ("model.pt", "config.yaml", "release.json"):
            if prefix + name in names:
                _extract(zf, prefix + name, dest / name)
        relocate_config(ctx, dest / "config.yaml", folder)
        releases.append(release)
    rec.releases = releases
    ctx.models.save(rec, touch=False)
    return mid


def relocate_config(ctx: "AppContext", config_path: Path, model_dir: Path) -> None:
    """Point an engine config at this machine's pretrained models and this model's folders."""
    conf = read_yaml(config_path)
    if not conf:
        return
    encoder = conf.get("data", {}).get("encoder")
    if encoder in ENCODER_RESOURCE:
        conf["data"]["encoder_ckpt"] = str(ctx.resources.path(ENCODER_RESOURCE[encoder]))
    conf.setdefault("vocoder", {})["ckpt"] = str(ctx.resources.path("nsf_hifigan"))
    conf["data"]["train_path"] = str(model_dir / "dataset" / "train")
    conf["data"]["valid_path"] = str(model_dir / "dataset" / "val")
    conf.setdefault("env", {})["expdir"] = str(model_dir / "exp")
    write_yaml(config_path, conf)

