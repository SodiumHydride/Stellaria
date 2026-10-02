# Stellaria backend (`stellaria_backend`)

FastAPI server behind the Stellaria desktop app. It owns the user's library, the job queue,
the audio pipeline, health checks, engine adapters (DDSP-SVC in ../engine), metrics,
downloads and add-ons. The contract it implements is `../docs/api.md` §1 (camelCase on the wire).

**The API process never imports torch.** Engine work runs in subprocesses: `preprocess.py` and
`train_reflow.py` with a generated config, a persistent render worker, a separation worker and a
one-off torch probe.

## Run

```bash
# dev mode against .dev/{library,data}: port 8765, token "dev", CORS *
scripts/dev-backend.sh            # real engine (needs torch + pretrained models)
scripts/dev-backend.sh --fake     # fake engine: no torch/GPU, everything still "alive"
# Windows: scripts\dev-backend.ps1 [-Fake]
```

Set `STELLARIA_PYTHON` (or activate a venv) with `requirements-runtime.txt` installed (plus
`torch==2.9.1`/`torchaudio==2.9.1` for real engine work). Directly:

```bash
python -m stellaria_backend --library L --data D --engine ../engine \
       --resources ../resources/manifest.json [--uv uv] [--port 0] [--token T] [--dev] [--fake-engine]
```

It prints one line, `STELLARIA_READY {"port": …, "token": "…", "version": "0.1.0"}`, then logs to
stderr and `<data>/logs/backend.log`. SIGTERM or `POST /api/shutdown` pause running trainings
(`exp/STOP`, ≤ ~20 s wait) and exit.

Environment knobs (not part of the shell contract):

| Variable | Effect |
|---|---|
| `STELLARIA_FAKE_SPEED` | fake training speed in steps/s (default 25) |
| `STELLARIA_ALLOW_CPU_TRAIN=1` | allow training without an NVIDIA GPU (tests, developers) |

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest                       # unit + API + fake-engine tests (no torch needed)
STELLARIA_E2E_MODELS=/path/to/models python -m pytest -m e2e -s   # real engine on CPU (needs torch)
```

`STELLARIA_E2E_MODELS` holds `rmvpe/model.pt`, `nsf/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/*`
and `hubert-soft-0d54a1f4.pt`; the test links them into `<data>/models` as the manifest lays them out
and trains a tiny `hubertsoft` model for 30 + 30 steps with a pause in between, then renders.
On CPU the engine needs ~8 s per step (its Muon optimizer runs Newton–Schulz in fp16, which is slow
on CPUs), so the test takes ~10–15 minutes.

## Package layout

| Module | Role |
|---|---|
| `__main__`, `app` | CLI, socket binding, READY line; FastAPI factory, token auth (`Bearer` or `?token=`), CORS, `{error: {code, message, detail, params}}` handlers, lifespan |
| `context` | `Options` + `AppContext` wiring every service; `FileTokens` for `/files/{token}` |
| `schemas` | pydantic mirrors of the contract types |
| `events` | WebSocket hub: thread-safe publish, topics (`gpu`, `job.log:<id>`), throttling |
| `jobs/` | queue, runner threads, GPU placement per `gpuPolicy` (`smart` ≥ 2.5 GB free / `cpu` / `queue`), lock keys, cancel, logs |
| `library/` | on-disk stores (atomic JSON; canvases migrated to v2 on load), clip editing, computed `Character`/`Model` views |
| `creation/` | canvas v3 graph (`canvas`: migration, ports, bypass, dependents), node updates and save ownership (`updates`), `/run` + song stems (`run`), mix v2 jobs (`mix`), process jobs through the dsp CLI (`process`), JSON-lines subprocesses (`cli`), sections/key + background song analysis (`analysis`) |
| `audio/` | decode (soundfile → bundled imageio-ffmpeg), wav/flac/mp3 writing, peaks, loudness helpers |
| `pipeline/` | raw/media imports (decoded once into `<character>/cache/decoded/`); organize: separate → normalize → engine `Slicer` → split long → drop → peak safety → dedupe → name → health; the synchronous slicing preview runs the same steps |
| `quality/` | per-clip analysis (numpy, parselmouth, MFCC; cached), health rules/scoring, validation pick |
| `engine/` | config generation, dataset assembly (hard links + fingerprint), preprocess/train runners, training service (STOP, pid file, re-attach, monitor), tfevents reader, validation audio + mel, checkpoints/releases, render worker + client, separation worker + client, mixdown, pitch/key, slim; external model import (`external`), base models + seeding (`basemodels`), checkpoint inspection in a torch subprocess (`inspect_worker`, `modelfiles`) |
| `resources/` | manifest, resumable multi-mirror downloader (Range, sha256, latency probe), add-on install via `uv pip` |
| `system` | OS/CPU/RAM, NVML GPUs (nvidia-smi fallback), torch probe subprocess, train blockers |
| `packs` | `.stella` export/import |
| `fake/` | stand-ins for `preprocess.py`/`train_reflow.py` (same CLI, STOP, max_steps, real tfevents) |

`slicer.py` imports torch at module level, so `pipeline/slicing.py` compiles only its `Slicer`
class from the engine source — the API process stays torch-free while slicing exactly like the engine.

## Notes for the frontend

Job `title` codes: `job.import`, `job.import_media`, `job.organize`, `job.health`, `job.train`,
`job.render`, `job.quick_render`, `job.separate`, `job.mixdown`, `job.download`, `job.install_addon`,
`job.export`, `job.import_pack`, `job.slim`, `job.mix`, `job.import_local`, `job.import_model`, `job.import_base`,
`job.process.<op>` (`trim|split|pitch|fx|separate|comp|concat|export`), `job.diagnostics` (params `{name}` or `{resource}`).

Job `stage` codes: `stage.waiting_gpu`, `stage.import {name,index,total}`, `stage.organize {name,index,total}`,
`stage.analyze {index,total}`, `stage.music`, `stage.health`, `stage.prepare`, `stage.preprocess {done,total}`,
`stage.train {step,target}`, `stage.render`, `stage.render_cpu` (render placed on the CPU next to a training),
`stage.separate` (`{device: 'cpu'}` when placed on the CPU), `stage.mixdown`, `stage.download {source}`,
`stage.extract`, `stage.install`, `stage.export`, `stage.import_pack`, `stage.slim_dataset`, `stage.slim_checkpoints`,
`stage.mix.analyze|vocal|reverb|master` / `stage.mix.plugin` (`{variant, variants}`), `stage.import_local` (model imports:
`{file, index, total}`), `stage.verify`, `stage.inspect` (checkpoint validation), `stage.seed_base` (writing a base model as `model_0.pt`),
`stage.dsp.<op>` / `stage.dsp.<name>` (process jobs, params as the dsp CLI reports them), `stage.diagnostics`.

Job `result` shapes:

| Job | `result` |
|---|---|
| `import` (raw) | `{rawIds}` |
| `import` (project media) | `{mediaIds}` |
| `organize` | `{clipIds, rawIds, dropped: {tooShort, silent, clipping, duplicate}, health: {score, level}}` |
| `health` | `{score, level, blocking}` |
| `train` | `{paused: true, step}` or `{done: true, step}` (+ `reattached: true` after a backend restart) |
| `render` (node) | `{mediaId, nodeId, device}` (`device`: `cuda` \| `mps` \| `cpu`, as used) |
| `render` (quick render) | `{url, path, device}` — `url` is `/models/{id}/quick-render/{jobId}/audio` (`…/peaks` next to it) |
| `separate` | `{nodeId, stems, device}` (the source node's updated `stems`) |
| `mix` | `{nodeId, outputs: {mediaId, seed, lufs}[]}` |
| `mixdown` | `{mediaId, path?}` |
| `download` / `install_addon` (incl. local import) | `{resourceId, path}` |
| `export` (packs, media) | `{path}` (desktop) or `{downloadUrl, path}` (no `path` given) |
| `import_pack` | `{characterIds, modelIds}` |
| `slim` | `{freedBytes}` |
| `import_model` | `{characterId, characterIds, modelId, releaseIds}` |
| `import_base` | `{baseModelId}` |
| `process` | `{nodeId, outputs: {port: mediaId}, result}` (export: `outputs {}`, `result {path, downloadUrl}`) |
| `diagnostics` | `{path, downloadUrl}` |

Stable error codes include `audio.decode_failed`, `raw.not_ready`, `clip.bad_name`, `clip.name_taken`, `organize.bad_pattern`, `dataset.empty`, `dataset.no_val`, `dataset.f0_failed`,
`engine.missing_model` (detail `resource=<id>`, model error params `{resource}`), `engine.out_of_memory`,
`engine.preprocess_failed`, `engine.train_failed`, `train.need_nvidia` (also `train.torch_missing`,
`train.cuda_unavailable`), `train.health_blocking`, `train.config_mismatch`, `addon.missing`,
`model.busy`, `render.no_checkpoint`, `render.bad_checkpoint`, `node.input_missing` (`{nodeId, port}`), `node.not_runnable` (`{nodeId}`),
`node.busy` (`{nodeId, jobId}`), `process.input_count` (`{min, max}`), `process.failed`, `dsp.unavailable`, `stems.not_provided` (`{port}`),
`vst.crashed` (`{name}`), `vst.instrument` (`{name}`), `analysis.failed`, `mix.no_inputs`, `mix.no_vocals`,
`mix.too_many_vocals`, `mix.too_many_inputs`, `mix.processor_unavailable`, `mix.failed`, `model.import_incompatible` (`{reason}`),
`model.import_bad_checkpoint` (`{file}`), `model.inspect_failed`, `import.not_found`, `base.in_use` (`{models}`), `base.read_only`, `resource.checksum`, `resource.incomplete`, `download.failed`, `request.invalid`, `auth.invalid_token`,
`<kind>_not_found` (e.g. `clip_not_found`), `internal`.

`Resource.name`/`description` are `Msg`s whose `params` carry the manifest texts per language
(`{"zh-CN": …, "en": …}`). `GpuInfo.vramTotal/vramUsed` and `SystemInfo.ramTotal` are bytes.
