# Creation v2, mixing, plugins, custom Python: contract

This document is the source of truth for the next round of work. It extends `api.md`; when a
part is implemented, merge it into `api.md` and delete it from here. Conventions as in `api.md`
(camelCase on the wire, `Msg` for user-facing text, errors as `{code, message, detail}`).

Goals, in the user's words: creating should feel comfortable, **every edge on the canvas
should mean something**, one song should easily be sung by several voices with different
settings, friends' mixing scripts should plug in, and advanced users should be able to point
the app at their own Python.

---------------------------------------------------------------------------------------------

## 1. Canvas v2: edges are the data flow

> **Status — backend: done** (merged into `api.md`: types, migration, port resolution, separate →
> `stems`, render `input`/`stale`, mix job, `/mix/processors`, media analysis; plus
> `POST /projects/{id}/media/{mediaId}/export`). Frontend: open.

There are no free-floating edges any more. **Edges are derived from node data** (the frontend
draws them from `render.input`, `mix.vocals[]` and `mix.backing`). `canvas.edges` stays in the
schema for compatibility and is always written as `[]`.

```ts
type NodeType = 'source' | 'render' | 'mix' | 'note' | 'group'      // 'stem' is gone (migrated)
type SourcePort = 'mix' | 'vocals' | 'inst'                          // outputs of a source node

interface SourceData {
  mediaId: ID; name: string
  stems: {
    status: 'none' | 'queued' | 'running' | 'done' | 'failed'
    vocals: ID | null; inst: ID | null                                // media ids once separated
    jobId: ID | null; error: Msg | null
  }
}

interface RenderData {
  input: { nodeId: ID; port: SourcePort }    // replaces inputNodeId
  characterId: ID; modelId: ID; releaseId: ID | null
  params: RenderParams; mediaId: ID | null
  status: 'draft' | 'queued' | 'running' | 'done' | 'failed'; jobId: ID | null; error: Msg | null
  rating: number; favorite: boolean; label: string
  stale: boolean            // input or params changed after the last successful render (frontend sets it)
  batchId: ID | null        // takes created together by "多人试唱"
}

interface MixVocal { nodeId: ID; gainDb: number; pan: number }       // a render node (pan -1..1)
interface MixData {
  vocals: MixVocal[]                                                  // 1+ takes (duets allowed)
  backing: { nodeId: ID; port: SourcePort; gainDb: number } | null    // usually a source's 'inst'
  processor: string                                                   // 'builtin' or a plugin id
  preset: string | null                                               // builtin preset id
  params: Record<string, number | string | boolean>                   // per the processor's ParamSpec[]
  variants: number                                                    // 1..4
  outputs: { mediaId: ID; seed: number; lufs: number | null }[]
  selected: number                                                    // index into outputs
  status: 'draft' | 'queued' | 'running' | 'done' | 'failed'; jobId: ID | null; error: Msg | null
  stale: boolean; label: string
}
```

Port → media resolution (backend, for render inputs and mix backing):
`mix` → `source.mediaId`; `vocals` → `source.stems.vocals`; `inst` → `source.stems.inst`.
A missing stem is a 409 `node.input_missing` (`{port}` in params).

**Migration** (backend, on project load, idempotent): every `stem` node is folded into its
source (`stems.vocals/inst` = the stem's mediaId, status `done` if both exist); render nodes
whose `inputNodeId` pointed at a stem get `input = {nodeId: <source>, port: <stem kind
'instrumental' → 'inst'>}`; `inputNodeId` pointing at a source becomes `port: 'mix'`; old
edges are dropped; missing `stems`/`stale`/`batchId` get defaults. The migrated project is
saved once.

### Endpoints (changes)

| Method & path | Body | Response |
|---|---|---|
| POST `/projects/{id}/nodes/{nodeId}/separate` | — | `Job` (type `separate`). Updates the **source node's `stems`** (queued → running → done/failed) via `project.nodes` events. No new nodes. |
| POST `/projects/{id}/nodes/{nodeId}/render` | — | `Job` as before; input media from `input` (port resolution above). Clears `stale`. |
| POST `/projects/{id}/nodes/{nodeId}/mix` | — | `Job` (type `mix`). Reads the mix node, resolves vocals (each render's `mediaId`, must be `done`) and backing, runs the processor (§2/§3), writes `variants` outputs as media (`kind: 'mix'`), updates the node (`outputs`, `selected: 0`, status). Result `{outputs}`. |
| GET `/projects/{id}/media/{mediaId}/analysis` | — | `{bpm: number \| null, lufs: number, peakDb: number, duration: number}` (cached per media). |
| GET `/mix/processors` | — | `MixProcessor[]` — the built-in mixer first, then enabled plugins of kind `mix`. |

```ts
interface ParamSpec {
  id: string; label: Msg | string; type: 'number' | 'boolean' | 'choice' | 'text' | 'folder' | 'file'
  default: number | string | boolean | null
  min?: number; max?: number; step?: number; unit?: string
  options?: { value: string; label: Msg | string }[]
}
interface MixProcessor {
  id: string                       // 'builtin' | plugin id
  name: Msg | string
  kind: 'builtin' | 'plugin'
  presets: { id: string; name: Msg | string }[]    // builtin only; [] for plugins
  params: ParamSpec[]
  maxVocals: number                // builtin: 4; command plugins: 1 unless the manifest says more
  available: boolean; error: Msg | null           // plugin not runnable (missing python/packages)
}
```

`Media.kind` gains `'mix'`. `project.nodes` events carry source/render/mix updates alike.

---------------------------------------------------------------------------------------------

## 2. Built-in mixer ("一键混音")

Pure numpy/scipy (+ librosa for BPM), **no VST, no GPL dependencies** (pedalboard is GPL-3 —
do not use it). Runs in a subprocess: `python -m stellaria_backend.mixing.run <job.json>`,
stdout JSON lines `{"progress": 0.4, "stage": "mix.reverb"}` / `{"log": "..."}` /
`{"result": {...}}`. Input/output files are 44.1 kHz float WAV (stereo out).

Chain (inspired by a hand-tuned VST chain; values are starting points, presets adjust them):

- Analysis: BPM from the backing (librosa beat tracking on a 22.05 kHz mono copy, halved
  above 100 BPM); integrated LUFS (BS.1770 K-weighting + gating) of each input; short-term
  RMS distribution of the vocal → compressor/de-esser thresholds.
- Vocal: gain staging to ~-20 dBFS RMS → high-pass 90 Hz → cleanup EQ (gentle low cut
  ~230 Hz 6 dB/oct, -2 dB bell 2.7 kHz) → de-esser (split band ≥6.5 kHz, threshold from the
  band's envelope stats, max ~6 dB) → presence EQ (+3 dB bell 1.4 kHz Q1.15, +2.5 dB 8.5 kHz
  Q1.0, scaled by `brightness`) → RMS compressor (attack ~19 ms, release ≈ the 1/8-note value
  nearest 350–500 ms at the song's BPM, ratio ~3–5, knee 4.5 dB, threshold from analysis,
  make-up gain).
- Space: BPM-synced pre-delay and decay; reverb by FFT convolution with **synthesized impulse
  responses** (early reflections + exponentially decaying filtered noise, stereo decorrelated)
  for room / plate / chamber / hall, blended per preset; reverb return high-pass 150–300 Hz,
  low-pass 8 kHz, bus EQ (-4 dB 1.45 kHz, +5 dB 2.3 kHz wide); optional 1/4-note delay throw
  (`airy` preset).
- Backing: loudness-matched so the processed vocal sits `vocalLevel` LU above the backing in
  integrated loudness; a static -1.5 dB pocket around 2.5 kHz.
- Duets: each vocal processed separately, panned per `MixVocal.pan`, gains per `gainDb`.
- Master: sum → soft clipper (-3.6 dB, ~20 % softness) → look-ahead limiter (ceiling -1 dBFS
  sample peak with 2× oversampling estimate, release synced) → normalize to `targetLufs`.
- Variants: `variants > 1` re-runs with small seeded jitter (EQ ±0.5 dB, ±150 Hz, compressor
  attack ±6 ms, release ±60 ms, reverb pre-delay/decay ±10–30 %), like the original script's
  random mode. Seed 0 = no jitter.

Presets (`preset`): `natural` (default), `pop`, `airy`, `dry`. Params (ParamSpec ids):
`vocalLevel` (dB, -6..+6, default +1), `reverb` (0..1, default by preset), `brightness`
(-1..1, default 0), `width` (0..1, reverb stereo width), `targetLufs` (-23..-9, default -14),
`backingGain` (dB offset, -12..+6, default 0).

Quality bars (tests): output true peak ≤ -0.9 dBFS, integrated loudness within ±0.5 LU of
`targetLufs`, no NaN/inf, a 3-minute stereo song mixes in **< 15 s** on a laptop CPU.

---------------------------------------------------------------------------------------------

## 3. Plugins — ✅ done

Implemented (backend: `stellaria_backend/plugins/`, router `api/plugins.py`) and merged into
`api.md` → Plugins (types `Plugin`, `PythonCheck`, `PluginWizard`; endpoints, manifest, tokens,
runtimes, interpreters, statuses, error codes, `plugin`/`plugin.deleted` events). How to write a
plugin, with a template for a friend-style `auto_mixing.py`: `docs/plugins.md`. Frontend
(Settings → 插件, mix-node processor list): open.

-----

## 4. Use your own Python for the app runtime (shell) — ✅ done

Implemented in the shell and merged into `api.md` §2: `shell.json` `runtime {mode, pythonPath}`,
`ShellState.runtime.mode` / `pythonPath`, commands `check_python`, `set_runtime_python`,
`install_runtime_requirements` (`PythonCheck` is defined there too). Onboarding: a successful
check + `set_runtime_python` skips the managed install (and `bootstrap_start` completes at once
while a passing custom interpreter is selected).

The user's environment is never modified: packages it lacks (and add-ons) go into Stellaria's
layer over it (`<data>/runtime/layer`, `api.md` §2) — a venv, or a shadow interpreter for
embeddable Pythons. When a layer cannot work, the way out is the app's own runtime. Onboarding and Settings check the environments found in the background and
mark the best one (torch with CUDA).

---------------------------------------------------------------------------------------------

## 5. Use local model files (Resources)

> **Status — backend: done** (merged into `api.md`; also accepts a DDSP-SVC `pretrain/<name>/`
> folder for zip resources).

| Method & path | Body | Response |
|---|---|---|
| POST `/resources/{id}/import` | `{path}` | `Job` (type `download`, `stage.import_local`): copies the file (or extracts the zip) into place, verifies sha256 when the manifest has one (`resource.checksum` on mismatch). |

For people who already have `pretrain/` folders from DDSP-SVC, or no access to the download
mirrors.

---------------------------------------------------------------------------------------------

## 6. Implementation split and internal interfaces

Work happens in parallel; these Python interfaces are fixed so the pieces meet.

**Mixer** — package `stellaria_backend/mixing/` (owner: mixer team). Must not import torch.

```python
# stellaria_backend/mixing/__init__.py
def builtin_processor() -> dict            # MixProcessor wire dict for 'builtin' (presets, params with Msg labels)
def analyze(path: Path) -> dict            # {"bpm": float | None, "lufs": float, "peakDb": float, "duration": float}
# CLI: python -m stellaria_backend.mixing.run <job.json>
# job.json = {"vocals": [{"path": str, "gainDb": float, "pan": float}], "backing": {"path": str, "gainDb": float} | None,
#             "preset": str, "params": dict, "seed": int, "output": str}
# stdout JSON lines: {"progress": 0..1, "stage": "mix.analyze" | "mix.vocal" | "mix.reverb" | "mix.master"}
#                    {"log": str}   and finally   {"result": {"lufs": float, "peakDb": float, "bpm": float | None}}
# exit code 0 on success; on failure a final {"error": {"code": str, "message": str}} line and exit code 1.
```

**Plugins** — package `stellaria_backend/plugins/` + router `api/plugins.py` (owner: plugins team) —
**done**. `run_plugin` also takes optional `names` (input path → readable file stem),
`on_progress` (callback 0..1 instead of `jctx.progress`, e.g. to fold variants into one job) and
`require_enabled` (default True → `plugin.disabled`). `check_python` also takes `torch`,
`min_version`, `env`, `timeout`.

```python
# stellaria_backend/plugins/registry.py
class PluginRegistry:                       # ctx.plugins (constructed in context.py)
    def list(self) -> list[Plugin]; def get(self, pid: str) -> Plugin
    def mix_processors(self) -> list[dict]  # MixProcessor wire dicts for enabled plugins of kind 'mix'
# stellaria_backend/plugins/runner.py
def run_plugin(ctx, jctx, plugin_id: str, *, vocals: list[Path], backing: Path | None,
               input: Path | None, output: Path, params: dict) -> Path
    # runs in a subprocess, streams logs to jctx, honours cancel, returns the produced audio file
    # (output, or the first glob match copied to output); raises ApiError(plugin.*) on failure
# stellaria_backend/plugins/pycheck.py
def check_python(path: Path, packages: list[str] | None = None) -> dict    # PythonCheck wire dict
```

**Projects / mix job / resources** (owner: backend team) — **done**: migration, stems on sources,
port inputs, `POST …/nodes/{id}/mix` (calls the mixer CLI per variant or `run_plugin`),
`GET /mix/processors` (`[builtin_processor(), *ctx.plugins.mix_processors()]`), media
analysis endpoint (caches `analyze()`), `POST /resources/{id}/import`. Code:
`creation/canvas.py`, `creation/mix.py`, `engine/rendering.py`, `resources/manager.py`. Until
`mixing/` lands, the builtin processor description, `analyze()` and the mixer fall back to
in-backend stand-ins (plain sum + peak normalize; no BPM).

**Shell** (owner: shell team): §4.

**Frontend** (owner: frontend): canvas v2, mix node + inspector, 多人试唱 batch dialog,
N-way compare, Settings → 插件 / 运行环境, Resources → 使用本地文件.

---------------------------------------------------------------------------------------------

## 7. Base models (底模) and adjustable training parameters

> **Status — backend: done** (merged into `api.md`: `BaseModel`, `/base-models*`,
> `TrainConfig.baseModelId`, seeding `model_0.pt` (`stage.seed_base`), the PATCH rules, manifest
> kind `base`). Also done: `POST /models/import-external` (same validation subprocess),
> `RenderParams.f0Min/f0Max`, and error `params` on HTTP and job errors. Frontend: open.

A base model is a reflow DDSP-SVC checkpoint (usually trained on many voices) that a new
voice's training starts from, so small datasets train faster and sound better. DDSP-SVC resumes
from the newest `model_<step>.pt` in the experiment folder, so seeding works by writing the
base as `model_0.pt` before the first training run.

```ts
interface BaseModel {
  id: ID; name: string; notes: string; createdAt: string
  source: 'download' | 'local' | 'release'      // manifest download, imported .pt, or one of the user's releases
  resourceId: ID | null                           // source 'download': the manifest resource (status/progress live there)
  status: 'ready' | 'missing' | 'downloading' | 'failed'
  encoder: TrainConfig['encoder']; nLayers: number; nChans: number; nAuxLayers: number; nAuxChans: number
  nSpk: number; step: number; size: number
}
// TrainConfig gains: baseModelId: ID | null
```

| Method & path | Body | Response |
|---|---|---|
| GET `/base-models` | — | `BaseModel[]` (manifest entries of kind `base` included, `status: 'missing'` until downloaded via `/resources/{id}/install`) |
| POST `/base-models/import` | `{ckptPath, configPath?, name}` | `Job` (type `import_base`; validates in a subprocess: reflow model, encoder/vocoder compatible, loads; result `{baseModelId}`) |
| POST `/base-models/from-release` | `{modelId, releaseId, name}` | `BaseModel` (copies that release's checkpoint + config) |
| PATCH `/base-models/{id}` | `{name?, notes?}` | `BaseModel` |
| DELETE `/base-models/{id}` | — | `{ok}` (409 `base.in_use` while a model that hasn't started training references it) |

Training rules:
- `POST /models` / `PATCH /models/{id}` with `config.baseModelId`: the structural fields
  (`encoder`, `nLayers`, `nChans`, `nAuxLayers`, `nAuxChans`) are taken from the base model
  (request values ignored).
- On the first training run (no checkpoints yet) with a base model: write `model_0.pt` from the
  base — `global_step = 0`, optimizer state dropped, speaker-embedding tensors dropped when
  `n_spk` differs (the engine loads with `strict=False`; only same-shape tensors may remain).
  Log what was kept/dropped. Stage code `stage.seed_base`.
- `PATCH /models/{id}` `config` while the model isn't training: non-structural fields
  (`batchSize`, `lr`, `ampDtype`, `cacheDevice`, `numWorkers`, `intervalVal`,
  `intervalForceSave`, `usePitchAug`, `useFusedKernels`, `f0Extractor` (re-preprocess)) are
  always allowed; structural fields or `baseModelId` after the first checkpoint → 409
  `train.config_mismatch`.
- Manifest: support `kind: "base"` resources (same download/mirror machinery); none are
  listed until verified sources exist.
