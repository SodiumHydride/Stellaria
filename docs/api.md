# Stellaria API contract

Two interfaces:

1. **Backend HTTP + WebSocket** (Python FastAPI) — used by the frontend for everything about the
   library, jobs, audio, metrics and downloads.
2. **Shell bridge** (Tauri commands + events) — runtime bootstrap, backend lifecycle, window
   and OS integration.

Shapes are given as TypeScript; the canonical copy is `desktop/src/api/types.ts` and the
backend mirrors them with pydantic models (camelCase on the wire).

---

## 1. Backend

### Process

```
python -m stellaria_backend
    --library <dir>          library root (created if missing)
    --data <dir>             app data dir (models/, logs/, preferences.json)
    --engine <dir>           DDSP-SVC sources (contains train_reflow.py; engine/ in this repo)
    --resources <file>       manifest.json
    [--uv <path>]            uv binary (used to install add-on packages)
    [--port 0] [--token T]   0 = pick a free port
    [--dev]                  fixed port 8765 if --port not given, token "dev", CORS *
    [--fake-engine]          simulate preprocess/train/render (UI development and tests)
```

On startup it prints exactly one line to stdout:
`STELLARIA_READY {"port": 51234, "token": "…", "version": "0.1.0"}` and then logs to stderr
and `<data>/logs/backend.log`.

### Conventions

* Base URL `http://127.0.0.1:<port>/api`. Auth: header `Authorization: Bearer <token>`, or
  query `?token=` (needed for `<audio src>` and `<img src>`).
* JSON uses camelCase. Times are ISO-8601 strings in UTC. Durations are seconds (float).
* Errors: HTTP 4xx/5xx with body `{"error": {"code": "clip_not_found", "message": "…", "detail": "…", "params": {}}}`.
  `code` is stable and localized by the frontend (with `params`, like a `Msg`); `message` is an English fallback.
  Failed jobs carry the same `{code, message, detail, params}` in `Job.error`.
* Localizable messages anywhere are `Msg = {code: string, params?: Record<string, string|number>}`.
* Media endpoints support HTTP Range.

### Types

```ts
type ID = string
type Msg = { code: string; params?: Record<string, string | number> }

// ---------- system ----------
interface GpuInfo { index: number; name: string; vramTotal: number; vramUsed?: number; utilization?: number; temperature?: number; computeCapability?: string; driver?: string }
interface SystemInfo {
  version: string; os: 'windows' | 'macos' | 'linux'; osVersion: string; arch: string
  cpu: string; cpuCount: number; ramTotal: number
  gpus: GpuInfo[]
  torch: { available: boolean; version?: string; cuda?: string; device: 'cuda' | 'mps' | 'cpu'; triton: boolean } // probed once in a subprocess
  canTrain: boolean; trainBlockers: Msg[]            // e.g. {code:'train.need_nvidia'}
  libraryPath: string; dataPath: string; engineVersion: string
  fakeEngine: boolean
}
interface Preferences {
  language: 'zh-CN' | 'en'
  mirrorRegion: 'auto' | 'cn' | 'global'
  separationModel: string | null        // vocal separation model for organizing, health checks and separations
                                        // that name none (GET /resources/separation); null = the best installed
  cleanupModel: string | null           // cleanup (dereverb) model for organizing and dereverb nodes that name
                                        // none (GET /resources/cleanup); null = the best installed
  gpuPolicy: 'smart' | 'cpu' | 'queue'  // render/separate while training: smart (default) = on the GPU next to
                                        // training if ≥ 2.5 GB VRAM is free, else on the CPU at once; cpu = always
                                        // on the CPU; queue = wait (waiting_gpu) until training pauses or ends
  gpuIdleReleaseMin: number             // 0–240, default 10: minutes without singing after which the render
                                        // worker exits (its GPU / memory is freed); 0 = keep it loaded
  lowPowerWhileTraining: boolean         // frontend reduces effects while a train job runs
  autoSlimAfterTraining: boolean
  notifyOnFinish: boolean
  glass: boolean
  exportFolder: string | null            // default folder of export nodes and "导出"; null = <library>/exports
}

// ---------- characters & material ----------
type CharacterStage = 'empty' | 'material' | 'ready' | 'training' | 'trained'
interface Character {
  id: ID; name: string; color: string; hasAvatar: boolean; avatarVersion: number
  tags: string[]; notes: string
  consent: { source: string; authorized: boolean; note: string }
  pinned: boolean; archived: boolean
  createdAt: string; updatedAt: string; lastOpenedAt: string | null
  stage: CharacterStage
  stats: { rawCount: number; clipCount: number; activeDuration: number; trainCount: number; valCount: number; trashedCount: number }
  health: { score: number; level: HealthLevel; checkedAt: string; stale: boolean } | null
  modelIds: ID[]                      // models containing this character
  primaryModelId: ID | null           // single-character model, if any
  pitch: { low: number; high: number; median: number } | null   // Hz, from health check (p5/p95/median)
}
interface RawImport { id: ID; name: string; ext: string; size: number; duration: number | null; importedAt: string; status: 'imported' | 'organizing' | 'organized' | 'failed'; error: Msg | null; clipCount: number }
type ClipFlag = 'too_short' | 'too_long' | 'clipping' | 'silent' | 'noisy' | 'outlier' | 'music' | 'f0_failed' | 'duplicate'
interface ClipMetrics { peakDb: number; rmsDb: number; silenceRatio: number; clippingRatio: number; snrDb: number | null; f0Low: number | null; f0High: number | null; f0Median: number | null; voicedRatio: number | null; outlierScore: number | null; musicRatio: number | null }
interface Clip {
  id: ID; name: string; file: string            // file relative to the character dir
  rawId: ID | null; rawName: string | null; start: number | null; end: number | null
  duration: number; status: 'active' | 'trashed'; split: 'train' | 'val'
  flags: ClipFlag[]; metrics: ClipMetrics | null; reviewed: boolean
  createdAt: string
}
interface OrganizeOptions {
  separateVocals: boolean                       // needs add-on "separator"
  dereverb: boolean                             // default false: slice the dry voice (after separation when
                                                // separateVocals), made by the default cleanup model; needs a
                                                // ready one (the job fails with cleanup.model_missing otherwise)
  normalize: boolean; targetDb: number          // RMS-based loudness target, default -20 dBFS
  slice: { enabled: boolean; thresholdDb: number; minLength: number; minInterval: number; hopSize: number; maxSilKept: number }  // seconds; slicer.py semantics; enabled=false keeps already-cut files as they are
  maxLength: number                             // longer slices are split again at quiet points, default 15 s
  minDuration: number                           // default 2.0 (engine requirement)
  dropClipping: boolean; dropSilent: boolean; dedupe: boolean
  namePattern: string                           // default "{name}_{index:04d}"; tokens {name} (character), {index}
                                                // ({index:03}, {index:04} …), {source} (raw file stem); needs {index}
  valCount: number                              // default 10 (clamped to 2..20 and ≤ 5% of clips)
  rawIds: ID[] | null                           // null = all raw imports not organized yet
}
type HealthLevel = 'excellent' | 'good' | 'fair' | 'poor' | 'blocked'
interface HealthIssue {
  code: string                    // e.g. 'data.too_little', 'clips.clipping', 'clips.too_short', 'val.too_few'
  severity: 'info' | 'warning' | 'error'      // error = blocks training
  params: Record<string, string | number>
  clipIds: ID[]
  fix: { action: 'trash_clips' | 'organize' | 'add_material' | 'pick_val' | 'install_addon'; params?: Record<string, unknown> } | null
}
interface HealthReport {
  score: number; level: HealthLevel; blocking: boolean; checkedAt: string
  totals: { clips: number; duration: number; trainDuration: number; valCount: number }
  pitch: { histogram: { midi: number; seconds: number }[]; low: number | null; high: number | null; median: number | null } // Hz
  issues: HealthIssue[]
  checks: { basic: boolean; pitch: boolean; outlier: boolean; music: boolean }   // which checks ran
}

// ---------- models & training ----------
type TrainPresetId = 'vram6' | 'vram8' | 'vram12' | 'vram24' | 'custom'
interface TrainConfig {
  preset: TrainPresetId
  targetSteps: number; batchSize: number; lr: number
  ampDtype: 'fp16' | 'bf16' | 'fp32'; cacheDevice: 'cpu' | 'cuda'; numWorkers: number
  intervalVal: number; intervalForceSave: number
  f0Extractor: 'rmvpe' | 'fcpe' | 'parselmouth' | 'harvest' | 'dio' | 'crepe'
  encoder: 'contentvec768l12tta2x' | 'contentvec768l12' | 'hubertsoft'
  nLayers: number; nChans: number; nAuxLayers: number; nAuxChans: number   // structural (see PATCH /models/{id})
  usePitchAug: boolean; useFusedKernels: 'auto' | boolean
  baseModelId: ID | null          // training starts from this base model (底模); structure comes from it
}
interface BaseModel {             // 底模: a reflow checkpoint a new voice's training starts from
  id: ID; name: string; notes: string; createdAt: string
  source: 'download' | 'local' | 'release'      // manifest resource, imported .pt, or a copy of one of the user's releases
  resourceId: ID | null                           // source 'download': the Resource (install/progress live there)
  status: 'ready' | 'missing' | 'downloading' | 'failed'   // local/release: always 'ready'
  encoder: TrainConfig['encoder']; nLayers: number; nChans: number; nAuxLayers: number; nAuxChans: number
  nSpk: number; step: number; size: number
}
interface TrainPreset { id: TrainPresetId; vram: number; config: Partial<TrainConfig>; recommended: boolean }
type ModelStatus = 'idle' | 'preparing' | 'preprocessing' | 'training' | 'stopping' | 'paused' | 'done' | 'failed'
interface Checkpoint { step: number; file: string; size: number; createdAt: string; kind: 'regular' | 'permanent' | 'pause'; releaseId: ID | null }
interface Release { id: ID; name: string; step: number; createdAt: string; notes: string; size: number }
interface Model {
  id: ID; name: string; kind: 'single' | 'constellation'
  characters: { characterId: ID; spkId: number }[]
  config: TrainConfig
  status: ModelStatus; error: Msg | null
  step: number; targetSteps: number
  dataset: { status: 'none' | 'stale' | 'ready'; preparedAt: string | null; clipCount: number; duration: number }
  checkpoints: Checkpoint[]; releases: Release[]
  createdAt: string; updatedAt: string
  trainingTime: number           // seconds spent training (sum of runs)
  speed: number | null           // steps/s, recent
  eta: number | null             // seconds to targetSteps
  diskUsage: { dataset: number; checkpoints: number; releases: number; total: number } | null
}
interface ScalarSeries { tag: string; points: [step: number, value: number, wallTime: number][] }
interface ValidationSample { name: string; gtUrl: string; predUrl: string; melUrl: string }  // urls relative to API base
interface ValidationSnapshot { step: number; samples: ValidationSample[] }

// ---------- creation ----------
interface Media {
  id: ID; name: string; kind: 'import' | 'stem' | 'render' | 'mix' | 'mixdown' | 'process'
  duration: number; sampleRate: number; channels: number; createdAt: string
  offset: number                 // seconds: where this audio starts on its song's timeline (0 for imports; see "Offsets")
}
interface RenderParams {
  key: number; formant: number; register: number          // semitones
  f0Extractor: 'rmvpe' | 'fcpe' | 'parselmouth' | 'harvest' | 'dio' | 'crepe'
  steps: number; method: 'euler' | 'rk4'; tStart: number
  threshold: number                                       // dB, response threshold
  spkMix: Record<string, number> | null                   // characterId → weight (constellations)
  region: { start: number; end: number } | null
  f0Min?: number | null; f0Max?: number | null            // Hz, pitch search range (engine defaults 50 / 1100); 422 request.invalid unless f0Min < f0Max
  pitchEdit?: PitchEdit | null                            // the pitch editor's changes
}
// Semitone offsets on a regular grid (rate per second, 20–400; the editor uses its analysis frames,
// 44100/512 ≈ 86.13), only where something changed; seconds from the start of the node's input.
// Applied to the extracted f0 before the key shift (unvoiced frames interpolated first, as the engine
// does); values clamped to ±24, at most 30 min of values; 422 request.invalid when malformed.
interface PitchEdit { rate: number; segments: { start: number; values: number[] }[] }
// Canvas v3 (docs/canvas-v3.md §1): every edge is audio; edges are derived from node data
// (render.input, mix.inputs[].ref, process.inputs[]); canvas.edges is always stored as [].
type NodeType = 'source' | 'audio' | 'render' | 'mix' | 'process' | 'note' | 'group'   // v1 'stem' nodes are migrated away
type StemPort = 'vocals' | 'inst' | 'drums' | 'bass' | 'other'
type SourcePort = 'mix' | StemPort                                   // outputs of a song node
interface PortRef { nodeId: ID; port: string }                       // an output port: a SourcePort, 'out', 'part2', …
interface CanvasNode {
  id: ID; type: NodeType; position: { x: number; y: number }
  width?: number; height?: number; parentId?: ID
  data: SourceData | AudioData | RenderData | MixData | ProcessData | NoteData | GroupData
}
interface SourceData {                                                // a song
  mediaId: ID; name: string
  stems: {                                                            // filled by running the node (separation) or by stems import
    status: 'none' | 'queued' | 'running' | 'done' | 'failed'
    vocals: ID | null; inst: ID | null                                // media ids
    drums: ID | null; bass: ID | null; other: ID | null               // 4-stem split only (inst = drums + bass + other)
    provided: StemPort[]                                              // imported by the user; a split never overwrites them
    model: 'htdemucs' | 'htdemucs_ft' | null                          // what produced the separated stems
    jobId: ID | null; error: Msg | null
  }
  bpm: number | null; key: string | null                              // filled in the background after the import ('C#m', …)
}
interface AudioData {                                                 // an imported track that isn't a song to split
  mediaId: ID; name: string
  role: 'backing' | 'vocal' | 'other'                                 // hint for a mix input's default role
}
interface RenderData {
  input: PortRef                     // any node's output port
  characterId: ID; modelId: ID; releaseId: ID | null   // null = latest checkpoint
  params: RenderParams; mediaId: ID | null
  status: 'draft' | 'queued' | 'running' | 'done' | 'failed'; jobId: ID | null; error: Msg | null
  rating: number; favorite: boolean; label: string
  stale: boolean            // inputs or params changed after the last successful run (see "Staleness")
  batchId: ID | null        // takes created together by "多人试唱"
}
interface MixInput {
  id: ID; ref: PortRef
  role: 'lead' | 'harmony' | 'backing' | 'other'
  gainDb: number; pan: number                                         // pan -1..1
  mute: boolean; solo: boolean
}
interface MixData {
  inputs: MixInput[]                                                  // replaces v2 vocals[] + backing
  processor: string                                                   // 'builtin' (auto mix) | 'bus' (plain sum) | a plugin id
  preset: string | null                                               // builtin preset id
  params: Record<string, number | string | boolean>                   // per the processor's ParamSpec[]
  variants: number                                                    // 1..4 (bus: always 1)
  outputs: { mediaId: ID; seed: number; lufs: number | null }[]
  selected: number                                                    // index into outputs (port 'out' plays this one)
  status: 'draft' | 'queued' | 'running' | 'done' | 'failed'; jobId: ID | null; error: Msg | null
  stale: boolean; label: string
}
type ProcessOp = 'trim' | 'split' | 'pitch' | 'fx' | 'separate' | 'dereverb' | 'comp' | 'concat' | 'export'
interface ProcessData {
  op: ProcessOp; label: string
  inputs: PortRef[]                     // trim/split/pitch/fx/separate/dereverb/export: exactly 1; comp: 2–8; concat: 2–32
  params: TrimParams | SplitParams | PitchParams | FxParams | SeparateParams | DereverbParams | CompParams | ConcatParams | ExportParams
  outputs: Record<string, ID | null>    // port → media id (kind 'process')
  status: 'draft' | 'queued' | 'running' | 'done' | 'failed'; jobId: ID | null; error: Msg | null
  stale: boolean
  bypass: boolean                       // pitch/fx: the node passes its input through (no job)
  result: Record<string, unknown> | null   // export: {path, downloadUrl}; others: op facts (pitch: {engine}, separate: {model, stems}, dereverb: {model})
}
// op → output ports, params (seconds unless noted)
// trim     out                 {start, end, fadeIn, fadeOut}                    (within the input)
// split    part1 … partN       {cuts: number[], fade, labels: string[]}         (N = cuts + 1)
// pitch    out                 {semitones, cents, tempo: 0.5–2, preserveFormant: boolean, quality: 'fast' | 'high'}
// fx       out                 {chain: FxSlot[]}                                (FxSlot/EffectSpec: docs/canvas-v3.md §3)
// separate vocals, inst (+ drums, bass, other)   {model: 'htdemucs' | 'htdemucs_ft', stems: 2 | 4}
// dereverb dry, reverb         {model: 'auto' | cleanup model id}                (reverb = input − dry)
// comp     out                 {segments: {start, end, input}[], crossfade}     (song-timeline seconds; input -1 = silence)
// concat   out                 {gap, crossfade}
// export   —                   ExportParams
interface ExportParams {
  format: 'wav' | 'flac' | 'mp3'; sampleRate: 44100 | 48000; bitDepth: 16 | 24 | 32; mp3Bitrate: 192 | 256 | 320
  loudness: number | null               // LUFS target, null = keep
  folder: string | null                 // null = Preferences.exportFolder, else <library>/exports
  fileName: string                      // template: {project} {song} {voice} {label} {date} {index}
  keepPosition: boolean                 // pad `offset` seconds of silence in front
  tags: { title: string; artist: string } | null
}
interface ParamSpec {
  id: string; label: Msg | string; type: 'number' | 'boolean' | 'choice' | 'text' | 'folder' | 'file'
  default: number | string | boolean | null
  min?: number; max?: number; step?: number; unit?: string
  options?: { value: string; label: Msg | string }[]
  group?: string                   // fine-grained setting, shown folded under "advanced" by group
}
interface MixProcessor {
  id: string                       // 'builtin' | 'bus' | plugin id
  name: Msg | string
  kind: 'builtin' | 'plugin'       // 'bus' is kind 'builtin' too
  presets: { id: string; name: Msg | string;      // builtin only; [] for bus and plugins
             params: Record<string, number | string | boolean> }[]  // the preset's value of every
                                   // param whose default is null (what the controls show while unset)
  params: ParamSpec[]
  maxVocals: number                // lead + harmony inputs (builtin: 4; command plugins: 1 unless the manifest says more —
                                   // more vocals than that are pre-summed for plugins)
  maxInputs?: number               // builtin 12, bus 16
  available: boolean; error: Msg | null           // plugin not runnable (missing python/packages)
}
interface NoteData { text: string; color: string }
interface GroupData { label: string; color: string }
interface CanvasEdge { id: ID; source: ID; target: ID }
interface TimelineClip { id: ID; mediaId: ID; nodeId: ID | null; start: number; offset: number; duration: number; gainDb: number; fadeIn: number; fadeOut: number; mute?: boolean }
interface Track { id: ID; name: string; kind: 'vocal' | 'instrumental' | 'other'; gainDb: number; pan: number; mute: boolean; solo: boolean; clips: TimelineClip[] }
interface Project {
  id: ID; name: string; createdAt: string; updatedAt: string; coverColor: string   // from a muted palette
  canvas: { nodes: CanvasNode[]; edges: CanvasEdge[]; viewport: { x: number; y: number; zoom: number } }
  timeline: { tracks: Track[]; length: number; loop?: { start: number; end: number } | null; masterGainDb?: number }
}
interface ProjectSummary { id: ID; name: string; createdAt: string; updatedAt: string; coverColor: string; renderCount: number; characterIds: ID[] }

// ---------- jobs ----------
type JobType = 'import' | 'organize' | 'health' | 'prepare' | 'preprocess' | 'train' | 'render' | 'separate' | 'mix' | 'mixdown' | 'download' | 'install_addon' | 'export' | 'import_pack' | 'slim' | 'install_plugin' | 'plugin_test' | 'import_model' | 'import_base' | 'vst_scan' | 'vst_editor' | 'process' | 'diagnostics'
type JobStatus = 'queued' | 'waiting_gpu' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled'
interface Job {
  id: ID; type: JobType; title: Msg
  target: { characterId?: ID; modelId?: ID; projectId?: ID; nodeId?: ID; resourceId?: ID; pluginId?: ID; vstPluginId?: ID }
  status: JobStatus; progress: number | null; stage: Msg | null; eta: number | null
  usesGpu: boolean; createdAt: string; startedAt: string | null; finishedAt: string | null
  error: { code: string; message: string; detail: string; params: Record<string, string | number> } | null
  result: Record<string, unknown> | null
}

// ---------- resources ----------
interface Resource {
  id: string                      // 'contentvec' | 'rmvpe' | 'nsf_hifigan' | 'hubertsoft' | 'separator' | ...
  kind: 'model' | 'addon' | 'base'; required: boolean   // 'base': a downloadable base model (see /base-models)
  name: Msg; description: Msg; size: number
  status: 'missing' | 'downloading' | 'installing' | 'ready' | 'failed'
  progress: number | null; bytesDone: number | null; speed: number | null; source: string | null
  error: Msg | null; path: string | null
  license: string | null; homepage: string | null   // from the manifest (Resources and Credits show them)
}

// ---------- plugins ----------
interface PythonCheck {
  ok: boolean                     // usable (error null) and nothing missing
  executable: string              // sys.executable as the interpreter reports it (or the path tried)
  version: string | null          // '3.11.9'
  missing: string[]               // requirements not installed (by normalized distribution name; markers and
                                  // version specifiers honoured when `packaging` is importable there)
  error: Msg | null               // python.not_found {path} | python.failed {path, reason} | python.check_failed {reason}
                                  // | python.timeout {seconds} | python.too_old {version, min} (python-runtime plugins
                                  // need 3.8+) | plugin.env_failed {reason}. Same shape as the shell's check_python.
  torch: { version: string; cuda: string | null; cudaAvailable: boolean } | null   // null: not installed / not probed
}
interface Plugin {
  id: ID; name: string; version: string; kind: 'mix' | 'effect'; runtime: 'command' | 'python'
  folder: string; manifestSource: 'folder' | 'wizard'
  interpreter: 'builtin' | 'custom'; pythonPath: string | null
  status: 'ready' | 'needs_setup' | 'error'; check: PythonCheck | null   // null for commands that do not use {python}
  params: ParamSpec[]; settings: ParamSpec[]
  settingValues: Record<string, unknown>   // effective values (manifest defaults + the user's)
  enabled: boolean; addedAt: string
  error: Msg | null                        // why status is not 'ready' (see Plugins below)
  requirements: string[]; maxVocals: number; checkedAt: string | null
}
interface PluginWizard {                   // POST /plugins/wizard
  folder: string; name: string; kind: 'mix' | 'effect'
  entry: string                            // file in the folder (.py → ["{python}", "{folder}/<entry>", ...args])
  args?: string[]                          // default ['{backing}', '{vocal}'] (mix) / ['{input}', '{output}'] (effect)
  outputs?: string[]                       // default ['output/*.flac', 'output/*.wav', 'output/*.mp3'] unless args use {output}/{outdir}
  cwd?: 'temp' | 'folder'                  // default 'temp'
  runtime?: 'command' | 'python'           // default 'command'; 'python' = entry has process(job)
  requirements?: string[]; timeoutSec?: number; maxVocals?: number
}

// ---------- VST3 / Audio Unit host (add-on `vsthost`) ----------
interface VstParam {
  id: string                       // pedalboard's Python attribute name (key for FxSlot.params)
  name: string                     // as the plugin shows it
  kind: 'float' | 'bool' | 'choice'
  min: number | null; max: number | null                 // float only (display units, not 0..1)
  default: number | string | boolean | null              // display value of the plugin's default
  unit: string | null              // e.g. 'dB', 'Hz' (null if the plugin reports none)
  choices: string[] | null         // choice only
}
interface VstPlugin {
  id: ID                           // stable: sha1(normalized path + (pluginName ?? ''))
  name: string; vendor: string | null; version: string | null
  format: 'vst3' | 'au'; path: string
  pluginName: string | null        // set for plugins inside a shell bundle (e.g. Waves); null otherwise
  category: 'effect' | 'instrument'
  params: VstParam[]               // [] when failed
  status: 'ok' | 'failed'
  error: Msg | null                // vst.crashed {name} | vst.timeout {name, seconds} | vst.load_failed {name, reason}
  scannedAt: string
}
interface VstStatus {
  hostInstalled: boolean; hostVersion: string | null   // pedalboard importable in the runtime (detected without importing it)
  formats: ('vst3' | 'au')[]                           // ['vst3', 'au'] on macOS, ['vst3'] elsewhere
  folders: string[]                                    // the user's folders (PATCH /vst/settings)
  defaultFolders: string[]                             // always scanned when they exist
  editorSupported: boolean                             // Windows and macOS
  lastScan: string | null; scanning: boolean
}
```

### Endpoints

All paths below are relative to `/api`.

**System**

| Method & path | Body / query | Response |
|---|---|---|
| GET `/health` | — | `{ok: true, version}` (no auth) |
| GET `/system` | — | `SystemInfo` |
| GET `/system/gpu` | — | `GpuInfo[]` (live NVML/nvidia-smi values, cached ≤ 1 s) |
| GET `/system/render` | — | `RenderStatus` `{loaded, busy, device: 'cuda'\|'mps'\|'cpu'\|null, idleSeconds: number\|null, releaseAfterMin}`: the render worker (singing models, pitch analysis). Changes are pushed as `render.status` events. |
| POST `/system/render/release` | — | `{released: boolean, ...RenderStatus}`: the worker exits now and frees its GPU / memory (the next render starts it again, a few seconds slower); `released: false` while it is singing or nothing is loaded. The tray's "free GPU memory" calls this. |
| GET `/preferences` / PUT `/preferences` | `Partial<Preferences>` | `Preferences` (`separationModel` / `cleanupModel`: `'auto'` or `''` = null; an unknown id is 422 `separation.unknown_model` / `cleanup.unknown_model`) |
| GET `/disk` | — | `{libraryFree, libraryTotal, characters: {id, bytes}[], models: {id, bytes}[], projects: {id, bytes}[]}` |
| POST `/shutdown` | — | `{ok}`; stops jobs gracefully (training gets STOP) and exits |
| POST `/system/diagnostics` | — | `Job` (type `diagnostics`, title `job.diagnostics`, stage `stage.diagnostics`); result `{path, downloadUrl}` of a zip in `<data>/exports/diagnostics/` with backend + shell logs (`<data>/logs/`, large files: their last 20 MB), `about.json` (versions, platform), `system.json`, `preferences.json`, `resources.json`, `jobs.json` (type/status/error), `library.json` (counts, sizes, node kinds per project) — no audio, no models |

**Characters**

| Method & path | Body / query | Response |
|---|---|---|
| GET `/characters` | `?archived=0\|1` | `Character[]` |
| POST `/characters` | `{name, color?, tags?}` | `Character` |
| GET `/characters/{id}` | — | `Character` |
| PATCH `/characters/{id}` | `Partial<{name,color,tags,notes,consent,pinned,archived}>` | `Character` |
| DELETE `/characters/{id}` | `?deleteModels=0\|1` | `{ok}` |
| PUT `/characters/{id}/avatar` | multipart `file` (png/jpg/webp, resized to 256²) | `Character` |
| GET `/characters/{id}/avatar` | `?v=` | image |
| POST `/characters/{id}/opened` | — | `Character` (updates lastOpenedAt) |
| GET `/characters/{id}/raw` | — | `RawImport[]` |
| POST `/characters/{id}/raw` | JSON `{paths: string[]}` (absolute paths; folders are walked) **or** multipart `files[]` | `Job` (type `import`; converts/validates; result `{rawIds}`) |
| DELETE `/characters/{id}/raw/{rawId}` | `?withClips=0\|1` | `{ok}` |
| GET `/characters/{id}/clips` | `?status=active\|trashed\|all` | `Clip[]` |
| PATCH `/characters/{id}/clips` | `{ids: ID[], set: Partial<{status, split, reviewed, name}>}` | `Clip[]` (changed). `name`: one clip at a time; 422 `clip.bad_name` if unusable as a file name, 409 `clip.name_taken` if another clip of the character has it |
| POST `/characters/{id}/clips/{clipId}/split` | `{at: number}` (seconds) | `Clip[]` (the two new clips; original trashed) |
| POST `/characters/{id}/clips/empty-trash` | — | `{deleted: number}` |
| POST `/characters/{id}/clips/rename` | `{pattern?: string}` (tokens as `namePattern`: `{name}`, `{index}`, `{index:03}`, `{index:04}`, `{source}`) | `Clip[]` (renumbers active clips; 422 `organize.bad_pattern`) |
| POST `/characters/{id}/clips/pick-val` | `{count?: number}` | `Clip[]` (re-picks validation clips: diverse lengths/pitch, never flagged) |
| GET `/characters/{id}/clips/{clipId}/audio` | — | audio/wav |
| GET `/characters/{id}/clips/{clipId}/peaks` | `?n=800` | `{peaks: number[] /* 0..1, n values */, duration}` |
| GET `/characters/{id}/organize/defaults` | — | `OrganizeOptions` |
| POST `/characters/{id}/organize/preview` | `{rawId, slice: OrganizeOptions['slice'], maxLength, separateVocals?: false, n?: 600}` (optional, defaults as `OrganizeOptions`: `normalize, targetDb, minDuration, dropSilent, dropClipping`) | `{duration, peaks: number[] /* n values 0..1 */, segments: {start, end, dropped: null \| 'too_short' \| 'silent' \| 'clipping'}[], separated: boolean}` — synchronous; same slicer and max-length splitting as organize on that raw file (vocals only if organize already separated them: `separated`); 409 `raw.not_ready` while the file is still being imported |
| POST `/characters/{id}/organize` | `Partial<OrganizeOptions>` | `Job` (type `organize`; ends by running a health check) |
| POST `/characters/{id}/health` | `{advanced?: boolean}` | `Job` (type `health`) |
| GET `/characters/{id}/health` | — | `HealthReport \| null` |
| POST `/characters/{id}/export` | `{includeMaterial: boolean, path?: string}` | `Job` (type `export`, result `{path}` of the `.stella` file; in browser mode `{downloadUrl}`) |
| POST `/packs/import` | `{path}` or multipart `file` | `Job` (type `import_pack`, result `{characterIds, modelIds}`) |

Stage rules: `empty` (no raw, no clips) → `material` (has clips but last health is blocking or
missing) → `ready` (health not blocking) → `training` (a model containing it is preparing,
preprocessing or training) → `trained` (some model containing it has a release or a checkpoint).

**Models & training**

| Method & path | Body / query | Response |
|---|---|---|
| GET `/train/presets` | — | `TrainPreset[]` (recommended one based on the GPU) |
| GET `/models` | `?characterId=` | `Model[]` |
| POST `/models` | `{name?, characterIds: ID[], config?: Partial<TrainConfig>}` | `Model` (spkId assigned 1..n in given order). With `config.baseModelId` the structural fields come from that base model (request values ignored; 404 `base_not_found`) |
| POST `/models/import-external` | `ExternalImport` (below) | `Job` (type `import_model`; result `{characterId, characterIds, modelId, releaseIds}`) — a DDSP-SVC reflow model trained elsewhere becomes a voice without material |
| GET `/models/{id}` | — | `Model` |
| PATCH `/models/{id}` | `{name?, config?: Partial<TrainConfig>, targetSteps?}` | `Model` (409 `model.busy` while training; structural rules below) |
| DELETE `/models/{id}` | — | `{ok}` |
| POST `/models/{id}/train` | `{fresh?: boolean}` | `Job` (type `train`). Runs `prepare` (assemble dataset; skipped if fingerprint unchanged) → `preprocess` (skipped if unchanged) → train (resumes from the latest checkpoint unless `fresh`). One job covers all three; `stage` tells which. |
| POST `/models/{id}/pause` | — | `Model` (writes `exp/STOP`; job ends as `succeeded` with result `{paused: true}`; status → `paused`) |
| POST `/models/{id}/stop` | — | same as pause, kept for clarity in UI ("finish now") |
| GET `/models/{id}/metrics` | `?tags=a,b&since=<step>&maxPoints=1500` | `ScalarSeries[]` (all tags when `tags` omitted; downsampled keeping min/max) |
| GET `/models/{id}/validation` | `?step=` (default latest) | `ValidationSnapshot \| null` |
| GET `/models/{id}/validation/steps` | — | `number[]` |
| GET `/models/{id}/validation/{step}/{name}/{gt\|pred}.wav` | — | audio/wav (extracted from tfevents, cached) |
| GET `/models/{id}/validation/{step}/{name}/mel` | — | `{frames, bins, data /* base64 uint8, row-major frames×bins, 0=-14 dB..255=3.5 dB */, gtData}` — mels of pred and gt computed from audio |
| GET `/models/{id}/log` | `?tail=500` | `{lines: string[]}` (exp/log_info.txt + captured stdout) |
| POST `/models/{id}/releases` | `{step, name?, notes?}` | `Release` (copies the checkpoint + config into releases/) |
| PATCH `/models/{id}/releases/{relId}` | `{name?, notes?}` | `Release` |
| DELETE `/models/{id}/releases/{relId}` | — | `{ok}` |
| DELETE `/models/{id}/checkpoints/{step}` | — | `{ok}` |
| POST `/models/{id}/slim` | `{dataset: boolean, checkpoints: 'none' \| 'keep_latest' \| 'releases_only'}` | `Job` (type `slim`) |
| POST `/models/{id}/export` | `{releaseId, path?}` | `Job` (`.stella` with characters' metadata + this release) |
| POST `/models/{id}/quick-render` | `{path}` or multipart `file`, plus `{characterId, releaseId?, params?}` | `Job` (type `render`; result `{url, device}` — `url` of the output wav; for "try it" on the model page) |
| GET `/models/{id}/quick-render/{jobId}/audio` · `/peaks` | `?n=` for peaks | the quick render's wav (`result.url`) / `{peaks, duration}` |
| GET `/base-models` | — | `BaseModel[]` (local ones by creation time, then manifest resources of kind `base`, `status: 'missing'` until installed via `/resources/{id}/install` or `/import`) |
| GET `/base-models/{id}` | — | `BaseModel` (404 `base_not_found`) |
| POST `/base-models/import` | `{ckptPath, configPath?, name}` | `Job` (type `import_base`; `configPath` defaults to `config.yaml` next to the checkpoint; validated like external models; stages `stage.inspect` → `stage.import_local`; result `{baseModelId}`) |
| POST `/base-models/from-release` | `{modelId, releaseId, name}` | `BaseModel` (copies that release's checkpoint + config; empty `name` → the release's name, prefixed with the model's name unless it already starts with it; 422 `model.import_incompatible` for non-reflow releases) |
| PATCH `/base-models/{id}` | `{name?, notes?}` | `BaseModel` (409 `base.read_only` for manifest bases) |
| DELETE `/base-models/{id}` | — | `{ok}` (409 `base.in_use` `{models}` — comma-separated names — while a model that hasn't saved a checkpoint uses it; 409 `base.read_only` for manifest bases, remove those under Resources) |

```ts
interface ExternalImport {
  ckptPaths: string[]             // one or more .pt checkpoints of the same model (min 1)
  configPath?: string             // default: config.yaml next to the first checkpoint
  characterId?: ID                // attach speaker 1 to this voice …
  newCharacter?: { name: string; color?: string }   // … or create it
  characterIds?: ID[]             // multi-speaker models: speaker i+1 → characterIds[i] (overrides characterId)
  name?: string                   // model name; default newCharacter.name, else the config's folder name
}
```

External models (`/models/import-external`): validated in a subprocess (the API process never
imports torch): the config must be a reflow model (`model.type: RectifiedFlow`) at 44.1 kHz / hop
512 with an `nsf-hifigan` vocoder and a supported encoder (`contentvec768l12tta2x`,
`contentvec768l12`, `hubertsoft`, with matching `encoder_out_channels`), else the job fails with
`model.import_incompatible` `{reason: 'not_reflow' | 'sample_rate' | 'vocoder' | 'encoder' | 'model'}`
(detail: the config summary as JSON). Every checkpoint is then loaded strictly (as rendering
does); one that doesn't load fails with `model.import_bad_checkpoint` `{file}` (detail: the load
error). 422 `import.not_found` (sync) for missing files. The model gets the config's structure
(`encoder`, `nLayers`, `nChans`, `nAuxLayers`, `nAuxChans`, `usePitchAug`, a known `f0Extractor`)
and `targetSteps` = the highest step; its `glu_type` is kept (configs without one were trained
with `atanglu`, the engine's rule). Each checkpoint is copied to `exp/model_<step>.pt` (step =
`global_step` in the checkpoint, else the file name `model_<step>.pt`, else made up; made unique)
and published as a Release named `<name> · <step>`; status `done`. The voices' stage becomes
`trained` (no clips needed). `n_spk > 1` → a constellation: speaker *i* goes to `characterIds[i-1]`
(or `characterId`/`newCharacter` for speaker 1); unmapped speakers get new voices named
`<name> spk<i>` (`<name>` alone for a single speaker). `kind` follows the speaker count. On
failure nothing is left behind (the model and voices it created are removed). Stages:
`stage.inspect` → `stage.import_local` `{file, index, total}`.

Training config rules (`PATCH /models/{id}` `config`, only while not training):
- Non-structural fields (`batchSize`, `lr`, `ampDtype`, `cacheDevice`, `numWorkers`,
  `intervalVal`, `intervalForceSave`, `usePitchAug`, `useFusedKernels`, `f0Extractor` (re-runs
  preprocessing), `targetSteps`, `preset`) are always allowed.
- Structural fields (`encoder`, `nLayers`, `nChans`, `nAuxLayers`, `nAuxChans`) and
  `baseModelId` → 409 `train.config_mismatch` (detail: the changed fields) once training saved a
  checkpoint (a step-0 base seed doesn't count). With a `baseModelId` the structural fields always
  come from the base model, so sending them changes nothing.
- Base models: on a training run that starts without checkpoints, the base is written as
  `exp/model_0.pt` first (stage `stage.seed_base`): `global_step` 0, no optimizer state, only
  tensors whose name and shape match the new model (the speaker embedding is dropped when the
  speaker count differs; the engine loads with `strict=False`); the job log lists what was kept
  and dropped. The seed is deleted once training saved a newer checkpoint, and when the base or
  structure changes before that. A base that is gone fails the job before preprocessing
  (`base_not_found`, or `engine.missing_model` `{resource}` for an uninstalled manifest base).
- Manifest resources of kind `base` carry `base: {encoder, nLayers, nChans, nAuxLayers,
  nAuxChans, nSpk, step, gluType?, config?}` (`config`: the config file name next to the
  checkpoint, default `config.yaml`); none are listed until verified sources exist.

**Projects (creation)**

| Method & path | Body / query | Response |
|---|---|---|
| GET `/projects` | — | `ProjectSummary[]` |
| POST `/projects` | `{name?}` | `Project` |
| GET `/projects/{id}` | — | `Project` |
| PUT `/projects/{id}` | `{name?, canvas?, timeline?, coverColor?}` | `Project` (frontend saves debounced; server bumps updatedAt; `canvas.edges` is stored as `[]`) |
| DELETE `/projects/{id}` | — | `{ok}` |
| POST `/projects/{id}/duplicate` | — | `Project` |
| GET `/projects/{id}/media` | — | `Media[]` |
| POST `/projects/{id}/media` | `{paths}` or multipart `files[]` | `Job` (type `import`; result `{mediaIds}`; media `offset: 0`). Each imported file is then analysed in the background (bpm + key) and song nodes playing it get `bpm`/`key` by `project.nodes` (a song node saved later takes them from the cache at once) |
| GET `/projects/{id}/media/{mediaId}/audio` | — | audio |
| GET `/projects/{id}/media/{mediaId}/peaks` | `?n=` | `{peaks, duration}` |
| GET `/projects/{id}/media/{mediaId}/pitch` | — | `{hop: number /*s*/, f0: number[] /* Hz, 0 = unvoiced */}` (cached) |
| GET `/projects/{id}/media/{mediaId}/analysis` | — | `{bpm: number \| null, lufs: number, peakDb: number, duration: number, key: string \| null}` (cached per media; `key` like `'C#m'`, Krumhansl profile on chroma) |
| GET `/projects/{id}/media/{mediaId}/sections` | — | `{bpm, beats: number[], downbeats: number[], phrases: {start, end}[], sections: {start, end, label: 'intro' \| 'verse' \| 'chorus' \| 'bridge' \| 'outro' \| null}[]}` (cached; phrases from the song node's vocals stem when it has one) — for the split/comp/trim editors |
| POST `/projects/{id}/media/{mediaId}/export` | `{format: 'wav'\|'flac'\|'mp3', path?: string, normalize?: false}` | `Job` (type `export`; result `{path}`, or `{downloadUrl, path}` without `path` — download via `/files/{token}`) |
| POST `/projects/{id}/nodes/{nodeId}/run` | `{stems?: 2 \| 4}` (songs only) | `Job` for any runnable node (reads the node from the saved project): song → separation (type `separate`, 2 stems unless `stems: 4`), render (type `render`), mix (type `mix`), process (type `process`, title `job.process.<op>` `{name}`; `separate` and `dereverb` ops use the GPU queue like separations). 409 `node.input_missing` `{nodeId, port}`, 409 `node.not_runnable` `{nodeId}` (note, group, audio, a bypassed fx/pitch node, a song whose stems were all provided), 409 `node.busy` `{nodeId, jobId}` (its job is still queued/running), 422 `process.input_count` `{min, max}`, 409 `dsp.unavailable` |
| POST `/projects/{id}/nodes/{nodeId}/render` · `/separate` · `/mix` | as `/run` | v2 aliases of `/run` (with their old type errors 422 `node.not_render` / `node.not_source` / `node.not_mix`) |
| GET `/projects/{id}/nodes/{nodeId}/pitch` | — | the pitch line of a render node's input: `{frameS, duration, midi: (number \| null)[] (per frame, null = unvoiced), notes: {start, end, pitch}[], key: {tonic 0–11, mode, confidence} \| null, madeFrom: {mediaId, f0Extractor, f0Min, f0Max}, current, offset}` — `current` = still made from the node's input and pitch settings; `offset` = the input's place in the song. 404 `pitch.not_analyzed`, 409 `node.input_missing`, 422 `node.not_render` |
| POST `/projects/{id}/nodes/{nodeId}/pitch/analyze` | — | `Job` (type `pitch`, GPU queue, CPU fallback, one at a time with renders): the render worker tracks the input's pitch with the node's extractor on the analysis grid (512 samples at 44.1 kHz, shared with the render f0 cache); stored in `<project>/pitch/<nodeId>.json` |
| POST `/projects/{id}/nodes/{nodeId}/pitch/preview` | `{start, end, pitchEdit?}` (seconds of the input) | `Job` (type `pitch`): sings `start`–`end` (≤ 60 s) with the node's voice and settings and `pitchEdit`; result `{nodeId, start, end, url}` — `url` = `/projects/{id}/pitch-preview/{jobId}/audio` (WAV). The node is not changed. 422 `request.invalid` (empty range), `node.incomplete`, `render.no_checkpoint` |
| POST `/projects/{id}/nodes/{nodeId}/stems` | JSON `{paths: [path], port}` / `{path, port}`, or multipart `files` + form field `port` (same shapes as `/media`) | the song `CanvasNode`: the file becomes that stem (media `kind: 'import'`, offset 0, named `<song> · <stem>`), is added to `stems.provided`, readers of that port turn stale (`project.nodes`). With both `vocals` and `inst` present, `status` becomes `done`. 422 `request.invalid` (port), 422 `node.not_source`, 422 `import.not_found` |
| DELETE `/projects/{id}/nodes/{nodeId}/stems/{port}` | — | the song `CanvasNode` without that stem; only provided stems (409 `stems.not_provided` `{port}`) |
| GET `/mix/processors` | — | `MixProcessor[]` — `builtin` (auto mix), `bus` (plain sum: no presets, params `normalize`, `targetLufs`), then enabled plugins of kind `mix` |
| POST `/presets/export` | `{path, file}` | `{path}` — writes the user's presets to a `.stellapreset` file (suffix added when missing). `file` = `{format: 'stellaria-presets', version: 1, mix: {name, processor, preset, params}[], fx: {name, chain: FxSlot[]}[]}`; every preset needs a name. Errors `preset.invalid_file {reason}`, `preset.bad_path` (not absolute), `preset.write_failed` |
| POST `/presets/import` | `{path}` | the file's `{format, version, mix, fx}` (checked as above, ≤ 2 MB); the frontend adds them (same names are replaced). Errors `preset.not_found` (404), `preset.invalid_file {reason: 'json' \| 'format' \| 'version' \| 'size' \| 'mix' \| 'fx'}` |
| GET `/fx/effects` | — | `EffectSpec[]` (built-in effects, docs/canvas-v3.md §3; `[]` until the effects package is installed) |
| GET `/projects/{id}/suggest-key` | `?mediaId=&characterId=` | `{key: number, sourceRange: {low, high, median}, characterRange: {low, high, median}, confidence: 'high'\|'low'}` |
| POST `/projects/{id}/mixdown` | `{format: 'wav'\|'flac'\|'mp3', path?: string, normalize?: boolean (peak −1 dBFS, default true), loudness?: number (LUFS −40…−5, true peak ≤ −1 dBTP; wins over normalize), sampleRate?: 44100\|48000, bitDepth?: 16\|24\|32 (32 = float WAV; FLAC ≤ 24), mp3Bitrate?: 192\|256\|320, start?: number, end?: number (both = only that range), stems?: boolean, timeline?: object (the arrangement as on screen; else the saved one)}` | `Job` (type `mixdown`; renders the timeline: clip `mute`, fades/offsets, track gain/pan/mute/solo, `timeline.masterGainDb`; a 2 ms ramp where a clip cuts into its audio, none where a split continues. `path` = the file (in a folder: a new name, never an overwrite); none = the exports folder + `downloadUrl`. Result `{mediaId?, path, downloadUrl?}` (`mediaId`: the whole song is kept as `mixdown` media). `stems`: one file per audible track, `"{project} - 01 {track}"`, all moved by the gain that levels their sum, into the folder `path` (none: a zip download); result `{paths, folder, path?, downloadUrl?}`. 409 `mixdown.empty`) |

Port → media resolution (every job input): song `mix` → `mediaId`, `vocals|inst|drums|bass|other`
→ `stems[port]`; audio/render `out` → `mediaId`; mix `out` → `outputs[selected].mediaId`; process
→ `outputs[port]`, and a bypassed fx/pitch node resolves to its own input. Anything unresolved is a
409 `node.input_missing` `{nodeId, port}` naming the node that lacks the audio (past bypassed
nodes). v2's `port: 'mix'` on a non-source node is read as `out`.

Offsets (`Media.offset`, seconds on the song's timeline): imports, stems imports 0; separation =
input offset; trim/split part = input offset + region start; render = input offset (+
`params.region.start` — only the region is written); pitch with `tempo ≠ 1` = input offset ÷
tempo; fx = input offset; mix (builtin, bus, plugins) and comp = the earliest playing input's
offset, inputs placed at `offset − min`; concat 0. Export writes the audio as is
(`keepPosition` pads `offset` seconds of silence in front).

Canvas migration (on load, idempotent, saved once; a `PUT` from an older client is migrated
too): v1 `stem` nodes fold into their source's `stems`, render `inputNodeId` becomes `input`;
`port: 'mix'` on a non-source node becomes `out`; mix `vocals[]` become `inputs` (`id = nodeId`,
`ref {nodeId, port: 'out'}`, role `lead`) and `backing` a `backing` input (`id = "<nodeId>:<port>"`,
`pan 0`), `mute/solo` false; songs gain `stems.drums/bass/other: null`, `provided: []`, `model:
null`, `bpm/key: null`; audio/process/mix/render nodes get missing defaults; edges are dropped;
media without `offset` read as 0.

Staleness and ownership (`PUT /projects/{id}` merge, `project.nodes` events): server-owned
fields — `status`, `jobId`, `error`, `outputs`, `result`, `mediaId`, `stems`, `bpm`, `key` — keep
their stored values for every node type when the client saves an existing node (a new node is
stored as sent). `stale` = incoming OR stored: the client can set it, only a successful run clears
it. When a run succeeds with new outputs, the node's direct dependents (also those reading
through bypassed fx/pitch nodes; for songs only readers of the ports that changed) get `stale:
true` in the same `project.nodes` event; the frontend runner handles the transitive closure.
A mix's `selected` (reset to 0 by a run) is re-applied to saves until the client echoes it.

Mix jobs (CPU, the `dsp` pool): job JSON v2 for the mixer (`inputs` with `path, role, gainDb, pan,
offset, mute, solo`; `reference {vocals, mix, offset}` when the first playing lead input traces
back — through any nodes — to a song with a `vocals` stem; `processor: 'builtin' | 'bus'`).
Plugin processors get vocals = the playing lead + harmony inputs (one pre-summed file when there
are more than the plugin's `maxVocals`) and backing = the sum of the backing/other inputs, all
placed on one timeline (gain/pan applied). Errors: 409 `mix.no_inputs`, 409 `mix.no_vocals`
(auto mix and plugins need a playing lead/harmony), 422 `mix.too_many_vocals` / `mix.too_many_inputs`
(`{max}`), 409 `mix.processor_unavailable` (`{processor}`), and the mixer's own codes (e.g.
`mix.failed`). Stages `stage.mix.analyze|vocal|reverb|master` (mixer) or `stage.mix.plugin`
(`{processor}`), with params `{variant, variants}`. `bus` makes one variant. Outputs: media
`kind: 'mix'`, 44.1 kHz stereo; node `outputs`, `selected: 0`, `status`; result `{nodeId, outputs}`.

Process jobs (`docs/canvas-v3.md` §2): every op but `separate` and `dereverb` runs `python -m
stellaria_backend.dsp.run <job.json>` in the `dsp` CPU pool (`max(1, cpu_count // 4)` process +
mix jobs at once; no torch). The job gets the inputs (`path`, `offset`), `params`, the song's
`bpm`, `vstPlugins` (the `VstPlugin` of every active VST slot: 409 `vst.host_missing`, 404
`vst.not_found` `{name}`, 422 `vst.instrument` `{name}` are checked when the run starts) and
`names` for export file names (`project`, `song` = the song upstream, `voice` = the nearest take's
voice, `label` = the node's label or the song, `date` = YYYY-MM-DD, `index` = this export node's
1-based position among export nodes). Outputs become media `kind: 'process'` at the offsets the
op computed; the node gets `outputs`, `result`, `status: 'done'`; result `{nodeId, outputs,
result}`. Stages `stage.dsp.<op>` (or `stage.dsp.<name>` as the processor reports them, with its
extra fields as params). Errors: the processor's own codes with params (e.g. `dsp.*`, `vst.*`),
`process.failed`, and `vst.crashed` `{name}` when an fx run with VST slots dies without an error
line (a plugin crashed the process). `export` writes into `params.folder` →
`Preferences.exportFolder` → `<library>/exports` (created as needed), never overwrites (`name
(2).flac`), and has `result {path, downloadUrl}` and no outputs. `separate` ops use the
separation add-on (GPU queue, `gpuPolicy`), `stems: 4` keeps drums/bass/other (from Demucs on the
instrumental when the vocals come from a Roformer), `model` is any separation model id (`auto`/absent
= `Preferences.separationModel` → the best installed); result `{model: <resolved id>, stems}`.
`dereverb` ops use the same worker with a cleanup model (GPU queue, CPU fallback, stage
`stage.dereverb` `{device?}`): port `dry` is the voice without reverb (and echo, by model), `reverb`
the rest (input − dry), both 44.1 kHz with the input's channel count; `model` is any cleanup model id
(`auto`/absent = `Preferences.cleanupModel` → the best installed); result `{model: <resolved id>}`.
Errors: 409 `addon.missing` / `addon.outdated` `{resource: 'separator'}`, 409 `cleanup.model_missing`
`{model, resource?}` (for `auto` with none ready: the recommended download), 409 `cleanup.unavailable`
(no cleanup model exists at all), 422 `cleanup.unknown_model` `{model}`, 500 `cleanup.failed` /
`cleanup.model_mismatch`.

Render semantics: input audio = the media behind `input`; output written to a new media
(`kind: render`, offset as above); the node is updated server-side and sent as `project.nodes`.
While a training holds the GPU, renders and separations follow `gpuPolicy`; a render placed on
the CPU that way shows `stage: {code: 'stage.render_cpu'}`, and render results carry
`device: 'cuda' | 'mps' | 'cpu'` (the device actually used). `releaseId: null` renders the latest
checkpoint, also while that model is still training (a completely written copy is used).

**Jobs**

| Method & path | Body / query | Response |
|---|---|---|
| GET `/jobs` | `?active=1` | `Job[]` (newest first, last 200) |
| GET `/jobs/{id}` | — | `Job` |
| GET `/jobs/{id}/log` | `?tail=500` | `{lines: string[]}` |
| POST `/jobs/{id}/cancel` | — | `Job` |
| POST `/jobs/clear` | — | `{ok}` (removes finished jobs from the list) |

**Resources**

| Method & path | Body / query | Response |
|---|---|---|
| GET `/resources` | — | `Resource[]` |
| POST `/resources/{id}/install` | `{source?: string}` | `Job` (type `download` or `install_addon`) |
| POST `/resources/{id}/import` | `{path}` | `Job` (type `download`, stage `stage.import_local`): the model file itself, the official zip, or a folder laid out like the zip's contents (e.g. DDSP-SVC's `pretrain/nsf_hifigan/`; a file inside it works too). sha256 is verified when the manifest has one for that form (files, official zips): 422 `resource.checksum` on mismatch; `resource.incomplete` (`{files}`) when files are missing. Licence/notice texts are optional. |
| POST `/resources/import-folder` | `{path}` | `{found: [{id, path, jobId}], missing: [id]}`: every model not installed yet that is found in a folder someone already has (a DDSP-SVC checkout or bundle, its `pretrain/`, or one model's folder), each started as an `/import` job. Searched up to 4 levels deep by folder name (`contentvec`, `rmvpe`, `*hifigan*`, `hubert`) and file names; Python environments (`pyvenv.cfg`, `conda-meta`) and datasets are skipped; the older vocoder file `model` counts as `model.ckpt`. On the same drive files are hard-linked instead of copied. Also returns `separation: SeparationImportResult` for vocal separation and cleanup checkpoints found there (see below). 422 `import.not_found` when `path` is not a folder. |
| GET `/resources/separation` | – | `{models: SeparationModel[], defaultModel}`. A model is `{id, engine: demucs\|roformer, arch, source: addon\|resource\|local, resourceId, name: Msg (inline zh-CN/en), quality: standard\|high, sdr, ready, missing: addon\|addon_update\|model\|null, size, path}`; Demucs models first, then Roformer checkpoints (best first), then imported ones (newest first). `defaultModel` is the `separationModel` preference while that model is ready, else the best ready one (Roformer by SDR, imported Roformers, then `htdemucs`). Changes are pushed as `separation` events. |
| POST `/resources/separation/import` | `{path, config?}` | `{found: [{id, name, arch, role: separation\|cleanup, kind, path, config, jobId, resourceId}], skipped: [{path, reason: Msg}]}`: BS-/Mel-Band-Roformer checkpoints in a Music-Source-Separation-Training (or MSST-GUI / MSST-WebUI) folder, or one checkpoint with its YAML: vocal models (an output named vocals) and cleanup models (an output named dry / noreverb, or a two-instrument model returning the other one); `kind` (cleanup only) is `dereverb_echo` / `denoise` / `dereverb` from the file name. Configs are matched by MSST-GUI's `model_config_*.json` or by file name (`<ckpt>.yaml`, `<ckpt stem>.yaml`, `config_<stem>.yaml`); checkpoints named like a manifest checkpoint import as that resource. Each starts a `download` job (`job.import_separation` / `job.import_cleanup`); files are hard-linked on the same drive. Skip reasons: `separation.skip.no_config \| bad_config \| arch \| params \| no_vocals \| duplicate`. |
| DELETE `/resources/separation/{id}` | – | removes an imported separation model; `SeparationModels`. 409 `separation.read_only` for built-in ones, 404 `separation_model_not_found`. |
| GET `/resources/cleanup` | – | `{models: CleanupModel[], defaultModel: string \| null}`: vocal cleanup models (the dry voice without reverb / echo / noise). A model is `{id, kind: dereverb\|dereverb_echo\|denoise, arch, source: resource\|local, resourceId, name: Msg (inline zh-CN/en), sdr, ready, missing: addon\|addon_update\|model\|null, size, path}`; manifest checkpoints in manifest order (the recommended one first), then imported ones (newest first). They run in the `separator` add-on. `defaultModel` is the `cleanupModel` preference while that model is ready, else the first ready `dereverb` / `dereverb_echo` model, else `null` (nothing ready). Changes are pushed as `cleanup` events. Kept apart from `/resources/separation` because the two lists feed different pickers and defaults; one folder import finds both. |
| DELETE `/resources/cleanup/{id}` | – | removes an imported cleanup model; `CleanupModels`. 409 `separation.read_only` for downloadable ones (remove those as resources), 404 `cleanup_model_not_found`. |
| DELETE `/resources/{id}` | — | `Resource` |
| POST `/resources/install-required` | — | `Job[]` (all missing required models) |
| GET `/resources/sources` | — | `{region, measured: {sourceId, latencyMs \| null}[]}` |

<a id="plugins"></a>**Plugins** (Settings → 插件; how to write one: [`plugins.md`](plugins.md))

| Method & path | Body / query | Response |
|---|---|---|
| GET `/plugins` | — | `Plugin[]` (in the order they were added; enabled plugins whose check is stale — interpreter or requirements changed, builtin environment gone — are re-checked first, once) |
| GET `/plugins/{id}` | — | `Plugin` |
| POST `/plugins` | `{folder}` (absolute) | `Plugin` when the folder has a `stellaria-plugin.json` (checks the interpreter at once; adding the same folder again returns it). 409 `plugin.no_manifest` with `detail: {candidates: string[], requirements: string[], folder}` — script files found (`.py` with a `__main__` guard first, then other `.py`, then programs) and third-party imports guessed from the folder's `.py` files (prefill for the wizard). 422 `plugin.bad_manifest`, 409 `plugin.exists` (`{id, folder}`: another folder already uses that id), 404 `plugin.folder_missing` |
| POST `/plugins/wizard` | `PluginWizard` | `Plugin` (id from the name, e.g. `sodium`, `sodium-2`; writes `<data>/plugins/<id>.json`, never into the user's folder) |
| PATCH `/plugins/{id}` | `{enabled?, interpreter?, pythonPath?, settingValues?, name?}` | `Plugin`. Re-checks when the interpreter or `pythonPath` changes. `settingValues` is merged (`null` removes a value); `name: null`/`''` restores the manifest's name; `interpreter: 'custom'` without a `pythonPath` → 422 `plugin.python_missing`. `pythonPath` may be an environment folder (venv/conda) |
| DELETE `/plugins/{id}` | — | `{ok}` (removes the wizard manifest, the plugin's environment and test outputs; never the user's folder). 409 `plugin.busy` while an install/test job of it runs |
| POST `/plugins/{id}/check` | — | `Plugin` (re-runs the interpreter check; creates the builtin environment if missing) |
| POST `/plugins/{id}/install-requirements` | `{target: 'builtin' \| 'custom'}` | `Job` (type `install_plugin`, target `{pluginId}`; stages `stage.plugin_env` → `stage.install` → `stage.check`; result `{pluginId, target, installed: string[], status}`; skips the download when nothing is missing). Fails with `plugin.install_failed` (detail: installer output), or `plugin.packages_missing` if still missing afterwards. `custom` installs only the missing requirements into the user's interpreter (409 `plugin.python_missing` without one) |
| POST `/plugins/{id}/test` | `{params?}` | `Job` (type `plugin_test`, target `{pluginId}`; runs the plugin on the bundled 8-second test song + vocal (synthesized once into `<data>/plugins/testdata/`; effects get the vocal); also works while the plugin is disabled; result `{mediaUrl, pluginId, duration}`) |
| GET `/plugins/{id}/test/{jobId}/audio` · `/peaks` | `?n=` for peaks | the test output WAV (`result.mediaUrl`) / `{peaks, duration}`; the last 5 per plugin are kept. 404 `plugin.test_not_found` |
| POST `/python/check` | `{path, packages?: string[]}` | `PythonCheck` (always probes torch) |

Plugin semantics:

* Manifest (`stellaria-plugin.json`, or the wizard's copy in `<data>/plugins/<id>.json` with an extra
  `folder`): `{id, name, version, kind: 'mix'|'effect', runtime: 'command'|'python', command?: string[],
  entry?: string, cwd: 'temp'|'folder' (default 'temp'), outputs: string[] (globs relative to the working
  folder), maxVocals?: number (default 1), params: ParamSpec[], settings: ParamSpec[], requirements:
  string[], timeoutSec (default 1200)}`; `id` is `[a-z0-9][a-z0-9_-]{0,63}` and not `builtin`. Manifests are
  re-read on every request, so edits apply at once; `plugin.bad_manifest` `{reason}` names the problem
  (unknown token, `{vocals}` not a whole argument, `{input}` in a mix plugin, missing script/entry file,
  no way to find the output, …). Plugins whose manifest breaks later stay listed with `status: 'error'`.
* Command tokens: `{python}`, `{folder}`, `{backing}`, `{vocal}` (first vocal), `{vocals}` (a whole argument,
  one argument per vocal), `{input}` (effect), `{output}` (`<run>/out/output.wav`), `{outdir}`,
  `{params_json}` (`{params, settings, inputs, output, outdir, plugin: {id, name, folder}}`), `{setting:<id>}`,
  `{{`/`}}`. Paths are native (backslashes on Windows); arguments starting with a path token are normalized.
  A first argument that is not `{python}` is looked up in the plugin folder, then on PATH.
* A run (`stellaria_backend.plugins.run_plugin`): inputs are copied as WAVs with readable names into
  `<run>/in/` (outside the working folder), the process starts with no console window, UTF-8 I/O, no stdin,
  its own process group (cancel/timeout kill the tree) and the env vars `STELLARIA_PLUGIN_ID`,
  `STELLARIA_PLUGIN_FOLDER`, `STELLARIA_OUTPUT`, `STELLARIA_OUTDIR`, `STELLARIA_PARAMS_JSON` (and
  `STELLARIA_SEED` when params carry a numeric `seed`, as the mix job's variants do); cwd is a fresh
  `<run>/work/` (`cwd: 'temp'`) or the plugin folder (`'folder'`: only files written during the run count).
  stdout/stderr lines go to the job log; `STELLARIA_PROGRESS <0..1>` lines set progress. The result is
  `{output}` if written, else the first match of `outputs` (sorted by name), else any audio file in
  `{outdir}`; it is copied (WAV) or converted (FLAC/MP3/… at its own sample rate) to the requested WAV.
  Params are the manifest defaults overridden by the node's values (numbers clamped to min/max; unknown
  keys passed through). Job stage `stage.plugin` `{plugin}` unless the caller reports progress itself.
* `python` runtime: the backend runs `host.py` (stdlib only, Python ≥ 3.8) with the plugin's interpreter;
  it imports the entry by path (its folder on `sys.path`) and calls `process(job)` with `job.inputs`
  (`{vocals: Path[], backing: Path | None}` or `{input: Path}`), `job.output`, `job.outdir`, `job.workdir`,
  `job.folder`, `job.params`, `job.settings`, `job.progress(x)`, `job.log(text)`; returning a path makes that
  file the result. `print` goes to the log.
* Interpreters: `builtin` = a venv per plugin in `<data>/plugins/envs/<id>`, made from the app runtime
  (`uv venv --system-site-packages`, else `python -m venv`) plus a `.pth` that appends the runtime's
  site-packages, so numpy/scipy/librosa/soundfile/torch come from the app and the plugin's own installs win;
  re-created when the app runtime changes. Installs resolve with `uv pip compile` against the runtime's
  versions as constraints (fallback: only torch/torchaudio/numpy pinned), then install only what the runtime
  lacks (`--no-deps`), trying the PyPI mirrors in the add-on installer's order. `custom` = the user's
  interpreter, run as in their terminal (user site allowed, conda DLL folders on PATH on Windows); never
  installed into without an explicit `install-requirements {target: 'custom'}`.
* Status: `error` (manifest broken `plugin.bad_manifest`, folder gone `plugin.folder_missing`, interpreter
  unusable — `check.error` — or the program of a non-Python command not found `plugin.program_missing
  {program}`); `needs_setup` (`plugin.packages_missing {packages}`, `plugin.setting_missing {setting}` for a
  `{setting:<id>}` without a value, or `plugin.not_checked` when the interpreter/requirements changed since
  the last check); else `ready`. The check is persisted; `torch` is only probed for plugins that require torch.
  A run re-checks first when the last check is stale or failed. `GET /mix/processors` lists enabled `mix`
  plugins with `available = status == 'ready'` and `error` = the plugin's `error` (stale checks refreshed
  as in `GET /plugins`).
* Errors (`{code, message, detail}`; `params` reach node/job `Msg`s): `plugin.not_found` (404),
  `plugin.no_manifest`, `plugin.bad_manifest` `{reason}`, `plugin.exists` `{id, folder}`,
  `plugin.folder_missing` `{folder}`, `plugin.disabled`, `plugin.busy`, `plugin.python_missing` `{path, reason}`,
  `plugin.packages_missing` `{packages}`, `plugin.setting_missing` `{setting}`, `plugin.env_failed`,
  `plugin.install_failed`, `plugin.input_missing` `{input}`, `plugin.too_many_vocals` `{max}`,
  `plugin.timeout` `{seconds}`, `plugin.failed` `{exitCode}` (message: exit code or the exception; detail: the
  last 60 log lines), `plugin.no_output` `{outputs}` (detail: last 60 log lines), `plugin.test_not_found`.
* Job titles `job.install_plugin` / `job.plugin_test` `{plugin}`.

<a id="vst"></a>**VST3 / Audio Unit host** (optional add-on resource `vsthost` = `pedalboard==0.9.25`, GPL-3.0, installed with
`POST /resources/vsthost/install`; without it built-in effects still work and `vst` slots fail with `vst.host_missing`)

| Method & path | Body | Response |
|---|---|---|
| GET `/vst/status` | — | `VstStatus` |
| PATCH `/vst/settings` | `{folders: string[]}` (absolute; de-duplicated, empty entries dropped) | `VstStatus` (persisted in `<data>/vst/settings.json`) |
| POST `/vst/scan` | `{full?: boolean}` | `Job` (type `vst_scan`, no target; stage `stage.vst_scan` `{name, done, total}`, progress per file; result `{found, failed, scanned}` — `found`/`failed` count all known plugins, `scanned` the files loaded this time). Returns the running scan if there is one. 409 `vst.host_missing` |
| GET `/vst/plugins` | — | `VstPlugin[]` from the cache, sorted by name (instruments listed with `category: 'instrument'`; only effects can go in a chain) |
| GET `/vst/plugins/{id}` | — | `VstPlugin` (404 `vst.not_found`) |
| POST `/vst/plugins/{id}/editor` | `{state: string \| null}` | `Job` (type `vst_editor`, target `{vstPluginId}`, no progress); result `{state, params}` — the base64 `raw_state` and current values when the user closes the window; cancelling the job closes the window. 409 `vst.editor_unsupported` (Linux), `vst.host_missing` |
| POST `/vst/plugins/{id}/inspect` | `{state: string \| null}` | `{params: Record<string, number \| boolean \| string>}` — current values for a state (`null` = the plugin's defaults); runs a short subprocess (≤ 30 s) |

VST semantics:

* Folders scanned: `defaultFolders` + `folders`. Windows `%CommonProgramFiles%\VST3`,
  `%LOCALAPPDATA%\Programs\Common\VST3`; macOS `/Library/Audio/Plug-Ins/VST3`, `~/Library/Audio/Plug-Ins/VST3`,
  `/Library/Audio/Plug-Ins/Components`, `~/Library/Audio/Plug-Ins/Components`; Linux `~/.vst3`, `/usr/lib/vst3`,
  `/usr/local/lib/vst3`. Plugins are `*.vst3` (bundle or file) and, on macOS, `*.component`, found recursively
  (vendor sub-folders) but never inside a bundle; a plugin path can also be listed directly as a folder.
* Every plugin file is loaded in its own short-lived subprocess (no console window on Windows, own process
  group elsewhere) with a 30 s timeout, up to 3 at a time; shell bundles with several plugins
  (`get_plugin_names_for_file`) get one subprocess per plugin. A crash marks it `failed` with `vst.crashed`, a
  hang with `vst.timeout`, a load error with `vst.load_failed` `{name, reason}`; the scan continues.
  Cancelling the job kills the running loads and keeps what was scanned so far.
* Cache `<data>/vst/cache.json` keyed by path, with the file's mtime (bundles: newest mtime inside): a scan only
  loads new or changed files and drops uninstalled ones; `full: true` reloads everything (e.g. after
  activating a plugin that failed before).
* Parameters: `id` is pedalboard's attribute name; values in `FxSlot.params`, `inspect` and editor results are
  display values (numbers in `min..max`, booleans, choice strings). Processing clamps numbers to the range and
  ignores unknown ids and invalid choices.
* `state` is the plugin's own state (pedalboard `raw_state`, base64). Before reading it, the host runs a short
  block of silence so parameter changes made in the window reach the plugin's audio processor (VST3 delivers
  them there only while processing).
* Processing (inside the `dsp` subprocess, `stellaria_backend.vst.processing.apply`): load by `path` (+
  `pluginName`), once per process; restore `state` (or the plugin's default state when `null`); apply `params`;
  process with a reset at 44.1 kHz (48 kHz with resampling if the plugin rejects 44.1 kHz) plus up to 6 s of
  tail trimmed at −90 dBFS; mono input to a stereo-only plugin comes back stereo. pedalboard compensates the
  latency the plugin reports, so the output stays aligned with the input.
* Errors: `vst.host_missing` (409), `vst.not_found` `{name}` (404; also a plugin file that was uninstalled),
  `vst.load_failed` `{name, reason}`, `vst.crashed` `{name}`, `vst.timeout` `{name, seconds}`,
  `vst.editor_unsupported` (409), `vst.instrument` `{name}` (an instrument in an effect chain).
* Job titles `job.vst_scan`, `job.vst_editor` `{name}`.

**Filesystem helper (browser mode only)** — GET `/files/{token}` serves export results.

### WebSocket `/api/events?token=`

Server → client messages `{type, data}`:

| type | data |
|---|---|
| `hello` | `{version}` |
| `job` | `Job` (on every change; progress updates throttled to ≥ 250 ms per job) |
| `job.log` | `{jobId, lines: string[]}` (only for jobs the client subscribed to) |
| `character` | `Character` |
| `character.deleted` | `{id}` |
| `clips` | `{characterId}` (refetch clips/raw/health) |
| `health` | `{characterId, report: HealthReport}` |
| `model` | `Model` |
| `model.deleted` | `{id}` |
| `metrics` | `{modelId, step, scalars: Record<string, number>}` (appended points, ≤ 1/s) |
| `validation` | `{modelId, step}` |
| `project.nodes` | `{projectId, nodes: CanvasNode[]}` (server-side updates of nodes to merge; each node is sent whole — a finished run sends the node first, then the dependents it marked stale) |
| `project.media` | `{projectId, media: Media}` |
| `resource` | `Resource` |
| `plugin` | `Plugin` (after add/patch/check/install) |
| `plugin.deleted` | `{id}` |
| `vst.status` | `VstStatus` (folders changed, scan started/finished) |
| `vst.plugins` | `VstPlugin[]` (after a scan) |
| `gpu` | `GpuInfo[]` (every 2 s while at least one client is subscribed) |
| `preferences` | `Preferences` |
| `render.status` | `RenderStatus` (worker started, a request finished, worker released) |
| `separation` | `SeparationModels` (`GET /resources/separation`: after installs, imports, deletes) |
| `cleanup` | `CleanupModels` (`GET /resources/cleanup`: after installs, imports, deletes) |

Client → server: `{type: 'subscribe', topics: ['gpu', 'job.log:<jobId>']}` and
`{type: 'unsubscribe', topics: [...]}`.

---

## 2. Shell bridge (Tauri)

Commands (`invoke(name, args)`), all async:

```ts
interface ShellState {
  platform: 'windows' | 'macos' | 'linux'; arch: string; appVersion: string
  firstRunDone: boolean; libraryPath: string; dataPath: string; language: 'zh-CN' | 'en' | null
  termsAccepted: number          // version of the terms of use accepted (0 = none); the UI asks once per version
  managedReady: boolean          // the app's own runtime is installed (the way back from a broken custom one)
  runtime: {
    status: 'missing' | 'outdated' | 'ready' | 'installing' | 'broken'; variant: TorchVariant | null
    python: string | null; torch: string | null; cuda: string | null
    mode: 'managed' | 'custom'     // custom = the user's own interpreter runs the backend
    pythonPath: string | null      // the user's interpreter if one was configured (kept while mode is 'managed')
  }
  backend: BackendStatus
}
type TorchVariant = 'cu128' | 'cu126' | 'cpu' | 'mac'
interface BackendStatus { status: 'stopped' | 'starting' | 'ready' | 'crashed'; url: string | null; token: string | null; error: string | null; restarts: number }
// url is the origin, e.g. "http://127.0.0.1:51234" (API under url + '/api'); restarts = automatic restarts after crashes
interface HardwareReport {
  os: string; osVersion: string; cpu: string; ramTotal: number
  gpus: { vendor: 'nvidia' | 'amd' | 'intel' | 'apple' | 'other'; name: string; vram: number | null /* bytes */; driver: string | null; computeCapability: string | null }[]
  appleSilicon: boolean
  recommendedVariant: TorchVariant; canTrain: boolean
  notes: { code: string; params?: Record<string, string | number> }[]
  // 'gpu.no_nvidia' | 'gpu.low_vram' {name, vramGb} | 'gpu.too_old' {name, computeCapability} (pre-Maxwell)
  // | 'driver.too_old' {found, required} | 'mac.train_unsupported' | 'mac.intel_unsupported'
}
interface PythonCheck {         // same shape as the backend's POST /python/check
  ok: boolean; executable: string; version: string | null
  missing: string[]               // requirements-runtime.txt entries not installed (by distribution name)
  error: Msg | null               // first problem, see "Shell notes"
  torch: { version: string; cuda: string | null; cudaAvailable: boolean } | null
}
interface BootstrapProgress {
  stage: 'prepare' | 'python' | 'venv' | 'torch' | 'packages' | 'verify' | 'done' | 'error' | 'cancelled'
  stageIndex: number; stageCount: number
  progress: number | null        // 0..1 within the stage
  overall: number | null         // 0..1 for the whole bootstrap
  message: string                // short human text (English, frontend maps stage → localized title)
  detail: string | null          // e.g. current package
  bytesDone: number | null; bytesTotal: number | null; speed: number | null
  source: string | null          // mirror in use
  error: { code: string; message: string; detail: string } | null
}
```

| Command | Args | Returns |
|---|---|---|
| `get_shell_state` | — | `ShellState` |
| `detect_hardware` | — | `HardwareReport` |
| `disk_space` | `{path}` | `{free, total}` |
| `set_library_path` | `{path}` | `ShellState` (creates the folder; restarts backend if running) |
| `set_language` | `{language: 'zh-CN' \| 'en' \| null}` | `ShellState` (`null` = follow the system; also relabels the tray) |
| `bootstrap_start` | `{variant: TorchVariant, region: 'auto'\|'cn'\|'global'}` | `null` (progress via events) |
| `bootstrap_cancel` | — | `null` |
| `bootstrap_repair` | — | `null` (re-verifies and reinstalls missing pieces) |
| `finish_onboarding` | — | `ShellState` |
| `backend_start` / `backend_restart` / `backend_stop` | — | `BackendStatus` (a failed start resolves with `status: 'crashed'` + `error`; rejects only for unmet preconditions such as `runtime.not_ready`) |
| `backend_logs` | `{tail}` | `string[]` |
| `set_close_to_tray` | `{enabled}` | `null` |
| `accept_terms` | `{version}` | `ShellState`; stores the accepted terms version (never lowers it) |
| `set_tray_status` | `{text: string \| null, progress: number \| null}` | `null` (tooltip + taskbar progress) |
| `reveal_path` | `{path}` | `null` (opens the folder / selects the file) |
| `open_logs_folder` | — | `null` |
| `quit_app` | `{force?: boolean}` | `null` (asks backend to stop jobs gracefully unless force; waits ≤ 25 s, then exits) |
| `frontend_pong` | — | internal: answers the shell's liveness ping, not for app code |
| `check_python` | `{path}` | `PythonCheck` (Python 3.10–3.12, torch import/version/CUDA, missing runtime packages; ≤ 180 s; the result is cached per interpreter for `ShellState.runtime`). When Stellaria's layer sits on `path` (see below) the check runs through it: `layered: true`. Every interpreter can carry a layer (embeddable ones get a shadow, see below). |
| `find_pythons` | `{folder: string \| null}` | `PythonCandidate[]` `{path, source: 'folder' \| 'conda' \| 'system', label}`, nothing run. With a folder: interpreters inside it and its direct subfolders (`venv`, `.venv`, `env`, `runtime`, `workenv`, `python`, …, as bundles keep them). Without: conda environments (`~/.conda/environments.txt`, `envs_dirs` of `~/.condarc` and the usual install folders), interpreters registered under `Software\Python` on Windows (PEP 514: Anaconda / Miniconda / python.org installers) and python.org / Homebrew / PATH installs of 3.10–3.12 (not the Microsoft Store stub). At most 24. |
| `set_runtime_python` | `{path: string \| null}` | `ShellState`. With a path: must pass `check_python` (else rejects `runtime.custom_invalid`, `detail` = the check's error as JSON), is saved as `runtime {mode: 'custom', pythonPath}` and the backend is (re)started with it — through Stellaria's layer over it: one is created when there is none (nothing is downloaded). If the interpreter does not pass the check through a layer, the layer is removed and the call rejects `runtime.layer_failed`: the user's interpreter is never run bare, so nothing can be installed into it (the UI offers the app's own runtime instead). If the backend does not become ready, rejects `runtime.custom_failed` (`detail` = backend output) and, when the managed runtime is installed, switches back to it (else the custom choice stays). `null` = back to the managed runtime. Rejects `bootstrap.running` while a runtime job runs. |
| `install_runtime_requirements` | `{path}` | `null`; installs what `requirements-runtime.txt` needs **without torch** with the bundled uv (same mirrors/region as the bootstrap) into Stellaria's layer over that interpreter, which is only read: a `uv pip install --dry-run` against it (with every requirement it already has reduced to its bare name, so its versions are kept) lists what it lacks, and exactly that goes into the layer with `--no-deps`. Nothing is ever installed into the interpreter itself. If the interpreter no longer imports torch through the new layer, the layer is removed and the job fails with `runtime.layer_failed`. Progress on `bootstrap://progress` as one stage `packages` (`overall` = that stage), cancel with `bootstrap_cancel`; rejects `bootstrap.running` while another runtime job runs. A backend running from that interpreter is stopped for the install and started again afterwards. Only call after an explicit confirmation in the UI. |

Events (`listen(name)`):

| Event | Payload |
|---|---|
| `bootstrap://progress` | `BootstrapProgress` |
| `backend://status` | `BackendStatus` |
| `backend://log` | `{stream: 'stdout' \| 'stderr', line: string}` |
| `tray://action` | `{action: 'show' \| 'pause_training' \| 'quit'}` (`show`/`quit` are already carried out by the shell — `quit` is a graceful quit; `pause_training` is left to the frontend, the shell pauses via the API itself only when the frontend does not respond) |
| `app://close-requested` | `{activeJobs: boolean}` (frontend decides: hide to tray / confirm quit). Also sent for Cmd+Q / Dock → Quit. If the frontend does not answer the shell's ping, the shell hides to the tray when `closeToTray` and work is running, else quits gracefully. `activeJobs` includes a running bootstrap. |

Plugins used from JS directly: `dialog` (open/save), `notification`, `opener`, `window`
(minimize/maximize/close/startDragging), `drag-drop` events via `getCurrentWebview().onDragDropEvent`.

Shell notes:

* Error codes (`{code, message, detail}`, also in `BootstrapProgress.error`): `bootstrap.network`,
  `bootstrap.disk_full`, `bootstrap.uv_missing`, `bootstrap.requirements_missing`,
  `bootstrap.unsupported_platform` (Intel Mac), `bootstrap.variant_unsupported`,
  `bootstrap.python_failed`, `bootstrap.venv_failed`, `bootstrap.venv_locked`,
  `bootstrap.torch_failed`, `bootstrap.packages_failed`, `bootstrap.verify_failed`,
  `bootstrap.vcredist_missing` (Windows: install the MSVC x64 runtime), `bootstrap.checksum`,
  `bootstrap.timeout`, `bootstrap.running`, `bootstrap.spawn_failed`, `manifest.missing`,
  `manifest.invalid`, `runtime.not_ready`, `backend.start_failed`, `backend.spawn_failed`,
  `library.invalid`, `library.create_failed`, `library.not_writable`, `path.not_found`,
  `path.open_failed`, `disk.full`, `io`, `internal`, `runtime.custom_invalid`,
  `runtime.custom_failed`, `runtime.install_failed`, `runtime.layer_failed`. After a failed mirror, the first (preferred)
  mirror's error is reported and the others are listed in `detail`.
* `PythonCheck.error` codes (`Msg`, first problem wins): `python.not_found` {path},
  `python.failed` {reason}, `python.timeout` {seconds}, `python.version_unsupported`
  {version, min, max}, `python.torch_missing` {reason}, `python.packages_missing`
  {packages (comma-separated), count}, `python.check_failed` {reason}.
* Custom runtime (`runtime.mode = 'custom'`): `ShellState.runtime.status` is `ready` while the
  interpreter exists and its last check (made by `check_python`, `set_runtime_python`, the
  requirements install and once at every launch) passed, `broken` otherwise, `installing` during
  `install_runtime_requirements`; `variant` is `null`, `python`/`torch`/`cuda` come from that
  check. The backend runs with the user's environment as is (user site-packages enabled,
  conda environments get their activation `PATH`), through Stellaria's layer over it when there
  is one.
* Stellaria's layer over the user's Python, in `<data>/runtime/layer`: a venv made from that
  interpreter — or, for an embeddable Python (a `python*._pth` next to it, which makes Python
  ignore venvs), a **shadow**: copies of its executable and its `python*.dll`/`vcruntime*.dll`
  with our own `._pth` (the base's entries made absolute, then `import site`). A `._pth` also
  drops `PYTHONPATH` and the script's folder from `sys.path`; `_stellaria_paths.pth` in the
  shadow's site-packages puts both back (`PYTHONUTF8`, `PYTHONIOENCODING`, `PYTHONUNBUFFERED`, …
  are read before the `._pth` and still apply). Builtin plugin environments need venvs, so a
  shadow refuses them (`plugin.env_unsupported`). `_stellaria_base.pth` in its site-packages adds the interpreter's site-packages
  (and user site) *after* the layer's own — on Windows conda environments it also adds their DLL
  folders (`Library\bin`, …) — so the layer holds only what the environment lacks, and the
  environment is never written to. `stellaria-layer.json` in the layer: `{base, version, sites,
  kind: 'venv' | 'shadow', createdAt}`; a layer for another interpreter or Python version is
  replaced. The backend runs with `STELLARIA_CUSTOM_RUNTIME=1` on a custom runtime: without a
  layer it refuses add-on installs (`addon.needs_layer`) rather than write into the user's
  environment.
* Startup with a custom runtime that is `broken` (or whose backend does not start): the frontend
  shows what changed (interpreter gone, Python version, torch gone, packages missing — from
  `check_python`) and the ways back: add the packages again (into the layer), the app's own
  runtime when `managedReady`, or choosing a runtime again (onboarding from the hardware step). The backend reads
  the marker: add-ons install into the layer only what it does not already see
  (`plugins/envs.install_into_env`), plugin environments get the user's site-packages too. `bootstrap_start` in custom mode completes
  at once when the interpreter passes the check, otherwise switches to `managed` and installs;
  `bootstrap_repair` in custom mode only re-checks (failing with `runtime.custom_invalid`).
* After the final `bootstrap://progress` event (`done` / `error` / `cancelled`) the job slot is
  already free: `get_shell_state` no longer reports `installing` and a new job can start.
* `BootstrapProgress`: `stageIndex` 0..5 for the working stages, 6 for `done`; on `error` /
  `cancelled` it keeps the index of the stage that stopped. `bytes*`/`speed` are exact for the
  torch wheel (downloaded by the shell, resumable, sha256-verified) and estimated for uv's own
  downloads. `source` is the mirror id (`pytorch`, `aliyun`, `tuna`, …).
* Shell logs in `<data>/logs/`: `shell.log`, `bootstrap.log`, `backend-console.log` (the
  backend's stdout/stderr as captured by the shell; `backend.log` belongs to the backend).
  `backend_logs` returns the in-memory tail of that capture.
* Backend launch (in addition to §1): cwd = data dir; env `PYTHONPATH=<resources>/backend`,
  `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`, `PYTHONUNBUFFERED=1`, `PYTHONNOUSERSITE=1`,
  `PYTHONPYCACHEPREFIX=<data>/runtime/pycache`, `VIRTUAL_ENV=<data>/runtime/venv`,
  `UV_CACHE_DIR`/`UV_PYTHON_INSTALL_DIR` (the runtime's), `UV_PYTHON_PREFERENCE=only-managed`,
  `UV_NO_CONFIG=1`, `UV_LINK_MODE=copy` (Windows) so add-on installs share the runtime layout, and
  `STELLARIA_SHELL_PID` (the backend may exit when that process disappears). Debug builds of the
  shell add `--dev` (CORS for the Vite origin; port/token stay explicit).
* Dev mode (debug builds): `STELLARIA_PYTHON=<python>` skips the bootstrap (runtime reported as
  `ready` with null versions; `bootstrap_start` completes immediately); engine = `engine/`,
  backend = `backend`, manifest = `resources/manifest.json`, `uv` from PATH;
  `STELLARIA_DATA_DIR` overrides the data dir.

Browser fallback (no Tauri): backend URL/token from `?backend=…&token=…` or
`VITE_BACKEND_URL`/`VITE_BACKEND_TOKEN`, onboarding shows the backend-side resource step only,
file picking uses `<input type=file>` + multipart uploads.
