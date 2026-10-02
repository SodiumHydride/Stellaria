// Canonical wire types — keep in sync with stellaria/docs/api.md (the contract).

export type ID = string
export type Msg = { code: string; params?: Record<string, string | number> }

// ---------- system ----------
export interface GpuInfo {
  index: number
  name: string
  vramTotal: number
  vramUsed?: number
  utilization?: number
  temperature?: number
  computeCapability?: string
  driver?: string
}

export interface SystemInfo {
  version: string
  os: 'windows' | 'macos' | 'linux'
  osVersion: string
  arch: string
  cpu: string
  cpuCount: number
  ramTotal: number
  gpus: GpuInfo[]
  torch: { available: boolean; version?: string; cuda?: string; device: 'cuda' | 'mps' | 'cpu'; triton: boolean }
  canTrain: boolean
  trainBlockers: Msg[]
  libraryPath: string
  dataPath: string
  engineVersion: string
  fakeEngine: boolean
}

export interface Preferences {
  language: 'zh-CN' | 'en'
  mirrorRegion: 'auto' | 'cn' | 'global'
  gpuPolicy: 'smart' | 'cpu' | 'queue'
  lowPowerWhileTraining: boolean
  autoSlimAfterTraining: boolean
  notifyOnFinish: boolean
  glass: boolean
  /** where export nodes write when they name no folder; null = the backend's default */
  exportFolder?: string | null
  /** separation model used by organizing, health checks and separations that name none;
   *  null = the best installed one (GET /resources/separation → defaultModel) */
  separationModel?: SeparationModelId | null
  /** cleanup (dereverb) model used by organizing and dereverb nodes that name none;
   *  null = the best installed one (GET /resources/cleanup → defaultModel) */
  cleanupModel?: CleanupModelId | null
  /** minutes without singing before the singing models leave the GPU / memory; 0 = keep them */
  gpuIdleReleaseMin: number
}

/** GET /system/render: the singing models' worker (it holds GPU memory while loaded) */
export interface RenderStatus {
  loaded: boolean
  busy: boolean
  device: string | null
  idleSeconds: number | null
  releaseAfterMin: number
}

export interface DiskUsage {
  libraryFree: number
  libraryTotal: number
  characters: { id: ID; bytes: number }[]
  models: { id: ID; bytes: number }[]
  projects: { id: ID; bytes: number }[]
}

// ---------- characters & material ----------
export type CharacterStage = 'empty' | 'material' | 'ready' | 'training' | 'trained'
export type HealthLevel = 'excellent' | 'good' | 'fair' | 'poor' | 'blocked'

export interface Character {
  id: ID
  name: string
  color: string
  hasAvatar: boolean
  avatarVersion: number
  tags: string[]
  notes: string
  consent: { source: string; authorized: boolean; note: string }
  pinned: boolean
  archived: boolean
  createdAt: string
  updatedAt: string
  lastOpenedAt: string | null
  stage: CharacterStage
  stats: { rawCount: number; clipCount: number; activeDuration: number; trainCount: number; valCount: number; trashedCount: number }
  health: { score: number; level: HealthLevel; checkedAt: string; stale: boolean } | null
  modelIds: ID[]
  primaryModelId: ID | null
  pitch: { low: number; high: number; median: number } | null
}

export interface RawImport {
  id: ID
  name: string
  ext: string
  size: number
  duration: number | null
  importedAt: string
  status: 'imported' | 'organizing' | 'organized' | 'failed'
  error: Msg | null
  clipCount: number
}

export type ClipFlag = 'too_short' | 'too_long' | 'clipping' | 'silent' | 'noisy' | 'outlier' | 'music' | 'f0_failed' | 'duplicate'

export interface ClipMetrics {
  peakDb: number
  rmsDb: number
  silenceRatio: number
  clippingRatio: number
  snrDb: number | null
  f0Low: number | null
  f0High: number | null
  f0Median: number | null
  voicedRatio: number | null
  outlierScore: number | null
  musicRatio: number | null
}

export interface Clip {
  id: ID
  name: string
  file: string
  rawId: ID | null
  rawName: string | null
  start: number | null
  end: number | null
  duration: number
  status: 'active' | 'trashed'
  split: 'train' | 'val'
  flags: ClipFlag[]
  metrics: ClipMetrics | null
  reviewed: boolean
  createdAt: string
}

export interface OrganizeOptions {
  separateVocals: boolean
  /** slice the dry voice (after separation when separateVocals), made by the default cleanup model */
  dereverb: boolean
  normalize: boolean
  targetDb: number
  slice: { enabled: boolean; thresholdDb: number; minLength: number; minInterval: number; hopSize: number; maxSilKept: number }
  maxLength: number
  minDuration: number
  dropClipping: boolean
  dropSilent: boolean
  dedupe: boolean
  namePattern: string
  valCount: number
  rawIds: ID[] | null
}

export type FixAction = 'trash_clips' | 'organize' | 'add_material' | 'pick_val' | 'install_addon'

export interface HealthIssue {
  code: string
  severity: 'info' | 'warning' | 'error'
  params: Record<string, string | number>
  clipIds: ID[]
  fix: { action: FixAction; params?: Record<string, unknown> } | null
}

export interface HealthReport {
  score: number
  level: HealthLevel
  blocking: boolean
  checkedAt: string
  totals: { clips: number; duration: number; trainDuration: number; valCount: number }
  pitch: { histogram: { midi: number; seconds: number }[]; low: number | null; high: number | null; median: number | null }
  issues: HealthIssue[]
  checks: { basic: boolean; pitch: boolean; outlier: boolean; music: boolean }
}

// ---------- models & training ----------
export type TrainPresetId = 'vram6' | 'vram8' | 'vram12' | 'vram24' | 'custom'
export type F0Extractor = 'rmvpe' | 'fcpe' | 'parselmouth' | 'harvest' | 'dio' | 'crepe'

export interface TrainConfig {
  preset: TrainPresetId
  targetSteps: number
  batchSize: number
  lr: number
  ampDtype: 'fp16' | 'bf16' | 'fp32'
  cacheDevice: 'cpu' | 'cuda'
  numWorkers: number
  intervalVal: number
  intervalForceSave: number
  f0Extractor: F0Extractor
  encoder: 'contentvec768l12tta2x' | 'contentvec768l12' | 'hubertsoft'
  nLayers: number
  nChans: number
  nAuxLayers: number
  nAuxChans: number
  usePitchAug: boolean
  useFusedKernels: 'auto' | boolean
  /** start from a base model (底模); structural fields then follow the base */
  baseModelId?: ID | null
}

export interface BaseModel {
  id: ID
  name: string
  notes: string
  createdAt: string
  source: 'download' | 'local' | 'release'
  resourceId: ID | null
  status: 'ready' | 'missing' | 'downloading' | 'failed'
  encoder: TrainConfig['encoder']
  nLayers: number
  nChans: number
  nAuxLayers: number
  nAuxChans: number
  nSpk: number
  step: number
  size: number
}

export interface TrainPreset {
  id: TrainPresetId
  vram: number
  config: Partial<TrainConfig>
  recommended: boolean
}

export type ModelStatus = 'idle' | 'preparing' | 'preprocessing' | 'training' | 'stopping' | 'paused' | 'done' | 'failed'

export interface Checkpoint {
  step: number
  file: string
  size: number
  createdAt: string
  kind: 'regular' | 'permanent' | 'pause'
  releaseId: ID | null
}

export interface Release {
  id: ID
  name: string
  step: number
  createdAt: string
  notes: string
  size: number
}

export interface Model {
  id: ID
  name: string
  kind: 'single' | 'constellation'
  characters: { characterId: ID; spkId: number }[]
  config: TrainConfig
  status: ModelStatus
  error: Msg | null
  step: number
  targetSteps: number
  dataset: { status: 'none' | 'stale' | 'ready'; preparedAt: string | null; clipCount: number; duration: number }
  checkpoints: Checkpoint[]
  releases: Release[]
  createdAt: string
  updatedAt: string
  trainingTime: number
  speed: number | null
  eta: number | null
  diskUsage: { dataset: number; checkpoints: number; releases: number; total: number } | null
}

export interface ScalarSeries {
  tag: string
  points: [step: number, value: number, wallTime: number][]
}

export interface ValidationSample {
  name: string
  gtUrl: string
  predUrl: string
  melUrl: string
}

export interface ValidationSnapshot {
  step: number
  samples: ValidationSample[]
}

export interface MelData {
  frames: number
  bins: number
  data: string
  gtData: string
}

// ---------- creation ----------
export interface Media {
  id: ID
  name: string
  kind: 'import' | 'stem' | 'render' | 'mixdown' | 'mix' | 'process'
  duration: number
  /** seconds: where this audio starts on its song's timeline (0 for imports) */
  offset?: number
  sampleRate: number
  channels: number
  createdAt: string
}

/** The pitch editor's changes: semitone offsets on a 100/s grid, only where something changed. */
export interface PitchEdit {
  rate: number
  segments: { start: number; values: number[] }[]
}

export interface RenderParams {
  key: number
  formant: number
  register: number
  f0Extractor: F0Extractor
  steps: number
  method: 'euler' | 'rk4'
  tStart: number
  threshold: number
  spkMix: Record<string, number> | null
  region: { start: number; end: number } | null
  /** pitch tracking range in Hz (optional; for very low or very high voices) */
  f0Min?: number | null
  f0Max?: number | null
  /** the pitch editor's changes (applied before the key shift) */
  pitchEdit?: PitchEdit | null
}

export type NodeType = 'source' | 'audio' | 'render' | 'mix' | 'process' | 'note' | 'group'
/** Separated (or user-provided) parts of a song. */
export type StemPort = 'vocals' | 'inst' | 'drums' | 'bass' | 'other'
/** Outputs of a song node: the original, and its stems. */
export type SourcePort = 'mix' | StemPort

/** Every edge is audio: an output port of one node feeding an input of another. */
export interface PortRef {
  nodeId: ID
  /** an output port name: a SourcePort, 'out', 'part2', … */
  port: string
}

/** How a content node looks on the canvas (user choices, saved with the node). */
export interface NodeLook {
  /** colour tag (a VOICE_COLORS value) */
  tag?: string | null
  /** shown as its header only */
  collapsed?: boolean
}

export interface SourceData extends NodeLook {
  mediaId: ID
  name: string
  stems: {
    status: 'none' | 'queued' | 'running' | 'done' | 'failed'
    vocals: ID | null
    inst: ID | null
    drums?: ID | null
    bass?: ID | null
    other?: ID | null
    /** stems the user imported themselves (a split never overwrites them) */
    provided?: StemPort[]
    model?: SeparationModelId | null
    jobId: ID | null
    error: Msg | null
  }
  bpm?: number | null
  key?: string | null
}

/** An imported track that isn't a song to split: an official instrumental, a harmony, SFX. */
export interface AudioData extends NodeLook {
  mediaId: ID
  name: string
  role: 'backing' | 'vocal' | 'other'
}

export type TakeStatus = 'draft' | 'queued' | 'running' | 'done' | 'failed'

export interface RenderData extends NodeLook {
  input: PortRef
  characterId: ID
  modelId: ID
  releaseId: ID | null
  params: RenderParams
  mediaId: ID | null
  status: TakeStatus
  jobId: ID | null
  error: Msg | null
  rating: number
  favorite: boolean
  label: string
  stale: boolean
  batchId: ID | null
}

export type MixRole = 'lead' | 'harmony' | 'backing' | 'other'

export interface MixInput {
  id: ID
  ref: PortRef
  role: MixRole
  gainDb: number
  /** -1..1 */
  pan: number
  mute: boolean
  solo: boolean
}

export interface MixData extends NodeLook {
  inputs: MixInput[]
  /** 'builtin' (auto mix) | 'bus' (plain sum) | a plugin id */
  processor: string
  preset: string | null
  params: Record<string, number | string | boolean>
  variants: number
  outputs: { mediaId: ID; seed: number; lufs: number | null }[]
  selected: number
  status: TakeStatus
  jobId: ID | null
  error: Msg | null
  stale: boolean
  label: string
}

export interface ParamSpec {
  id: string
  label: Msg | string
  type: 'number' | 'boolean' | 'choice' | 'text' | 'folder' | 'file'
  default: number | string | boolean | null
  min?: number
  max?: number
  step?: number
  unit?: string
  options?: { value: string; label: Msg | string }[]
  /** fine-grained settings, shown folded under their group */
  group?: string
}

export interface PythonCheck {
  ok: boolean
  executable: string
  version: string | null
  missing: string[]
  error: Msg | null
  torch: { version: string; cuda: string | null; cudaAvailable: boolean } | null
}

export interface Plugin {
  id: ID
  name: string
  version: string
  kind: 'mix' | 'effect'
  runtime: 'command' | 'python'
  folder: string
  manifestSource: 'folder' | 'wizard'
  interpreter: 'builtin' | 'custom'
  pythonPath: string | null
  status: 'ready' | 'needs_setup' | 'error'
  check: PythonCheck | null
  params: ParamSpec[]
  settings: ParamSpec[]
  settingValues: Record<string, unknown>
  enabled: boolean
  addedAt: string
  error?: Msg | null
  requirements?: string[]
  maxVocals?: number
  checkedAt?: string | null
}

export interface MixProcessor {
  id: string
  name: Msg | string
  kind: 'builtin' | 'plugin'
  /** each with its value of every param whose default is `null` */
  presets: { id: string; name: Msg | string; params?: Record<string, number | string | boolean> }[]
  params: ParamSpec[]
  maxVocals: number
  /** all inputs, voices included */
  maxInputs?: number
  available: boolean
  error: Msg | null
}

export type ProcessOp = 'trim' | 'split' | 'pitch' | 'fx' | 'separate' | 'dereverb' | 'comp' | 'concat' | 'export'

export interface TrimParams {
  start: number
  end: number
  fadeIn: number
  fadeOut: number
}
export interface SplitParams {
  cuts: number[]
  fade: number
  labels: string[]
}
export interface PitchParams {
  semitones: number
  cents: number
  tempo: number
  preserveFormant: boolean
  quality: 'fast' | 'high'
}
export type FxSlot = { id: ID; bypass: boolean; mix: number } & (
  | { kind: 'builtin'; effect: string; params: Record<string, number | boolean | string> }
  | { kind: 'vst'; pluginId: ID; state: string | null; params: Record<string, number | boolean | string> }
)
export interface FxParams {
  chain: FxSlot[]
}
export interface SeparateParams {
  /** 'auto' (or null/absent) = Preferences.separationModel → the best installed model */
  model: SeparationModelId
  stems: 2 | 4
}
/** ports: dry (the voice without reverb / echo), reverb (input − dry) */
export interface DereverbParams {
  /** 'auto' (or null/absent) = Preferences.cleanupModel → the best installed model */
  model: CleanupModelId
}
export interface CompSegment {
  start: number
  end: number
  /** index into inputs; -1 = silence */
  input: number
}
export interface CompParams {
  segments: CompSegment[]
  crossfade: number
}
export interface ConcatParams {
  gap: number
  crossfade: number
}
export interface ExportParams {
  format: 'wav' | 'flac' | 'mp3'
  sampleRate: 44100 | 48000
  bitDepth: 16 | 24 | 32
  mp3Bitrate: 192 | 256 | 320
  loudness: number | null
  folder: string | null
  fileName: string
  keepPosition: boolean
  tags: { title: string; artist: string } | null
}
export interface ProcessParamsByOp {
  trim: TrimParams
  split: SplitParams
  pitch: PitchParams
  fx: FxParams
  separate: SeparateParams
  dereverb: DereverbParams
  comp: CompParams
  concat: ConcatParams
  export: ExportParams
}

export interface ProcessData<Op extends ProcessOp = ProcessOp> extends NodeLook {
  op: Op
  label: string
  inputs: PortRef[]
  params: ProcessParamsByOp[Op]
  /** port → media id */
  outputs: Record<string, ID | null>
  status: TakeStatus
  jobId: ID | null
  error: Msg | null
  stale: boolean
  /** pitch/fx: pass the input through unchanged */
  bypass: boolean
  result: Record<string, unknown> | null
}

export interface EffectSpec {
  id: string
  name: Msg
  category: 'eq' | 'dynamics' | 'space' | 'color' | 'utility'
  params: ParamSpec[]
}

export interface VstParam {
  id: string
  name: string
  kind: 'float' | 'bool' | 'choice'
  min: number | null
  max: number | null
  default: number | string | boolean | null
  unit: string | null
  choices: string[] | null
}
export interface VstPlugin {
  id: ID
  name: string
  vendor: string | null
  version: string | null
  format: 'vst3' | 'au'
  path: string
  pluginName: string | null
  category: 'effect' | 'instrument'
  params: VstParam[]
  status: 'ok' | 'failed'
  error: Msg | null
  scannedAt: string
}
export interface VstStatus {
  hostInstalled: boolean
  hostVersion: string | null
  formats: ('vst3' | 'au')[]
  folders: string[]
  defaultFolders: string[]
  editorSupported: boolean
  lastScan: string | null
  scanning: boolean
}

export interface MediaSections {
  bpm: number | null
  beats: number[]
  downbeats: number[]
  phrases: { start: number; end: number }[]
  sections: { start: number; end: number; label: 'intro' | 'verse' | 'chorus' | 'bridge' | 'outro' | null }[]
}

export interface NoteData {
  text: string
  color: string
}

export interface GroupData {
  label: string
  color: string
}

export type NodeData = SourceData | AudioData | RenderData | MixData | ProcessData | NoteData | GroupData

export interface CanvasNode {
  id: ID
  type: NodeType
  position: { x: number; y: number }
  width?: number
  height?: number
  parentId?: ID
  data: NodeData
}

/** Stored edges are always []; the canvas derives its edges from node data (see canvas/model.ts). */
export interface CanvasEdge {
  id: ID
  source: ID
  target: ID
}

export interface TimelineClip {
  id: ID
  mediaId: ID
  nodeId: ID | null
  start: number
  offset: number
  duration: number
  gainDb: number
  fadeIn: number
  fadeOut: number
  /** silent in playback and export */
  mute?: boolean
}

export interface Track {
  id: ID
  name: string
  kind: 'vocal' | 'instrumental' | 'other'
  gainDb: number
  pan: number
  mute: boolean
  solo: boolean
  clips: TimelineClip[]
}

/** The arrangement's loop range (seconds); `on` = playback repeats it. */
export interface TimelineLoop {
  start: number
  end: number
  on?: boolean
}

export interface Timeline {
  tracks: Track[]
  length: number
  loop?: TimelineLoop | null
  /** the master fader (dB, default 0) */
  masterGainDb?: number
}

/** POST /projects/{id}/mixdown */
export interface MixdownRequest {
  format: 'wav' | 'flac' | 'mp3'
  /** desktop: the file (a folder with `stems`); omitted: a download */
  path?: string
  /** peak to −1 dBFS (`loudness` wins when set) */
  normalize?: boolean
  /** LUFS target, true peak ≤ −1 dBTP */
  loudness?: number | null
  sampleRate?: 44100 | 48000
  /** 32 = float */
  bitDepth?: 16 | 24 | 32
  mp3Bitrate?: 192 | 256 | 320
  /** both set: only this range (seconds) */
  start?: number
  end?: number
  /** one file per audible track */
  stems?: boolean
  /** the arrangement as on screen (else the saved one) */
  timeline?: Timeline
}

export interface Project {
  id: ID
  name: string
  createdAt: string
  updatedAt: string
  coverColor: string
  canvas: { nodes: CanvasNode[]; edges: CanvasEdge[]; viewport: { x: number; y: number; zoom: number } }
  timeline: Timeline
}

export interface ProjectSummary {
  id: ID
  name: string
  createdAt: string
  updatedAt: string
  coverColor: string
  renderCount: number
  characterIds: ID[]
}

export interface KeySuggestion {
  key: number
  sourceRange: { low: number; high: number; median: number }
  characterRange: { low: number; high: number; median: number }
  confidence: 'high' | 'low'
}

// ---------- jobs ----------
export type JobType =
  | 'import'
  | 'organize'
  | 'health'
  | 'prepare'
  | 'preprocess'
  | 'train'
  | 'render'
  | 'separate'
  | 'mixdown'
  | 'download'
  | 'install_addon'
  | 'export'
  | 'import_pack'
  | 'slim'
  | 'mix'
  | 'install_plugin'
  | 'plugin_test'
  | 'import_model'
  | 'import_base'
  | 'process'
  | 'vst_scan'
  | 'vst_editor'
  | 'diagnostics'
  | 'pitch'

export type JobStatus = 'queued' | 'waiting_gpu' | 'running' | 'stopping' | 'succeeded' | 'failed' | 'cancelled'

export interface Job {
  id: ID
  type: JobType
  title: Msg
  target: { characterId?: ID; modelId?: ID; projectId?: ID; nodeId?: ID; resourceId?: ID }
  status: JobStatus
  progress: number | null
  stage: Msg | null
  eta: number | null
  usesGpu: boolean
  createdAt: string
  startedAt: string | null
  finishedAt: string | null
  error: { code: string; message: string; detail: string; params?: Record<string, string | number> } | null
  result: Record<string, unknown> | null
}

// ---------- resources ----------
export interface Resource {
  id: string
  kind: 'model' | 'addon'
  required: boolean
  name: Msg
  description: Msg
  size: number
  status: 'missing' | 'downloading' | 'installing' | 'ready' | 'failed'
  progress: number | null
  bytesDone: number | null
  speed: number | null
  source: string | null
  error: Msg | null
  path: string | null
}

// ---------- vocal separation models (GET /resources/separation) ----------
/** Demucs ('htdemucs', 'htdemucs_ft'), a Roformer checkpoint resource, or an imported model
 *  ('sep_…'); 'auto' = the default (where a model can be named) */
export type SeparationModelId =
  | 'auto'
  | 'htdemucs'
  | 'htdemucs_ft'
  | 'mel_roformer_kim'
  | 'bs_roformer_viperx'
  | (string & {})

export interface SeparationModel {
  id: string
  engine: 'demucs' | 'roformer'
  arch: 'htdemucs' | 'bs_roformer' | 'mel_band_roformer' | (string & {})
  /** addon: part of the separator add-on; resource: a downloadable checkpoint; local: imported */
  source: 'addon' | 'resource' | 'local'
  /** what to install: 'separator' or the checkpoint's resource */
  resourceId: string | null
  /** inline translations in params ({'zh-CN', en}) */
  name: Msg
  quality: 'standard' | 'high'
  /** vocals SDR (dB) reported for the checkpoint, when known */
  sdr: number | null
  ready: boolean
  /** why it can't run: the add-on is missing, was installed before Roformer support (reinstall it),
   *  or the checkpoint isn't installed */
  missing: 'addon' | 'addon_update' | 'model' | null
  size: number | null
  path: string | null
}

export interface SeparationModels {
  /** Demucs models, Roformer resources (best first), imported models (newest first) */
  models: SeparationModel[]
  /** what organizing, health checks and separations without a model use */
  defaultModel: string
}

// ---------- vocal cleanup models (GET /resources/cleanup) ----------
/** a downloadable checkpoint resource or an imported model ('cln_…'); 'auto' = the default */
export type CleanupModelId = 'auto' | 'dereverb_mbr_anvuew' | 'dereverb_echo_mbr_fused' | (string & {})
/** what a cleanup model removes */
export type CleanupKind = 'dereverb' | 'dereverb_echo' | 'denoise'

export interface CleanupModel {
  id: string
  kind: CleanupKind
  arch: 'bs_roformer' | 'mel_band_roformer' | (string & {})
  /** resource: a downloadable checkpoint; local: imported */
  source: 'resource' | 'local'
  resourceId: string | null
  /** inline translations in params ({'zh-CN', en}) */
  name: Msg
  /** SDR (dB) of the dry voice reported for the checkpoint, when known */
  sdr: number | null
  ready: boolean
  /** they run in the separator add-on: missing / installed before Roformer support / checkpoint not installed */
  missing: 'addon' | 'addon_update' | 'model' | null
  size: number | null
  path: string | null
}

export interface CleanupModels {
  /** downloadable checkpoints (recommended first), then imported models (newest first) */
  models: CleanupModel[]
  /** what dereverb nodes and organizing use; null = no dereverb model is ready */
  defaultModel: string | null
}

/** POST /resources/separation/import, and `separation` of POST /resources/import-folder */
export interface SeparationImportResult {
  /** each started as a job (type 'download'); manifest checkpoints import as their resource;
   *  role: a vocal separation or a cleanup (dereverb) model; kind: the cleanup kind */
  found: { id: string; name: string; arch: string | null; role: 'separation' | 'cleanup'; kind: CleanupKind | null; path: string; config: string | null; jobId: ID; resourceId: string | null }[]
  /** reason codes: separation.skip.no_config | .bad_config {reason} | .arch {arch} | .params {params}
   *  | .no_vocals {instruments} | .duplicate {model} */
  skipped: { path: string; reason: Msg }[]
}

// ---------- events ----------
export type ServerEvent =
  | { type: 'hello'; data: { version: string } }
  | { type: 'job'; data: Job }
  | { type: 'job.log'; data: { jobId: ID; lines: string[] } }
  | { type: 'character'; data: Character }
  | { type: 'character.deleted'; data: { id: ID } }
  | { type: 'clips'; data: { characterId: ID } }
  | { type: 'health'; data: { characterId: ID; report: HealthReport } }
  | { type: 'model'; data: Model }
  | { type: 'model.deleted'; data: { id: ID } }
  | { type: 'metrics'; data: { modelId: ID; step: number; scalars: Record<string, number> } }
  | { type: 'validation'; data: { modelId: ID; step: number } }
  | { type: 'project.nodes'; data: { projectId: ID; nodes: CanvasNode[] } }
  | { type: 'project.media'; data: { projectId: ID; media: Media } }
  | { type: 'resource'; data: Resource }
  | { type: 'plugin'; data: Plugin }
  | { type: 'plugin.deleted'; data: { id: ID } }
  | { type: 'base_model'; data: BaseModel }
  | { type: 'base_model.deleted'; data: { id: ID } }
  | { type: 'vst.status'; data: VstStatus }
  | { type: 'vst.plugins'; data: VstPlugin[] }
  | { type: 'gpu'; data: GpuInfo[] }
  | { type: 'preferences'; data: Preferences }
  | { type: 'render.status'; data: RenderStatus }
  | { type: 'separation'; data: SeparationModels }
  | { type: 'cleanup'; data: CleanupModels }

export function isActiveJob(job: Job): boolean {
  return job.status === 'queued' || job.status === 'waiting_gpu' || job.status === 'running' || job.status === 'stopping'
}
