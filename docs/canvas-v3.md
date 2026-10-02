# Canvas v3, effects & VST3, mixer v2: contract

Source of truth for the round after `creation-v2.md`. Same conventions as `api.md` (camelCase on
the wire, `Msg` for user-facing text, errors `{code, message, detail}`). When a part lands, merge
it into `api.md` and mark it done here.

The user's ask: the infinite canvas should be able to do what an infinite canvas *can* do —
nodes connect to different kinds of things, you can pre-mix the music / an imported
instrumental and then mix again, cut first, and so on; creating should feel good; the built-in
auto mix should get better; paid VST3s (Cenozoix, Pro-Q 3, Valhalla, …) should be usable; and
the UI should be complete, not an MVP.

Design rules that keep this from turning into a DAW-in-a-graph mess:

1. **Every edge is audio.** One port type. An output port can feed any input port.
2. **Every node says what it makes.** Output ports have names (`vocals`, `out`, `part2`), and
   the node shows its result as a waveform the moment it exists.
3. **Time stays aligned.** Audio derived from a song remembers where it sits on that song
   (`Media.offset`), so a chorus cut out, sung by three voices and mixed with the backing lands
   on the right bar without anyone thinking about it.
4. **Run is one button.** Anything runnable goes through one endpoint; the frontend runner
   walks the graph in dependency order (as in v2) and re-runs only what is stale.

---------------------------------------------------------------------------------------------

## 1. Graph model — ✅ done (backend)

> **Status — backend: done**, merged into `api.md` (canvas v3 types, `Media.offset`,
> `/nodes/{id}/run` + aliases, stems import/delete, `/sections`, analysis `key`, port resolution,
> offsets, migration, staleness and save ownership, mix v2 jobs, process jobs, `exportFolder`,
> `/system/diagnostics`, `/fx/effects`, the `dsp` CPU pool). The process jobs call the §6 dsp CLI
> and `dsp.analysis`, and `/fx/effects` calls `fx.effect_specs()`, as soon as those packages exist.
> Frontend: written against it.

### 1.1 Ports and references

```ts
interface PortRef { nodeId: ID; port: string }          // generalises v2's {nodeId, port: SourcePort}
```

| Node type | Output ports | Inputs |
|---|---|---|
| `source` (song) | `mix`, `vocals`, `inst`, and after a 4-stem split also `drums`, `bass`, `other` | — |
| `audio` (imported track: an official instrumental, a harmony, SFX) — **new** | `out` | — |
| `render` (a voice sings) | `out` | `input: PortRef` |
| `mix` | `out` (the selected variant) | `inputs: MixInput[]` |
| `process` — **new**, one node type with an `op` | per op, below | `inputs: PortRef[]` |
| `note`, `group` | — | — |

Resolution `(PortRef) → media` (backend, used by every job):

- source: `mix` → `mediaId`; `vocals|inst|drums|bass|other` → `stems[port]`
- audio: `out` → `mediaId`
- render: `out` → `mediaId`
- mix: `out` → `outputs[selected].mediaId`
- process: `outputs[port]`; when `bypass` is true (pitch/fx only) the node resolves to its input
- anything unresolved → 409 `node.input_missing` `{nodeId, port}`

v2 compatibility: a render `input` with `port: 'mix'` pointing at a non-source node means `out`
(migration rewrites it).

### 1.2 Media offset

```ts
interface Media { …; offset: number }   // seconds: where this audio starts on its song's timeline (0 for imports)
```

| Producer | Output offset |
|---|---|
| import, separation, stems import | input offset (0 for a song) |
| trim / split part | input offset + region start |
| render | input offset (+ `params.region.start` when a region is rendered and only the region is written) |
| pitch with `tempo ≠ 1` | input offset ÷ tempo |
| fx | input offset (effects keep length; reverb/delay tails extend the end) |
| mix (`builtin`, `bus`, plugins) | min input offset; inputs are placed at `offset − min` |
| comp | min input offset |
| concat | 0 |

Export writes the audio as is; `keepPosition: true` pads `offset` seconds of silence in front.

### 1.3 Node data

```ts
type NodeType = 'source' | 'audio' | 'render' | 'mix' | 'process' | 'note' | 'group'
type StemPort = 'vocals' | 'inst' | 'drums' | 'bass' | 'other'

interface SourceData {
  mediaId: ID; name: string
  stems: {
    status: 'none' | 'queued' | 'running' | 'done' | 'failed'
    vocals: ID | null; inst: ID | null
    drums: ID | null; bass: ID | null; other: ID | null      // 4-stem split only
    provided: StemPort[]          // stems the user imported themselves (never overwritten by a split)
    model: 'htdemucs' | 'htdemucs_ft' | null                  // what produced the separated stems
    jobId: ID | null; error: Msg | null
  }
  bpm: number | null; key: string | null                      // filled by analysis (C#m etc.), shown on the node
}

interface AudioData {
  mediaId: ID; name: string
  role: 'backing' | 'vocal' | 'other'                          // a hint for the mix node's default role
}

interface RenderData {                                         // v2 + PortRef input
  input: PortRef
  …everything else unchanged…
}

interface MixInput {
  id: ID; ref: PortRef
  role: 'lead' | 'harmony' | 'backing' | 'other'
  gainDb: number; pan: number                                  // -1..1
  mute: boolean; solo: boolean
}
interface MixData {
  inputs: MixInput[]                                           // replaces v2 vocals[] + backing
  processor: string                                            // 'builtin' (auto mix) | 'bus' (plain sum) | plugin id
  preset: string | null; params: Record<string, number | string | boolean>
  variants: number
  outputs: { mediaId: ID; seed: number; lufs: number | null }[]; selected: number
  status: 'draft' | 'queued' | 'running' | 'done' | 'failed'; jobId: ID | null; error: Msg | null
  stale: boolean; label: string
}

type ProcessOp = 'trim' | 'split' | 'pitch' | 'fx' | 'separate' | 'comp' | 'concat' | 'export'
interface ProcessData {
  op: ProcessOp; label: string
  inputs: PortRef[]                     // trim/split/pitch/fx/separate/export: exactly 1; comp: 2–8; concat: 2–32
  params: TrimParams | SplitParams | PitchParams | FxParams | SeparateParams | CompParams | ConcatParams | ExportParams
  outputs: Record<string, ID | null>    // port → media id
  status: 'draft' | 'queued' | 'running' | 'done' | 'failed'; jobId: ID | null; error: Msg | null
  stale: boolean
  bypass: boolean                       // pitch/fx: resolve to the input, no job
  result: Record<string, unknown> | null   // export: {path}; others: op-specific facts for display
}
```

Process ops (ports and params):

| op | Output ports | Params |
|---|---|---|
| `trim` 剪切 | `out` | `{start, end, fadeIn, fadeOut}` — seconds within the input |
| `split` 分段 | `part1` … `partN` (N = cuts + 1) | `{cuts: number[], fade: number, labels: string[]}` |
| `pitch` 变调变速 | `out` | `{semitones, cents, tempo (0.5–2), preserveFormant: boolean, quality: 'fast' \| 'high'}` |
| `fx` 效果链 | `out` | `{chain: FxSlot[]}` (§3) |
| `separate` 分离 | `vocals`, `inst` (+ `drums`, `bass`, `other` for 4 stems) | `{model: 'htdemucs' \| 'htdemucs_ft', stems: 2 \| 4}` |
| `comp` 拼接择优 | `out` | `{segments: {start, end, input}[], crossfade: number}` — song-timeline seconds; `input` indexes `inputs`, −1 = silence. Inputs are aligned by offset. For duets (verse A, chorus B) and best-take comping. |
| `concat` 串联 | `out` | `{gap, crossfade}` — inputs one after another (medleys) |
| `export` 导出 | — | `ExportParams` below; `result.path` after a run |

```ts
interface ExportParams {
  format: 'wav' | 'flac' | 'mp3'; sampleRate: 44100 | 48000; bitDepth: 16 | 24 | 32; mp3Bitrate: 192 | 256 | 320
  loudness: number | null               // LUFS target, null = keep
  folder: string | null                 // null = Settings → default export folder
  fileName: string                      // template: {project} {song} {voice} {label} {date} {index}
  keepPosition: boolean
  tags: { title: string; artist: string } | null
}
```

### 1.4 Staleness

- The **frontend** sets `stale` when it changes a node's params or inputs (as in v2).
- The **backend** sets `stale: true` on every node that references node X when X finishes a run
  with new outputs (direct dependents only; the runner handles the transitive closure), and
  emits the change in the same `project.nodes` event. A run of X clears X's own `stale`.

### 1.5 Migration (backend, on load, idempotent, saved once)

- render `input.port: 'mix'` on a non-source node → `'out'`
- mix `vocals[]` → `inputs` with `role: 'lead'`, `ref {nodeId, port: 'out'}`; `backing` →
  `role: 'backing'` input; `mute/solo` false
- source `stems` gains `drums/bass/other: null`, `provided: []`, `model`; source gains `bpm/key: null`
- media without `offset` → 0

### 1.6 Endpoints

| Method & path | Body | Response |
|---|---|---|
| POST `/projects/{id}/nodes/{nodeId}/run` | `{stems?: 2 \| 4}` (sources only) | `Job` for any runnable node: source → separate (2 stems by default; 4 when asked, e.g. because something uses `drums`/`bass`/`other`), render, mix, process. 409 `node.input_missing`, 409 `node.not_runnable` (note/group/audio/bypassed), 409 `node.busy`. v2's `/separate`, `/render`, `/mix` stay as aliases. Job types: `separate`, `render`, `mix`, `process` (new; title `job.process.<op>`). |
| POST `/projects/{id}/nodes/{nodeId}/stems` | `{port: StemPort, path}` | `Source` node: imports the file as that stem (media `kind: 'import'`, offset 0), adds it to `provided`. Lets people use the official instrumental instead of separating. |
| DELETE `/projects/{id}/nodes/{nodeId}/stems/{port}` | — | removes a provided stem |
| GET `/projects/{id}/media/{mediaId}/sections` | — | `{bpm, beats: number[], downbeats: number[], phrases: {start, end}[], sections: {start, end, label: 'intro' \| 'verse' \| 'chorus' \| 'bridge' \| 'outro' \| null}[]}` — for split/comp/trim editors (snap to phrase/bar). Cached. Phrases come from vocal activity (the vocals stem when the media is a source with stems); sections from self-similarity novelty on chroma + loudness. Labels are best-effort, `null` when unsure. |
| GET `/projects/{id}/media/{mediaId}/analysis` | — | v2 + `key: string \| null` (Krumhansl profile on chroma) |
| GET `/fx/effects` | — | `EffectSpec[]` (§3) |
| GET `/mix/processors` | — | v2 + the `bus` processor (`kind: 'builtin'`, no presets, params `normalize` (bool), `targetLufs`) |

Process jobs run in a subprocess like the mixer: `python -m stellaria_backend.dsp.run job.json`,
same JSON-lines protocol as `mixing.run` (`progress`, `stage`, `log`, `result`, `error`).
`separate` keeps using the existing separation worker (Demucs add-on); 4 stems keep all of
Demucs' outputs instead of summing them away.

Also in this round (backend):

| Method & path | Body | Response |
|---|---|---|
| `Preferences.exportFolder` | — | `string \| null` — default folder for export nodes and "导出"; `null` = `<library>/exports` |
| POST `/system/diagnostics` | — | `Job` (type `diagnostics`); result `{path}` of a zip with backend/shell logs, `system` report, preferences, library summary (counts, sizes) — no audio, no models |

CPU jobs (process, mix) run concurrently up to `max(1, cpu_count // 4)`; renders keep the GPU
queue. A process job never imports torch except `separate`.

---------------------------------------------------------------------------------------------

## 2. DSP ops (`stellaria_backend/dsp/`) — ✅ done (backend)

Pure numpy/scipy, reusing `mixing/` building blocks (filters, dynamics, loudness, reverb,
audioio). 44.1 kHz float32 internally; inputs resampled; output written as float WAV with the
computed `offset` in the result.

- **trim**: sample-accurate cut, equal-power fades (default 10 ms when 0 would click).
- **split**: like trim per part; `fade` applied at each cut.
- **pitch**: preference order, first available wins:
  1. Rubber Band via `pedalboard.time_stretch` when the VST host add-on (§3) is installed (best);
  2. Signalsmith Stretch (MIT) through the `python-stretch` wheel if it exists for
     win-x64 / macOS arm64+x64 / py3.10–3.12 (verify on PyPI; add to requirements only if so);
  3. a built-in phase vocoder with identity phase locking + transient preservation (fallback).
  `preserveFormant` only where the backend supports it (Rubber Band, Signalsmith); `result`
  says which engine ran (`{engine: 'rubberband' | 'signalsmith' | 'builtin'}`).
- **fx**: runs the chain slot by slot (§3); `mix` per slot is a dry/wet blend; output may be
  longer than the input by the reverb/delay tail (≤ 6 s, trimmed at −90 dBFS).
- **comp**: aligns inputs by offset, builds the segment timeline, equal-power crossfades of
  `crossfade` ms at each boundary, silence for `input: −1`.
- **concat**: sequential, `gap` seconds of silence or `crossfade` overlap.
- **export**: resample/dither (TPDF at 16-bit), loudness-normalise if asked (true-peak ceiling
  −1 dBTP), encode WAV/FLAC via soundfile, MP3 via `lameenc` (LGPL, wheels for all targets —
  verify) or ffmpeg if present; never overwrite: `name (2).flac`. Tags via `mutagen` when
  installed, else skipped.
- **sections / key analysis** (for §1.6): librosa beat tracking + chroma; cached per media.

Quality bars: a 4-minute stereo song trims/splits/comps in < 1 s, pitch-shifts (builtin) in
< 20 s, a 6-slot built-in fx chain in < 10 s on a laptop CPU; no clicks at boundaries (max
sample step at a cut < 0.05 for a 1 kHz sine at −6 dBFS); pitch ±12 semitones keeps duration
within 1 ms when tempo = 1.

### 2.1 As built

- **CLI job**: `{op, inputs: [{path, offset}], params, outputDir, bpm, vstPlugins, names,
  defaultFolder?}`; `outputDir` is required except for export, `defaultFolder` (export only)
  is used and created when `params.folder` is null. Inputs: trim/split/pitch/fx/export 1,
  comp 2–8, concat 2–32. Outputs are `<outputDir>/<port>.wav`, 32-bit float, 44.1 kHz.
- **Lines**: `{progress, stage}` (fx adds `slot` and `effect` or `plugin`), `{log}`,
  `{result: {outputs: {port: {path, offset, duration}}, result}}` (exit 0),
  `{error: {code, message, params}}` (exit 1). Stages: `dsp.read|write|<op>|fx.vst|
  export.loudness|export.encode` (the job runner shows them as `stage.dsp.*`).
- **Errors**: `dsp.bad_job`, `dsp.input_missing {path}`, `dsp.failed`, `dsp.bad_params {reason}`
  with reasons `inputs.count`, `trim.range|fade`, `split.cuts|fade`, `pitch.range`, `fx.chain`,
  `comp.segments|crossfade`, `concat.gap|crossfade`, `export.format|loudness`; export adds
  `export.folder_missing {folder}`, `export.encoder_missing {format}`,
  `export.write_failed {folder, reason}`, `export.encode_failed {format}`.
- **Results**: trim `{start, end, fadeIn, fadeOut}` as applied; split `{parts: [{port, start, end,
  label}], fades}`; pitch `{engine: rubberband|signalsmith|builtin|none, formantPreserved,
  semitones, tempo}`; fx `{slots, tail}`; comp `{segments, crossfades}` after resolution; concat
  `{starts, crossfades, gap}`; export `{path, format, sampleRate, bitDepth, mp3Bitrate, lufs,
  truePeakDb, duration, encoder, tagged, limited}` (the graph adds `downloadUrl`).
- **Differences from the plan above**:
  - Signalsmith has no formant control in its Python binding: `preserveFormant` skips it (Rubber
    Band or the built-in engine). `params.engine` may force an engine (falls back with a log line).
  - split fades are centred on each cut, so neighbouring parts overlap by the fade; a part's
    offset is the cut minus half the fade, which crossfades when parts are placed back.
  - trim fades lie inside the region; a 0 fade at a cut through audible audio becomes 10 ms.
  - comp `crossfade` is in seconds; overlapping segments: the later-starting one wins.
  - concat: `crossfade` > 0 overrides `gap` and is limited to half the shorter neighbour.
  - export: 32-bit FLAC is written as 24-bit; with loudness "keep", an integer or MP3 export that
    would clip goes through a −0.1 dBFS limiter (`limited: true`); exclusive file creation makes
    `name (2).ext` race-free; stems ≤ 200 UTF-8 bytes, paths ≤ 250 characters; MP3 via lameenc
    with a LAME/Info frame (gapless, exact length), then bundled ffmpeg, then ffmpeg on PATH.
  - Tags: mutagen (GPL-2.0+) is not shipped; a built-in ID3v2.3 writer and soundfile's WAV/FLAC
    tags cover it.
- **Dependencies**: `python-stretch` (MIT) and `lameenc` (LGPL-3.0) are in
  `requirements-runtime.txt` (wheels for win-x64, macOS arm64/x64, py3.10–3.12); Rubber Band
  comes with the optional pedalboard add-on (GPL-3.0).
- **Timings** (4-minute stereo song, 4-core CPU): trim 0.24 s, split 0.22 s, comp 0.37 s, concat
  of two songs 1.3 s, pitch built-in 3.5–11 s / Signalsmith ≈ 5 s / Rubber Band ≈ 32 s, 6-slot fx
  chain 7–10 s, export WAV/FLAC 2–2.5 s (≈ 4 s with loudness), MP3 ≈ 7 s, sections ≈ 4 s, key
  0.6 s. The first analysis after install can take 15–20 s while librosa compiles its cache.

---------------------------------------------------------------------------------------------

## 3. Effects and VST3 / Audio Unit hosting

### 3.1 Effect chain slots

```ts
type FxSlot = { id: ID; bypass: boolean; mix: number /* wet 0..1 */ } & (
  | { kind: 'builtin'; effect: string; params: Record<string, number | boolean | string> }
  | { kind: 'vst'; pluginId: ID; state: string | null /* base64 plugin state */; params: Record<string, number | boolean | string> }
)
interface EffectSpec { id: string; name: Msg; category: 'eq' | 'dynamics' | 'space' | 'color' | 'utility'; params: ParamSpec[] }
```

Built-in effects (`GET /fx/effects`), parameters as `ParamSpec` with Msg labels:
`eq` (high-pass, low shelf, 3 bells, high shelf, low-pass), `compressor`, `deesser`, `smooth`
(resonance tamer: amount, band, max cut — the mixer's 7b), `gate` (expander), `reverb`
(room/plate/chamber/hall IRs from the mixer, pre-delay, decay, tone, tail modulation as 7c),
`delay` (time in ms or note value synced to the song's BPM, feedback, ping-pong, filters),
`saturation` (tanh/tube, 2× oversampled), `doubler` (stereo micro-pitch + short delays),
`gain`, `stereo` (width, pan, mono-maker below N Hz), `limiter`.

### 3.2 VST3 / AU host add-on — ✅ done (backend)

> **Status — backend: done**, merged into `api.md` → [VST3 / Audio Unit host](api.md#vst) (types
> `VstParam`/`VstPlugin`/`VstStatus`, endpoints, job types `vst_scan`/`vst_editor`, events
> `vst.status`/`vst.plugins`, errors). Add-on `vsthost` = `pedalboard==0.9.25`. Frontend: open.

Hosting third-party plugins uses **pedalboard** (Spotify, GPL-3.0). It is **not bundled**: it
is an optional add-on resource `vsthost` (like `separator`), installed into the runtime on
demand, with its license shown in Resources and Credits. Without it, built-in effects still
work and `vst` slots fail with `vst.host_missing`.

- Formats: VST3 on Windows and macOS, Audio Units on macOS. (Linux: VST3 where available, for tests.)
- Paid plugins (FabFilter Pro-Q 3, Valhalla, Accentize Cenozoix, iZotope, …) work when they are
  installed and activated on the machine; we only host them. iLok/online activation happens in
  the plugin's own window.
- Default folders: Windows `%CommonProgramFiles%\VST3`, `%LOCALAPPDATA%\Programs\Common\VST3`;
  macOS `/Library/Audio/Plug-Ins/VST3`, `~/Library/Audio/Plug-Ins/VST3`,
  `/Library/Audio/Plug-Ins/Components`, `~/Library/Audio/Plug-Ins/Components`; plus user
  folders from settings.
- **Scanning loads each plugin in its own short-lived subprocess** with a 30 s timeout (plugins
  can crash or hang on load); a crash marks it `failed` and the scan continues. Shell bundles
  with several plugins (Waves) are expanded via `get_plugin_names_for_file`. Results are cached
  in `<data>/vst/cache.json` keyed by path + mtime; a rescan only loads new/changed files.
- **Editor**: opening the plugin's own window runs in a dedicated subprocess (no console
  window) that loads the plugin, restores the slot's `state`, calls `show_editor()` and, when
  the user closes the window, returns the new `raw_state` (base64) and the current parameter
  values. Supported on Windows and macOS; elsewhere `vst.editor_unsupported`.
- **Processing** (in `dsp.run`): load by path (+ `pluginName`), restore `state`, then apply
  `params` overrides, process at 44.1 or 48 kHz (the plugin's preferred rate if it rejects
  44.1), reset between files, tail of up to 6 s. Latency is compensated with the plugin's
  reported latency when available.

```ts
interface VstParam { id: string; name: string; kind: 'float' | 'bool' | 'choice'; min: number | null; max: number | null; default: number | string | boolean | null; unit: string | null; choices: string[] | null }
interface VstPlugin {
  id: ID                                   // stable: sha1(path + pluginName)
  name: string; vendor: string | null; version: string | null
  format: 'vst3' | 'au'; path: string; pluginName: string | null
  category: 'effect' | 'instrument'
  params: VstParam[]
  status: 'ok' | 'failed'; error: Msg | null; scannedAt: string
}
interface VstStatus { hostInstalled: boolean; hostVersion: string | null; formats: ('vst3' | 'au')[]; folders: string[]; defaultFolders: string[]; editorSupported: boolean; lastScan: string | null; scanning: boolean }
```

| Method & path | Body | Response |
|---|---|---|
| GET `/vst/status` | — | `VstStatus` |
| PATCH `/vst/settings` | `{folders: string[]}` | `VstStatus` |
| POST `/vst/scan` | `{full?: boolean}` | `Job` (type `vst_scan`, progress per file, result `{found, failed}`) → `vst.plugins` event |
| GET `/vst/plugins` | — | `VstPlugin[]` (instruments are listed but marked; only effects can go in a chain) |
| POST `/vst/plugins/{id}/editor` | `{state: string \| null}` | `Job` (type `vst_editor`, no progress); result `{state, params}` when the window closes; cancelling the job closes the window |
| POST `/vst/plugins/{id}/inspect` | `{state: string \| null}` | `{params: Record<string, number \| boolean \| string>}` — current values for a state |

Errors: `vst.host_missing`, `vst.not_found` `{name}`, `vst.load_failed` `{name, reason}`,
`vst.crashed` `{name}`, `vst.timeout` `{name, seconds}`, `vst.editor_unsupported`,
`vst.instrument` `{name}` (an instrument in an effect chain).

The mixer plugins of `creation-v2` §3 (e.g. a friend's `auto_mixing.py` that loads VST3s
itself) keep working unchanged; the host add-on gives them nothing extra and needs nothing from them.

---------------------------------------------------------------------------------------------

## 4. Mixer v2 (`builtin` auto mix) and the `bus` processor

`mixing.run` job JSON v2 (v1 still accepted):

```jsonc
{
  "inputs": [{"path": "...", "role": "lead|harmony|backing|other", "gainDb": 0, "pan": 0, "offset": 12.5}],
  "reference": {"vocals": "...|null", "mix": "...|null", "offset": 0},   // the original song, when the vocals trace back to it
  "processor": "builtin|bus", "preset": "natural", "params": {...}, "seed": 0, "output": "..."
}
```

`reference` is filled by the backend by walking each lead input upstream (render → input → …)
to a `source`; when found and that source has a `vocals` stem, the original vocals and mix are
passed with their offset.

Improvements over v1 (all deterministic for `seed: 0`):

1. **Follow the original** (`match`, 0..1, default 0.7 when a reference exists): ride the lead
   vocal so its short-term loudness relative to the backing follows the original vocal/backing
   balance (3 s windows, smoothed, clamped ±4 dB around the static level); spectral match of
   the converted vocal's long-term 1/3-octave spectrum towards the original vocal (±4 dB,
   smoothed, scaled by `match`); default reverb amount estimated from the original vocal's
   decay tails; `matchLoudness` (bool) sets the master to the original mix's LUFS instead of
   `targetLufs`.
2. **Unmasking** (`unmask`, 0..1): dynamic EQ on backing/other at 1–4 kHz keyed by the lead's
   envelope (max −3 dB), plus a gentle 200–400 Hz dip when the vocal is dense.
3. **Gap control** (`gate`, bool, default on): downward expander on vocals in their gaps
   (−6..−12 dB, hold 80 ms) — converted vocals carry hiss and artefacts in silence.
4. **Two-stage dynamics**: fast peak compressor (≈2 ms attack, 4:1, catches transients) into the
   v1 leveling compressor; bus glue compressor (2:1, slow) before the limiter.
5. **Throws** (`throws`, bool): tempo-synced delay throws on phrase ends (phrases from vocal
   activity), filtered and ducked under the next phrase. On for `airy`/`pop`.
6. **Doubler** (`doubler`, 0..1): stereo double on harmony inputs (and on loud lead sections when > 0.5).
7. **Warmth** (`warmth`, 0..1): oversampled saturation on the vocal bus.
7b. **Tame harshness** (`smooth`, 0..1, after the de-esser): a cut-only spectral dynamic EQ
   (`mixing/spectral.py`). Each STFT frame (46 ms, 75 % overlap) is judged on a 24-per-octave
   log grid: a 1/6-octave smoothing (wide enough to merge a sung note's harmonics) against a
   1-octave one (the spectral shape); where the fine one bulges more than 2 dB, that band is
   pulled down in proportion, at most 6 dB, inside 2–10 kHz with half-octave fades, with 5 ms
   attack / 80 ms release. Lead with the full chain gets the full amount, others 60 %.
7c. **Reverb modulation** (`modulation`, 0..1, advanced, space group): the IR is split 80 ms after
   its first sound; the early part stays clean, the late part's output goes through a delay
   swept by two slow incommensurate LFOs per channel (≤ 0.9 ms, ≈ ±6 cents at 1; 4-point Lagrange
   reads). One overlap-add convolution pass serves both parts; `0` is exactly the plain reverb.
8. **Harmony/other roles**: harmonies get a lighter chain (more cut below 300 Hz, −4 dB, wider);
   `other` gets level only.
9. **Master**: true-peak limiting with 4× oversampled detection (ceiling −1 dBTP), output
   report `{lufs, truePeakDb, bpm, vocalToBackingDb}`.

New params (ParamSpec, Msg labels `mix.param.<id>`): `match`, `matchLoudness`, `unmask`,
`gate`, `throws`, `doubler`, `warmth`, `smooth`, `vocalChain` (`full` | `light` | `off` — `off` when the
user already processed the vocal with their own fx chain/VSTs), plus v1's.

**`bus` processor**: places inputs by offset, applies gain/pan/mute/solo, sums; optional
`normalize` to `targetLufs` with a true-peak ceiling; nothing else. This is the "pre-mix the
instrumental first" node (e.g. drums +2 dB, other −1 dB → a backing → the final auto mix).

Quality bars (tests): v1 bars still hold; with a reference, vocal-to-backing loudness ratio
within ±1.5 dB of the original's; with `gate` on, vocal noise floor in gaps drops ≥ 6 dB on a
synthetic hissy vocal; 4-minute song with 3 inputs < 20 s.

---------------------------------------------------------------------------------------------

## 5. Frontend (canvas v3 and the rest of the UI) — ✅ done

Canvas:

- **Add anything anywhere**: double-click empty canvas / right-click → 添加 / `Tab` opens a
  searchable node menu at the cursor (歌曲, 音频/伴奏, 演唱, 混音, 剪切, 分段, 变调变速, 效果链,
  分离, 拼接择优, 串联, 导出, 便签, 分组). Dragging from a port and dropping on empty canvas
  opens the same menu filtered to nodes that accept audio, already wired.
- **Insert on an edge**: drop a node onto an edge, or edge context menu → 插入… (e.g. an fx
  chain between a take and the mix). Edge menu also: 删除连线, 试听此处.
- **Context menus** on nodes: 运行, 运行到这里, 复制, 粘贴, 创建副本, 绕过 (pitch/fx), 重命名,
  颜色, 删除; on multi-selection: 分组, 对齐/分布, 运行所选, 批量试唱.
- **Undo/redo** (`Ctrl/⌘+Z`, `Ctrl/⌘+Shift+Z`) for every structural or parameter edit;
  **copy/paste/duplicate** (`Ctrl/⌘+C/V/D`) including across projects (paste keeps internal
  edges, drops dangling ones).
- **Templates** (new project or into the current one): 翻唱 (song → split stems → N voices →
  auto mix → export), 对唱 (song → stems → two voices → comp by sections → mix), 先混伴奏
  (song 4 stems → bus → auto mix), 只做干声 (song → vocals → voice → fx → export).
- **Port preview**: hover a port to see its waveform thumbnail; click the port dot with Alt to
  play it. Every node shows its main output's waveform; the inspector edits regions (trim,
  split cuts, comp segments) directly on a waveform that snaps to phrases/bars from `/sections`.
- **Comp editor**: lanes per input (voice colour), click a section to assign it to a lane,
  drag boundaries, crossfade handle; "按段落自动分配" alternates voices by section for duets.
- **FX chain editor**: slot list with drag reorder, bypass, wet, built-in effects by
  `EffectSpec`, VST slots with their parameters (searchable), "打开插件界面" (native editor),
  save/load chain presets (local), A/B against bypass.
- **Export node** + "导出全部" in the run menu; default export folder in Settings.
- Minimap toggle, snap to grid, align/distribute, collapse a node to its header.

Elsewhere:

- Settings → 效果器插件 (VST3/AU): host add-on status/installation, folders, scan, list with
  search, failed plugins with reasons, rescan; default export folder.
- Resources: the `vsthost` add-on with its GPL-3 note.
- Transport: loop a region in the inspector waveform, scrubbing, level meter while playing,
  "原唱对比" toggle (switch the playing take with the source vocal at the same position).
- Diagnostics: Settings → 关于 → "导出诊断信息" (zip of logs + system report, no audio).

---------------------------------------------------------------------------------------------

## 6. Implementation split and internal interfaces (backend)

Fixed so the pieces meet; all three packages must not import torch.

**Effects + DSP ops** — packages `stellaria_backend/fx/` and `stellaria_backend/dsp/` (owner:
mixer team, reusing `mixing/`).

```python
# stellaria_backend/fx/__init__.py
def effect_specs() -> list[dict]                       # EffectSpec wire dicts (Msg labels fx.effect.<id>, fx.param.<id>)
def apply_chain(x: np.ndarray, sr: int, chain: list[dict], *, bpm: float | None,
                vst_plugins: dict[str, dict]) -> np.ndarray
    # x: float32 (channels, samples). Built-in slots run here; 'vst' slots call
    # stellaria_backend.vst.processing.apply(x, sr, slot, vst_plugins[slot["pluginId"]]).
    # Unknown plugin id → ApiError('vst.not_found', name=...). Bypassed slots are skipped; 'mix' blends dry/wet.

# CLI: python -m stellaria_backend.dsp.run <job.json>   (same JSON-lines protocol as mixing.run)
# job.json = {"op": "trim|split|pitch|fx|comp|concat|export",
#             "inputs": [{"path": str, "offset": float}], "params": {...}, "outputDir": str,
#             "bpm": float | None, "vstPlugins": {id: VstPlugin wire dict}, "names": {...}}   # names: export template values
# result: {"outputs": {port: {"path": str, "offset": float, "duration": float}}, "result": {...}}  # export: outputs {} and result {"path"}
```

**VST host** — package `stellaria_backend/vst/` + router `api/vst.py` (owner: plugins team) — **done**.
`processing` also exports `load(plugin)` (per-process cache, so a chain using one plugin twice loads it once),
`clear_cache()`, `resample()` and `trim_tail()`. `apply()` returns float32 (channels, samples), longer than the
input by the tail (≤ 6 s) and stereo when a stereo-only plugin got mono; latency is compensated by pedalboard.

```python
# ctx.vst: VstRegistry
def status() -> dict                     # VstStatus wire
def plugins() -> list[dict]              # VstPlugin wire, from the cache
def get(plugin_id) -> dict | None
# stellaria_backend/vst/processing.py — imported inside the dsp subprocess only
def apply(x: np.ndarray, sr: int, slot: dict, plugin: dict) -> np.ndarray   # pedalboard; raises ApiError vst.*
def host_available() -> bool
```

The `vsthost` add-on is a resource in `resources/manifest.json` (kind `addon`, packages
`pedalboard`, license GPL-3.0) installed by the existing add-on installer.

**Graph** — `creation/` + `api/projects.py` (owner: backend team) — **done**: v3 migration, port resolution
with offsets, `/run`, stems import, sections/key analysis endpoints (calling helpers from `dsp`),
process jobs (resolve inputs → write the dsp job → run the CLI as a subprocess like the mixer →
media `kind: 'process'` with `offset` → node `outputs`/`result`/status), mix v2 inputs/roles/
reference resolution, `bus` processor in `/mix/processors`, stale propagation,
`Preferences.exportFolder`, `/system/diagnostics`, `/fx/effects` (→ `fx.effect_specs()`).
