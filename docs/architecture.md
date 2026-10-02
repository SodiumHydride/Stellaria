# 星籁 Stellaria — Architecture

Stellaria is a desktop studio (Windows + macOS) around the DDSP-SVC engine that lives in the
root of this repository. It covers the whole journey: preparing material → health check →
preprocessing → training → creating songs, without a terminal window ever showing up.

```
┌──────────────────────┐  invoke / events  ┌──────────────────────┐
│  Frontend (React)     │ ◄───────────────► │  Shell (Tauri 2/Rust) │
│  desktop/src          │                   │  desktop/src-tauri    │
└─────────┬────────────┘                   └─────────┬────────────┘
          │ HTTP + WebSocket (127.0.0.1, token)        │ spawns (no console window)
          ▼                                            ▼
┌──────────────────────┐  spawns subprocesses  ┌──────────────────────────────┐
│  Backend (FastAPI)    │ ────────────────────► │ Engine = DDSP-SVC (engine/)    │
│  backend/             │   preprocess / train  │ preprocess.py, train_reflow.py │
│  never imports torch  │   render worker       │ + our render worker            │
└──────────────────────┘                       └──────────────────────────────┘
```

* **Shell (Rust)**: window, tray, single instance, first-run bootstrap of the Python runtime
  (bundled `uv` → standalone Python → venv → torch matching the GPU → packages), starting and
  supervising the backend process, native dialogs/notifications. Every child process is
  started with `CREATE_NO_WINDOW` on Windows.
* **Backend (Python, FastAPI)**: the single source of truth for the user's library
  (characters, clips, models, projects), the job queue, the audio pipeline, quality checks,
  engine adapters, metrics reading (tfevents), downloads of pretrained models and add-ons.
  The API process itself never imports torch so it stays light and responsive; everything
  heavy runs in subprocesses.
* **Engine**: DDSP-SVC as-is. Training and preprocessing run the original scripts with a
  generated config. Rendering uses a persistent worker (`stellaria_backend/engine/render_worker.py`)
  that keeps models loaded and mirrors `main_reflow.py`'s inference loop.

The frontend also runs in a plain browser against a backend started with `--dev`
(useful for development and UI tests); all shell features have browser fallbacks.

## Directory layout

```
  docs/                 architecture.md (this file), api.md (HTTP/WS + shell bridge contract)
  backend/              Python package `stellaria_backend` + tests + requirements
  desktop/              Vite + React + TS frontend (src/) and Tauri shell (src-tauri/)
  resources/            manifest.json (runtime + downloadable models, mirrors, hashes)
  scripts/              dev helpers (run backend in dev mode, stage engine for bundling)
  engine/               DDSP-SVC (yxlllc, MIT) with Stellaria's engine fixes; see engine/README.md
```

CI is `.github/workflows/stellaria.yml`.

## Runtime layout on the user's machine

| What | Where |
|---|---|
| App install (read only) | shell binary, frontend assets, bundled resources: `uv`, `engine/` (copy of DDSP-SVC sources), `backend/` (copy of `stellaria_backend`), `manifest.json` |
| App data (`%LOCALAPPDATA%\Stellaria`, `~/Library/Application Support/Stellaria`) | `shell.json`, `runtime/` (python, venv, uv cache), `models/` (pretrained + add-ons), `logs/`, `preferences.json` |
| Library (user chosen, default `Documents/Stellaria`) | `characters/`, `models/`, `projects/`, `library.json` |

### Library format

```
<library>/
  library.json                       {"version": 1}
  characters/<chr_id>/
    character.json                   metadata (name, color, tags, consent, pinned, archived ...)
    avatar.png                       optional
    raw/<raw_id>.<ext>               imported originals, never modified
    raw.json                         list of RawImport
    clips/<name>.wav                 44.1 kHz mono material clips (named "<character>_0001.wav")
    clips.json                       list of Clip (status active|trashed, split train|val, metrics, flags)
    health.json                      last HealthReport
  models/<mdl_id>/
    model.json                       Model (characters + speaker ids, config, status, target steps)
    dataset/train/audio/[<spk>/]*.wav  assembled from active clips (hard links, copy fallback)
    dataset/val/audio/[<spk>/]*.wav
    dataset/... units/ f0/ mel/ ...  written by DDSP-SVC preprocess.py
    config.yaml                      generated engine config (absolute paths)
    exp/                             DDSP-SVC expdir (model_<step>.pt, logs/ tfevents, config.yaml)
    releases/<rel_id>/               model.pt + config.yaml + release.json (named, starred checkpoints)
    cache/validation/<step>/         wav files extracted from tfevents for playback
  projects/<prj_id>/
    project.json                     canvas + timeline
    media.json                       list of Media
    media/<med_id>.wav               imports (converted), stems, renders, mixdowns
```

All JSON writes are atomic (write to temp + rename). IDs are short random strings with a
type prefix (`chr_`, `clp_`, `raw_`, `mdl_`, `rel_`, `prj_`, `med_`, `nd_`, `job_`, `trk_`, `tc_`).

## Key product rules

* **Originals are never modified.** Organizing material creates clips from raw imports; every
  destructive-looking action (trash) is reversible until the user empties the trash.
* **Datasets never mix.** Each character owns its clips. A model lists the characters it
  contains; the training dataset is assembled per model. A multi-character model
  ("constellation") gets numbered speaker folders automatically; users only see names.
* **Training is never slowed down by the UI.** The UI only reads files (tfevents) and polls
  NVML; nothing is added to the training loop. Pausing writes a `STOP` file which the engine
  honours by saving a checkpoint first, so pausing never loses progress.
* **GPU scheduling.** Jobs that use the GPU (preprocess, train, render, separate) run one at a
  time. Renders and separations that arrive while a training holds the GPU follow the preference
  `gpuPolicy`: `smart` (default) runs them on the GPU next to the training when ≥ 2.5 GB VRAM is
  free and otherwise on the CPU right away; `cpu` always runs them on the CPU so training speed
  is untouched; `queue` makes them wait until the training pauses or ends (the UI offers to pause
  it). The render worker keeps its GPU models while serving such CPU requests and releases the
  CPU copies after ~2 minutes idle. Renders of a model's latest checkpoint during its training
  use a completely written copy of that checkpoint.
* **macOS** supports material preparation, health checks and creation (render on MPS with CPU
  fallback). Training requires an NVIDIA GPU; the UI explains this and offers model packages
  (`.stella`) to move models between machines.
* **No console windows, no raw tracebacks.** Errors reach the UI as `{code, message, detail}`;
  logs are viewable and copyable in-app.

## Brand and motion

* **Two marks.** 旋律星 (`Logo`, a phrase of melody whose highest note is a star) is the brand:
  app icon, title bar, boot screen, About, tray. While Stellaria works on something it runs:
  a light in the star's colour travels along the melody and the star twinkles (task dock, canvas run, node
  status, render / separate / train buttons). 声波星 (`LevelMark`, level bars standing in a
  star) moves like a level meter and is only for mixing and exporting audio (mix nodes, the
  export op, the export dialog, mix jobs in the dock). Short waits (loading a list, saving)
  keep the plain spinner.
* **Icons** are generated from `desktop/src-tauri/icons/*.svg` by `desktop/scripts/icons.sh`
  (`icon-macos.svg` is the same tile on Apple's 824 px grid, for `.icns` only).
* **Motion** (`--t-fast` 120 ms press/hover, `--t-med` 220 ms switch/expand, `--t-slow` 420 ms
  large entrances; exits faster than entrances). Only transform and opacity move, so playback
  and canvas dragging never drop frames. Loops (running marks, shimmering status text) exist
  only while a job runs and stop under reduced motion; they are too small to cost the training
  anything, so the low-power training mode (which only drops the glass blur) keeps them. One-shot
  feedback (`lib/motion.ts`) follows reduced motion. Audio editors (pitch, waveform, timeline)
  get no decorative motion beyond the playhead.

## Hardware → runtime variant

| Detected | torch variant | Training | Rendering |
|---|---|---|---|
| NVIDIA, compute capability ≥ 7.5 | `cu128` | yes (fused Triton kernels on) | CUDA |
| NVIDIA, compute capability < 7.5 | `cu126` | yes (fused kernels off) | CUDA |
| Apple Silicon | `mac` (PyPI default) | no | MPS → CPU fallback |
| Anything else | `cpu` | no (could, but far too slow) | CPU |

`torch==2.9.1`, `torchaudio==2.9.1`, Python 3.11 (pinned in `resources/manifest.json`).

## Download sources

Everything downloadable (standalone Python, pip index, torch index, pretrained models,
add-ons) has an ordered list of sources in `resources/manifest.json`, tagged `global` or `cn`.
Region `auto` measures latency and prefers the fastest reachable source; failures fall over to
the next source; downloads resume (HTTP Range) and are verified (size, sha256 when known).
