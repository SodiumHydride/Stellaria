"""Pydantic mirrors of ``docs/api.md`` types. camelCase on the wire and on disk."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

ID = str
Number = float | int


class Wire(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")

    def wire(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")


class Msg(Wire):
    code: str
    params: dict[str, Any] = Field(default_factory=dict)


def msg(code: str, **params: Any) -> Msg:
    return Msg(code=code, params=params)


# ---------- system ----------
class GpuInfo(Wire):
    index: int
    name: str
    vram_total: int  # bytes
    vram_used: int | None = None  # bytes
    utilization: float | None = None  # percent
    temperature: float | None = None  # °C
    compute_capability: str | None = None
    driver: str | None = None


class TorchInfo(Wire):
    available: bool = False
    version: str | None = None
    cuda: str | None = None
    device: Literal["cuda", "mps", "cpu"] = "cpu"
    triton: bool = False


class SystemInfo(Wire):
    version: str
    os: Literal["windows", "macos", "linux"]
    os_version: str
    arch: str
    cpu: str
    cpu_count: int
    ram_total: int
    gpus: list[GpuInfo]
    torch: TorchInfo
    can_train: bool
    train_blockers: list[Msg]
    library_path: str
    data_path: str
    engine_version: str
    fake_engine: bool


class Preferences(Wire):
    language: Literal["zh-CN", "en"] = "zh-CN"
    mirror_region: Literal["auto", "cn", "global"] = "auto"
    gpu_policy: Literal["smart", "cpu", "queue"] = "smart"
    low_power_while_training: bool = True
    auto_slim_after_training: bool = False
    notify_on_finish: bool = True
    glass: bool = True
    export_folder: str | None = None  # default folder of export nodes; None = <library>/exports
    separation_model: str | None = None  # default separation model id; None = the best installed one
    cleanup_model: str | None = None  # default cleanup (dereverb) model id; None = the best installed one
    # minutes without singing after which the singing models leave the GPU / memory (0 = keep them)
    gpu_idle_release_min: int = Field(default=10, ge=0, le=240)


# ---------- characters & material ----------
CharacterStage = Literal["empty", "material", "ready", "training", "trained"]
HealthLevel = Literal["excellent", "good", "fair", "poor", "blocked"]
ClipFlag = Literal["too_short", "too_long", "clipping", "silent", "noisy", "outlier", "music", "f0_failed", "duplicate"]


class Consent(Wire):
    source: str = ""
    authorized: bool = False
    note: str = ""


class CharacterStats(Wire):
    raw_count: int = 0
    clip_count: int = 0
    active_duration: float = 0.0
    train_count: int = 0
    val_count: int = 0
    trashed_count: int = 0


class CharacterHealthSummary(Wire):
    score: int
    level: HealthLevel
    checked_at: str
    stale: bool


class PitchRange(Wire):
    low: float
    high: float
    median: float


class CharacterRecord(Wire):
    """What ``character.json`` stores; ``Character`` adds computed fields."""

    id: ID
    name: str
    color: str
    has_avatar: bool = False
    avatar_version: int = 0
    tags: list[str] = Field(default_factory=list)
    notes: str = ""
    consent: Consent = Field(default_factory=Consent)
    pinned: bool = False
    archived: bool = False
    created_at: str
    updated_at: str
    last_opened_at: str | None = None
    clips_revision: int = 0


class Character(Wire):
    id: ID
    name: str
    color: str
    has_avatar: bool
    avatar_version: int
    tags: list[str]
    notes: str
    consent: Consent
    pinned: bool
    archived: bool
    created_at: str
    updated_at: str
    last_opened_at: str | None
    stage: CharacterStage
    stats: CharacterStats
    health: CharacterHealthSummary | None
    model_ids: list[ID]
    primary_model_id: ID | None
    pitch: PitchRange | None


class RawImport(Wire):
    id: ID
    name: str
    ext: str
    size: int
    duration: float | None = None
    imported_at: str
    status: Literal["imported", "organizing", "organized", "failed"] = "imported"
    error: Msg | None = None
    clip_count: int = 0


class ClipMetrics(Wire):
    peak_db: float
    rms_db: float
    silence_ratio: float
    clipping_ratio: float
    snr_db: float | None = None
    f0_low: float | None = None
    f0_high: float | None = None
    f0_median: float | None = None
    voiced_ratio: float | None = None
    outlier_score: float | None = None
    music_ratio: float | None = None


class Clip(Wire):
    id: ID
    name: str
    file: str
    raw_id: ID | None = None
    raw_name: str | None = None
    start: float | None = None
    end: float | None = None
    duration: float
    status: Literal["active", "trashed"] = "active"
    split: Literal["train", "val"] = "train"
    flags: list[ClipFlag] = Field(default_factory=list)
    metrics: ClipMetrics | None = None
    reviewed: bool = False
    created_at: str


class SliceOptions(Wire):
    enabled: bool = True
    threshold_db: float = -40.0
    min_length: float = 5.0
    min_interval: float = 0.3
    hop_size: float = 0.01
    max_sil_kept: float = 0.5


class OrganizeOptions(Wire):
    separate_vocals: bool = False
    dereverb: bool = False  # keep the dry voice (after separation): the default cleanup model
    normalize: bool = True
    target_db: float = -20.0
    slice: SliceOptions = Field(default_factory=SliceOptions)
    max_length: float = 15.0
    min_duration: float = 2.0
    drop_clipping: bool = False
    drop_silent: bool = True
    dedupe: bool = True
    name_pattern: str = "{name}_{index:04d}"
    val_count: int = 10
    raw_ids: list[ID] | None = None


class HealthFix(Wire):
    action: Literal["trash_clips", "organize", "add_material", "pick_val", "install_addon"]
    params: dict[str, Any] = Field(default_factory=dict)


class HealthIssue(Wire):
    code: str
    severity: Literal["info", "warning", "error"]
    params: dict[str, Any] = Field(default_factory=dict)
    clip_ids: list[ID] = Field(default_factory=list)
    fix: HealthFix | None = None


class HealthTotals(Wire):
    clips: int
    duration: float
    train_duration: float
    val_count: int


class PitchBin(Wire):
    midi: int
    seconds: float


class HealthPitch(Wire):
    histogram: list[PitchBin] = Field(default_factory=list)
    low: float | None = None
    high: float | None = None
    median: float | None = None


class HealthChecks(Wire):
    basic: bool = False
    pitch: bool = False
    outlier: bool = False
    music: bool = False


class HealthReport(Wire):
    score: int
    level: HealthLevel
    blocking: bool
    checked_at: str
    totals: HealthTotals
    pitch: HealthPitch
    issues: list[HealthIssue]
    checks: HealthChecks


# ---------- models & training ----------
TrainPresetId = Literal["vram6", "vram8", "vram12", "vram24", "custom"]
F0Extractor = Literal["rmvpe", "fcpe", "parselmouth", "harvest", "dio", "crepe"]
Encoder = Literal["contentvec768l12tta2x", "contentvec768l12", "hubertsoft"]
ModelStatus = Literal["idle", "preparing", "preprocessing", "training", "stopping", "paused", "done", "failed"]


class TrainConfig(Wire):
    preset: TrainPresetId = "vram8"
    target_steps: int = 100000
    batch_size: int = 48
    lr: float = 0.0005
    amp_dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    cache_device: Literal["cpu", "cuda"] = "cpu"
    num_workers: int = 2
    interval_val: int = 2000
    interval_force_save: int = 10000
    f0_extractor: F0Extractor = "rmvpe"
    encoder: Encoder = "contentvec768l12tta2x"
    n_layers: int = 6
    n_chans: int = 1024
    n_aux_layers: int = 6
    n_aux_chans: int = 512
    use_pitch_aug: bool = True
    use_fused_kernels: Literal["auto"] | bool = "auto"
    base_model_id: ID | None = None  # training starts from this base model (底模)


class TrainPreset(Wire):
    id: TrainPresetId
    vram: int  # GB
    config: dict[str, Any]
    recommended: bool


class Checkpoint(Wire):
    step: int
    file: str
    size: int
    created_at: str
    kind: Literal["regular", "permanent", "pause"]
    release_id: ID | None = None


class Release(Wire):
    id: ID
    name: str
    step: int
    created_at: str
    notes: str = ""
    size: int = 0


class ModelCharacter(Wire):
    character_id: ID
    spk_id: int


class DatasetInfo(Wire):
    status: Literal["none", "stale", "ready"] = "none"
    prepared_at: str | None = None
    clip_count: int = 0
    duration: float = 0.0


class DiskUsage(Wire):
    dataset: int
    checkpoints: int
    releases: int
    total: int


class ModelRecord(Wire):
    """What ``model.json`` stores; ``Model`` adds computed fields."""

    id: ID
    name: str
    kind: Literal["single", "constellation"]
    characters: list[ModelCharacter]
    config: TrainConfig
    status: ModelStatus = "idle"
    error: Msg | None = None
    target_steps: int = 100000
    releases: list[Release] = Field(default_factory=list)
    created_at: str
    updated_at: str
    training_time: float = 0.0
    glu_type: str | None = None  # engine model.glu_type when not the template's (imported or base models)


class Model(Wire):
    id: ID
    name: str
    kind: Literal["single", "constellation"]
    characters: list[ModelCharacter]
    config: TrainConfig
    status: ModelStatus
    error: Msg | None
    step: int
    target_steps: int
    dataset: DatasetInfo
    checkpoints: list[Checkpoint]
    releases: list[Release]
    created_at: str
    updated_at: str
    training_time: float
    speed: float | None
    eta: float | None
    disk_usage: DiskUsage | None


class ScalarSeries(Wire):
    tag: str
    points: list[tuple[int, float, float]]


class ValidationSample(Wire):
    name: str
    gt_url: str
    pred_url: str
    mel_url: str


class ValidationSnapshot(Wire):
    step: int
    samples: list[ValidationSample]


# ---------- creation ----------
class Media(Wire):
    id: ID
    name: str
    kind: Literal["import", "stem", "render", "mix", "mixdown", "process"]
    duration: float
    sample_rate: int
    channels: int
    created_at: str
    offset: float = 0.0  # seconds: where this audio starts on its song's timeline (0 for imports)


class Region(Wire):
    start: float
    end: float


class RenderParams(Wire):
    key: float = 0.0
    formant: float = 0.0
    vocal_register: float = Field(0.0, alias="register")  # `register` would shadow ABCMeta.register
    f0_extractor: F0Extractor = "rmvpe"
    steps: int = 50
    method: Literal["euler", "rk4"] = "euler"
    t_start: float = 0.0
    threshold: float = -60.0
    spk_mix: dict[str, float] | None = None
    region: Region | None = None
    f0_min: float | None = None  # Hz, pitch search range (worker default 50)
    f0_max: float | None = None  # Hz (worker default 1100)
    # the pitch editor's changes: {rate: 100, segments: [{start, values: [semitones]}]}
    pitch_edit: dict[str, Any] | None = None

    @field_validator("pitch_edit", mode="before")
    @classmethod
    def _pitch_edit(cls, value: Any) -> Any:
        from .engine.pitch_edit import parse

        return parse(value)


class Project(Wire):
    # canvas / timeline are owned by the frontend and stored verbatim
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")

    id: ID
    name: str
    created_at: str
    updated_at: str
    cover_color: str
    canvas: dict[str, Any]
    timeline: dict[str, Any]


class ProjectSummary(Wire):
    id: ID
    name: str
    created_at: str
    updated_at: str
    cover_color: str
    render_count: int
    character_ids: list[ID]


# ---------- jobs ----------
JobType = Literal[
    "import", "organize", "health", "prepare", "preprocess", "train", "render", "separate",
    "mixdown", "download", "install_addon", "export", "import_pack", "slim", "mix", "install_plugin", "plugin_test",
    "import_model", "import_base",
    "vst_scan", "vst_editor",
    "process", "diagnostics", "pitch",
]
JobStatus = Literal["queued", "waiting_gpu", "running", "stopping", "succeeded", "failed", "cancelled"]


class JobError(Wire):
    code: str
    message: str
    detail: str = ""
    params: dict[str, Any] = Field(default_factory=dict)  # message parameters, like Msg.params


class JobView(Wire):
    id: ID
    type: JobType
    title: Msg
    target: dict[str, str]
    status: JobStatus
    progress: float | None
    stage: Msg | None
    eta: float | None
    uses_gpu: bool
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: JobError | None
    result: dict[str, Any] | None


# ---------- resources ----------
class Resource(Wire):
    id: str
    kind: Literal["model", "addon", "base"]
    required: bool
    name: Msg
    description: Msg
    size: int
    status: Literal["missing", "downloading", "installing", "ready", "failed"]
    progress: float | None = None
    bytes_done: int | None = None
    speed: float | None = None
    source: str | None = None
    error: Msg | None = None
    path: str | None = None
    license: str | None = None
    homepage: str | None = None


class SeparationModel(Wire):
    """A vocal separation model (``GET /resources/separation``)."""

    id: str  # 'htdemucs' | 'htdemucs_ft' | a manifest resource id | 'sep_…' (imported)
    engine: Literal["demucs", "roformer"]
    arch: str  # 'htdemucs' | 'bs_roformer' | 'mel_band_roformer'
    source: Literal["addon", "resource", "local"]
    resource_id: str | None = None  # what to install: 'separator' or the checkpoint's resource
    name: Msg
    quality: Literal["standard", "high"]
    sdr: float | None = None  # vocals SDR (dB) reported for the checkpoint, when known
    ready: bool
    missing: Literal["addon", "addon_update", "model"] | None = None
    size: int | None = None  # checkpoint bytes
    path: str | None = None  # checkpoint file, when ready


class SeparationModels(Wire):
    models: list[SeparationModel]
    default_model: str  # what organizing, health checks and separations without a model use


class CleanupModel(Wire):
    """A vocal cleanup model: the dry voice without reverb / echo / noise (``GET /resources/cleanup``)."""

    id: str  # a manifest resource id | 'cln_…' (imported)
    kind: Literal["dereverb", "dereverb_echo", "denoise"]  # what it removes
    arch: str  # 'bs_roformer' | 'mel_band_roformer'
    source: Literal["resource", "local"]
    resource_id: str | None = None  # the checkpoint's resource
    name: Msg
    sdr: float | None = None  # SDR (dB) of the dry voice reported for the checkpoint, when known
    ready: bool
    missing: Literal["addon", "addon_update", "model"] | None = None
    size: int | None = None  # checkpoint bytes
    path: str | None = None  # checkpoint file, when ready


class CleanupModels(Wire):
    models: list[CleanupModel]  # resources (recommended first), then imported models (newest first)
    default_model: str | None  # what dereverb nodes and organizing use; None = no dereverb model is ready


# ---------- base models (底模) ----------
class BaseModelRecord(Wire):
    """``<data>/base-models/<id>/base.json`` (imported files or a copied release)."""

    id: ID
    name: str
    notes: str = ""
    created_at: str
    source: Literal["local", "release"]
    encoder: Encoder
    n_layers: int
    n_chans: int
    n_aux_layers: int
    n_aux_chans: int
    n_spk: int
    step: int
    size: int
    glu_type: str = "atanglu"


class BaseModelInfo(Wire):
    """Wire ``BaseModel`` (docs/creation-v2.md §7)."""

    id: ID
    name: str
    notes: str
    created_at: str
    source: Literal["download", "local", "release"]
    resource_id: ID | None = None
    status: Literal["ready", "missing", "downloading", "failed"]
    encoder: Encoder
    n_layers: int
    n_chans: int
    n_aux_layers: int
    n_aux_chans: int
    n_spk: int
    step: int
    size: int

