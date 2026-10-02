"""Wire descriptions of the built-in processors (``builtin`` auto mix, ``bus`` plain sum) and the
resolution of preset + params (+ seeded jitter) into concrete mixer settings.

Params whose value is missing or ``null`` take the preset's value (``reverb``, ``width``,
``throws``, ``doubler`` and ``warmth`` have ``default: null`` for that reason; each preset lists
its values under ``presets[].params``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from . import tempo
from .params import Param, boolean, choice, msg, number

PROCESSOR_ID = "builtin"
BUS_ID = "bus"
DEFAULT_PRESET = "natural"
MAX_VOCALS = 4            # lead + harmony inputs of the auto mix
MAX_INPUTS = 12           # all inputs of the auto mix
BUS_MAX_INPUTS = 16
VOCAL_CHAINS = ("full", "light", "off")

PARAMS: tuple[Param, ...] = (
    number("vocalLevel", 1.0, -6.0, 6.0, 0.5, "dB"),
    number("reverb", None, 0.0, 1.0, 0.01),
    number("brightness", 0.0, -1.0, 1.0, 0.05),
    number("width", None, 0.0, 1.0, 0.05),
    number("targetLufs", -14.0, -23.0, -9.0, 0.5, "LUFS"),
    number("backingGain", 0.0, -12.0, 6.0, 0.5, "dB"),
    number("match", 0.7, 0.0, 1.0, 0.05),
    boolean("matchLoudness", False),
    number("unmask", 0.5, 0.0, 1.0, 0.05),
    boolean("gate", True),
    boolean("throws", None),
    number("doubler", None, 0.0, 1.0, 0.05),
    number("warmth", None, 0.0, 1.0, 0.05),
    number("smooth", None, 0.0, 1.0, 0.05),
    choice("vocalChain", "full", VOCAL_CHAINS, option_prefix="mix.vocalChain"),
    # fine-grained settings (shown folded): `null` = the preset's value, which `presets[].params` lists
    number("lowCut", None, 40.0, 600.0, 1.0, "Hz", group="eq"),
    number("cutFreq", None, 800.0, 8000.0, 10.0, "Hz", group="eq"),
    number("cutGain", None, -12.0, 6.0, 0.5, "dB", group="eq"),
    number("presenceFreq", None, 500.0, 5000.0, 10.0, "Hz", group="eq"),
    number("presenceGain", None, -6.0, 9.0, 0.5, "dB", group="eq"),
    number("airFreq", None, 3000.0, 16000.0, 50.0, "Hz", group="eq"),
    number("airGain", None, -6.0, 9.0, 0.5, "dB", group="eq"),
    number("shelfFreq", None, 3000.0, 20000.0, 100.0, "Hz", group="eq"),
    number("shelfGain", None, -6.0, 6.0, 0.5, "dB", group="eq"),
    number("deessFreq", None, 3000.0, 12000.0, 50.0, "Hz", group="dynamics"),
    number("deessDepth", None, 0.0, 12.0, 0.5, "dB", group="dynamics"),
    number("compRatio", None, 1.0, 10.0, 0.1, group="dynamics"),
    number("compAttack", None, 1.0, 80.0, 0.5, "ms", group="dynamics"),
    number("compKnee", None, 0.0, 12.0, 0.5, "dB", group="dynamics"),
    number("compDepth", None, 0.0, 10.0, 0.5, "dB", group="dynamics"),
    number("roomMix", None, 0.0, 1.0, 0.05, group="space"),
    number("plateMix", None, 0.0, 1.0, 0.05, group="space"),
    number("chamberMix", None, 0.0, 1.0, 0.05, group="space"),
    number("hallMix", None, 0.0, 1.0, 0.05, group="space"),
    number("decayScale", None, 0.5, 2.0, 0.05, "×", group="space"),
    number("preDelayScale", None, 0.0, 3.0, 0.05, "×", group="space"),
    number("reverbLowCut", None, 80.0, 600.0, 5.0, "Hz", group="space"),
    number("modulation", None, 0.0, 1.0, 0.05, group="space"),
    number("clip", None, 0.0, 8.0, 0.1, "dB", group="master"),
    number("ceiling", None, -3.0, -0.1, 0.1, "dBTP", group="master"),
)
PARAM_BY_ID = {p.id: p for p in PARAMS}
REVERB_MIX = {"room": "roomMix", "plate": "plateMix", "chamber": "chamberMix", "hall": "hallMix"}

BUS_PARAMS: tuple[Param, ...] = (
    boolean("normalize", False),
    number("targetLufs", -14.0, -23.0, -9.0, 0.5, "LUFS"),
)
BUS_PARAM_BY_ID = {p.id: p for p in BUS_PARAMS}


@dataclass(frozen=True)
class DelayPreset:
    fraction: float = 0.25          # note value (1/4 note)
    feedback: float = 0.35
    level_db: float = -14.0


@dataclass(frozen=True)
class Preset:
    id: str
    reverb: float
    width: float
    throws: bool
    doubler: float
    warmth: float
    weights: dict[str, float]
    decay_window: tuple[float, float]      # reverb RT60 window (s) → nearest note value
    predelay_window: tuple[float, float]   # pre-delay window (s) → nearest note value
    return_highpass_hz: float
    presence_db: float                     # +dB bell at 1.4 kHz (scaled by brightness)
    air_db: float                          # +dB bell at 8.5 kHz (scaled by brightness)
    air_shelf_db: float                    # extra high shelf at 10 kHz
    comp_ratio: float
    comp_offset_db: float                  # compressor threshold below the median sung level
    deess_max_db: float
    delay: DelayPreset | None = None       # constant delay in the space (used when throws are off)
    throw_fraction: float = 0.25           # throw delay note value
    # the rest of the vocal chain and the master (shared by most presets)
    lowcut_hz: float = 230.0
    cut_hz: float = 2700.0
    cut_db: float = -2.0
    presence_hz: float = 1400.0
    air_hz: float = 8500.0
    air_shelf_hz: float = 10000.0
    deess_hz: float = 6500.0
    comp_attack_s: float = 0.019
    comp_knee_db: float = 4.5
    clip_db: float = 3.6
    ceiling_db: float = -1.0
    smooth: float = 0.3                    # harsh resonances tamed (spectral.smooth_resonances)
    modulation: float = 0.4                # reverb tail modulation

    def values(self) -> dict:
        """The preset's value of every param whose default is `null` (what the controls show)."""
        out = {"reverb": self.reverb, "width": self.width, "throws": self.throws,
               "doubler": self.doubler, "warmth": self.warmth, "smooth": self.smooth,
               "modulation": self.modulation,
               "lowCut": self.lowcut_hz, "cutFreq": self.cut_hz, "cutGain": self.cut_db,
               "presenceFreq": self.presence_hz, "presenceGain": self.presence_db,
               "airFreq": self.air_hz, "airGain": self.air_db,
               "shelfFreq": self.air_shelf_hz, "shelfGain": self.air_shelf_db,
               "deessFreq": self.deess_hz, "deessDepth": self.deess_max_db,
               "compRatio": self.comp_ratio, "compAttack": round(self.comp_attack_s * 1000.0, 1),
               "compKnee": self.comp_knee_db, "compDepth": self.comp_offset_db,
               "decayScale": 1.0, "preDelayScale": 1.0, "reverbLowCut": self.return_highpass_hz,
               "clip": self.clip_db, "ceiling": self.ceiling_db}
        out.update({key: float(self.weights.get(kind, 0.0)) for kind, key in REVERB_MIX.items()})
        return out

    def wire(self) -> dict:
        return {"id": self.id, "name": msg(f"mix.preset.{self.id}"), "params": self.values()}


PRESETS: dict[str, Preset] = {p.id: p for p in (
    Preset("natural", reverb=0.35, width=0.8, throws=False, doubler=0.3, warmth=0.2,
           weights={"room": 0.3, "plate": 0.4, "chamber": 0.3},
           decay_window=(1.3, 2.0), predelay_window=(0.015, 0.035), return_highpass_hz=220.0,
           presence_db=3.0, air_db=2.5, air_shelf_db=0.0, comp_ratio=3.5, comp_offset_db=2.0, deess_max_db=6.0),
    Preset("pop", reverb=0.3, width=0.9, throws=True, doubler=0.4, warmth=0.3,
           weights={"plate": 0.6, "hall": 0.25, "room": 0.15},
           decay_window=(1.4, 2.2), predelay_window=(0.02, 0.045), return_highpass_hz=250.0,
           presence_db=3.5, air_db=3.0, air_shelf_db=0.5, comp_ratio=4.5, comp_offset_db=4.0, deess_max_db=6.0,
           throw_fraction=0.1875, smooth=0.35, modulation=0.5),
    Preset("airy", reverb=0.5, width=1.0, throws=True, doubler=0.4, warmth=0.15,
           weights={"hall": 0.6, "plate": 0.4},
           decay_window=(2.2, 3.4), predelay_window=(0.03, 0.07), return_highpass_hz=200.0,
           presence_db=2.5, air_db=3.5, air_shelf_db=1.5, comp_ratio=3.5, comp_offset_db=3.0, deess_max_db=6.0,
           delay=DelayPreset(), modulation=0.6),
    Preset("dry", reverb=0.12, width=0.6, throws=False, doubler=0.0, warmth=0.25,
           weights={"room": 0.7, "chamber": 0.3},
           decay_window=(0.5, 0.9), predelay_window=(0.005, 0.015), return_highpass_hz=300.0,
           presence_db=2.5, air_db=2.0, air_shelf_db=0.0, comp_ratio=3.0, comp_offset_db=2.0, deess_max_db=5.0,
           smooth=0.25, modulation=0.2),
    # a classic hand-built vocal chain: firm de-essing, 5:1 leveling, a gentle top shelf, four reverbs
    # in parallel (energy shares from their send/return gains) and a soft clip into the limiter
    Preset("classic", reverb=0.35, width=0.8, throws=False, doubler=0.0, warmth=0.0,
           weights={"room": 0.4, "plate": 0.1, "chamber": 0.15, "hall": 0.35},
           decay_window=(1.1, 1.8), predelay_window=(0.01, 0.03), return_highpass_hz=200.0,
           presence_db=3.0, air_db=2.5, air_shelf_db=-2.0, comp_ratio=5.0, comp_offset_db=2.0, deess_max_db=12.0,
           air_shelf_hz=20000.0, deess_hz=6550.0, clip_db=3.65, smooth=0.0, modulation=0.5),
)}


def builtin_processor() -> dict:
    """``MixProcessor`` wire dict of the built-in auto mix (camelCase, ``Msg`` labels)."""
    return {
        "id": PROCESSOR_ID,
        "name": msg("mix.builtin"),
        "kind": "builtin",
        "presets": [p.wire() for p in PRESETS.values()],
        "params": [p.wire("mix.param") for p in PARAMS],
        "maxVocals": MAX_VOCALS,
        "maxInputs": MAX_INPUTS,
        "available": True,
        "error": None,
    }


def bus_processor() -> dict:
    """``MixProcessor`` wire dict of the plain-sum ``bus`` processor (no presets)."""
    return {
        "id": BUS_ID,
        "name": msg("mix.bus"),
        "kind": "builtin",
        "presets": [],
        "params": [p.wire("mix.param") for p in BUS_PARAMS],
        "maxVocals": BUS_MAX_INPUTS,
        "maxInputs": BUS_MAX_INPUTS,
        "available": True,
        "error": None,
    }


# ----------------------------------------------------------------------------- settings

@dataclass
class MixSettings:
    preset: str
    seed: int
    # user params (resolved)
    vocal_level: float
    reverb: float
    brightness: float
    width: float
    target_lufs: float
    backing_gain: float
    match: float = 0.7
    match_loudness: bool = False
    unmask: float = 0.5
    gate: bool = True
    throws: bool = False
    doubler: float = 0.3
    warmth: float = 0.2
    smooth: float = 0.3
    modulation: float = 0.4
    vocal_chain: str = "full"
    reverb_from_preset: bool = True        # the user did not set `reverb` (an estimate may replace it)
    # vocal chain
    hp_hz: float = 90.0
    lowcut_hz: float = 230.0
    cut_hz: float = 2700.0
    cut_db: float = -2.0
    cut_q: float = 1.0
    presence_hz: float = 1400.0
    presence_db: float = 3.0
    presence_q: float = 1.15
    air_hz: float = 8500.0
    air_db: float = 2.5
    air_q: float = 1.0
    air_shelf_hz: float = 10000.0
    air_shelf_db: float = 0.0
    deess_hz: float = 6500.0
    deess_max_db: float = 6.0
    peak_ratio: float = 4.0                # fast peak compressor ahead of the leveler
    peak_attack_s: float = 0.002
    peak_release_s: float = 0.05
    peak_above_median_db: float = 8.0
    comp_ratio: float = 3.5
    comp_knee_db: float = 4.5
    comp_attack_s: float = 0.019
    comp_offset_db: float = 2.0
    comp_release_offset_s: float = 0.0
    # harmony
    harmony_lowcut_hz: float = 300.0
    harmony_level_db: float = -4.0
    harmony_send_db: float = 3.0
    # space
    weights: dict[str, float] = field(default_factory=dict)
    decay_window: tuple[float, float] = (1.3, 2.0)
    predelay_window: tuple[float, float] = (0.015, 0.035)
    decay_factor: float = 1.0
    predelay_factor: float = 1.0
    return_highpass_hz: float = 220.0
    delay: DelayPreset | None = None
    throw_fraction: float = 0.25
    throw_level_db: float = -8.0
    # backing
    pocket_hz: float = 2500.0
    pocket_db: float = -1.5
    pocket_q: float = 1.0
    # master
    glue_ratio: float = 2.0
    glue_attack_s: float = 0.03
    glue_knee_db: float = 6.0
    ceiling_db: float = -1.0
    clip_db: float = 3.6
    clip_softness: float = 0.2

    def summary(self) -> dict:
        d = asdict(self)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


@dataclass
class Timings:
    bpm: float                    # timing BPM (halved above 100, 90 when unknown)
    comp_release_s: float
    predelay_s: float
    decay_s: float
    limiter_release_s: float
    delay_s: float | None
    throw_s: float = 0.0
    glue_release_s: float = 0.3


def param(params: dict | None, key: str, fallback: Any = None, table: dict[str, Param] | None = None) -> Any:
    spec = (table or PARAM_BY_ID)[key]
    raw = params.get(key) if isinstance(params, dict) else None
    return spec.parse(raw, fallback)


_UNSET = object()


def resolve(preset: str | None, params: dict | None, seed: int = 0) -> MixSettings:
    """Preset + params (+ seeded jitter for ``seed != 0``) → concrete settings."""
    p = PRESETS.get(preset or DEFAULT_PRESET, PRESETS[DEFAULT_PRESET])
    params = params or {}
    brightness = float(param(params, "brightness"))
    fine = p.values()

    def adv(key: str) -> float:
        return float(param(params, key, fine[key]))

    weights = {kind: adv(key) for kind, key in REVERB_MIX.items()}
    s = MixSettings(
        preset=p.id, seed=int(seed),
        vocal_level=float(param(params, "vocalLevel")),
        reverb=float(param(params, "reverb", p.reverb)),
        brightness=brightness,
        width=float(param(params, "width", p.width)),
        target_lufs=float(param(params, "targetLufs")),
        backing_gain=float(param(params, "backingGain")),
        match=float(param(params, "match")),
        match_loudness=bool(param(params, "matchLoudness")),
        unmask=float(param(params, "unmask")),
        gate=bool(param(params, "gate")),
        throws=bool(param(params, "throws", p.throws)),
        doubler=float(param(params, "doubler", p.doubler)),
        warmth=float(param(params, "warmth", p.warmth)),
        smooth=float(param(params, "smooth", p.smooth)),
        modulation=adv("modulation"),
        vocal_chain=str(param(params, "vocalChain")),
        reverb_from_preset=param(params, "reverb", _UNSET) is _UNSET,
        lowcut_hz=adv("lowCut"),
        cut_hz=adv("cutFreq"),
        cut_db=adv("cutGain"),
        presence_hz=adv("presenceFreq"),
        presence_db=adv("presenceGain") * (1.0 + 0.5 * brightness),
        air_hz=adv("airFreq"),
        air_db=adv("airGain") * (1.0 + brightness),
        air_shelf_hz=adv("shelfFreq"),
        air_shelf_db=adv("shelfGain") + 1.0 * brightness,
        deess_hz=adv("deessFreq"),
        deess_max_db=adv("deessDepth"),
        comp_ratio=adv("compRatio"),
        comp_attack_s=adv("compAttack") / 1000.0,
        comp_knee_db=adv("compKnee"),
        comp_offset_db=adv("compDepth"),
        weights={kind: w for kind, w in weights.items() if w > 0},
        decay_window=p.decay_window,
        predelay_window=p.predelay_window,
        decay_factor=adv("decayScale"),
        predelay_factor=adv("preDelayScale"),
        return_highpass_hz=adv("reverbLowCut"),
        delay=p.delay,
        throw_fraction=p.throw_fraction,
        clip_db=adv("clip"),
        ceiling_db=adv("ceiling"),
    )
    if s.seed:
        jitter(s, np.random.default_rng(s.seed))
    return s


def jitter(s: MixSettings, rng: np.random.Generator) -> None:
    """Small random variations for alternative takes (the draw order is fixed → deterministic)."""
    s.cut_db += rng.uniform(-0.5, 0.5)
    s.presence_db += rng.uniform(-0.5, 0.5)
    s.air_db += rng.uniform(-0.5, 0.5)
    s.pocket_db += rng.uniform(-0.5, 0.5)
    s.cut_hz += rng.uniform(-150.0, 150.0)
    s.presence_hz += rng.uniform(-150.0, 150.0)
    s.air_hz += rng.uniform(-150.0, 150.0)
    s.comp_attack_s = max(0.005, s.comp_attack_s + rng.uniform(-0.006, 0.006))
    s.comp_release_offset_s = rng.uniform(-0.06, 0.06)
    s.predelay_factor *= 1.0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 0.3)
    s.decay_factor *= 1.0 + rng.choice([-1.0, 1.0]) * rng.uniform(0.1, 0.3)


def timings(s: MixSettings, detected_bpm: float | None) -> Timings:
    bpm = tempo.timing_bpm(detected_bpm)
    delay_s = tempo.note_seconds(bpm, s.delay.fraction) if s.delay else None
    return Timings(
        bpm=bpm,
        comp_release_s=max(0.1, tempo.synced(bpm, 0.35, 0.5) + s.comp_release_offset_s),
        predelay_s=tempo.synced(bpm, *s.predelay_window) * s.predelay_factor,
        decay_s=tempo.synced(bpm, *s.decay_window) * s.decay_factor,
        limiter_release_s=tempo.synced(bpm, 0.05, 0.15),
        delay_s=delay_s,
        throw_s=tempo.note_seconds(bpm, s.throw_fraction),
        glue_release_s=tempo.synced(bpm, 0.2, 0.4),
    )
