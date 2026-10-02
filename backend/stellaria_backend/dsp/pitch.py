"""Pitch / tempo op with engine selection.

Preference (first available wins): Rubber Band through ``pedalboard.time_stretch`` (the optional
VST host add-on), Signalsmith Stretch through ``python-stretch``, then the built-in phase vocoder
(:mod:`.vocoder`). ``preserveFormant`` is honoured by Rubber Band and the built-in engine; the
``python-stretch`` binding (0.3.x) does not expose Signalsmith's formant control, so a request
that needs formants skips it in favour of the built-in engine.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .common import SR, bad_params, flag, number, read_audio
from .edit import Output

ENGINES = ("rubberband", "signalsmith", "builtin")
MAX_SEMITONES = 24.0
MIN_EXTERNAL = 4096        # shorter buffers go to the built-in engine (pedalboard guesses the
                           # channel layout from the shape, which a few samples make ambiguous)


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 1:
        y = y[None, :]
    if y.shape[-1] >= n:
        return np.ascontiguousarray(y[:, :n])
    out = np.zeros((y.shape[0], n), dtype=np.float32)
    out[:, : y.shape[-1]] = y
    return out


def rubberband_available() -> bool:
    try:
        import pedalboard

        return hasattr(pedalboard, "time_stretch")
    except Exception:  # noqa: BLE001 - a broken add-on install must not break the op
        return False


def signalsmith_available() -> bool:
    try:
        import python_stretch  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def available_engines() -> list[str]:
    out = []
    if rubberband_available():
        out.append("rubberband")
    if signalsmith_available():
        out.append("signalsmith")
    out.append("builtin")
    return out


def choose_engine(requested: str | None, need_formant: bool, available: list[str] | None = None) -> str:
    """The engine to run: the requested one when available, else the first available in the
    preference order that can do what is asked (formants)."""
    available = available_engines() if available is None else available
    if requested in available and not (need_formant and requested == "signalsmith"):
        return str(requested)
    for name in ENGINES:
        if name in available and not (need_formant and name == "signalsmith"):
            return name
    return "builtin"


def _rubberband(x: np.ndarray, semitones: float, tempo: float, quality: str, formant: bool) -> np.ndarray:
    import pedalboard

    return pedalboard.time_stretch(np.ascontiguousarray(x, dtype=np.float32), float(SR), stretch_factor=float(tempo),
                                   pitch_shift_in_semitones=float(semitones), high_quality=quality != "fast",
                                   preserve_formants=bool(formant))


def _signalsmith(x: np.ndarray, semitones: float, tempo: float, quality: str) -> np.ndarray:
    import python_stretch as ps

    st = ps.Signalsmith.Stretch()
    st.preset(int(x.shape[0]), float(SR), quality == "fast")
    st.setTransposeSemitones(float(semitones))
    st.timeFactor = float(tempo)
    return st.process(np.ascontiguousarray(x, dtype=np.float32))


def shift(x: np.ndarray, semitones: float, tempo: float, *, quality: str = "high", preserve_formant: bool = False,
          engine: str | None = None, log: Callable[[str], None] = lambda _m: None,
          progress: Callable[[float], None] | None = None) -> tuple[np.ndarray, str, bool]:
    """Run the pitch/tempo change. Returns ``(audio (C, round(n/tempo)), engine, formant_preserved)``."""
    n = x.shape[-1]
    target = int(round(n / tempo))
    need_formant = preserve_formant and abs(semitones) > 1e-9
    name = choose_engine(engine, need_formant, None if n >= MIN_EXTERNAL else ["builtin"])
    if engine and engine != name:
        log(f"pitch: engine '{engine}' is not available here, using '{name}'")
    if n == 0:
        return np.zeros((x.shape[0], target), dtype=np.float32), name, need_formant
    if name == "rubberband":
        try:
            return _fit(_rubberband(x, semitones, tempo, quality, need_formant), target), name, need_formant
        except Exception as e:  # noqa: BLE001 - fall through to the next engine
            log(f"pitch: Rubber Band failed ({type(e).__name__}: {e}); falling back")
            name = choose_engine(None, need_formant, [a for a in available_engines() if a != "rubberband"])
    if name == "signalsmith":
        try:
            return _fit(_signalsmith(x, semitones, tempo, quality), target), name, False
        except Exception as e:  # noqa: BLE001
            log(f"pitch: Signalsmith failed ({type(e).__name__}: {e}); falling back to the built-in engine")
    from .vocoder import pitch_shift

    y = pitch_shift(x, semitones, tempo, quality=quality, preserve_formant=need_formant, progress=progress)
    return _fit(y, target), "builtin", need_formant


def pitch(job, ctx) -> tuple[list[Output], dict]:
    """``{semitones, cents, tempo, preserveFormant, quality}`` → port ``out`` at offset ÷ tempo."""
    p, src = job.params, job.inputs[0]
    semis = number(p, "semitones", 0.0, lo=-MAX_SEMITONES, hi=MAX_SEMITONES, reason="pitch.range")
    cents = number(p, "cents", 0.0, lo=-100.0, hi=100.0, reason="pitch.range")
    tempo = number(p, "tempo", 1.0, lo=0.5, hi=2.0, reason="pitch.range")
    total = semis + cents / 100.0
    if abs(total) > MAX_SEMITONES:
        raise bad_params("pitch.range", f"A shift of {total:+.2f} semitones is outside ±{MAX_SEMITONES:g}")
    quality = p.get("quality") if p.get("quality") in ("fast", "high") else "high"
    formant = flag(p, "preserveFormant", False)
    requested = p.get("engine") if p.get("engine") in ENGINES else None
    ctx.progress(0.02, "dsp.read")
    x = read_audio(src.path)
    offset = src.offset / tempo
    if abs(total) < 1e-9 and abs(tempo - 1.0) < 1e-12:
        ctx.log("pitch: no change requested; copying the input")
        return [Output("out", x, offset)], {"engine": "none", "formantPreserved": False,
                                            "semitones": 0.0, "tempo": 1.0}
    ctx.progress(0.1, "dsp.pitch")
    ctx.log(f"pitch: {total:+.2f} semitones, tempo ×{tempo:g}, quality {quality}"
            f"{', formants preserved' if formant else ''}")

    def on_progress(f: float) -> None:
        ctx.progress(0.1 + 0.8 * f, "dsp.pitch")

    y, engine, kept = shift(x, total, tempo, quality=quality, preserve_formant=formant, engine=requested,
                            log=ctx.log, progress=on_progress)
    if formant and not kept and abs(total) > 1e-9:
        ctx.log(f"pitch: '{engine}' cannot preserve formants")
    ctx.log(f"pitch: engine {engine}, {x.shape[-1] / SR:.2f} s → {y.shape[-1] / SR:.2f} s")
    return [Output("out", y, offset)], {"engine": engine, "formantPreserved": bool(kept),
                                        "semitones": round(total, 4), "tempo": tempo}
