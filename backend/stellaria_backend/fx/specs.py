"""Built-in effects: ``EffectSpec`` wire dicts (``docs/canvas-v3.md`` §3.1).

Labels: effect names ``fx.effect.<id>``, parameters ``fx.param.<id>`` (a parameter id means the
same thing in every effect that has it), choice options ``fx.<option set>.<value>``. Defaults
are chosen so that an effect dropped on a vocal with no tweaks already sounds right.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..mixing.params import Param, boolean, choice, msg, number

REVERB_TYPES = ("room", "plate", "chamber", "hall")
DELAY_SYNC = ("off", "1/16", "1/8", "1/8d", "1/8t", "1/4", "1/4d", "1/2", "1/1")
SATURATION_MODES = ("tube", "tanh")
CATEGORIES = ("eq", "dynamics", "space", "color", "utility")


@dataclass(frozen=True)
class Effect:
    id: str
    category: str
    params: tuple[Param, ...]
    tail: bool = False                  # output can be longer than the input (reverb, delay)

    def wire(self) -> dict:
        return {"id": self.id, "name": msg(f"fx.effect.{self.id}"), "category": self.category,
                "params": [p.wire("fx.param") for p in self.params]}

    def parse(self, params: dict | None) -> dict:
        raw = params if isinstance(params, dict) else {}
        return {p.id: p.parse(raw.get(p.id)) for p in self.params}


EFFECTS: dict[str, Effect] = {e.id: e for e in (
    Effect("eq", "eq", (
        boolean("highPass", True), number("highPassFreq", 80.0, 20.0, 1000.0, 1.0, "Hz"),
        number("lowShelfFreq", 150.0, 40.0, 1000.0, 1.0, "Hz"), number("lowShelfGain", 0.0, -15.0, 15.0, 0.1, "dB"),
        number("band1Freq", 300.0, 40.0, 18000.0, 1.0, "Hz"), number("band1Gain", -1.5, -15.0, 15.0, 0.1, "dB"),
        number("band1Q", 1.0, 0.2, 10.0, 0.05),
        number("band2Freq", 1000.0, 40.0, 18000.0, 1.0, "Hz"), number("band2Gain", 0.0, -15.0, 15.0, 0.1, "dB"),
        number("band2Q", 1.0, 0.2, 10.0, 0.05),
        number("band3Freq", 3500.0, 40.0, 18000.0, 1.0, "Hz"), number("band3Gain", 1.5, -15.0, 15.0, 0.1, "dB"),
        number("band3Q", 1.0, 0.2, 10.0, 0.05),
        number("highShelfFreq", 10000.0, 1000.0, 20000.0, 10.0, "Hz"),
        number("highShelfGain", 1.5, -15.0, 15.0, 0.1, "dB"),
        boolean("lowPass", False), number("lowPassFreq", 18000.0, 1000.0, 22000.0, 10.0, "Hz"),
    )),
    Effect("compressor", "dynamics", (
        boolean("autoThreshold", True), number("threshold", -18.0, -60.0, 0.0, 0.5, "dB"),
        number("ratio", 3.0, 1.0, 20.0, 0.1), number("attack", 10.0, 0.1, 200.0, 0.1, "ms"),
        number("release", 120.0, 10.0, 2000.0, 1.0, "ms"), number("knee", 4.0, 0.0, 12.0, 0.5, "dB"),
        boolean("autoMakeup", True), number("makeup", 0.0, -12.0, 24.0, 0.5, "dB"),
    )),
    Effect("deesser", "dynamics", (
        number("frequency", 6500.0, 3000.0, 12000.0, 10.0, "Hz"), number("range", 6.0, 0.0, 12.0, 0.5, "dB"),
        number("sensitivity", 0.5, 0.0, 1.0, 0.05),
    )),
    Effect("smooth", "dynamics", (
        number("amount", 0.5, 0.0, 1.0, 0.05), number("bandLow", 2000.0, 500.0, 8000.0, 10.0, "Hz"),
        number("bandHigh", 10000.0, 3000.0, 18000.0, 50.0, "Hz"), number("maxCut", 6.0, 1.0, 12.0, 0.5, "dB"),
    )),
    Effect("gate", "dynamics", (
        boolean("autoThreshold", True), number("threshold", -50.0, -90.0, 0.0, 0.5, "dB"),
        number("range", 10.0, 0.0, 40.0, 0.5, "dB"), number("hold", 80.0, 0.0, 500.0, 1.0, "ms"),
        number("attack", 2.0, 0.1, 50.0, 0.1, "ms"), number("release", 60.0, 5.0, 1000.0, 1.0, "ms"),
    )),
    Effect("reverb", "space", (
        choice("type", "plate", REVERB_TYPES, option_prefix="fx.reverbType"),
        number("decay", 1.8, 0.2, 8.0, 0.05, "s"), number("preDelay", 20.0, 0.0, 250.0, 1.0, "ms"),
        number("tone", 0.0, -1.0, 1.0, 0.05), number("width", 1.0, 0.0, 1.0, 0.05),
        number("lowCut", 200.0, 20.0, 1000.0, 1.0, "Hz"), number("level", -16.0, -40.0, 0.0, 0.5, "dB"),
        number("modulation", 0.5, 0.0, 1.0, 0.05),
    ), tail=True),
    Effect("delay", "space", (
        choice("sync", "1/4", DELAY_SYNC, option_prefix="fx.delaySync"),
        number("time", 375.0, 1.0, 2000.0, 1.0, "ms"), number("feedback", 0.3, 0.0, 0.95, 0.01),
        boolean("pingPong", True), number("lowCut", 300.0, 20.0, 2000.0, 1.0, "Hz"),
        number("highCut", 5000.0, 1000.0, 20000.0, 10.0, "Hz"), number("level", -14.0, -40.0, 0.0, 0.5, "dB"),
    ), tail=True),
    Effect("saturation", "color", (
        choice("mode", "tube", SATURATION_MODES, option_prefix="fx.saturationMode"),
        number("drive", 6.0, 0.0, 24.0, 0.5, "dB"), number("tone", 0.0, -1.0, 1.0, 0.05),
        number("output", 0.0, -24.0, 12.0, 0.5, "dB"),
    )),
    Effect("doubler", "space", (
        number("amount", 0.5, 0.0, 1.0, 0.05), number("detune", 10.0, 0.0, 30.0, 0.5, "cents"),
        number("delay", 20.0, 5.0, 50.0, 0.5, "ms"), number("width", 1.0, 0.0, 1.0, 0.05),
    )),
    Effect("gain", "utility", (
        number("gain", 0.0, -36.0, 24.0, 0.1, "dB"), boolean("invert", False),
    )),
    Effect("stereo", "utility", (
        number("width", 1.0, 0.0, 2.0, 0.05), number("pan", 0.0, -1.0, 1.0, 0.01),
        number("monoBass", 0.0, 0.0, 500.0, 1.0, "Hz"),
    )),
    Effect("limiter", "dynamics", (
        number("ceiling", -1.0, -12.0, 0.0, 0.1, "dBTP"), number("gain", 0.0, 0.0, 24.0, 0.1, "dB"),
        number("release", 80.0, 1.0, 1000.0, 1.0, "ms"),
    )),
)}


def effect_specs() -> list[dict]:
    """``EffectSpec`` wire dicts of all built-in effects."""
    return [e.wire() for e in EFFECTS.values()]
