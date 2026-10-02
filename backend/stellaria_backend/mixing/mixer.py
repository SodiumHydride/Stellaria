"""The built-in auto mix ("一键混音", v2): vocal chains → shared space → backing → master.

Audio is float32, channel-first, 44.1 kHz. Inputs are placed on one timeline by their offsets
(``offset - min(offset)``). See ``docs/creation-v2.md`` §2 and ``docs/canvas-v3.md`` §4.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from . import activity, chains, filters, loudness, match, reverb, spec, tempo
from . import doubler as doubling
from . import dynamics as dyn
from .audioio import SR, MixError, place, read, read_check, timeline, to_mono, to_stereo, write
from .bus import mix_bus
from .chains import VOCAL_RMS_DB, pan_gains
from .job import MixInput, MixJob, parse_job
from .master import MasterSpec, master
from .throws import throws as make_throws
from . import unmask as unmasking

__all__ = ["VOCAL_RMS_DB", "MixError", "MixInput", "mix", "pan_gains", "parse_job", "process_vocal"]

ProgressFn = Callable[[float, str], None]
LogFn = Callable[[str], None]

REVERB_REF_LU = -7.0          # wet return loudness relative to the dry vocals at reverb = 1
END_FADE_S = 0.01
MATCH_LOUDNESS_RANGE = (-30.0, -6.0)


def _gain(db_value: float) -> float:
    return 10.0 ** (float(db_value) / 20.0)


def process_vocal(x: np.ndarray, sr: float, s: spec.MixSettings, t: spec.Timings,
                  log: LogFn = lambda _m: None, label: str = "vocal") -> np.ndarray:
    """The whole lead chain on a mono vocal (v1 name)."""
    return chains.process(x, sr, s, t, log, label)


class _Background:
    """Run ``fn`` in a daemon thread (tempo detection overlaps the vocal chain)."""

    def __init__(self, fn: Callable[[], Any]) -> None:
        self._result: Any = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(fn,), daemon=True, name="mix-bpm")
        self._thread.start()

    def _run(self, fn: Callable[[], Any]) -> None:
        try:
            self._result = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in result()
            self._error = e

    def result(self) -> Any:
        self._thread.join()
        if self._error is not None:
            raise self._error
        return self._result


@dataclass
class _Reference:
    vocals: np.ndarray | None     # (2, n) aligned to the output
    mix: np.ndarray | None        # (2, n)
    backing: np.ndarray | None    # (2, n): mix - vocals, or our own backing

    @property
    def vocals_mono(self) -> np.ndarray | None:
        return None if self.vocals is None else self.vocals.mean(axis=0)


def _load_reference(j: MixJob, base: float, n: int, sr: int, log: LogFn) -> _Reference | None:
    ref = j.reference
    if ref is None:
        return None
    out = {}
    for key in ("vocals", "mix"):
        path = getattr(ref, key)
        if path is None:
            out[key] = None
            continue
        try:
            audio = to_stereo(read(path, sr)[0])
        except MixError as e:
            log(f"reference {key} ignored: {e.message}")
            out[key] = None
            continue
        out[key] = match.align(audio, ref.offset, base, n, sr)
    if out["vocals"] is None and out["mix"] is None:
        return None
    backing = out["mix"] - out["vocals"] if out["vocals"] is not None and out["mix"] is not None else None
    return _Reference(out["vocals"], out["mix"], backing)


def _warmth(x: np.ndarray, sr: float, amount: float) -> np.ndarray:
    """What parallel 2× oversampled tube saturation adds to ``x`` (mono): ``amount`` (0..1) sets
    the drive (3–12 dB) and the blend. Returns the added signal (harmonics, softened peaks)."""
    sample = np.abs(x[::7])
    ref = float(np.percentile(sample[sample > 1e-6], 99.9)) if np.any(sample > 1e-6) else 0.0
    if ref <= 0:
        return np.zeros_like(x)
    sat = dyn.saturate(x / np.float32(ref), sr, 3.0 + 9.0 * amount, mode="tube") * np.float32(ref)
    return ((sat - x) * np.float32(amount)).astype(np.float32)


def _loud_sections(x: np.ndarray, sr: float, n: int) -> np.ndarray:
    """0..1 per sample: the loudest ~30 % of the sung stretches (3 s windows), 1 s ramps."""
    st = loudness.short_term(x, sr, hop_s=0.5)
    finite = np.isfinite(st)
    if finite.sum() < 4:
        return np.zeros(n, dtype=np.float32)
    active = finite & (st > np.percentile(st[finite], 95) - 20.0)
    if not active.any():
        return np.zeros(n, dtype=np.float32)
    loud = (active & (st > np.percentile(st[active], 70))).astype(np.float64)
    loud = np.convolve(np.pad(loud, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    return dyn.uniform_interp(loud, 0.5 * sr, n)


def mix(job: dict | MixJob, progress: ProgressFn | None = None, log: LogFn | None = None,
        buses: dict | None = None) -> dict:
    """Run one mix. ``progress(fraction, stage)``; returns the report
    ``{lufs, truePeakDb, peakDb, bpm, vocalToBackingDb, offset}``. ``buses`` (tests) receives the
    pre-master ``vocals`` and ``backing`` buses."""
    progress = progress or (lambda _p, _s: None)
    log = log or (lambda _m: None)
    j = job if isinstance(job, MixJob) else parse_job(job)
    if j.processor == spec.BUS_ID:
        return mix_bus(j, progress, log)
    s = spec.resolve(j.preset, j.params, j.seed)
    sr = SR
    log(f"preset {s.preset}, seed {s.seed}, chain {s.vocal_chain}, reverb {s.reverb:.2f}, width {s.width:.2f}, "
        f"brightness {s.brightness:+.2f}, vocal level {s.vocal_level:+.1f} LU, target {s.target_lufs:.1f} LUFS, "
        f"match {s.match:.2f}{' +loudness' if s.match_loudness else ''}, unmask {s.unmask:.2f}, "
        f"gate {'on' if s.gate else 'off'}, throws {'on' if s.throws else 'off'}, doubler {s.doubler:.2f}, "
        f"warmth {s.warmth:.2f}")

    # ---- inputs on one timeline ------------------------------------------------------------
    progress(0.0, "mix.analyze")
    for i in j.inputs:                                   # fail fast before any heavy work
        read_check(i.path)
    active = j.active()
    leads = [i for i in active if i.role == "lead"]
    harmonies = [i for i in active if i.role == "harmony"]
    if not leads:
        leads, harmonies = harmonies, []
    beds = [i for i in active if i.role in ("backing", "other")]
    raw: dict[int, np.ndarray] = {}
    for i in beds:
        raw[id(i)] = to_stereo(read(i.path, sr)[0])
    for i in leads + harmonies:
        raw[id(i)] = to_mono(read(i.path, sr)[0])
    placed_inputs = leads + harmonies + beds
    base, n, starts = timeline([(raw[id(i)].shape[-1], i.offset) for i in placed_inputs], sr)
    start_of = {id(i): st for i, st in zip(placed_inputs, starts)}

    back = None
    if beds:
        back = np.zeros((2, n), dtype=np.float32)
        for i in beds:
            left, right = chains.balance_gains(i.pan)
            p = place(raw.pop(id(i)), start_of[id(i)], n)
            back[0] += p[0] * np.float32(left * _gain(i.gain_db))
            back[1] += p[1] * np.float32(right * _gain(i.gain_db))
    vocal_raw = {id(i): raw.pop(id(i)) for i in leads + harmonies}     # chains run on each input's own length
    log(f"inputs: {len(leads)} lead, {len(harmonies)} harmony, {len(beds)} backing/other; {n / sr:.1f} s "
        f"from {base:.2f} s")
    if back is not None:
        bpm_job = _Background(lambda b=back: tempo.detect_bpm(b, sr))
    else:
        vsum = np.sum([place(vocal_raw[id(i)], start_of[id(i)], n) for i in leads + harmonies], axis=0)
        bpm_job = _Background(lambda v=vsum: tempo.detect_bpm(v, sr))
    ref = _load_reference(j, base, n, sr, log)
    if ref is not None and ref.backing is None and ref.vocals is not None and back is not None:
        ref.backing = back                                # the instrumental we were given
    progress(0.05, "mix.vocal")

    # ---- vocal chains, part 1 (tempo detected meanwhile) -----------------------------------
    chain = s.vocal_chain
    prepared: dict[int, chains.Prepared] = {}
    vocal_inputs = leads + harmonies
    for k, i in enumerate(vocal_inputs):
        progress(0.05 + 0.2 * k / len(vocal_inputs), "mix.vocal")
        role = "lead" if i in leads else "harmony"
        prepared[id(i)] = chains.prepare(vocal_raw.pop(id(i)), sr, s, role, chain)

    # ---- following the original --------------------------------------------------------------
    m_amount = s.match if (ref is not None and ref.vocals is not None) else 0.0
    orig_ratio = None
    tail_db = None
    if m_amount > 0:
        ref_v = ref.vocals_mono
        tail_db = match.tail_ratio_db(ref_v, sr) if s.reverb_from_preset else None
        if chain != "off":
            target_bands = match.band_levels(ref_v, sr)
            for i in leads:
                p = prepared[id(i)]
                if p.level_db is None:
                    continue
                curve = match.match_curve(match.band_levels(p.audio, sr), target_bands, m_amount)
                p.audio = match.apply_fir(p.audio, match.match_fir(curve, sr))
                log(f"reference: spectral match ±{float(np.max(np.abs(curve))):.1f} dB "
                    f"(lows {curve[:4].mean():+.1f}, mids {curve[8:14].mean():+.1f}, highs {curve[-5:].mean():+.1f})")
        if ref.backing is not None:
            ref_vk, ref_bk = loudness.k_weight(ref.vocals, sr), loudness.k_weight(ref.backing, sr)
            lv, lb = loudness.integrated_weighted(ref_vk, sr), loudness.integrated_weighted(ref_bk, sr)
            orig_ratio = lv - lb if math.isfinite(lv) and math.isfinite(lb) else None
            if orig_ratio is not None:
                log(f"reference: original vocal-to-backing {orig_ratio:+.1f} LU")
    bands = None
    if back is not None:
        back = filters.apply(filters.peaking(s.pocket_hz, s.pocket_db, s.pocket_q, sr), back)
        if s.unmask > 0.001:
            bands = unmasking.split(back, sr)          # voice-independent part of the dynamic EQ
    bpm = bpm_job.result()
    t = spec.timings(s, bpm)
    if tail_db is not None:
        est = match.reverb_for_tail(tail_db, t.decay_s)
        before = s.reverb
        s.reverb = (1.0 - m_amount) * s.reverb + m_amount * est
        log(f"reference: phrase-end tails {tail_db:.1f} dB → reverb {est:.2f}; {before:.2f} → {s.reverb:.2f}")
    log(f"tempo: {'%.1f BPM' % bpm if bpm else 'not detected'} → timing {t.bpm:.1f} BPM; leveler release "
        f"{t.comp_release_s * 1000:.0f} ms, pre-delay {t.predelay_s * 1000:.0f} ms, decay {t.decay_s:.2f} s")

    # ---- vocal chains, part 2 ------------------------------------------------------------------
    voice: dict[int, np.ndarray] = {}
    for k, i in enumerate(vocal_inputs):
        progress(0.25 + 0.2 * k / len(vocal_inputs), "mix.vocal")
        voice[id(i)] = place(chains.finish(prepared.pop(id(i)), sr, s, t, log, label=f"input {i.index + 1}"),
                             start_of[id(i)], n)
    lead_sum = np.sum([voice[id(i)] for i in leads], axis=0).astype(np.float32)
    if m_amount > 0 and ref.backing is not None and back is not None:
        ride_db, info = match.ride(lead_sum, back, ref_vk, ref_bk, sr, m_amount)
        del ref_vk, ref_bk
        if info.get("frames"):
            g = np.power(np.float32(10.0), ride_db / np.float32(20.0))
            for i in leads:
                voice[id(i)] = (voice[id(i)] * g).astype(np.float32)
            lead_sum = (lead_sum * g).astype(np.float32)
            log(f"reference: level riding on {info['frames']} windows, up to ±{info['maxDb']:.1f} dB")
    act = activity.detect(lead_sum, sr)

    # ---- vocal bus: pan, doubles, warmth --------------------------------------------------------
    dry = np.zeros((2, n), dtype=np.float32)
    send = np.zeros(n, dtype=np.float32)
    mono = np.zeros(n, dtype=np.float32) if s.warmth > 0.01 else None
    gains = {id(i): i.gain_db + (s.harmony_level_db if i in harmonies else 0.0) for i in vocal_inputs}
    unity = {id(i): (s.harmony_level_db if i in harmonies else 0.0) for i in vocal_inputs}
    same_offset = len({gains[id(i)] - unity[id(i)] for i in vocal_inputs}) == 1
    dry_ref = None if same_offset else np.zeros((2, n), dtype=np.float32)
    doubles = 0
    for i in vocal_inputs:
        y = voice.pop(id(i))
        left, right = pan_gains(i.pan)
        g = _gain(gains[id(i)])
        layer = np.stack([y * np.float32(left), y * np.float32(right)])
        if i in harmonies and s.doubler > 0.01:
            layer += doubling.voices(y, sr, seed=s.seed + i.index) * np.float32(0.5 * s.doubler)
            doubles += 1
        dry += layer * np.float32(g)
        if dry_ref is not None:
            dry_ref += layer * np.float32(_gain(unity[id(i)]))
        send += y * np.float32(g * (_gain(s.harmony_send_db) if i in harmonies else 1.0))
        if mono is not None:
            mono += y * np.float32(g)
    if s.doubler > 0.5:
        amount = (s.doubler - 0.5) * 2.0
        env = _loud_sections(lead_sum, sr, n)
        if env.any():
            lead_doubles = doubling.voices(lead_sum, sr, seed=s.seed, where=env > 0.01)
            lead_doubles *= env * np.float32(0.35 * amount)
            dry += lead_doubles
            if dry_ref is not None:
                dry_ref += lead_doubles
            log(f"doubler: lead doubled in its loudest sections ({float(np.mean(env > 0.5)) * 100:.0f}% of the song)")
    if doubles:
        log(f"doubler: {doubles} harmony input(s) doubled at {20 * math.log10(0.5 * s.doubler):.1f} dB")
    if s.warmth > 0.01:
        # saturating the mono vocal sum and adding the excess in the centre equals saturating the
        # bus for a centred lead and costs one mono pass however many vocals there are
        excess = _warmth(mono, sr, s.warmth) * np.float32(math.sqrt(0.5))
        dry += excess
        if dry_ref is not None:
            dry_ref += excess
        del excess, mono
    l_dry = loudness.integrated_loudness(dry, sr)
    if dry_ref is None:
        offset = gains[id(vocal_inputs[0])] - unity[id(vocal_inputs[0])]
        l_ref = l_dry - offset
    else:
        l_ref = loudness.integrated_loudness(dry_ref, sr)
        del dry_ref

    # ---- space: reverb (+ constant delay when there are no throws), throws ----------------------
    progress(0.45, "mix.reverb")
    vocal_bus = dry
    if s.reverb > 0.001 and math.isfinite(l_dry):
        space = reverb.SpaceSpec(
            weights=s.weights, decay_s=t.decay_s, predelay_s=t.predelay_s, width=s.width,
            return_highpass_hz=s.return_highpass_hz,
            delay=reverb.DelaySpec(time_s=t.delay_s, feedback=s.delay.feedback, level_db=s.delay.level_db)
            if s.delay and t.delay_s and not s.throws else None,
        )
        ir = reverb.space_ir(space, sr)
        progress(0.5, "mix.reverb")
        wet = reverb.convolve_modulated(send, ir, n, sr=sr, cutoff_hz=space.return_lowpass_hz, modulation=s.modulation)
        l_wet = loudness.integrated_loudness(wet, sr)
        if math.isfinite(l_wet):
            wet_db = l_dry + 20.0 * math.log10(s.reverb) + REVERB_REF_LU - l_wet
            wet *= np.float32(_gain(wet_db))
            vocal_bus = dry + wet
            log(f"space: {', '.join(f'{k} {w:.2f}' for k, w in s.weights.items())}; "
                f"{ir.shape[-1] / sr:.2f} s IR, wet {l_wet + wet_db - l_dry:+.1f} LU vs dry")
        del wet
    del send
    if s.throws:
        lead_gained = lead_sum * np.float32(_gain(np.mean([gains[id(i)] for i in leads])))
        th, count = make_throws(lead_gained, sr, act, t.throw_s, level_db=s.throw_level_db)
        if count:
            vocal_bus = vocal_bus + th
            log(f"throws: {count} phrase ends, {t.throw_s * 1000:.0f} ms echoes")
    progress(0.62, "mix.master")

    # ---- backing: pocket, unmasking, level ---------------------------------------------------
    report_ratio = None
    if back is not None:
        if bands is not None and act.active.any():
            density = float(np.clip((act.active.mean() - 0.3) / 0.5, 0.0, 1.0))
            back, info = unmasking.unmask(back, act.key(sr, n), sr, s.unmask, density, bands)
            del bands
            log(f"unmask: up to -{info['maxDb']:.1f} dB at 1–4 kHz, -{info['lowDb']:.1f} dB at 200–400 Hz while singing")
        l_back = loudness.integrated_loudness(back, sr)
        weight = float(np.clip(2.0 * m_amount, 0.0, 1.0))
        l_bus = loudness.integrated_loudness(vocal_bus, sr)
        if orig_ratio is not None and weight > 0 and math.isfinite(l_back):
            target_ratio = s.vocal_level + weight * (orig_ratio - 1.0)
            back_db = (l_bus - target_ratio - l_back if math.isfinite(l_bus) else 0.0) + s.backing_gain
            log(f"backing: {loudness.report_lufs(l_back):.1f} LUFS, gain {back_db:+.1f} dB "
                f"(following the original: vocals {target_ratio:+.1f} LU above)")
        else:
            matched = (l_ref - s.vocal_level - l_back) if math.isfinite(l_ref) and math.isfinite(l_back) else 0.0
            back_db = matched + s.backing_gain
            log(f"backing: {loudness.report_lufs(l_back):.1f} LUFS, gain {back_db:+.1f} dB "
                f"(vocals {loudness.report_lufs(l_ref):.1f} LUFS, {s.vocal_level:+.1f} LU above)")
        back *= np.float32(_gain(back_db))
        if math.isfinite(l_bus) and math.isfinite(l_back):
            report_ratio = round(l_bus - (l_back + back_db), 2)       # loudness scales exactly with gain
    if buses is not None:
        buses["vocals"] = vocal_bus.copy()
        buses["backing"] = None if back is None else back.copy()
    mixbus = vocal_bus if back is None else vocal_bus + back
    del vocal_bus, back
    progress(0.7, "mix.master")

    # ---- master ----------------------------------------------------------------------------------
    target = s.target_lufs
    if s.match_loudness and ref is not None and ref.mix is not None:
        l_orig = loudness.integrated_loudness(ref.mix, sr)
        if math.isfinite(l_orig):
            target = float(np.clip(l_orig, *MATCH_LOUDNESS_RANGE))
            log(f"reference: master loudness follows the original, {target:.1f} LUFS")
    ms = MasterSpec(target_lufs=target, ceiling_db=s.ceiling_db, limiter_release_s=t.limiter_release_s,
                    glue_ratio=s.glue_ratio, glue_attack_s=s.glue_attack_s, glue_release_s=t.glue_release_s,
                    glue_knee_db=s.glue_knee_db, clip_db=s.clip_db, clip_softness=s.clip_softness)
    r = master(mixbus, sr, ms, log, lambda p: progress(0.7 + 0.25 * p, "mix.master"))
    del mixbus
    y = r.audio
    fade = min(y.shape[-1], int(END_FADE_S * sr))
    if fade > 1:
        y[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    progress(0.96, "mix.master")
    write(j.output, y, sr)
    progress(1.0, "mix.master")
    return {"lufs": loudness.report_lufs(r.lufs), "peakDb": round(r.peak_db, 2),
            "truePeakDb": round(r.true_peak_db, 2), "bpm": round(bpm, 2) if bpm else None,
            "vocalToBackingDb": report_ratio, "offset": round(base, 6)}

