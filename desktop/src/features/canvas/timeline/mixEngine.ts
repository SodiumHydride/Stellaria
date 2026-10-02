import type { Timeline, Track } from '../../../api/types'
import { audibleTracks, clipEnd, dbToGain, DECLICK, envelope, joins, loopOn } from './ops'

/** ramp where playback itself starts, stops or wraps inside a clip (seek, loop, an edit) */
const CUT = 0.004
/** how far ahead loop passes are scheduled (seconds) */
const AHEAD = 1.5
/** gain changes glide this fast (time constant, seconds): quick, but never a click */
const GLIDE = 0.012

export interface Level {
  peak: [number, number]
  rms: [number, number]
}

interface Meter {
  analysers: [AnalyserNode, AnalyserNode]
  buf: Float32Array<ArrayBuffer>
}

/** a track's channel: volume → balance (L/R gains) → mute/solo gate → master; metered after the gate */
interface Chain {
  input: GainNode
  left: GainNode
  right: GainNode
  gate: GainNode
  out: AudioNode
  meter: Meter
}

interface Voice {
  src: AudioBufferSourceNode
  env: GainNode
  ends: number
}

/** from context time `at`, timeline second `from` plays, up to `to` */
interface Pass {
  at: number
  from: number
  to: number
}

/**
 * Plays the arrangement with WebAudio. The mixer graph (one channel per track + the master) is
 * persistent, so volume / pan / mute / solo / master move while playing without a restart; clips
 * are scheduled as buffer sources with their fades, and a loop is scheduled pass after pass ahead
 * of time so it wraps without a gap. Decoded buffers are cached per URL.
 */
export class MixEngine {
  private ctx: AudioContext | null = null
  private buffers = new Map<string, Promise<AudioBuffer | null>>()
  private decoded = new Map<string, AudioBuffer>()
  private chains = new Map<string, Chain>()
  private master: { gain: GainNode; meter: Meter } | null = null
  private voices: Voice[] = []
  private passes: Pass[] = []
  private timeline: Timeline | null = null
  private urlFor: (mediaId: string) => string = (id) => id
  private timer = 0
  /** bumped by every play / stop: one still loading when a newer one came gives up */
  private generation = 0
  /** the same for reschedules (they never cancel a play) */
  private rescheduling = 0
  private offset = 0
  playing = false

  private context(): AudioContext {
    if (!this.ctx) {
      const ctx = new AudioContext({ latencyHint: 'playback' })
      const gain = ctx.createGain()
      gain.connect(ctx.destination)
      this.ctx = ctx
      this.master = { gain, meter: this.meter(gain) }
      if (this.timeline) this.setMix(this.timeline)
    }
    return this.ctx
  }

  private meter(from: AudioNode): Meter {
    const ctx = this.ctx!
    const split = ctx.createChannelSplitter(2)
    const analysers: [AnalyserNode, AnalyserNode] = [ctx.createAnalyser(), ctx.createAnalyser()]
    for (const a of analysers) a.fftSize = 2048
    from.connect(split)
    split.connect(analysers[0], 0)
    split.connect(analysers[1], 1)
    return { analysers, buf: new Float32Array(2048) }
  }

  private chain(trackId: string): Chain {
    let ch = this.chains.get(trackId)
    if (ch) return ch
    const ctx = this.context()
    // two channels in: mono clips are spread to both sides first (as the mixdown does)
    const input = new GainNode(ctx, { channelCount: 2, channelCountMode: 'explicit', channelInterpretation: 'speakers' })
    const split = ctx.createChannelSplitter(2)
    const left = ctx.createGain()
    const right = ctx.createGain()
    const merge = ctx.createChannelMerger(2)
    const gate = ctx.createGain()
    input.connect(split)
    split.connect(left, 0)
    split.connect(right, 1)
    left.connect(merge, 0, 0)
    right.connect(merge, 0, 1)
    merge.connect(gate)
    gate.connect(this.master!.gain)
    ch = { input, left, right, gate, out: gate, meter: this.meter(gate) }
    this.chains.set(trackId, ch)
    return ch
  }

  /** move a gain smoothly; a channel made just now starts at its value */
  private glide(param: AudioParam, value: number, fresh = false) {
    const now = this.ctx!.currentTime
    param.cancelScheduledValues(now)
    if (fresh) param.setValueAtTime(value, now)
    else param.setTargetAtTime(value, now, GLIDE)
  }

  /** Volume, pan, mute / solo and the master fader, applied live (no restart). */
  setMix(tl: Timeline): void {
    this.timeline = tl
    if (!this.ctx) return
    const audible = audibleTracks(tl.tracks)
    for (const tr of tl.tracks) {
      const fresh = !this.chains.has(tr.id)
      const ch = this.chain(tr.id)
      const pan = Math.max(-1, Math.min(1, tr.pan))
      this.glide(ch.input.gain, dbToGain(tr.gainDb), fresh)
      this.glide(ch.left.gain, Math.min(1, 1 - pan), fresh)
      this.glide(ch.right.gain, Math.min(1, 1 + pan), fresh)
      this.glide(ch.gate.gain, audible.has(tr.id) ? 1 : 0, fresh)
    }
    for (const [id, ch] of this.chains) {
      if (tl.tracks.some((tr) => tr.id === id)) continue
      ch.gate.disconnect()
      ch.input.disconnect()
      this.chains.delete(id)
    }
    this.glide(this.master!.gain.gain, dbToGain(tl.masterGainDb ?? 0))
  }

  private buffer(url: string): Promise<AudioBuffer | null> {
    let p = this.buffers.get(url)
    if (!p) {
      p = fetch(url)
        .then((r) => {
          if (!r.ok) throw new Error(String(r.status))
          return r.arrayBuffer()
        })
        .then((data) => this.context().decodeAudioData(data))
        .then((buf) => {
          this.decoded.set(url, buf)
          return buf
        })
        .catch(() => {
          this.buffers.delete(url)
          return null
        })
      this.buffers.set(url, p)
    }
    return p
  }

  /** Decode what the arrangement needs (all at once, before anything starts). */
  private async load(tl: Timeline): Promise<void> {
    const urls = new Set(tl.tracks.flatMap((tr) => tr.clips.filter((c) => !c.mute).map((c) => this.urlFor(c.mediaId))))
    await Promise.all([...urls].map((u) => this.buffer(u)))
  }

  /** Timeline second at context time `when` (the passes map one to the other). */
  private positionAt(when: number): number {
    if (!this.playing || !this.passes.length) return this.offset
    let p = this.passes[0]
    for (const x of this.passes) if (x.at <= when) p = x
    if (when < p.at) return p.from
    return Math.min(p.from + (when - p.at), p.to)
  }

  position(): number {
    return this.ctx ? this.positionAt(this.ctx.currentTime) : this.offset
  }

  seek(t: number): void {
    this.offset = Math.max(0, t)
  }

  async play(tl: Timeline, from: number, urlFor: (mediaId: string) => string): Promise<boolean> {
    const gen = ++this.generation
    this.urlFor = urlFor
    this.silence()
    const ctx = this.context()
    if (ctx.state === 'suspended') await ctx.resume()
    this.setMix(tl)
    await this.load(tl)
    if (gen !== this.generation) return false
    this.offset = Math.max(0, from)
    this.playing = true
    this.start(ctx.currentTime + 0.05, this.offset)
    window.clearInterval(this.timer)
    this.timer = window.setInterval(() => this.pump(), 100)
    return true
  }

  /** Clips changed while playing: carry on from where it is with the new arrangement. */
  async reschedule(tl: Timeline): Promise<void> {
    this.setMix(tl)
    if (!this.playing) return
    const gen = this.generation
    const mine = ++this.rescheduling
    await this.load(tl)
    if (gen !== this.generation || mine !== this.rescheduling || !this.playing || !this.ctx) return
    const at = this.ctx.currentTime + 0.03
    const from = this.positionAt(at)
    this.silence()
    this.start(at, from)
  }

  stop(): void {
    this.generation++
    if (this.playing) this.offset = this.position()
    this.playing = false
    this.passes = []
    window.clearInterval(this.timer)
    this.silence()
  }

  private start(at: number, from: number) {
    this.passes = []
    const loop = this.timeline ? loopOn(this.timeline) : null
    this.schedule({ at, from, to: loop && from < loop.end ? loop.end : Infinity })
    this.pump()
  }

  /** Keep the loop scheduled ahead; forget what has finished. */
  private pump() {
    const ctx = this.ctx
    const tl = this.timeline
    if (!ctx || !tl || !this.playing) return
    const loop = loopOn(tl)
    let last = this.passes[this.passes.length - 1]
    while (loop && last && last.to !== Infinity && last.at + (last.to - last.from) - ctx.currentTime < AHEAD) {
      last = { at: last.at + (last.to - last.from), from: loop.start, to: loop.end }
      this.schedule(last)
    }
    const now = ctx.currentTime
    this.voices = this.voices.filter((v) => v.ends > now - 0.5)
    while (this.passes.length > 2 && this.passes[1].at < now - 1) this.passes.shift()
  }

  private schedule(pass: Pass) {
    const ctx = this.ctx!
    const tl = this.timeline
    if (!tl) return
    this.passes.push(pass)
    for (const tr of tl.tracks) this.scheduleTrack(ctx, tr, pass)
  }

  private scheduleTrack(ctx: AudioContext, tr: Track, pass: Pass) {
    const { starts, ends } = joins(tr.clips)
    const out = this.chain(tr.id).input
    for (const c of tr.clips) {
      if (c.mute) continue
      const a = Math.max(c.start, pass.from)
      const b = Math.min(clipEnd(c), pass.to)
      if (b - a < 0.001) continue
      const buf = this.decoded.get(this.urlFor(c.mediaId))
      if (!buf) continue
      const s0 = a - c.start
      const s1 = b - c.start
      const t0 = pass.at + (a - pass.from)
      if (c.offset + s0 >= buf.duration) continue
      const src = ctx.createBufferSource()
      src.buffer = buf
      const env = ctx.createGain()
      env.gain.value = 0
      const level = dbToGain(c.gainDb)
      const points = envelope(c, s0, s1, starts.has(c.id) ? 0 : DECLICK, ends.has(c.id) ? 0 : DECLICK, s0 > 0.0005 ? CUT : 0, s1 < c.duration - 0.0005 ? CUT : 0)
      env.gain.setValueAtTime(points[0].g * level, t0)
      for (const p of points.slice(1)) env.gain.linearRampToValueAtTime(p.g * level, t0 + (p.t - s0))
      src.connect(env).connect(out)
      src.start(t0, c.offset + s0, s1 - s0)
      this.voices.push({ src, env, ends: t0 + (s1 - s0) })
    }
  }

  /** Fade out and stop everything scheduled (a few ms, no click). */
  private silence() {
    const ctx = this.ctx
    if (!ctx) return
    const now = ctx.currentTime
    for (const v of this.voices) {
      try {
        const g = v.env.gain
        if (typeof g.cancelAndHoldAtTime === 'function') g.cancelAndHoldAtTime(now)
        else {
          g.cancelScheduledValues(now)
          g.setValueAtTime(g.value, now)
        }
        g.linearRampToValueAtTime(0, now + CUT * 2)
        v.src.stop(now + CUT * 2 + 0.002)
      } catch {
        /* never started, or already stopped */
      }
    }
    this.voices = []
  }

  /** Peak and RMS of a track (its id) or of the master ('master'), post-fader; null before anything played. */
  levels(id: string): Level | null {
    const meter = id === 'master' ? this.master?.meter : this.chains.get(id)?.meter
    if (!meter) return null
    const out: Level = { peak: [0, 0], rms: [0, 0] }
    meter.analysers.forEach((a, ch) => {
      a.getFloatTimeDomainData(meter.buf)
      let peak = 0
      let sum = 0
      for (let i = 0; i < meter.buf.length; i++) {
        const v = meter.buf[i]
        const abs = v < 0 ? -v : v
        if (abs > peak) peak = abs
        sum += v * v
      }
      out.peak[ch] = peak
      out.rms[ch] = Math.sqrt(sum / meter.buf.length)
    })
    return out
  }

  dispose(): void {
    this.stop()
    void this.ctx?.close()
    this.ctx = null
    this.master = null
    this.chains.clear()
    this.buffers.clear()
    this.decoded.clear()
  }
}
