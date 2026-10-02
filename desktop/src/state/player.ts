import { create } from 'zustand'
import { useShallow } from 'zustand/react/shallow'

/**
 * One global audio player: starting a clip stops the previous one, every waveform can show
 * its own playhead, and A/B comparison switches sources while keeping the position.
 */
interface PlayerState {
  currentId: string | null
  url: string | null
  playing: boolean
  time: number
  duration: number
  /** when set, the takes being compared; `currentId` is the one that is audible */
  group: { id: string; url: string }[] | null
  toggle: (id: string, url: string, startAt?: number) => void
  play: (id: string, url: string, startAt?: number) => void
  pause: () => void
  stop: () => void
  seek: (t: number) => void
  compare: (a: { id: string; url: string }, b: { id: string; url: string }) => void
  /** compare any number of takes; switching keeps the playback position */
  compareGroup: (items: { id: string; url: string }[], startIndex?: number) => void
  switchTo: (index: number) => void
  flip: () => void
  /** repeat a region of the current clip (seconds); null = play through */
  loop: { start: number; end: number } | null
  setLoop: (loop: { start: number; end: number } | null) => void
}

const audio: HTMLAudioElement | null = typeof Audio !== 'undefined' ? new Audio() : null
if (audio) {
  audio.preload = 'auto'
  audio.crossOrigin = 'anonymous'
}

// Level metering: the element is routed through Web Audio once something plays (it needs a user
// gesture to start), then two analysers report each channel's peak.
let meter: { ctx: AudioContext; analysers: AnalyserNode[]; buf: Float32Array<ArrayBuffer> } | null = null
let meterFailed = false

function ensureMeter() {
  if (!audio || meter || meterFailed) {
    if (meter?.ctx.state === 'suspended') void meter.ctx.resume()
    return
  }
  try {
    const Ctx = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!Ctx) throw new Error('no Web Audio')
    const ctx = new Ctx()
    const source = ctx.createMediaElementSource(audio)
    const split = ctx.createChannelSplitter(2)
    const analysers = [ctx.createAnalyser(), ctx.createAnalyser()]
    for (const a of analysers) a.fftSize = 1024
    source.connect(ctx.destination)
    source.connect(split)
    split.connect(analysers[0], 0)
    split.connect(analysers[1], 1)
    meter = { ctx, analysers, buf: new Float32Array(1024) }
    if (ctx.state === 'suspended') void ctx.resume()
  } catch {
    meterFailed = true // playback keeps working without a meter
  }
}

/** Current peak of each channel (0..1, linear), or null when there is no meter. */
export function readLevels(): [number, number] | null {
  if (!meter) return null
  const out: number[] = []
  for (const a of meter.analysers) {
    a.getFloatTimeDomainData(meter.buf)
    let peak = 0
    for (let i = 0; i < meter.buf.length; i++) {
      const v = Math.abs(meter.buf[i])
      if (v > peak) peak = v
    }
    out.push(peak)
  }
  // a mono file leaves the second channel silent: show it on both
  if (out[1] === 0 && out[0] > 0) return [out[0], out[0]]
  return [out[0], out[1]]
}

let raf = 0
function tick() {
  if (!audio) return
  const loop = usePlayer.getState().loop
  if (loop && loop.end - loop.start > 0.05 && (audio.currentTime >= loop.end || audio.currentTime < loop.start - 0.25)) audio.currentTime = loop.start
  usePlayer.setState({ time: audio.currentTime, duration: isFinite(audio.duration) ? audio.duration : 0 })
  raf = requestAnimationFrame(tick)
}

function startTicking() {
  cancelAnimationFrame(raf)
  raf = requestAnimationFrame(tick)
}

function stopTicking() {
  cancelAnimationFrame(raf)
}

export const usePlayer = create<PlayerState>((set, get) => ({
  currentId: null,
  url: null,
  playing: false,
  time: 0,
  duration: 0,
  group: null,
  loop: null,
  setLoop: (loop) => set({ loop }),

  play: (id, url, startAt) => {
    if (!audio) return
    if (get().url !== url) {
      audio.src = url
      set({ url, time: 0, duration: 0, loop: null })
    }
    if (startAt !== undefined) audio.currentTime = startAt
    set({ currentId: id, playing: true })
    if (get().group && !get().group!.some((x) => x.id === id)) set({ group: null })
    ensureMeter()
    audio.play().catch(() => set({ playing: false }))
    startTicking()
  },

  toggle: (id, url, startAt) => {
    const s = get()
    if (s.currentId === id && s.playing) s.pause()
    else s.play(id, url, s.currentId === id ? startAt : (startAt ?? 0))
  },

  pause: () => {
    audio?.pause()
    set({ playing: false })
    stopTicking()
  },

  stop: () => {
    if (audio) {
      audio.pause()
      audio.currentTime = 0
    }
    set({ playing: false, currentId: null, time: 0, group: null })
    stopTicking()
  },

  seek: (t) => {
    if (!audio) return
    audio.currentTime = Math.max(0, t)
    set({ time: audio.currentTime })
  },

  compare: (a, b) => get().compareGroup([a, b]),

  compareGroup: (items, startIndex = 0) => {
    if (items.length === 0) return
    const s = get()
    const keepTime = s.group && s.currentId && items.some((x) => x.id === s.currentId) ? audio?.currentTime : 0
    set({ group: items })
    const first = items[Math.min(startIndex, items.length - 1)]
    if (s.currentId && items.some((x) => x.id === s.currentId)) get().switchTo(items.findIndex((x) => x.id === first.id))
    else get().play(first.id, first.url, keepTime ?? 0)
  },

  flip: () => {
    const s = get()
    if (!s.group) return
    const i = s.group.findIndex((x) => x.id === s.currentId)
    s.switchTo((i + 1) % s.group.length)
  },

  switchTo: (index) => {
    const s = get()
    if (!s.group || !audio) return
    const next = s.group[index]
    if (!next || next.id === s.currentId) return
    if (!s.currentId) {
      s.play(next.id, next.url, 0)
      return
    }
    const t = audio.currentTime
    const wasPlaying = s.playing
    audio.src = next.url
    set({ url: next.url, currentId: next.id })
    const resume = () => {
      audio.currentTime = Math.min(t, isFinite(audio.duration) ? audio.duration : t)
      if (wasPlaying) audio.play().catch(() => set({ playing: false }))
    }
    audio.addEventListener('loadedmetadata', resume, { once: true })
  },
}))

if (audio) {
  audio.addEventListener('ended', () => {
    usePlayer.setState({ playing: false, time: 0 })
    stopTicking()
  })
  audio.addEventListener('error', () => {
    usePlayer.setState({ playing: false })
    stopTicking()
  })
}

/** Progress 0..1 of a given playable, or null when it isn't the current one. */
export function usePlayback(id: string): { playing: boolean; progress: number | null; time: number } {
  return usePlayer(
    useShallow((s) =>
      s.currentId === id
        ? { playing: s.playing, progress: s.duration > 0 ? s.time / s.duration : 0, time: s.time }
        : NOT_CURRENT,
    ),
  )
}

const NOT_CURRENT = { playing: false, progress: null, time: 0 }
