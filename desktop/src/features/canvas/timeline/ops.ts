// Pure arrangement edits: every function takes a timeline and returns a new one (or the same
// object when nothing changed), so a gesture can preview edits and commit the last one.

import type { Timeline, TimelineClip, Track } from '../../../api/types'

/** shortest clip a trim or split leaves (seconds) */
export const MIN_CLIP = 0.05
/** a fader at or below this is off (−∞) */
export const SILENT_DB = -60
export const MAX_DB = 6
export const CLIP_GAIN_MIN = -36
export const CLIP_GAIN_MAX = 12
/** ramp where a clip starts or ends in the middle of its audio (the mixdown uses the same) */
export const DECLICK = 0.002
/** clips this close that continue the same audio play as one (a split) */
const JOIN = 0.0005

export const round3 = (s: number) => Math.round(s * 1000) / 1000
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))
export const clipEnd = (c: TimelineClip) => c.start + c.duration
export const dbToGain = (db: number) => (db <= SILENT_DB ? 0 : Math.pow(10, db / 20))
export const gainToDb = (g: number) => (g <= 0 ? -Infinity : 20 * Math.log10(g))

export type NewId = () => string

export interface Located {
  clip: TimelineClip
  track: Track
  ti: number
}

/** every clip with its track, in track order */
export function allClips(tl: Timeline): Located[] {
  return tl.tracks.flatMap((track, ti) => track.clips.map((clip) => ({ clip, track, ti })))
}

export function findClip(tl: Timeline, id: string): Located | undefined {
  for (let ti = 0; ti < tl.tracks.length; ti++) {
    const clip = tl.tracks[ti].clips.find((c) => c.id === id)
    if (clip) return { clip, track: tl.tracks[ti], ti }
  }
  return undefined
}

/** The timeline with its `length` (end of the last clip) brought up to date. */
export function withLength(tl: Timeline): Timeline {
  let end = 0
  for (const tr of tl.tracks) for (const c of tr.clips) end = Math.max(end, clipEnd(c))
  end = round3(end)
  return end === tl.length ? tl : { ...tl, length: end }
}

/** Fades that fit the clip: each ≥ 0 and together no longer than it. */
export function fitFades(c: TimelineClip): TimelineClip {
  const fadeIn = round3(clamp(c.fadeIn, 0, c.duration))
  const fadeOut = round3(clamp(c.fadeOut, 0, c.duration - fadeIn))
  return fadeIn === c.fadeIn && fadeOut === c.fadeOut ? c : { ...c, fadeIn, fadeOut }
}

/** Replace the clips in `ids` with `fn(clip)`; tracks without them keep their identity. */
export function mapClips(tl: Timeline, ids: Iterable<string>, fn: (c: TimelineClip) => TimelineClip): Timeline {
  const set = new Set(ids)
  let changed = false
  const tracks = tl.tracks.map((tr) => {
    if (!tr.clips.some((c) => set.has(c.id))) return tr
    const clips = tr.clips.map((c) => {
      if (!set.has(c.id)) return c
      const next = fn(c)
      if (next !== c) changed = true
      return next
    })
    return { ...tr, clips }
  })
  return changed ? withLength({ ...tl, tracks }) : tl
}

export function patchClips(tl: Timeline, ids: Iterable<string>, patch: Partial<TimelineClip>): Timeline {
  return mapClips(tl, ids, (c) => fitFades({ ...c, ...patch }))
}

export function removeClips(tl: Timeline, ids: Iterable<string>): Timeline {
  const set = new Set(ids)
  if (!allClips(tl).some((x) => set.has(x.clip.id))) return tl
  return withLength({ ...tl, tracks: tl.tracks.map((tr) => (tr.clips.some((c) => set.has(c.id)) ? { ...tr, clips: tr.clips.filter((c) => !set.has(c.id)) } : tr)) })
}

/**
 * Move clips `dt` seconds and `dTrack` tracks (both kept in range for the whole group), or with
 * `copy` (old id → new id) place copies there and leave the originals. Returns the moved (or new) clips' ids.
 */
export function moveClips(tl: Timeline, ids: string[], dt: number, dTrack: number, copy?: (id: string) => string): { tl: Timeline; ids: string[] } {
  const set = new Set(ids)
  const moving = allClips(tl).filter((x) => set.has(x.clip.id))
  if (!moving.length) return { tl, ids: [] }
  const shift = Math.max(dt, -Math.min(...moving.map((x) => x.clip.start)))
  const lane = clamp(dTrack, -Math.min(...moving.map((x) => x.ti)), tl.tracks.length - 1 - Math.max(...moving.map((x) => x.ti)))
  if (!copy && shift === 0 && lane === 0) return { tl, ids: moving.map((x) => x.clip.id) }
  const added = new Map<number, TimelineClip[]>()
  const out: string[] = []
  for (const { clip, ti } of moving) {
    const id = copy ? copy(clip.id) : clip.id
    out.push(id)
    const list = added.get(ti + lane) ?? []
    list.push({ ...clip, id, start: round3(clip.start + shift) })
    added.set(ti + lane, list)
  }
  const tracks = tl.tracks.map((tr, ti) => {
    const kept = copy ? tr.clips : tr.clips.filter((c) => !set.has(c.id))
    const extra = added.get(ti)
    if (!extra && kept.length === tr.clips.length) return tr
    return { ...tr, clips: extra ? [...kept, ...extra] : kept }
  })
  return { tl: withLength({ ...tl, tracks }), ids: out }
}

/**
 * Trim the start (`left`) or end of clips by `dt` seconds: the start keeps the audio where it
 * is on the timeline; neither edge goes past the audio's ends or makes a clip shorter than MIN_CLIP.
 */
export function trimClips(tl: Timeline, ids: string[], edge: 'left' | 'right', dt: number, mediaDuration: (mediaId: string) => number | undefined): Timeline {
  return mapClips(tl, ids, (c) => {
    if (edge === 'right') {
      const total = mediaDuration(c.mediaId) ?? c.offset + c.duration
      const duration = round3(clamp(c.duration + dt, Math.min(MIN_CLIP, c.duration), Math.max(c.duration, total - c.offset)))
      return duration === c.duration ? c : fitFades({ ...c, duration })
    }
    const shift = clamp(dt, Math.max(-c.offset, -c.start), Math.max(0, c.duration - MIN_CLIP))
    if (shift === 0) return c
    return fitFades({ ...c, start: round3(c.start + shift), offset: round3(c.offset + shift), duration: round3(c.duration - shift) })
  })
}

/** Clips that `t` falls inside (not on an edge), on one track or all. */
export function clipsAt(tl: Timeline, t: number, trackId?: string | null): string[] {
  return allClips(tl)
    .filter((x) => (!trackId || x.track.id === trackId) && x.clip.start + MIN_CLIP / 2 < t && t < clipEnd(x.clip) - MIN_CLIP / 2)
    .map((x) => x.clip.id)
}

/**
 * Split clips at `t`: the left part keeps the id and its fade-in, the right part gets a new id
 * and the fade-out; nothing fades at the cut, so the two still play as one. Returns both parts' ids.
 */
export function splitClips(tl: Timeline, ids: string[], t: number, newId: NewId): { tl: Timeline; ids: string[] } {
  const set = new Set(clipsAt(tl, t).filter((id) => ids.includes(id)))
  if (!set.size) return { tl, ids }
  const out: string[] = []
  const tracks = tl.tracks.map((tr) => {
    if (!tr.clips.some((c) => set.has(c.id))) return tr
    const clips = tr.clips.flatMap((c) => {
      if (!set.has(c.id)) return [c]
      const cut = round3(t - c.start)
      const left = fitFades({ ...c, duration: cut, fadeOut: 0 })
      const right = fitFades({ ...c, id: newId(), start: round3(t), offset: round3(c.offset + cut), duration: round3(c.duration - cut), fadeIn: 0 })
      out.push(left.id, right.id)
      return [left, right]
    })
    return { ...tr, clips }
  })
  const untouched = ids.filter((id) => !set.has(id))
  return { tl: withLength({ ...tl, tracks }), ids: [...untouched, ...out] }
}

/** Copies of the clips as one block right after it (Ctrl/Cmd+D). Returns the copies' ids. */
export function duplicateClips(tl: Timeline, ids: string[], newId: NewId): { tl: Timeline; ids: string[] } {
  const set = new Set(ids)
  const sel = allClips(tl).filter((x) => set.has(x.clip.id))
  if (!sel.length) return { tl, ids: [] }
  const span = Math.max(...sel.map((x) => clipEnd(x.clip))) - Math.min(...sel.map((x) => x.clip.start))
  return moveClips(tl, ids, round3(span), 0, newId)
}

export interface ClipBoard {
  projectId: string
  /** clips with their start relative to the earliest and their track relative to the topmost */
  items: { track: number; clip: TimelineClip }[]
}

export function copyClips(tl: Timeline, ids: string[], projectId: string): ClipBoard | null {
  const set = new Set(ids)
  const sel = allClips(tl).filter((x) => set.has(x.clip.id))
  if (!sel.length) return null
  const t0 = Math.min(...sel.map((x) => x.clip.start))
  const top = Math.min(...sel.map((x) => x.ti))
  return { projectId, items: sel.map((x) => ({ track: x.ti - top, clip: { ...x.clip, start: round3(x.clip.start - t0) } })) }
}

/** Paste at `at` seconds from track `trackIndex` down; tracks that don't exist yet are made with `makeTrack`. */
export function pasteClips(tl: Timeline, board: ClipBoard, at: number, trackIndex: number, newId: NewId, makeTrack: () => Track): { tl: Timeline; ids: string[] } {
  if (!board.items.length) return { tl, ids: [] }
  const tracks = tl.tracks.slice()
  const ids: string[] = []
  const first = clamp(trackIndex, 0, tracks.length)
  for (const { track, clip } of board.items) {
    const ti = first + track
    while (tracks.length <= ti) tracks.push(makeTrack())
    const id = newId()
    ids.push(id)
    tracks[ti] = { ...tracks[ti], clips: [...tracks[ti].clips, { ...clip, id, start: round3(Math.max(0, at) + clip.start) }] }
  }
  return { tl: withLength({ ...tl, tracks }), ids }
}

/** Put one clip on track `trackIndex` (a new track from `makeTrack` past the last one). */
/**
 * Put a new clip (a drop, an import) on track `trackIndex`. When something already plays there
 * at that time it goes to the next track that is free then, or to a new track right below, so
 * two sounds never pile up on one track by accident (moving clips may still overlap on purpose).
 */
export function placeClip(tl: Timeline, trackIndex: number, clip: TimelineClip, makeTrack: () => Track): Timeline {
  const tracks = tl.tracks.slice()
  const end = clip.start + clip.duration
  const free = (tr: Track) => tr.clips.every((c) => c.start + c.duration <= clip.start + 1e-3 || c.start >= end - 1e-3)
  let ti = clamp(trackIndex, 0, tracks.length)
  if (ti < tracks.length && !free(tracks[ti])) {
    const next = tracks.findIndex((tr, i) => i > ti && free(tr))
    if (next !== -1) ti = next
    else {
      tracks.splice(ti + 1, 0, makeTrack())
      ti += 1
    }
  }
  if (ti === tracks.length) tracks.push(makeTrack())
  tracks[ti] = { ...tracks[ti], clips: [...tracks[ti].clips, fitFades(clip)] }
  return withLength({ ...tl, tracks })
}

export function setFade(tl: Timeline, id: string, which: 'in' | 'out', seconds: number): Timeline {
  return mapClips(tl, [id], (c) => {
    const max = c.duration - (which === 'in' ? c.fadeOut : c.fadeIn)
    const v = round3(clamp(seconds, 0, max))
    if (which === 'in') return v === c.fadeIn ? c : { ...c, fadeIn: v }
    return v === c.fadeOut ? c : { ...c, fadeOut: v }
  })
}

export function setClipGain(tl: Timeline, ids: string[], db: number): Timeline {
  const v = Math.round(clamp(db, CLIP_GAIN_MIN, CLIP_GAIN_MAX) * 10) / 10
  return mapClips(tl, ids, (c) => (c.gainDb === v ? c : { ...c, gainDb: v }))
}

/** Change each clip's gain by `delta` dB. */
export function nudgeClipGain(tl: Timeline, ids: string[], delta: number): Timeline {
  return mapClips(tl, ids, (c) => {
    const v = Math.round(clamp(c.gainDb + delta, CLIP_GAIN_MIN, CLIP_GAIN_MAX) * 10) / 10
    return v === c.gainDb ? c : { ...c, gainDb: v }
  })
}

// ---------- tracks ----------

export function patchTrack(tl: Timeline, id: string, patch: Partial<Track>): Timeline {
  let changed = false
  const tracks = tl.tracks.map((tr) => {
    if (tr.id !== id) return tr
    const next = { ...tr, ...patch }
    changed = (Object.keys(patch) as (keyof Track)[]).some((k) => next[k] !== tr[k])
    return next
  })
  return changed ? { ...tl, tracks } : tl
}

export function insertTrack(tl: Timeline, track: Track, index = tl.tracks.length): Timeline {
  const tracks = tl.tracks.slice()
  tracks.splice(clamp(index, 0, tracks.length), 0, track)
  return { ...tl, tracks }
}

export function removeTrack(tl: Timeline, id: string): Timeline {
  if (!tl.tracks.some((tr) => tr.id === id)) return tl
  return withLength({ ...tl, tracks: tl.tracks.filter((tr) => tr.id !== id) })
}

export function moveTrack(tl: Timeline, id: string, to: number): Timeline {
  const from = tl.tracks.findIndex((tr) => tr.id === id)
  const dest = clamp(to, 0, tl.tracks.length - 1)
  if (from === -1 || from === dest) return tl
  const tracks = tl.tracks.slice()
  const [tr] = tracks.splice(from, 1)
  tracks.splice(dest, 0, tr)
  return { ...tl, tracks }
}

export function duplicateTrack(tl: Timeline, id: string, newId: (kind: 'track' | 'clip') => string, name: string): { tl: Timeline; id: string | null } {
  const i = tl.tracks.findIndex((tr) => tr.id === id)
  if (i === -1) return { tl, id: null }
  const src = tl.tracks[i]
  const copy: Track = { ...src, id: newId('track'), name, solo: false, clips: src.clips.map((c) => ({ ...c, id: newId('clip') })) }
  return { tl: insertTrack(tl, copy, i + 1), id: copy.id }
}

export function audibleTracks(tracks: Track[]): Set<string> {
  const soloed = tracks.some((tr) => tr.solo)
  return new Set(tracks.filter((tr) => !tr.mute && (!soloed || tr.solo)).map((tr) => tr.id))
}

// ---------- loop ----------

export function loopOn(tl: Timeline): { start: number; end: number } | null {
  const l = tl.loop
  return l && l.on !== false && l.end - l.start >= MIN_CLIP ? { start: l.start, end: l.end } : null
}

// ---------- playback envelope ----------

/** Ids of clips whose start / end continues another clip back to back with the same audio. */
export function joins(clips: TimelineClip[]): { starts: Set<string>; ends: Set<string> } {
  const starts = new Set<string>()
  const ends = new Set<string>()
  for (const a of clips) {
    if (a.mute) continue
    for (const b of clips) {
      if (a === b || b.mute || a.mediaId !== b.mediaId) continue
      if (Math.abs(clipEnd(a) - b.start) < JOIN && Math.abs(a.offset + a.duration - b.offset) < JOIN) {
        ends.add(a.id)
        starts.add(b.id)
      }
    }
  }
  return { starts, ends }
}

/**
 * Gain breakpoints (clip seconds, 0…1) of a clip played from `s0` to `s1` seconds into it:
 * linear fades (at least `rampIn` / `rampOut`, the declick), times `cutIn` / `cutOut` ramps
 * where playback itself starts or stops inside the clip. Starting inside a fade continues it.
 */
export function envelope(c: TimelineClip, s0: number, s1: number, rampIn: number, rampOut: number, cutIn = 0, cutOut = 0): { t: number; g: number }[] {
  const d = c.duration
  const fi = Math.max(c.fadeIn, rampIn)
  const fo = Math.max(c.fadeOut, rampOut)
  const gain = (t: number) =>
    (fi > 0 ? clamp(t / fi, 0, 1) : 1) * (fo > 0 ? clamp((d - t) / fo, 0, 1) : 1) * (cutIn > 0 ? clamp((t - s0) / cutIn, 0, 1) : 1) * (cutOut > 0 ? clamp((s1 - t) / cutOut, 0, 1) : 1)
  const ts = [s0, s1, fi, d - fo, s0 + cutIn, s1 - cutOut].filter((t) => t >= s0 && t <= s1)
  return Array.from(new Set(ts.map((t) => Math.round(t * 1e6) / 1e6)))
    .sort((a, b) => a - b)
    .map((t) => ({ t, g: gain(t) }))
}

// ---------- snapping and the grid ----------

export type GridMode = 'sec' | 'bars'

export interface Grid {
  /** snap step and fine lines (seconds) */
  minor: number
  /** labelled ticks (seconds) */
  major: number
  /** bars mode: seconds per beat and per bar */
  beat?: number
  bar?: number
}

const SEC_STEPS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600]

/** Grid for a zoom (pixels per second): seconds, or bars and beats (4/4) at `bpm`. */
export function gridFor(pps: number, mode: GridMode, bpm?: number | null): Grid {
  const pick = (steps: number[], px: number) => steps.find((s) => s * pps >= px) ?? steps[steps.length - 1]
  if (mode === 'bars' && bpm && bpm > 20) {
    const beat = 60 / bpm
    const bar = beat * 4
    const steps = [beat / 4, beat / 2, beat, bar, bar * 2, bar * 4, bar * 8, bar * 16, bar * 32, bar * 64]
    return { minor: pick(steps, 12), major: pick(steps.slice(2), 56), beat, bar }
  }
  return { minor: pick(SEC_STEPS, 12), major: pick(SEC_STEPS, 64) }
}

/**
 * Snap `t` to the nearest of `targets` (clip edges, the playhead, loop bounds) within `tol`
 * seconds, else to the nearest grid line within `tol`. `at` is where it snapped (for the guide).
 */
export function snapTime(t: number, targets: number[], grid: number | null, tol: number): { t: number; at: number | null } {
  let best: number | null = null
  for (const x of targets) if (Math.abs(x - t) <= tol && (best === null || Math.abs(x - t) < Math.abs(best - t))) best = x
  if (best === null && grid) {
    const g = Math.round(t / grid) * grid
    if (Math.abs(g - t) <= tol) best = round3(g)
  }
  return best === null ? { t, at: null } : { t: best, at: best }
}

/** The edges of every clip not in `exclude` (to snap to). */
export function clipEdges(tl: Timeline, exclude: Set<string>): number[] {
  const out: number[] = []
  for (const tr of tl.tracks) for (const c of tr.clips) if (!exclude.has(c.id)) out.push(c.start, clipEnd(c))
  return out
}

// ---------- faders ----------

/** dB ↔ fader travel (0…1): unity at 0.8, −∞ at the bottom, +6 dB at the top */
const FADER: [number, number][] = [
  [SILENT_DB, 0],
  [-48, 0.06],
  [-36, 0.14],
  [-24, 0.28],
  [-18, 0.38],
  [-12, 0.5],
  [-6, 0.64],
  [0, 0.8],
  [MAX_DB, 1],
]

export function faderPos(db: number): number {
  if (db <= SILENT_DB) return 0
  if (db >= MAX_DB) return 1
  for (let i = 1; i < FADER.length; i++) {
    const [d1, p1] = FADER[i]
    if (db <= d1) {
      const [d0, p0] = FADER[i - 1]
      return p0 + ((db - d0) / (d1 - d0)) * (p1 - p0)
    }
  }
  return 1
}

export function faderDb(pos: number): number {
  if (pos <= 0) return SILENT_DB
  if (pos >= 1) return MAX_DB
  for (let i = 1; i < FADER.length; i++) {
    const [d1, p1] = FADER[i]
    if (pos <= p1) {
      const [d0, p0] = FADER[i - 1]
      return Math.round((d0 + ((pos - p0) / (p1 - p0)) * (d1 - d0)) * 10) / 10
    }
  }
  return MAX_DB
}

/** "+1.5", "0.0", "−3.0", "−∞" */
export function fmtDb(db: number): string {
  if (db <= SILENT_DB) return '−∞'
  const v = Math.round(db * 10) / 10
  return `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}`
}

/** "-3.5", "−3,5", "-inf", "+2" → dB (null when it isn't a number) */
export function parseDb(text: string): number | null {
  const s = text.trim().replace(/[−–]/g, '-').replace(',', '.').replace(/db$/i, '').trim()
  if (/^-?(inf|∞)/i.test(s)) return SILENT_DB
  const v = Number(s)
  return Number.isFinite(v) && s !== '' ? clamp(v, SILENT_DB, MAX_DB) : null
}
