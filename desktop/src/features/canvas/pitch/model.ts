// The pitch editor's model: the input's pitch line (from the backend), the user's changes as a
// semitone offset per analysis frame, and the operations on them. Pure functions: every edit
// returns a new offset array, so undo is just keeping the previous one.

import type { PitchEdit } from '../../../api/types'

export interface PitchAnalysis {
  /** seconds per frame */
  frameS: number
  duration: number
  /** MIDI per frame (fractional), null where unvoiced */
  midi: (number | null)[]
  notes: { start: number; end: number; pitch: number }[]
  key: { tonic: number; mode: 'major' | 'minor'; confidence: number } | null
  /** still made from the node's current input and pitch settings */
  current: boolean
}

export interface Note {
  id: number
  /** frame indices, inclusive */
  first: number
  last: number
  /** the original centre (median MIDI) */
  pitch: number
}

export type Offsets = Float32Array

export const EDIT_RATE = 100
const EPS = 1e-3
const SCALES = { major: [0, 2, 4, 5, 7, 9, 11], minor: [0, 2, 3, 5, 7, 8, 10] }
export const NOTE_NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']

export function noteName(midi: number): string {
  const m = Math.round(midi)
  return `${NOTE_NAMES[((m % 12) + 12) % 12]}${Math.floor(m / 12) - 1}`
}

export function isBlackKey(midi: number): boolean {
  return [1, 3, 6, 8, 10].includes(((Math.round(midi) % 12) + 12) % 12)
}

export function notesOf(a: PitchAnalysis): Note[] {
  const n = a.midi.length
  return a.notes.map((x, id) => ({
    id,
    first: Math.max(0, Math.min(n - 1, Math.round(x.start / a.frameS))),
    last: Math.max(0, Math.min(n - 1, Math.round(x.end / a.frameS) - 1)),
    pitch: x.pitch,
  }))
}

// ---------- the stored edit ↔ offsets per frame ----------

export function offsetsFromEdit(edit: PitchEdit | null | undefined, frames: number, frameS: number): Offsets {
  const out = new Float32Array(frames)
  if (!edit?.segments?.length) return out
  const step = 1 / (edit.rate || EDIT_RATE)
  for (const seg of edit.segments) {
    const v = seg.values
    if (!v.length) continue
    const end = seg.start + (v.length - 1) * step
    const i0 = Math.max(0, Math.ceil(seg.start / frameS - 1e-9))
    const i1 = Math.min(frames - 1, Math.floor(end / frameS + 1e-9))
    for (let i = i0; i <= i1; i++) {
      const x = (i * frameS - seg.start) / step
      const k = Math.min(v.length - 1, Math.max(0, Math.floor(x)))
      const f = Math.min(1, Math.max(0, x - k))
      out[i] = k + 1 < v.length ? v[k] * (1 - f) + v[k + 1] * f : v[k]
    }
  }
  return out
}

/** Runs of changed frames (gaps under 50 ms merged), on the analysis grid itself (so loading
 * an edit back gives exactly the same offsets). */
export function editFromOffsets(off: Offsets, frameS: number): PitchEdit | null {
  const segments: PitchEdit['segments'] = []
  const n = off.length
  const gap = Math.max(1, Math.round(0.05 / frameS))
  let i = 0
  while (i < n) {
    if (Math.abs(off[i]) < EPS) {
      i++
      continue
    }
    let j = i
    let quiet = 0
    while (j + 1 < n && quiet <= gap) {
      j++
      quiet = Math.abs(off[j]) < EPS ? quiet + 1 : 0
    }
    const last = Math.abs(off[j]) < EPS ? j - quiet : j
    // one unchanged frame each side, so the edit ramps from 0 exactly where it did
    const a = Math.max(0, i - 1)
    const b = Math.min(n - 1, last + 1)
    const values: number[] = []
    for (let k = a; k <= b; k++) values.push(Math.round(off[k] * 1000) / 1000)
    segments.push({ start: a * frameS, values })
    i = last + 1
  }
  return segments.length ? { rate: 1 / frameS, segments } : null
}

export function changedCount(off: Offsets, notes: Note[]): number {
  return notes.filter((n) => {
    for (let i = n.first; i <= n.last; i++) if (Math.abs(off[i]) >= EPS) return true
    return false
  }).length
}

// ---------- what a note sounds like now ----------

export function editedMidi(a: PitchAnalysis, off: Offsets, i: number): number | null {
  const m = a.midi[i]
  return m == null ? null : m + off[i]
}

/** The note's current centre: its median edited pitch. */
export function centerOf(a: PitchAnalysis, off: Offsets, note: Note): number {
  const v: number[] = []
  for (let i = note.first; i <= note.last; i++) {
    const m = a.midi[i]
    if (m != null) v.push(m + off[i])
  }
  if (!v.length) return note.pitch
  v.sort((x, y) => x - y)
  return v[v.length >> 1]
}

// ---------- operations (each returns new offsets) ----------

function forNotes(off: Offsets, notes: Note[], fn: (out: Offsets, note: Note) => void): Offsets {
  const out = new Float32Array(off)
  for (const n of notes) fn(out, n)
  return out
}

/** Short ramps at a note's edges so a moved note glides instead of jumping (Melodyne-style). */
function edgeWeight(i: number, note: Note, ramp: number): number {
  const fromStart = i - note.first
  const toEnd = note.last - i
  return Math.min(1, (Math.min(fromStart, toEnd) + 1) / ramp)
}

export function shiftNotes(off: Offsets, notes: Note[], semitones: number, frameS: number): Offsets {
  const ramp = Math.max(1, Math.round(0.02 / frameS))
  return forNotes(off, notes, (out, n) => {
    for (let i = n.first; i <= n.last; i++) out[i] += semitones * edgeWeight(i, n, ramp)
  })
}

/** Moves each note so its centre lands on `target(centre)`. */
function retune(a: PitchAnalysis, off: Offsets, notes: Note[], target: (center: number) => number, amount = 1): Offsets {
  const ramp = Math.max(1, Math.round(0.02 / a.frameS))
  return forNotes(off, notes, (out, n) => {
    const c = centerOf(a, off, n)
    const by = (target(c) - c) * amount
    for (let i = n.first; i <= n.last; i++) out[i] += by * edgeWeight(i, n, ramp)
  })
}

export function snapToSemitone(a: PitchAnalysis, off: Offsets, notes: Note[]): Offsets {
  return retune(a, off, notes, Math.round)
}

export function nearestInKey(midi: number, tonic: number, mode: 'major' | 'minor'): number {
  const base = Math.round(midi)
  let best = base
  let dist = Infinity
  for (let m = base - 2; m <= base + 2; m++) {
    const pc = (((m - tonic) % 12) + 12) % 12
    if (!SCALES[mode].includes(pc)) continue
    const d = Math.abs(m - midi)
    if (d < dist) {
      dist = d
      best = m
    }
  }
  return best
}

export function snapToKey(a: PitchAnalysis, off: Offsets, notes: Note[], tonic: number, mode: 'major' | 'minor'): Offsets {
  return retune(a, off, notes, (c) => nearestInKey(c, tonic, mode))
}

/** Pulls the line toward each note's centre: drift and vibrato reduced by `amount` (0..1). */
export function straighten(a: PitchAnalysis, off: Offsets, notes: Note[], amount: number): Offsets {
  const ramp = Math.max(1, Math.round(0.04 / a.frameS))
  return forNotes(off, notes, (out, n) => {
    const c = centerOf(a, off, n)
    for (let i = n.first; i <= n.last; i++) {
      const m = a.midi[i]
      if (m == null) continue
      out[i] += (c - (m + off[i])) * amount * edgeWeight(i, n, ramp)
    }
  })
}

/** Adds vibrato (depth in semitones, peak) that fades in over the first third of each note. */
export function addVibrato(off: Offsets, notes: Note[], depth: number, rateHz: number, frameS: number): Offsets {
  return forNotes(off, notes, (out, n) => {
    const len = n.last - n.first + 1
    const fade = Math.max(1, Math.round(len / 3))
    for (let i = n.first; i <= n.last; i++) {
      const k = i - n.first
      const env = Math.min(1, k / fade) * Math.min(1, (n.last - i + 1) / Math.max(1, Math.round(0.05 / frameS)))
      out[i] += depth * env * Math.sin(2 * Math.PI * rateHz * k * frameS)
    }
  })
}

export function resetRange(off: Offsets, first: number, last: number): Offsets {
  const out = new Float32Array(off)
  out.fill(0, Math.max(0, first), Math.min(out.length, last + 1))
  return out
}

export function resetNotes(off: Offsets, notes: Note[]): Offsets {
  return forNotes(off, notes, (out, n) => out.fill(0, n.first, n.last + 1))
}

/**
 * The pen: frames `i0..i1` get the target pitch from linearly interpolating (i0, m0)→(i1, m1);
 * only voiced frames change. Called for each pointer move with the previous point.
 */
export function drawLine(a: PitchAnalysis, off: Offsets, i0: number, m0: number, i1: number, m1: number): void {
  const lo = Math.max(0, Math.min(i0, i1))
  const hi = Math.min(off.length - 1, Math.max(i0, i1))
  for (let i = lo; i <= hi; i++) {
    const m = a.midi[i]
    if (m == null) continue
    const f = i1 === i0 ? 1 : (i - i0) / (i1 - i0)
    off[i] = m0 + (m1 - m0) * f - m
  }
}

/** Smooths the pen stroke's edges into the untouched line (applied when the stroke ends). */
export function blendEdges(off: Offsets, before: Offsets, lo: number, hi: number, frameS: number): void {
  const ramp = Math.max(2, Math.round(0.06 / frameS))
  for (let k = 1; k <= ramp; k++) {
    const w = 1 - k / (ramp + 1)
    const l = lo - k
    const r = hi + k
    if (l >= 0) off[l] = before[l] + (off[lo] - before[lo]) * w
    if (r < off.length) off[r] = before[r] + (off[hi] - before[hi]) * w
  }
}

export function keyName(tonic: number, mode: 'major' | 'minor'): string {
  return `${NOTE_NAMES[tonic]} ${mode === 'major' ? 'major' : 'minor'}`
}
