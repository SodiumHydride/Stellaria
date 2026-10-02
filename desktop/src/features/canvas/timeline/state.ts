import { createContext, useCallback, useContext, useState } from 'react'
import { cssVar } from '../../../lib/color'
import type { Timeline, Track } from '../../../api/types'
import type { MixEngine } from './mixEngine'
import type { GridMode } from './ops'

/** A value that changes every frame (the playhead, scrolling): read it in listeners, not in render. */
export class Signal<T> {
  private subs = new Set<(v: T) => void>()
  constructor(public value: T) {}
  set(v: T) {
    this.value = v
    for (const fn of this.subs) fn(v)
  }
  subscribe(fn: (v: T) => void): () => void {
    this.subs.add(fn)
    return () => void this.subs.delete(fn)
  }
}

export type MenuKind = { kind: 'clip'; clipId: string } | { kind: 'track'; trackId: string } | { kind: 'lane'; trackIndex: number; time: number }

/** What the arrangement's parts can do; stable for the life of the timeline. */
export interface TimelineApi {
  /** commit an edit now (one undo step, or coalesced with the last one of the same `key`) */
  edit: (fn: (tl: Timeline) => Timeline, key?: string) => void
  /** show an edit while a gesture runs; `commit` makes it one undo step, `cancel` drops it */
  preview: (fn: (tl: Timeline) => Timeline) => void
  commit: (key?: string) => void
  cancel: () => void
  select: (ids: string[]) => void
  focusTrack: (id: string | null) => void
  seek: (t: number) => void
  /** the timeline as on screen (a gesture's preview included) */
  current: () => Timeline
  openMenu: (e: { clientX: number; clientY: number }, what: MenuKind) => void
  mediaDuration: (mediaId: string) => number | undefined
  newTrack: (name?: string, kind?: Track['kind']) => Track
  /** put media (a take, a mix…) on track `trackIndex` at `start` (past the last track: a new one); the clip's id */
  placeMedia: (item: { mediaId: string; nodeId: string | null; name?: string }, trackIndex: number, start: number) => string | null
  importAt: (files: string[] | File[], trackIndex: number, time: number) => void
}

export interface TimelineShared {
  engine: MixEngine
  /** playhead (seconds) */
  pos: Signal<number>
  /** ticks every animation frame while playing (and a moment after, so meters fall back) */
  frame: Signal<number>
  api: TimelineApi
  playing: boolean
  snap: boolean
  grid: GridMode
  bpm: number | null
}

export const TimelineContext = createContext<TimelineShared | null>(null)

export function useTimeline(): TimelineShared {
  const ctx = useContext(TimelineContext)
  if (!ctx) throw new Error('TimelineContext missing')
  return ctx
}

/** A view preference remembered across sessions (this device). */
export function useStored<T extends string | number | boolean>(key: string, fallback: T, valid?: (v: unknown) => boolean) {
  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      if (raw === null) return fallback
      const v = JSON.parse(raw) as unknown
      return typeof v === typeof fallback && (!valid || valid(v)) ? (v as T) : fallback
    } catch {
      return fallback
    }
  })
  const save = useCallback(
    (v: T) => {
      setValue(v)
      try {
        localStorage.setItem(key, JSON.stringify(v))
      } catch {
        /* private mode */
      }
    },
    [key],
  )
  return [value, save] as const
}

const palettes = new Map<string, Palette>()
export type Palette = Record<'ink' | 'text1' | 'text3' | 'text4' | 'grid' | 'axis' | 'font' | 'wave', string>

/** The theme's colours for canvas drawing, read once per theme (reading styles while dragging is slow). */
export function palette(theme: string): Palette {
  let p = palettes.get(theme)
  if (!p) {
    p = {
      ink: cssVar('--ink', '52 47 42'),
      text1: cssVar('--text-1'),
      text3: cssVar('--text-3'),
      text4: cssVar('--text-4'),
      grid: cssVar('--chart-grid'),
      axis: cssVar('--chart-axis'),
      font: cssVar('--font', 'system-ui'),
      wave: cssVar('--wave'),
    }
    palettes.set(theme, p)
  }
  return p
}

/** layout (px) */
export const HEADER_W = 212
export const RULER_H = 30
export const LOOP_H = 11
export const TRACK_H = 66
export const CLIP_INSET = 4
export const MIN_PPS = 2
export const MAX_PPS = 480
