import clsx from 'clsx'
import { FileAudio, Plus } from 'lucide-react'
import { memo, useEffect, useId, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { Timeline, TimelineClip, TimelineLoop, Track } from '../../../api/types'
import { useT } from '../../../i18n'
import { DROP_FILES_EVENT } from '../../../lib/dropZones'
import { useEvent, useLatest } from '../../../lib/useEvent'
import { CLIP_DRAG_TYPE } from '../context'
import { newId } from '../useProject'
import { ClipView, fmtSec, type ClipLook, type View } from './ClipView'
import {
  allClips,
  clamp,
  clipEdges,
  clipEnd,
  findClip,
  fmtDb,
  gridFor,
  MIN_CLIP,
  moveClips,
  moveTrack,
  nudgeClipGain,
  round3,
  setFade,
  snapTime,
  trimClips,
  type GridMode,
} from './ops'
import { clock, GridLines, Ruler } from './Ruler'
import { HEADER_W, LOOP_H, MAX_PPS, MIN_PPS, RULER_H, Signal, TRACK_H, useTimeline } from './state'
import { TrackHeader } from './TrackHeader'

export interface ArrangeControls {
  /** zoom by `factor` around a screen x (default: the playhead when in view, else the middle) */
  zoomBy: (factor: number, clientX?: number) => void
  /** the whole song in view */
  fit: () => void
}

type Snapper = (t: number, bypass: boolean) => { t: number; at: number | null }

type Gesture =
  | { kind: 'move'; ids: string[]; anchor: TimelineClip; ti: number; t0: number; x0: number; y0: number; base: Timeline; started: boolean; toggle: string | null; only: string | null; copies: Map<string, string>; result: string[]; snap: Snapper }
  | { kind: 'trim'; ids: string[]; edge: 'left' | 'right'; anchor: TimelineClip; t0: number; base: Timeline; snap: Snapper }
  | { kind: 'fade'; clip: TimelineClip; which: 'in' | 'out'; t0: number; base: Timeline }
  | { kind: 'gain'; ids: string[]; anchor: TimelineClip; y0: number; base: Timeline }
  | { kind: 'band'; x0: number; y0: number; additive: boolean; base: string[]; started: boolean; snap: Snapper }
  | { kind: 'scrub'; snap: Snapper; last: number }
  | { kind: 'loop'; mode: 'new' | 'start' | 'end' | 'move'; t0: number; x0: number; loop: TimelineLoop | null; base: Timeline; started: boolean; snap: Snapper }
  | { kind: 'track'; id: string; from: number; y0: number; started: boolean }


/**
 * The arrangement: a ruler (playhead, loop range) over one row per track (header + lane of
 * clips). Gestures preview their edit and commit it once when let go; snapping (grid, playhead,
 * clip edges, loop) is magnetic within 8 px and Shift bypasses it.
 */
export function Arrange({
  tl,
  pps,
  setPps,
  selection,
  focusTrack,
  follow,
  gridMode,
  controls,
  onHoverTrack,
  lookFor,
  trackColor,
  mediaDuration,
  projectId,
}: {
  tl: Timeline
  pps: number
  setPps: (v: number) => void
  selection: string[]
  focusTrack: string | null
  follow: boolean
  gridMode: GridMode
  controls: React.MutableRefObject<ArrangeControls | null>
  onHoverTrack: (id: string | null) => void
  lookFor: (c: TimelineClip) => ClipLook
  trackColor: (tr: Track) => string | null
  mediaDuration: (mediaId: string) => number | undefined
  projectId: string
}) {
  const t = useT()
  const { engine, pos, api, snap, bpm, playing } = useTimeline()
  const scroller = useRef<HTMLDivElement>(null)
  const gridEl = useRef<HTMLDivElement>(null)
  const tracksEl = useRef<HTMLDivElement>(null)
  const view = useMemo(() => new Signal<View>({ left: 0, width: 800, top: 0, height: 200 }), [])
  const [viewW, setViewW] = useState(800)
  const [viewH, setViewH] = useState(200)
  const [guide, setGuide] = useState<number | null>(null)
  const [band, setBand] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const [drop, setDrop] = useState<{ t: number; ti: number } | null>(null)
  const [reorder, setReorder] = useState<{ id: string; to: number; from: number } | null>(null)
  const [readout, setReadout] = useState<{ x: number; y: number; text: string } | null>(null)
  const gesture = useRef<Gesture | null>(null)
  const anchor = useRef<{ t: number; x: number } | null>(null)
  const hover = useRef<string | null>(null)
  const keysId = useId()
  const grid = gridFor(pps, gridMode, bpm)
  const n = tl.tracks.length
  const contentW = Math.ceil(Math.max(viewW - HEADER_W, (tl.length + 30) * pps))
  const tracksH = Math.max(n * TRACK_H + TRACK_H, viewH - RULER_H)
  const sel = useMemo(() => new Set(selection), [selection])
  const latest = useLatest({ tl, pps, grid, snap, selection, follow, n })
  const empty = !tl.tracks.some((tr) => tr.clips.length > 0)

  // ---------- geometry ----------
  const lanesLeft = () => (gridEl.current?.getBoundingClientRect().left ?? 0) + HEADER_W
  const timeAt = (clientX: number) => (clientX - lanesLeft()) / latest.current.pps
  const trackAt = (clientY: number) => Math.floor((clientY - (tracksEl.current?.getBoundingClientRect().top ?? 0)) / TRACK_H)

  /** snap to the grid, the playhead, loop bounds and other clips' edges (not `exclude`'s) */
  const snapper = (base: Timeline, exclude: Set<string>): Snapper => {
    const { pps: z, grid: g, snap: on } = latest.current
    const targets = [0, pos.value, ...clipEdges(base, exclude)]
    if (base.loop) targets.push(base.loop.start, base.loop.end)
    return (time, bypass) => (on && !bypass ? snapTime(time, targets, g.minor, 8 / z) : { t: time, at: null })
  }

  // ---------- scrolling and zoom ----------
  const syncView = useEvent(() => {
    const sc = scroller.current
    if (sc) view.set({ left: sc.scrollLeft, width: Math.max(0, sc.clientWidth - HEADER_W), top: sc.scrollTop, height: Math.max(0, sc.clientHeight - RULER_H) })
  })
  useLayoutEffect(() => {
    const sc = scroller.current
    if (!sc) return
    const ro = new ResizeObserver(() => {
      setViewW(sc.clientWidth)
      setViewH(sc.clientHeight)
      syncView()
    })
    ro.observe(sc)
    return () => ro.disconnect()
  }, [syncView])
  // after a zoom: the time under the pointer (or the playhead) stays where it was
  useLayoutEffect(() => {
    const sc = scroller.current
    const a = anchor.current
    anchor.current = null
    if (sc && a) sc.scrollLeft = Math.max(0, a.t * pps - a.x)
    syncView()
  }, [pps, syncView])

  const zoomBy = useEvent((factor: number, clientX?: number) => {
    const sc = scroller.current
    if (!sc) return
    const z = latest.current.pps
    const next = clamp(z * factor, MIN_PPS, MAX_PPS)
    if (next === z) return
    const w = sc.clientWidth - HEADER_W
    let x = clientX !== undefined ? clientX - sc.getBoundingClientRect().left - HEADER_W : pos.value * z - sc.scrollLeft
    if (clientX === undefined && (x < 0 || x > w)) x = w / 2
    anchor.current = { t: (sc.scrollLeft + x) / z, x }
    setPps(next)
  })
  const fit = useEvent(() => {
    const sc = scroller.current
    if (!sc) return
    const w = sc.clientWidth - HEADER_W - 40
    anchor.current = { t: 0, x: 0 }
    setPps(clamp(w / Math.max(latest.current.tl.length, 4), MIN_PPS, MAX_PPS))
  })
  useLayoutEffect(() => {
    controls.current = { zoomBy, fit }
  }, [controls, zoomBy, fit])

  // Ctrl / Cmd + wheel (and pinch) zooms around the pointer; Shift + wheel scrolls time
  const onWheel = useEvent((e: WheelEvent) => {
    const sc = scroller.current
    if (!sc) return
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault()
      zoomBy(Math.exp(-clamp(e.deltaY, -60, 60) * 0.006), e.clientX)
    } else if (e.shiftKey && Math.abs(e.deltaX) < Math.abs(e.deltaY)) {
      e.preventDefault()
      sc.scrollLeft += e.deltaY
    }
  })
  useEffect(() => {
    const sc = scroller.current
    sc?.addEventListener('wheel', onWheel, { passive: false })
    return () => sc?.removeEventListener('wheel', onWheel)
  }, [onWheel])

  // follow the playhead page by page while playing (not while the user is dragging something)
  useEffect(
    () =>
      pos.subscribe((p) => {
        const sc = scroller.current
        if (!sc || !latest.current.follow || !engine.playing || gesture.current) return
        const x = p * latest.current.pps
        const w = sc.clientWidth - HEADER_W
        if (x < sc.scrollLeft || x > sc.scrollLeft + w - 12) sc.scrollLeft = Math.max(0, x - w * 0.06)
      }),
    [pos, engine, latest],
  )

  const autoScroll = (e: PointerEvent) => {
    const sc = scroller.current
    if (!sc) return
    const r = sc.getBoundingClientRect()
    if (e.clientX > r.right - 28) sc.scrollLeft += 14
    else if (e.clientX < r.left + HEADER_W + 28) sc.scrollLeft -= 14
    if (e.clientY > r.bottom - 20) sc.scrollTop += 10
    else if (e.clientY < r.top + RULER_H + 16) sc.scrollTop -= 10
  }

  // ---------- gestures ----------
  const focusRoot = (el: HTMLElement) => {
    const root = el.closest<HTMLElement>('.timeline')
    if (root && !root.contains(document.activeElement)) root.focus({ preventScroll: true })
    else if (document.activeElement instanceof HTMLElement && document.activeElement.matches('input, textarea')) root?.focus({ preventScroll: true })
  }

  const finish = useEvent(() => {
    gesture.current = null
    setGuide(null)
    setBand(null)
    setReadout(null)
    setReorder(null)
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', onUp)
    window.removeEventListener('pointercancel', onCancel)
    window.removeEventListener('keydown', onEscape, true)
  })
  const begin = (g: Gesture) => {
    if (gesture.current) finish()
    gesture.current = g
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onCancel)
    window.addEventListener('keydown', onEscape, true)
  }
  const onCancel = useEvent(() => {
    if (gesture.current?.kind !== 'band' && gesture.current?.kind !== 'scrub') api.cancel()
    finish()
  })
  const onEscape = useEvent((e: KeyboardEvent) => {
    if (e.key !== 'Escape') return
    e.preventDefault()
    e.stopPropagation()
    onCancel()
  })
  const show = (e: PointerEvent, text: string) => setReadout({ x: e.clientX, y: e.clientY, text })

  const onMove = useEvent((e: PointerEvent) => {
    const g = gesture.current
    if (!g) return
    const { tl: cur, n: count } = latest.current
    if (g.kind === 'move') {
      if (!g.started) {
        if (Math.hypot(e.clientX - g.x0, e.clientY - g.y0) < 4) return
        g.started = true
      }
      autoScroll(e)
      let dt = timeAt(e.clientX) - g.t0
      const a = g.snap(g.anchor.start + dt, e.shiftKey)
      const b = g.snap(clipEnd(g.anchor) + dt, e.shiftKey)
      let at: number | null = null
      if (a.at !== null && (b.at === null || Math.abs(a.t - g.anchor.start - dt) <= Math.abs(b.t - clipEnd(g.anchor) - dt))) {
        dt = a.t - g.anchor.start
        at = a.at
      } else if (b.at !== null) {
        dt = b.t - clipEnd(g.anchor)
        at = b.at
      }
      const dTrack = clamp(trackAt(e.clientY), 0, count - 1) - g.ti
      const copy = e.altKey
        ? (id: string) => {
            if (!g.copies.has(id)) g.copies.set(id, newId('tc'))
            return g.copies.get(id)!
          }
        : undefined
      const r = moveClips(g.base, g.ids, dt, dTrack, copy)
      g.result = r.ids
      api.preview(() => r.tl)
      setGuide(at)
      const moved = r.ids.length ? findClip(r.tl, r.ids[g.ids.indexOf(g.anchor.id)] ?? r.ids[0])?.clip : undefined
      show(e, `${copy ? '+ ' : ''}${clock(moved?.start ?? g.anchor.start)}`)
    } else if (g.kind === 'trim') {
      autoScroll(e)
      const edge0 = g.edge === 'left' ? g.anchor.start : clipEnd(g.anchor)
      const s = g.snap(edge0 + timeAt(e.clientX) - g.t0, e.shiftKey)
      const next = trimClips(g.base, g.ids, g.edge, s.t - edge0, mediaDuration)
      api.preview(() => next)
      setGuide(s.at)
      const c = findClip(next, g.anchor.id)?.clip
      if (c) show(e, `${clock(g.edge === 'left' ? c.start : clipEnd(c))} · ${fmtSec(c.duration)}`)
    } else if (g.kind === 'fade') {
      // relative to where it was grabbed: the handle doesn't jump under the pointer
      const dt = timeAt(e.clientX) - g.t0
      const v = g.which === 'in' ? g.clip.fadeIn + dt : g.clip.fadeOut - dt
      const next = setFade(g.base, g.clip.id, g.which, v)
      api.preview(() => next)
      const c = findClip(next, g.clip.id)?.clip
      if (c) show(e, `${t(g.which === 'in' ? 'tl.fade_in' : 'tl.fade_out')} ${fmtSec(g.which === 'in' ? c.fadeIn : c.fadeOut)}`)
    } else if (g.kind === 'gain') {
      const delta = (g.y0 - e.clientY) * (e.shiftKey ? 0.02 : 0.1)
      const next = nudgeClipGain(g.base, g.ids, delta)
      api.preview(() => next)
      const c = findClip(next, g.anchor.id)?.clip
      if (c) show(e, `${fmtDb(c.gainDb)} dB`)
    } else if (g.kind === 'band') {
      if (!g.started) {
        if (Math.hypot(e.clientX - g.x0, e.clientY - g.y0) < 4) return
        g.started = true
      }
      autoScroll(e)
      const r = gridEl.current!.getBoundingClientRect()
      const x0 = g.x0 - r.left
      const y0 = g.y0 - r.top
      const x1 = e.clientX - r.left
      const y1 = e.clientY - r.top
      setBand({ x: Math.min(x0, x1), y: Math.min(y0, y1), w: Math.abs(x1 - x0), h: Math.abs(y1 - y0) })
      const ta = (Math.min(x0, x1) - HEADER_W) / latest.current.pps
      const tb = (Math.max(x0, x1) - HEADER_W) / latest.current.pps
      const ra = Math.floor((Math.min(y0, y1) - RULER_H) / TRACK_H)
      const rb = Math.floor((Math.max(y0, y1) - RULER_H) / TRACK_H)
      const inside = allClips(cur)
        .filter((x) => x.ti >= ra && x.ti <= rb && x.clip.start < tb && clipEnd(x.clip) > ta)
        .map((x) => x.clip.id)
      api.select(g.additive ? Array.from(new Set([...g.base, ...inside])) : inside)
    } else if (g.kind === 'scrub') {
      const s = g.snap(Math.max(0, timeAt(e.clientX)), e.shiftKey)
      g.last = s.t
      setGuide(s.at)
      // while playing the playhead only shows where it goes; playback jumps there when let go
      if (engine.playing) pos.set(s.t)
      else api.seek(s.t)
    } else if (g.kind === 'loop') {
      if (!g.started) {
        if (Math.abs(e.clientX - g.x0) < 3) return
        g.started = true
      }
      const s = g.snap(Math.max(0, timeAt(e.clientX)), e.shiftKey)
      setGuide(s.at)
      let start: number
      let end: number
      const l = g.loop
      if (g.mode === 'start' && l) [start, end] = [Math.min(s.t, l.end - MIN_CLIP), l.end]
      else if (g.mode === 'end' && l) [start, end] = [l.start, Math.max(s.t, l.start + MIN_CLIP)]
      else if (g.mode === 'move' && l) {
        const len = l.end - l.start
        const raw = Math.max(0, l.start + timeAt(e.clientX) - g.t0)
        const ss = g.snap(raw, e.shiftKey)
        const se = g.snap(raw + len, e.shiftKey)
        start = ss.at !== null ? ss.t : se.at !== null ? Math.max(0, se.t - len) : raw
        setGuide(ss.at ?? se.at)
        end = start + len
      } else [start, end] = [Math.min(g.t0, s.t), Math.max(g.t0, s.t)]
      const loop = { start: round3(start), end: round3(end), on: true }
      api.preview((x) => ({ ...g.base, tracks: x.tracks, loop }))
      show(e, `${clock(loop.start)} – ${clock(loop.end)}`)
    } else if (g.kind === 'track') {
      if (!g.started) {
        if (Math.abs(e.clientY - g.y0) < 4) return
        g.started = true
      }
      autoScroll(e)
      setReorder({ id: g.id, from: g.from, to: clamp(trackAt(e.clientY), 0, count - 1) })
    }
  })

  const onUp = useEvent((e: PointerEvent) => {
    const g = gesture.current
    if (!g) return
    finish()
    if (g.kind === 'move') {
      if (!g.started) {
        api.cancel()
        if (g.toggle) api.select(latest.current.selection.filter((x) => x !== g.toggle))
        else if (g.only) api.select([g.only])
        return
      }
      api.commit()
      api.select(g.result)
      const focus = g.result[g.ids.indexOf(g.anchor.id)] ?? g.result[0]
      requestAnimationFrame(() => document.querySelector<HTMLElement>(`.tl-clip[data-clip="${CSS.escape(focus)}"]`)?.focus({ preventScroll: true }))
    } else if (g.kind === 'trim' || g.kind === 'fade' || g.kind === 'gain') {
      api.commit()
    } else if (g.kind === 'band') {
      if (g.started) return
      // a click on an empty lane: deselect, and the playhead goes there
      if (!g.additive) api.select([])
      api.seek(g.snap(Math.max(0, timeAt(g.x0)), e.shiftKey).t)
    } else if (g.kind === 'scrub') {
      api.seek(g.last)
    } else if (g.kind === 'loop') {
      if (g.started) api.commit()
      else {
        api.cancel()
        // a click inside the loop range switches looping on / off
        const l = g.loop
        if (l && g.t0 >= l.start && g.t0 <= l.end) api.edit((x) => ({ ...x, loop: { ...l, on: l.on === false } }))
      }
    } else if (g.kind === 'track') {
      if (!g.started) return
      const to = clamp(trackAt(e.clientY), 0, latest.current.n - 1)
      if (to !== g.from) api.edit((x) => moveTrack(x, g.id, to))
    }
  })

  useEffect(() => finish, [finish])

  const onLanesPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    const el = e.target as HTMLElement
    if (el.closest('.tl-head, .tl-add-row button')) return
    const cur = latest.current.tl
    const clipEl = el.closest<HTMLElement>('[data-clip]')
    if (clipEl) {
      const loc = findClip(cur, clipEl.dataset.clip!)
      if (!loc) return
      e.preventDefault()
      clipEl.focus({ preventScroll: true })
      const id = loc.clip.id
      const additive = e.shiftKey || e.metaKey || e.ctrlKey
      const was = latest.current.selection
      const ids = was.includes(id) ? was : additive ? [...was, id] : [id]
      if (ids !== was) api.select(ids)
      api.focusTrack(loc.track.id)
      const handle = el.closest<HTMLElement>('[data-handle]')?.dataset.handle
      const t0 = timeAt(e.clientX)
      if (handle === 'fade-in' || handle === 'fade-out') begin({ kind: 'fade', clip: loc.clip, which: handle === 'fade-in' ? 'in' : 'out', t0, base: cur })
      else if (handle === 'gain') begin({ kind: 'gain', ids, anchor: loc.clip, y0: e.clientY, base: cur })
      else if (handle === 'trim-l' || handle === 'trim-r') {
        const edge = handle === 'trim-l' ? 'left' : 'right'
        begin({ kind: 'trim', ids, edge, anchor: loc.clip, t0, base: cur, snap: snapper(cur, new Set(ids)) })
      } else
        begin({
          kind: 'move',
          ids,
          anchor: loc.clip,
          ti: loc.ti,
          t0,
          x0: e.clientX,
          y0: e.clientY,
          base: cur,
          started: false,
          toggle: additive && was.includes(id) ? id : null,
          only: !additive && was.includes(id) && was.length > 1 ? id : null,
          copies: new Map(),
          result: ids,
          snap: snapper(cur, new Set(ids)),
        })
      return
    }
    e.preventDefault()
    focusRoot(e.currentTarget as HTMLElement)
    const ti = trackAt(e.clientY)
    if (ti >= 0 && ti < cur.tracks.length) api.focusTrack(cur.tracks[ti].id)
    const additive = e.shiftKey || e.metaKey || e.ctrlKey
    begin({ kind: 'band', x0: e.clientX, y0: e.clientY, additive, base: latest.current.selection, started: false, snap: snapper(cur, new Set()) })
  }

  const onRulerPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    e.preventDefault()
    focusRoot(e.currentTarget as HTMLElement)
    const cur = latest.current.tl
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const time = Math.max(0, timeAt(e.clientX))
    const s = snapper(cur, new Set())
    if (e.clientY - r.top < LOOP_H + 2) {
      const l = cur.loop ?? null
      const near = (x: number) => Math.abs(x - time) * latest.current.pps <= 6
      const mode = l && near(l.start) ? 'start' : l && near(l.end) ? 'end' : l && time > l.start && time < l.end ? 'move' : 'new'
      const t0 = mode === 'new' ? s(time, e.shiftKey).t : time
      begin({ kind: 'loop', mode, t0, x0: e.clientX, loop: l, base: cur, started: false, snap: s })
      return
    }
    const at = s(time, e.shiftKey).t
    if (engine.playing) pos.set(at)
    else api.seek(at)
    begin({ kind: 'scrub', snap: s, last: at })
  }

  const onReorderStart = useEvent((e: React.PointerEvent, trackId: string) => {
    if (e.button !== 0) return
    e.preventDefault()
    e.stopPropagation()
    const from = latest.current.tl.tracks.findIndex((tr) => tr.id === trackId)
    begin({ kind: 'track', id: trackId, from, y0: e.clientY, started: false })
  })

  const onContextMenu = (e: React.MouseEvent) => {
    const el = e.target as HTMLElement
    if (el.closest('.tl-head, input')) return
    e.preventDefault()
    const cur = latest.current.tl
    const clipEl = el.closest<HTMLElement>('[data-clip]')
    if (clipEl) {
      const id = clipEl.dataset.clip!
      if (!latest.current.selection.includes(id)) api.select([id])
      api.openMenu(e, { kind: 'clip', clipId: id })
      return
    }
    const ti = clamp(trackAt(e.clientY), 0, cur.tracks.length)
    if (ti < cur.tracks.length) api.focusTrack(cur.tracks[ti].id)
    api.openMenu(e, { kind: 'lane', trackIndex: ti, time: Math.max(0, timeAt(e.clientX)) })
  }

  const onLanesPointerMove = (e: React.PointerEvent) => {
    if (gesture.current) return
    const id = latest.current.tl.tracks[trackAt(e.clientY)]?.id ?? null
    if (id !== hover.current) {
      hover.current = id
      onHoverTrack(id)
    }
  }

  // ---------- drag and drop: takes / mixes from the side panel, audio files ----------
  const dropTarget = (clientX: number, clientY: number, bypass: boolean) => {
    const cur = latest.current.tl
    return { t: snapper(cur, new Set())(Math.max(0, timeAt(clientX)), bypass).t, ti: clamp(trackAt(clientY), 0, cur.tracks.length) }
  }
  const accepts = (e: React.DragEvent) => e.dataTransfer.types.includes(CLIP_DRAG_TYPE) || e.dataTransfer.types.includes('Files')
  const onDragOver = (e: React.DragEvent) => {
    if (!accepts(e)) return
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
    const d = dropTarget(e.clientX, e.clientY, e.shiftKey)
    if (!drop || d.t !== drop.t || d.ti !== drop.ti) setDrop(d)
  }
  const onDragLeave = (e: React.DragEvent) => {
    if (!(e.currentTarget as HTMLElement).contains(e.relatedTarget as Node | null)) setDrop(null)
  }
  const onDrop = (e: React.DragEvent) => {
    if (!accepts(e)) return
    e.preventDefault()
    e.stopPropagation()
    setDrop(null)
    const { t: at, ti } = dropTarget(e.clientX, e.clientY, e.shiftKey)
    const raw = e.dataTransfer.getData(CLIP_DRAG_TYPE)
    if (raw) {
      let item: { mediaId?: string; nodeId?: string | null; name?: string } = {}
      try {
        item = JSON.parse(raw)
      } catch {
        return
      }
      if (item.mediaId) api.placeMedia({ mediaId: item.mediaId, nodeId: item.nodeId ?? null, name: item.name }, ti, at)
      return
    }
    const files = Array.from(e.dataTransfer.files)
    if (files.length) api.importAt(files, ti, at)
  }
  // the desktop shell's OS drops arrive as an event on the zone (see DROP_FILES_EVENT)
  const onDropFiles = useEvent((ev: Event) => {
    const { x, y, items } = (ev as CustomEvent<{ x: number; y: number; items: string[] | File[] }>).detail
    const d = dropTarget(x, y, false)
    api.importAt(items, d.ti, d.t)
  })
  useEffect(() => {
    const el = tracksEl.current
    el?.addEventListener(DROP_FILES_EVENT, onDropFiles)
    return () => el?.removeEventListener(DROP_FILES_EVENT, onDropFiles)
  }, [onDropFiles])

  // ---------- render ----------
  const loop = tl.loop
  const reorderLine = reorder ? (reorder.to > reorder.from ? reorder.to + 1 : reorder.to) * TRACK_H : null

  return (
    <div className="tl-arrange">
      <span id={keysId} className="sr-only">
        {t('tl.keys.clip')}
      </span>
      <div className="tl-scroll" ref={scroller} onScroll={syncView}>
        <div className="tl-grid" ref={gridEl} style={{ width: HEADER_W + contentW }}>
          <div className="tl-ruler-row">
            <div className="tl-corner" style={{ width: HEADER_W }}>
              <span className="tl-corner-hint">{t(gridMode === 'bars' && bpm ? 'tl.grid_bars_hint' : 'tl.grid_sec_hint')}</span>
            </div>
            <Ruler view={view} pps={pps} grid={grid} width={contentW} length={tl.length} loop={loop} onPointerDown={onRulerPointerDown} />
          </div>
          <div
            className={clsx('tl-tracks', drop && 'dropping')}
            ref={tracksEl}
            style={{ height: tracksH }}
            data-drop-zone="timeline"
            onPointerDown={onLanesPointerDown}
            onPointerMove={onLanesPointerMove}
            onPointerLeave={() => {
              hover.current = null
              onHoverTrack(null)
            }}
            onContextMenu={onContextMenu}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <div className="tl-lanes-bg" style={{ left: HEADER_W, width: contentW, height: tracksH }}>
              <GridLines view={view} pps={pps} grid={grid} />
              {loop && loop.end > loop.start && <div className={clsx('tl-loop-shade', loop.on !== false && 'on')} style={{ left: loop.start * pps, width: (loop.end - loop.start) * pps }} />}
            </div>
            {tl.tracks.map((tr, i) => (
              <TrackRow
                key={tr.id}
                track={tr}
                index={i}
                pps={pps}
                focused={focusTrack === tr.id}
                lifting={reorder?.id === tr.id}
                color={trackColor(tr)}
                selection={selection}
                sel={sel}
                lookFor={lookFor}
                view={view}
                projectId={projectId}
                mediaDuration={mediaDuration}
                keysId={keysId}
                onReorderStart={onReorderStart}
              />
            ))}
            <div className="tl-add-row" style={{ height: tracksH - n * TRACK_H }}>
              <div className="tl-add-head" style={{ width: HEADER_W }}>
                <button className="btn btn-sm btn-ghost" onClick={() => api.edit((x) => ({ ...x, tracks: [...x.tracks, api.newTrack()] }))}>
                  <Plus />
                  {t('tl.add_track')}
                </button>
              </div>
            </div>
            {empty && (
              <div className="tl-empty" style={{ left: HEADER_W, top: n ? n * TRACK_H : 0 }}>
                <FileAudio />
                <span>{t('tl.empty')}</span>
                <span className="muted">{t('tl.empty_drop')}</span>
              </div>
            )}
            {drop && (
              <>
                <div className="tl-drop-row" style={{ top: drop.ti * TRACK_H, left: HEADER_W }} />
                <div className="tl-drop-at" style={{ top: drop.ti * TRACK_H, left: HEADER_W + drop.t * pps }}>
                  <span>{drop.ti >= n ? t('tl.drop_new_track') : clock(drop.t)}</span>
                </div>
              </>
            )}
            {reorderLine !== null && <div className="tl-reorder-line" style={{ top: reorderLine - 1 }} />}
            <Playhead pps={pps} playing={playing} />
            {guide !== null && <div className="tl-guide" style={{ left: HEADER_W + guide * pps }} />}
          </div>
          {band && <div className="tl-band" style={{ left: band.x, top: band.y, width: band.w, height: band.h }} />}
        </div>
      </div>
      {readout &&
        createPortal(
          <div className="tl-readout tabular" style={{ left: readout.x + 14, top: readout.y - 30 }}>
            {readout.text}
          </div>,
          document.body,
        )}
    </div>
  )
}

/** One track's row; only rows whose track (or the selection) changed render again while dragging. */
const TrackRow = memo(function TrackRow({
  track,
  index,
  pps,
  focused,
  lifting,
  color,
  selection,
  sel,
  lookFor,
  view,
  projectId,
  mediaDuration,
  keysId,
  onReorderStart,
}: {
  track: Track
  index: number
  pps: number
  focused: boolean
  lifting: boolean
  color: string | null
  selection: string[]
  sel: Set<string>
  lookFor: (c: TimelineClip) => ClipLook
  view: Signal<View>
  projectId: string
  mediaDuration: (mediaId: string) => number | undefined
  keysId: string
  onReorderStart: (e: React.PointerEvent, trackId: string) => void
}) {
  return (
    <div className={clsx('tl-track', focused && 'focused', track.mute && 'muted-track', lifting && 'lifting')} style={{ height: TRACK_H }}>
      <TrackHeader track={track} index={index} color={color} focused={focused} onReorderStart={onReorderStart} />
      <div className="tl-lane">
        {track.clips.map((c) => (
          <ClipView
            key={c.id}
            clip={c}
            pps={pps}
            selected={sel.has(c.id)}
            look={lookFor(c)}
            view={view}
            projectId={projectId}
            mediaDuration={mediaDuration(c.mediaId)}
            selection={sel.has(c.id) ? selection : null}
            keysId={keysId}
          />
        ))}
      </div>
    </div>
  )
})

/** The playhead line over the lanes; it moves without re-rendering the arrangement. */
function Playhead({ pps, playing }: { pps: number; playing: boolean }) {
  const { pos } = useTimeline()
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const move = (p: number) => {
      if (ref.current) ref.current.style.transform = `translateX(${p * pps}px)`
    }
    move(pos.value)
    return pos.subscribe(move)
  }, [pos, pps])
  return <div ref={ref} className={clsx('tl-playhead', playing && 'playing')} style={{ left: HEADER_W }} aria-hidden />
}
