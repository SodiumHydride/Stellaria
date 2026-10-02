import clsx from 'clsx'
import { Maximize2, Pause, Play, Repeat } from 'lucide-react'
import { useEffect, useId, useRef, useState } from 'react'
import { mediaUrl } from '../../../api/client'
import type { CompSegment } from '../../../api/types'
import { loadPeaks } from '../../../components/Waveform'
import { Dialog, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { cssVar, waveColor } from '../../../lib/color'
import { clockPrecise } from '../../../lib/format'
import { usePlayer } from '../../../state/player'
import { useTheme } from '../../../state/theme'
import { mediaAudioPath, mediaPeaksPath } from '../context'

export type RegionMode =
  | { kind: 'range'; start: number; end: number; onChange: (start: number, end: number) => void }
  | { kind: 'cuts'; cuts: number[]; onChange: (cuts: number[]) => void }
  | { kind: 'segments'; segments: CompSegment[]; lanes: { label: string; color: string }[]; onChange: (segments: CompSegment[]) => void }

export interface RegionEditorProps {
  projectId: string
  mediaId: string
  /** length of the audio, seconds */
  duration: number
  /** times in `mode` are on this timeline: 0 = the media's start (trim, split) or the song's start (comp) */
  timeBase?: number
  /** points worth snapping to (phrase edges, downbeats), in the same time as `mode` */
  snaps?: number[]
  mode: RegionMode
  height?: number
  /** the expand button opens a big editor */
  expandable?: boolean
  title?: string
}

const SNAP_PX = 7
const MIN_LEN = 0.05

const round3 = (s: number) => Math.round(s * 1000) / 1000

/**
 * Edit time on a waveform: a range (trim), cut lines (split) or voice segments (comp). Drags
 * snap to phrases and bars (hold Alt to place freely); the loop button repeats the region.
 * Every handle is also a keyboard slider; in split and comp the waveform itself takes Enter to
 * cut at the playhead.
 */
export function RegionEditor(props: RegionEditorProps) {
  const t = useT()
  const [big, setBig] = useState(false)
  return (
    <>
      <RegionCanvas {...props} onExpand={props.expandable ? () => setBig(true) : undefined} />
      {big && (
        <Dialog open onOpenChange={(o) => !o && setBig(false)} title={props.title ?? t('region.title')} wide>
          <div className="region-big">
            <RegionCanvas {...props} height={150} />
            <p className="muted region-help">{t(`region.help.${props.mode.kind}` as const)}</p>
          </div>
        </Dialog>
      )}
    </>
  )
}

function RegionCanvas({ projectId, mediaId, duration, timeBase = 0, snaps = [], mode, height = 64, title, onExpand }: RegionEditorProps & { onExpand?: () => void }) {
  const t = useT()
  const wrap = useRef<HTMLDivElement>(null)
  const canvas = useRef<HTMLCanvasElement>(null)
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const [width, setWidth] = useState(300)
  const [drag, setDrag] = useState<null | { what: string; x0: number; t0: number; v0: number[] }>(null)
  const theme = useTheme((s) => s.resolved)
  const playerId = `region:${mediaId}`
  const current = usePlayer((s) => s.currentId === playerId)
  const playing = usePlayer((s) => s.currentId === playerId && s.playing)
  const time = usePlayer((s) => (s.currentId === playerId ? s.time : null))
  const loop = usePlayer((s) => s.loop)
  const total = Math.max(duration, 0.1)
  const ids = useId()
  const refocus = useRef<string | null>(null)

  useEffect(() => {
    let cancelled = false
    loadPeaks(mediaPeaksPath(projectId, mediaId), 400)
      .then((r) => !cancelled && setPeaks(r.peaks))
      .catch(() => !cancelled && setPeaks([]))
    return () => {
      cancelled = true
    }
  }, [projectId, mediaId])

  useEffect(() => {
    const el = wrap.current
    if (!el) return
    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)
    return () => ro.disconnect()
  }, [])

  // media time ↔ x
  const toX = (sec: number) => ((sec - timeBase) / total) * width
  const toT = (x: number) => Math.max(timeBase, Math.min(timeBase + total, timeBase + (x / width) * total))
  const snap = (sec: number, alt: boolean) => {
    if (alt) return sec
    let best = sec
    let bestPx = SNAP_PX
    for (const s of snaps) {
      const d = Math.abs(toX(s) - toX(sec))
      if (d < bestPx) {
        best = s
        bestPx = d
      }
    }
    return best
  }

  useEffect(() => {
    const c = canvas.current
    if (!c || !peaks) return
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = Math.round(width * dpr)
    c.height = Math.round(height * dpr)
    c.style.width = `${width}px`
    c.style.height = `${height}px`
    const ctx = c.getContext('2d')!
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, width, height)
    if (peaks.length === 0) return
    const mid = height / 2
    const bw = Math.max(1, width / peaks.length - 0.6)
    ctx.fillStyle = waveColor(cssVar('--fill'))
    peaks.forEach((p, i) => {
      const x = (i * width) / peaks.length
      const hh = Math.max(0.5, p * (height - 4) * 0.5)
      ctx.fillRect(x, mid - hh, bw, hh * 2)
    })
  }, [peaks, width, height, theme])

  const xOf = (e: { clientX: number }) => e.clientX - (wrap.current?.getBoundingClientRect().left ?? 0)

  const startDrag = (what: string, e: React.PointerEvent, values: number[]) => {
    e.stopPropagation()
    e.preventDefault()
    ;(e.currentTarget as HTMLElement).focus({ preventScroll: true })
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
    setDrag({ what, x0: xOf(e), t0: toT(xOf(e)), v0: values })
  }

  const onMove = (e: React.PointerEvent) => {
    if (!drag) return
    const now = snap(toT(xOf(e)), e.altKey)
    if (mode.kind === 'range') {
      const [s0, e0] = drag.v0
      if (drag.what === 'start') mode.onChange(Math.min(now, e0 - MIN_LEN), e0)
      else if (drag.what === 'end') mode.onChange(s0, Math.max(now, s0 + MIN_LEN))
      else {
        const len = e0 - s0
        const delta = toT(xOf(e)) - drag.t0
        const s = Math.max(timeBase, Math.min(timeBase + total - len, s0 + delta))
        mode.onChange(snap(s, e.altKey), snap(s, e.altKey) + len)
      }
    } else if (mode.kind === 'cuts') {
      const i = Number(drag.what)
      const lo = (drag.v0[i - 1] ?? timeBase) + MIN_LEN
      const hi = (drag.v0[i + 1] ?? timeBase + total) - MIN_LEN
      const next = drag.v0.slice()
      next[i] = Math.max(lo, Math.min(hi, now))
      mode.onChange(next)
    } else {
      const i = Number(drag.what) // boundary between segment i and i+1
      const segs = mode.segments.map((s) => ({ ...s }))
      const lo = segs[i].start + MIN_LEN
      const hi = segs[i + 1].end - MIN_LEN
      const v = Math.max(lo, Math.min(hi, now))
      segs[i].end = v
      segs[i + 1].start = v
      mode.onChange(segs)
    }
  }

  const playFrom = (sec: number) => {
    usePlayer.getState().play(playerId, mediaUrl(mediaAudioPath(projectId, mediaId)), Math.max(0, sec - timeBase))
  }

  const regionPlay = () => {
    const st = usePlayer.getState()
    if (playing) return st.pause()
    if (mode.kind === 'range') playFrom(mode.start)
    else playFrom(current && time != null ? time + timeBase : timeBase)
  }

  const loopRegion = () => {
    const st = usePlayer.getState()
    if (loop) return st.setLoop(null)
    if (mode.kind !== 'range') return
    playFrom(mode.start)
    st.setLoop({ start: mode.start - timeBase, end: mode.end - timeBase })
  }

  // keep a running loop in step with the handles (the loop itself is read, not watched: setting it
  // makes a new one)
  const rangeStart = mode.kind === 'range' ? mode.start : null
  const rangeEnd = mode.kind === 'range' ? mode.end : null
  useEffect(() => {
    const st = usePlayer.getState()
    if (rangeStart === null || rangeEnd === null || st.currentId !== playerId || !st.loop) return
    st.setLoop({ start: rangeStart - timeBase, end: rangeEnd - timeBase })
  }, [rangeStart, rangeEnd, playerId, timeBase])

  /** Adds a cut (split) or splits the segment under `sec` (comp); returns the new handle's index. */
  const cutAt = (sec: number): number | null => {
    if (mode.kind === 'cuts') {
      const next = [...mode.cuts, sec].sort((a, b) => a - b).filter((c, i, a) => i === 0 || c - a[i - 1] > MIN_LEN)
      mode.onChange(next)
      const i = next.indexOf(sec)
      return i >= 0 ? i : null
    }
    if (mode.kind === 'segments') {
      const i = mode.segments.findIndex((s) => sec > s.start + MIN_LEN && sec < s.end - MIN_LEN)
      if (i < 0) return null
      const s = mode.segments[i]
      mode.onChange([...mode.segments.slice(0, i), { ...s, end: sec }, { ...s, start: sec }, ...mode.segments.slice(i + 1)])
      return i
    }
    return null
  }

  const onBackgroundClick = (e: React.MouseEvent) => {
    if (drag) return
    const sec = snap(toT(xOf(e)), e.altKey)
    if (mode.kind !== 'range' && e.detail === 2) cutAt(sec)
    else if (e.detail === 1) playFrom(sec)
  }

  // comp: a click (or Enter) cycles who sings a segment, voices then silence
  const cycleVoice = (i: number) => {
    if (mode.kind !== 'segments') return
    const s = mode.segments[i]
    const nextInput = s.input + 1 >= mode.lanes.length ? -1 : s.input + 1
    mode.onChange(mode.segments.map((x, j) => (j === i ? { ...x, input: nextInput } : x)))
  }
  const mergeAt = (i: number) => {
    if (mode.kind !== 'segments') return
    const segs = mode.segments.slice()
    segs.splice(i, 2, { ...segs[i], end: segs[i + 1].end })
    mode.onChange(segs)
  }
  const laneName = (input: number) => (mode.kind === 'segments' ? mode.lanes[input]?.label : undefined) ?? t('region.silence')

  // ---------- keyboard ----------
  // after each render: focus what a keyboard edit asked for (set just before its update)
  useEffect(() => {
    const key = refocus.current
    if (!key) return
    refocus.current = null
    const el = key === 'track' ? wrap.current : wrap.current?.querySelector<HTMLElement>(`[data-handle="${key}"]`)
    el?.focus({ preventScroll: true })
  })

  /** Where a key moves a handle at `v` (bounded by lo…hi): arrows step 0.1 s, Shift 1 s, Alt 0.01 s; Page Up/Down jump to the next snap point. */
  const keyTarget = (e: React.KeyboardEvent, v: number, lo: number, hi: number): number | null => {
    const step = e.shiftKey ? 1 : e.altKey ? 0.01 : 0.1
    let next: number
    if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') next = v - step
    else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') next = v + step
    else if (e.key === 'Home') next = lo
    else if (e.key === 'End') next = hi
    else if (e.key === 'PageUp' || e.key === 'PageDown') {
      const up = e.key === 'PageUp'
      const ahead = snaps.filter((s) => (up ? s > v + 1e-3 : s < v - 1e-3))
      next = ahead.length ? (up ? Math.min(...ahead) : Math.max(...ahead)) : v + (up ? 5 : -5)
    } else return null
    return round3(Math.max(lo, Math.min(hi, next)))
  }

  /** Slider semantics and keys for a handle at `v`; `onDelete` makes Delete / Backspace remove it. */
  const handleProps = (key: string, label: string, v: number, lo: number, hi: number, set: (x: number) => void, onDelete?: () => void) => ({
    'data-handle': key,
    role: 'slider' as const,
    tabIndex: 0,
    'aria-label': label,
    'aria-describedby': onDelete ? `${ids}-nudge ${ids}-del` : `${ids}-nudge`,
    'aria-valuemin': round3(lo),
    'aria-valuemax': round3(hi),
    'aria-valuenow': round3(v),
    'aria-valuetext': clockPrecise(v),
    onKeyDown: (e: React.KeyboardEvent) => {
      if (onDelete && (e.key === 'Delete' || e.key === 'Backspace')) onDelete()
      else {
        const next = keyTarget(e, v, lo, hi)
        if (next === null) return
        set(next)
      }
      e.preventDefault()
      e.stopPropagation()
    },
  })

  /** On the waveform (split, comp): Enter cuts at the playhead, or halves the longest part; Space plays. */
  const onTrackKey = (e: React.KeyboardEvent) => {
    if (e.target !== e.currentTarget || mode.kind === 'range') return
    if (e.key === 'Enter') {
      let sec = current && time != null ? time + timeBase : null
      if (sec === null) {
        const edges = mode.kind === 'cuts' ? [timeBase, ...mode.cuts, timeBase + total] : [mode.segments[0]?.start ?? timeBase, ...mode.segments.map((s) => s.end)]
        let best = 0
        for (let i = 1; i < edges.length - 1; i++) if (edges[i + 1] - edges[i] > edges[best + 1] - edges[best]) best = i
        sec = round3((edges[best] + edges[best + 1]) / 2)
      }
      const i = cutAt(sec)
      if (i !== null) refocus.current = mode.kind === 'cuts' ? `cut-${i}` : `b-${i}`
    } else if (e.key === ' ') regionPlay()
    else return
    e.preventDefault()
    e.stopPropagation()
  }

  const playX = time != null ? toX(time + timeBase) : null

  return (
    <div className="region">
      <span id={`${ids}-nudge`} className="sr-only">
        {t('region.keys.nudge')}
      </span>
      {mode.kind !== 'range' && (
        <>
          <span id={`${ids}-del`} className="sr-only">
            {t(mode.kind === 'cuts' ? 'region.keys.remove_cut' : 'region.keys.merge')}
          </span>
          <span id={`${ids}-track`} className="sr-only">
            {t(mode.kind === 'cuts' ? 'region.keys.add_cut' : 'region.keys.split')}
          </span>
        </>
      )}
      {mode.kind === 'segments' && (
        <span id={`${ids}-seg`} className="sr-only">
          {t('region.keys.segment')}
        </span>
      )}
      <div
        ref={wrap}
        className={clsx('region-track nodrag', drag && 'dragging')}
        style={{ height }}
        onPointerMove={onMove}
        onPointerUp={() => setDrag(null)}
        onPointerCancel={() => setDrag(null)}
        onClick={onBackgroundClick}
        {...(mode.kind !== 'range' ? { role: 'group', tabIndex: 0, 'aria-label': title ?? t('region.title'), 'aria-describedby': `${ids}-track`, onKeyDown: onTrackKey } : {})}
      >
        <canvas ref={canvas} />
        {mode.kind === 'range' && (
          <>
            <div className="region-dim" style={{ left: 0, width: Math.max(0, toX(mode.start)) }} />
            <div className="region-dim" style={{ left: toX(mode.end), right: 0 }} />
            <div className="region-sel" style={{ left: toX(mode.start), width: Math.max(2, toX(mode.end) - toX(mode.start)) }} onPointerDown={(e) => startDrag('move', e, [mode.start, mode.end])} />
            <div
              className="region-handle start"
              style={{ left: toX(mode.start) }}
              onPointerDown={(e) => startDrag('start', e, [mode.start, mode.end])}
              {...handleProps('start', t('region.start'), mode.start, timeBase, mode.end - MIN_LEN, (v) => mode.onChange(v, mode.end))}
            />
            <div
              className="region-handle end"
              style={{ left: toX(mode.end) }}
              onPointerDown={(e) => startDrag('end', e, [mode.start, mode.end])}
              {...handleProps('end', t('region.end'), mode.end, mode.start + MIN_LEN, timeBase + total, (v) => mode.onChange(mode.start, v))}
            />
          </>
        )}
        {mode.kind === 'cuts' &&
          mode.cuts.map((c, i) => (
            <div
              key={i}
              className="region-cut"
              style={{ left: toX(c) }}
              onPointerDown={(e) => startDrag(String(i), e, mode.cuts)}
              onDoubleClick={(e) => {
                e.stopPropagation()
                mode.onChange(mode.cuts.filter((_, j) => j !== i))
              }}
              {...handleProps(
                `cut-${i}`,
                t('region.cut', { n: i + 1 }),
                c,
                (mode.cuts[i - 1] ?? timeBase) + MIN_LEN,
                (mode.cuts[i + 1] ?? timeBase + total) - MIN_LEN,
                (v) => mode.onChange(mode.cuts.map((x, j) => (j === i ? v : x))),
                () => {
                  refocus.current = mode.cuts.length > 1 ? `cut-${Math.min(i, mode.cuts.length - 2)}` : 'track'
                  mode.onChange(mode.cuts.filter((_, j) => j !== i))
                },
              )}
            />
          ))}
        {mode.kind === 'segments' && (
          <>
            {mode.segments.map((s, i) => {
              const lane = mode.lanes[s.input]
              return (
                <div
                  key={i}
                  className={clsx('region-seg', s.input < 0 && 'silent')}
                  style={{ left: toX(s.start), width: Math.max(1, toX(s.end) - toX(s.start)), ['--seg' as string]: lane?.color ?? 'transparent' }}
                  title={lane?.label ?? t('region.silence')}
                  role="button"
                  tabIndex={0}
                  aria-label={`${lane?.label ?? t('region.silence')} · ${clockPrecise(s.start)} – ${clockPrecise(s.end)}`}
                  aria-describedby={`${ids}-seg`}
                  onClick={(e) => {
                    if (e.detail !== 1) return
                    e.stopPropagation()
                    cycleVoice(i)
                  }}
                  onKeyDown={(e) => {
                    if (e.key !== 'Enter' && e.key !== ' ') return
                    e.preventDefault()
                    e.stopPropagation()
                    cycleVoice(i)
                  }}
                >
                  <span className="region-seg-label">{lane?.label ?? '—'}</span>
                </div>
              )
            })}
            {mode.segments.slice(0, -1).map((s, i) => (
              <div
                key={`b${i}`}
                className="region-cut"
                style={{ left: toX(s.end) }}
                onPointerDown={(e) => startDrag(String(i), e, [])}
                onDoubleClick={(e) => {
                  e.stopPropagation()
                  mergeAt(i)
                }}
                {...handleProps(
                  `b-${i}`,
                  t('region.boundary', { a: laneName(mode.segments[i].input), b: laneName(mode.segments[i + 1].input) }),
                  s.end,
                  s.start + MIN_LEN,
                  mode.segments[i + 1].end - MIN_LEN,
                  (v) => mode.onChange(mode.segments.map((x, j) => (j === i ? { ...x, end: v } : j === i + 1 ? { ...x, start: v } : x))),
                  () => {
                    refocus.current = mode.segments.length > 2 ? `b-${Math.min(i, mode.segments.length - 3)}` : 'track'
                    mergeAt(i)
                  },
                )}
              />
            ))}
          </>
        )}
        {playX != null && <div className="region-playhead" style={{ left: playX }} />}
      </div>
      <div className="region-bar">
        <button className="icon-btn sm" onClick={regionPlay} aria-label={playing ? t('common.pause') : t('common.play')}>
          {playing ? <Pause /> : <Play />}
        </button>
        {mode.kind === 'range' && (
          <Tip label={t('region.loop')}>
            <button className={clsx('icon-btn sm', loop && current && 'on')} onClick={loopRegion} aria-label={t('region.loop')} aria-pressed={!!(loop && current)}>
              <Repeat />
            </button>
          </Tip>
        )}
        <span className="tabular muted region-time">
          {mode.kind === 'range' ? `${clockPrecise(mode.start)} – ${clockPrecise(mode.end)} · ${clockPrecise(mode.end - mode.start)}` : mode.kind === 'cuts' ? t('region.parts', { n: mode.cuts.length + 1 }) : t('region.segments', { n: mode.segments.length })}
        </span>
        <span className="spacer" />
        {onExpand && (
          <Tip label={t('region.expand')}>
            <button className="icon-btn sm" onClick={onExpand} aria-label={t('region.expand')}>
              <Maximize2 />
            </button>
          </Tip>
        )}
      </div>
    </div>
  )
}
