import clsx from 'clsx'
import { useEffect, useRef } from 'react'
import type { TimelineLoop } from '../../../api/types'
import { useT } from '../../../i18n'
import { useLatest } from '../../../lib/useEvent'
import { useTheme } from '../../../state/theme'
import type { View } from './ClipView'
import type { Grid } from './ops'
import { LOOP_H, palette, RULER_H, useTimeline, type Palette, type Signal } from './state'

const pad2 = (n: number) => String(n).padStart(2, '0')

/** A time for the ruler: "1:05", "0:07.5" (fine steps); bars mode "12", "12.3" (bar.beat). */
export function rulerLabel(t: number, grid: Grid): string {
  if (grid.bar && grid.beat) {
    const beats = Math.round(t / grid.beat)
    const bar = Math.floor(beats / 4) + 1
    const beat = (beats % 4) + 1
    return grid.major >= grid.bar - 1e-9 ? String(bar) : `${bar}.${beat}`
  }
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return grid.major >= 1 ? `${m}:${pad2(Math.round(s))}` : `${m}:${s.toFixed(grid.major >= 0.1 ? 1 : 2).padStart(grid.major >= 0.1 ? 4 : 5, '0')}`
}

/** Clock for the transport: "1:05.32"; bars mode adds "12.3.2" (bar.beat.sixteenth). */
export function clock(t: number): string {
  const m = Math.floor(t / 60)
  const s = t - m * 60
  return `${m}:${s.toFixed(2).padStart(5, '0')}`
}

export function barClock(t: number, bpm: number): string {
  const sixteenths = Math.floor((t * bpm * 4) / 60 + 1e-6)
  return `${Math.floor(sixteenths / 16) + 1}.${(Math.floor(sixteenths / 4) % 4) + 1}.${(sixteenths % 4) + 1}`
}

/** Grid steps visible in `view`: each line's time and whether it is a labelled (major) one. */
function* lines(view: View, pps: number, grid: Grid) {
  const first = Math.floor(view.left / pps / grid.minor)
  const last = Math.ceil((view.left + view.width) / pps / grid.minor)
  for (let i = Math.max(0, first); i <= last; i++) {
    const t = i * grid.minor
    const k = t / grid.major
    const bar = grid.bar ? Math.abs(t / grid.bar - Math.round(t / grid.bar)) < 1e-6 : false
    yield { t, x: t * pps - view.left, major: Math.abs(k - Math.round(k)) < 1e-6, bar }
  }
}

/**
 * A canvas over the visible part of the lanes, redrawn as they scroll (and when `key` changes);
 * `height` null: as tall as the visible tracks, following vertical scrolling too.
 */
function useViewCanvas(view: Signal<View>, height: number | null, draw: (ctx: CanvasRenderingContext2D, v: View, h: number, colors: Palette) => void, key: string) {
  const ref = useRef<HTMLCanvasElement>(null)
  const theme = useTheme((s) => s.resolved)
  const drawLatest = useLatest(draw)
  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    let raf = 0
    const colors = palette(theme)
    const paint = () => {
      raf = 0
      const v = view.value
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      const w = Math.max(1, Math.ceil(v.width))
      const h = Math.max(1, Math.ceil(height ?? v.height))
      canvas.width = Math.round(w * dpr)
      canvas.height = Math.round(h * dpr)
      canvas.style.left = `${v.left}px`
      canvas.style.top = height === null ? `${v.top}px` : '0'
      canvas.style.width = `${w}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)
      drawLatest.current(ctx, v, h, colors)
    }
    paint()
    const off = view.subscribe(() => {
      if (!raf) raf = requestAnimationFrame(paint)
    })
    return () => {
      off()
      cancelAnimationFrame(raf)
    }
  }, [view, height, theme, key, drawLatest])
  return ref
}

/** Faint lines behind the clips: the snap grid, stronger on labelled steps (bars). */
export function GridLines({ view, pps, grid }: { view: Signal<View>; pps: number; grid: Grid }) {
  const ref = useViewCanvas(
    view,
    null,
    (ctx, v, height, colors) => {
      const minor = colors.grid
      const major = colors.axis
      if (grid.minor * pps < 6) return
      for (const l of lines(v, pps, grid)) {
        ctx.fillStyle = l.major || l.bar ? major : minor
        ctx.fillRect(Math.round(l.x), 0, 1, height)
      }
    },
    `${pps}|${grid.minor}|${grid.major}|${grid.bar}`,
  )
  return <canvas ref={ref} className="tl-gridlines" aria-hidden />
}

/**
 * The ruler: click or drag in the lower part to move the playhead; drag in the top band to set
 * the loop range (drag its ends or its middle to change it, click it to switch looping).
 */
export function Ruler({
  view,
  pps,
  grid,
  width,
  length,
  loop,
  onPointerDown,
}: {
  view: Signal<View>
  pps: number
  grid: Grid
  width: number
  length: number
  loop: TimelineLoop | null | undefined
  onPointerDown: (e: React.PointerEvent) => void
}) {
  const t = useT()
  const { pos } = useTimeline()
  const marker = useRef<HTMLSpanElement>(null)
  const root = useRef<HTMLDivElement>(null)
  const ref = useViewCanvas(
    view,
    RULER_H,
    (ctx, v, _h, colors) => {
      const tick = colors.axis
      ctx.font = `500 10.5px ${colors.font}`
      ctx.textBaseline = 'alphabetic'
      for (const l of lines(v, pps, grid)) {
        const x = Math.round(l.x)
        if (l.major) {
          ctx.fillStyle = tick
          ctx.fillRect(x, LOOP_H + 1, 1, RULER_H - LOOP_H - 1)
          ctx.fillStyle = colors.text3
          ctx.fillText(rulerLabel(l.t, grid), x + 4, RULER_H - 6)
        } else {
          ctx.fillStyle = tick
          ctx.fillRect(x, RULER_H - (l.bar ? 8 : 5), 1, l.bar ? 8 : 5)
        }
      }
    },
    `${pps}|${grid.minor}|${grid.major}|${grid.bar}`,
  )

  useEffect(() => {
    let said = -1
    const move = (p: number) => {
      if (marker.current) marker.current.style.transform = `translateX(${p * pps}px)`
      // the slider's value for screen readers, a few times a second at most
      const tenth = Math.round(p * 10) / 10
      if (root.current && Math.abs(tenth - said) >= 0.5) {
        said = tenth
        root.current.setAttribute('aria-valuenow', String(tenth))
        root.current.setAttribute('aria-valuetext', clock(p))
      }
    }
    move(pos.value)
    return pos.subscribe(move)
  }, [pos, pps])

  return (
    // the arrangement's keys move it (arrows, Home / End); see tl.keys.timeline
    <div
      ref={root}
      className="tl-ruler"
      style={{ width }}
      onPointerDown={onPointerDown}
      role="slider"
      tabIndex={0}
      aria-label={t('tl.playhead')}
      aria-describedby="tl-keys"
      aria-valuemin={0}
      aria-valuemax={Math.round(length * 10) / 10}
    >
      <div className="tl-loopband" title={t('tl.loop_hint')}>
        {loop && loop.end > loop.start && (
          <span className={clsx('tl-loop-range', loop.on !== false && 'on')} style={{ left: loop.start * pps, width: (loop.end - loop.start) * pps }}>
            <span className="tl-loop-edge l" />
            <span className="tl-loop-edge r" />
          </span>
        )}
      </div>
      <canvas ref={ref} className="tl-ruler-ticks" aria-hidden />
      <span className="tl-ruler-marker" ref={marker} aria-hidden />
    </div>
  )
}
