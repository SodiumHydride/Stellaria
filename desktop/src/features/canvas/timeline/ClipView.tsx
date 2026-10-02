import clsx from 'clsx'
import { VolumeX } from 'lucide-react'
import { memo, useEffect, useRef, useState } from 'react'
import { api } from '../../../api/client'
import type { TimelineClip } from '../../../api/types'
import { useT } from '../../../i18n'
import { waveColor } from '../../../lib/color'
import { useTheme } from '../../../state/theme'
import { mediaPeaksPath } from '../context'
import { clamp, dbToGain, fmtDb, moveClips, setClipGain, setFade, trimClips } from './ops'
import { CLIP_INSET, palette, TRACK_H, useTimeline, type Signal } from './state'

export const CLIP_H = TRACK_H - CLIP_INSET * 2
/** the clip's name strip; fade and gain handles sit in it */
const HEAD_H = 15

type Peaks = { peaks: number[]; duration: number }
const peakCache = new Map<string, Promise<Peaks | null>>()

/** Peaks of a media file at ~100 per second (the backend's limit is 20000), shared by every clip. */
function loadPeaks(projectId: string, mediaId: string, duration: number | undefined): Promise<Peaks | null> {
  const key = `${projectId}/${mediaId}`
  let p = peakCache.get(key)
  if (!p) {
    const n = clamp(Math.ceil((duration ?? 60) * 100), 256, 20000)
    p = api.get<Peaks>(mediaPeaksPath(projectId, mediaId), { n }).catch(() => {
      peakCache.delete(key)
      return null
    })
    peakCache.set(key, p)
  }
  return p
}

export interface ClipLook {
  label: string
  color: string
}

/** visible part of the lanes, in lane pixels (top / height: of the tracks' area) */
export type View = { left: number; width: number; top: number; height: number }

export const fmtSec = (s: number) => `${s < 10 ? s.toFixed(2) : s.toFixed(1)} s`

/** One clip: waveform (shaped by its fades and gain), name, trim edges, fade and gain handles. */
export const ClipView = memo(function ClipView({
  clip,
  pps,
  selected,
  look,
  view,
  projectId,
  mediaDuration,
  selection,
  keysId,
}: {
  clip: TimelineClip
  pps: number
  selected: boolean
  look: ClipLook
  view: Signal<View>
  projectId: string
  mediaDuration: number | undefined
  /** the selection when this clip is part of it (keyboard edits move all of it) */
  selection: string[] | null
  keysId: string
}) {
  const t = useT()
  const { api: tl } = useTimeline()
  const left = clip.start * pps
  const width = Math.max(4, clip.duration * pps)
  const ids = selection ?? [clip.id]

  const onKey = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 1 : e.altKey ? 0.01 : 0.1
    const trim = e.altKey ? 0.01 : 0.1
    if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') tl.edit((x) => moveClips(x, ids, e.key === 'ArrowLeft' ? -step : step, 0).tl, 'nudge')
    else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
      tl.edit((x) => moveClips(x, ids, 0, e.key === 'ArrowUp' ? -1 : 1).tl, 'nudge')
      // on another track it is a new element: keep the focus on it
      requestAnimationFrame(() => document.querySelector<HTMLElement>(`.tl-clip[data-clip="${CSS.escape(clip.id)}"]`)?.focus({ preventScroll: true }))
    }
    else if (e.key === 'Home') tl.edit((x) => moveClips(x, [clip.id], -clip.start, 0).tl)
    // [ / ] take in the start / end; with Shift they let it out again
    else if (e.code === 'BracketLeft') tl.edit((x) => trimClips(x, [clip.id], 'left', e.shiftKey ? -trim : trim, () => mediaDuration), 'trim')
    else if (e.code === 'BracketRight') tl.edit((x) => trimClips(x, [clip.id], 'right', e.shiftKey ? trim : -trim, () => mediaDuration), 'trim')
    else if (e.key === 'Enter') tl.seek(clip.start)
    else return
    e.preventDefault()
    e.stopPropagation()
  }

  const showHandles = width >= 44
  return (
    <div
      data-clip={clip.id}
      className={clsx('tl-clip', selected && 'selected', clip.mute && 'muted')}
      style={{ left, width, ['--clip' as string]: look.color }}
      role="slider"
      tabIndex={0}
      aria-label={look.label || t('tl.clip')}
      aria-describedby={keysId}
      aria-valuemin={0}
      aria-valuenow={Math.round(clip.start * 1000) / 1000}
      aria-valuetext={`${fmtSec(clip.start)} – ${fmtSec(clip.start + clip.duration)}${clip.mute ? ` · ${t('tl.clip_muted')}` : ''}`}
      onKeyDown={onKey}
      onFocus={() => !selected && tl.select([clip.id])}
    >
      <ClipWave projectId={projectId} clip={clip} pps={pps} left={left} color={look.color} view={view} mediaDuration={mediaDuration} />
      <div className="tl-clip-head">
        {clip.mute && <VolumeX className="tl-clip-icon" />}
        <span className="tl-clip-label">{look.label}</span>
        {clip.gainDb !== 0 && width >= 90 && <span className="tl-clip-gain-val tabular">{fmtDb(clip.gainDb)} dB</span>}
      </div>
      <span className="tl-edge l" data-handle="trim-l" aria-hidden />
      <span className="tl-edge r" data-handle="trim-r" aria-hidden />
      {showHandles && (
        <>
          <span
            className="tl-fade-h in"
            data-handle="fade-in"
            style={{ left: Math.min(clip.fadeIn * pps, width - 12) }}
            title={`${t('tl.fade_in')} ${fmtSec(clip.fadeIn)} · ${t('tl.dblclick_reset')}`}
            aria-hidden
            onDoubleClick={() => tl.edit((x) => setFade(x, clip.id, 'in', 0))}
          />
          <span
            className="tl-fade-h out"
            data-handle="fade-out"
            style={{ right: Math.min(clip.fadeOut * pps, width - 12) }}
            title={`${t('tl.fade_out')} ${fmtSec(clip.fadeOut)} · ${t('tl.dblclick_reset')}`}
            aria-hidden
            onDoubleClick={() => tl.edit((x) => setFade(x, clip.id, 'out', 0))}
          />
          <span
            className="tl-gain-h"
            data-handle="gain"
            title={`${t('tl.clip_gain')} ${fmtDb(clip.gainDb)} dB · ${t('tl.gain_drag')}`}
            aria-hidden
            onDoubleClick={() => tl.edit((x) => setClipGain(x, ids.includes(clip.id) ? ids : [clip.id], 0))}
          />
        </>
      )}
    </div>
  )
})

/**
 * The waveform, drawn only around the visible part of the clip (long clips at high zoom would
 * not fit one canvas), redrawn when scrolling uncovers more. Its height follows the clip's gain
 * and fades; the fades' curves are drawn over it.
 */
function ClipWave({ projectId, clip, pps, left, color, view, mediaDuration }: { projectId: string; clip: TimelineClip; pps: number; left: number; color: string; view: Signal<View>; mediaDuration: number | undefined }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const [peaks, setPeaks] = useState<Peaks | null>(null)
  const theme = useTheme((s) => s.resolved)
  const { mediaId, offset, duration, fadeIn, fadeOut, gainDb, mute } = clip

  useEffect(() => {
    let live = true
    void loadPeaks(projectId, mediaId, mediaDuration).then((p) => live && setPeaks(p))
    return () => {
      live = false
    }
  }, [projectId, mediaId, mediaDuration])

  useEffect(() => {
    const canvas = ref.current
    if (!canvas) return
    const w = duration * pps
    const h = CLIP_H - 2
    let drawn: { a: number; b: number } | null = null
    const colors = palette(theme)
    const ink = colors.ink
    const fill = mute ? colors.text4 : waveColor(color.startsWith('#') ? color : colors.wave)
    const line = colors.text1
    const draw = (force: boolean) => {
      const v = view.value
      const visA = clamp(v.left - left, 0, w)
      const visB = clamp(v.left + v.width - left, 0, w)
      if (visB - visA < 0.5) {
        canvas.style.display = 'none'
        drawn = null
        return
      }
      if (!force && drawn && drawn.a <= visA && drawn.b >= visB) return
      const pad = Math.max(200, v.width * 0.5)
      const a = Math.floor(Math.max(0, visA - pad))
      const b = Math.ceil(Math.min(w, visB + pad))
      const cw = Math.max(1, b - a)
      const dpr = Math.min(2, window.devicePixelRatio || 1)
      canvas.width = Math.round(cw * dpr)
      canvas.height = Math.round(h * dpr)
      canvas.style.display = ''
      canvas.style.left = `${a}px`
      canvas.style.width = `${cw}px`
      canvas.style.height = `${h}px`
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, cw, h)
      drawn = { a, b }
      const level = dbToGain(gainDb)
      const env = (x: number) => {
        const s = x / pps
        return (fadeIn > 0 ? clamp(s / fadeIn, 0, 1) : 1) * (fadeOut > 0 ? clamp((duration - s) / fadeOut, 0, 1) : 1)
      }
      const top = HEAD_H
      const mid = top + (h - top) / 2
      const half = (h - top) / 2 - 1
      if (peaks && peaks.duration > 0 && peaks.peaks.length) {
        const n = peaks.peaks.length
        const per = n / peaks.duration
        const amp: number[] = []
        for (let x = a; x < b; x++) {
          const t0 = offset + x / pps
          const i0 = Math.floor(t0 * per)
          const i1 = Math.max(i0 + 1, Math.floor((t0 + 1 / pps) * per))
          let m = 0
          for (let i = Math.max(0, i0); i < Math.min(n, i1); i++) if (peaks.peaks[i] > m) m = peaks.peaks[i]
          amp.push(Math.min(1, m * level) * env(x + 0.5) * half)
        }
        ctx.fillStyle = fill
        ctx.globalAlpha = 0.85
        ctx.beginPath()
        ctx.moveTo(0, mid)
        amp.forEach((v, i) => ctx.lineTo(i + 0.5, mid - Math.max(0.5, v)))
        for (let i = amp.length - 1; i >= 0; i--) ctx.lineTo(i + 0.5, mid + Math.max(0.5, amp[i]))
        ctx.closePath()
        ctx.fill()
        ctx.globalAlpha = 1
      }
      // fades: what they take away is shaded, the curve drawn on top
      ctx.lineWidth = 1.25
      ctx.strokeStyle = line
      ctx.fillStyle = `rgb(${ink} / 0.13)`
      const fi = fadeIn * pps - a
      if (fadeIn > 0 && fi > 0) {
        ctx.beginPath()
        ctx.moveTo(-a, 0)
        ctx.lineTo(fi, 0)
        ctx.lineTo(-a, h)
        ctx.closePath()
        ctx.fill()
        ctx.beginPath()
        ctx.moveTo(-a, h)
        ctx.lineTo(fi, 0)
        ctx.stroke()
      }
      const fo = (duration - fadeOut) * pps - a
      if (fadeOut > 0 && fo < cw) {
        ctx.beginPath()
        ctx.moveTo(w - a, 0)
        ctx.lineTo(fo, 0)
        ctx.lineTo(w - a, h)
        ctx.closePath()
        ctx.fill()
        ctx.beginPath()
        ctx.moveTo(fo, 0)
        ctx.lineTo(w - a, h)
        ctx.stroke()
      }
    }
    draw(true)
    return view.subscribe(() => draw(false))
  }, [peaks, offset, duration, fadeIn, fadeOut, gainDb, mute, pps, left, color, view, theme])

  return <canvas ref={ref} className="tl-clip-wave" />
}
