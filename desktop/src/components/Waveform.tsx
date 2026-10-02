import clsx from 'clsx'
import { Pause, Play } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { api, mediaUrl } from '../api/client'
import { useT } from '../i18n'
import { cssVar, waveColor } from '../lib/color'
import { usePlayback, usePlayer } from '../state/player'
import { useTheme } from '../state/theme'
import './waveform.css'

const peaksCache = new Map<string, Promise<{ peaks: number[]; duration: number }>>()

export function loadPeaks(path: string, n: number) {
  const key = `${path}?n=${n}`
  let p = peaksCache.get(key)
  if (!p) {
    p = api.get<{ peaks: number[]; duration: number }>(path, { n })
    p.catch(() => peaksCache.delete(key))
    peaksCache.set(key, p)
  }
  return p
}

export function invalidatePeaks(prefix: string) {
  for (const k of peaksCache.keys()) if (k.startsWith(prefix)) peaksCache.delete(k)
}

/** Lazily loads peaks when scrolled into view, draws mirrored bars, shows the shared playhead. */
export function Waveform({
  id,
  peaksPath,
  audioPath,
  height = 40,
  color,
  bars = 160,
  interactive = true,
  showButton = false,
  className,
}: {
  id: string
  peaksPath: string
  audioPath: string
  height?: number
  color?: string
  bars?: number
  interactive?: boolean
  showButton?: boolean
  className?: string
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [peaks, setPeaks] = useState<number[] | null>(null)
  const [duration, setDuration] = useState(0)
  const [visible, setVisible] = useState(false)
  const { playing, progress } = usePlayback(id)
  const toggle = usePlayer((s) => s.toggle)
  const play = usePlayer((s) => s.play)
  const seek = usePlayer((s) => s.seek)
  const theme = useTheme((s) => s.resolved)
  const t = useT()

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const io = new IntersectionObserver((entries) => entries.some((e) => e.isIntersecting) && setVisible(true), { rootMargin: '200px' })
    io.observe(el)
    return () => io.disconnect()
  }, [])

  useEffect(() => {
    if (!visible) return
    let cancelled = false
    loadPeaks(peaksPath, bars)
      .then((r) => {
        if (cancelled) return
        setPeaks(r.peaks)
        setDuration(r.duration)
      })
      .catch(() => !cancelled && setPeaks([]))
    return () => {
      cancelled = true
    }
  }, [visible, peaksPath, bars])

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap || !peaks) return
    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2)
      const w = wrap.clientWidth - (showButton ? 34 : 0)
      canvas.width = Math.max(1, Math.round(w * dpr))
      canvas.height = Math.round(height * dpr)
      canvas.style.width = `${w}px`
      canvas.style.height = `${height}px`
      const ctx = canvas.getContext('2d')!
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, height)
      if (peaks.length === 0) return
      const gap = 1.5
      const bw = Math.max(1, w / peaks.length - gap)
      const mid = height / 2
      const played = progress ?? -1
      const ink = waveColor(color)
      const rest = cssVar('--wave-rest')
      peaks.forEach((p, i) => {
        const x = (i * w) / peaks.length
        const hh = Math.max(1, p * (height - 2) * 0.5)
        ctx.fillStyle = i / peaks.length <= played ? ink : rest
        const r = Math.min(bw / 2, 1.5)
        ctx.beginPath()
        ctx.roundRect(x, mid - hh, bw, hh * 2, r)
        ctx.fill()
      })
    }
    draw()
    const ro = new ResizeObserver(draw)
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [peaks, height, color, progress, showButton, theme])

  const url = () => mediaUrl(audioPath)

  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!interactive) return
    const rect = e.currentTarget.getBoundingClientRect()
    const frac = (e.clientX - rect.left) / rect.width
    const st = usePlayer.getState()
    if (st.currentId === id && st.duration > 0) {
      seek(frac * st.duration)
      if (!st.playing) play(id, url())
    } else {
      play(id, url(), frac * duration)
    }
  }

  return (
    <div className={clsx('waveform', className)} ref={wrapRef} style={{ height }}>
      {showButton && (
        <button
          className={clsx('wave-play', playing && 'playing')}
          onClick={(e) => {
            e.stopPropagation()
            toggle(id, url())
          }}
          aria-label={playing ? t('common.pause') : t('common.play')}
        >
          {playing ? <Pause /> : <Play />}
        </button>
      )}
      {peaks ? (
        <canvas ref={canvasRef} onClick={onClick} className={clsx(interactive && 'clickable')} />
      ) : (
        <div className="skeleton" style={{ flex: 1, height: Math.max(6, height * 0.4) }} />
      )}
    </div>
  )
}

/** Small round play button for lists. */
export function PlayButton({ id, audioPath, size = 28, className }: { id: string; audioPath: string; size?: number; className?: string }) {
  const t = useT()
  const { playing, progress } = usePlayback(id)
  const toggle = usePlayer((s) => s.toggle)
  return (
    <button
      className={clsx('play-btn', playing && 'playing', className)}
      style={{ width: size, height: size, ['--p' as string]: progress ?? 0 }}
      onClick={(e) => {
        e.stopPropagation()
        toggle(id, mediaUrl(audioPath))
      }}
      aria-label={playing ? t('common.pause') : t('common.play')}
    >
      {playing ? <Pause /> : <Play />}
    </button>
  )
}
