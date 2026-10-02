import clsx from 'clsx'
import { useEffect, useRef } from 'react'
import { useT } from '../../../i18n'
import { faderPos } from './ops'
import { useTimeline } from './state'

/** linear level → meter travel, on the faders' dB scale (0 dBFS where unity sits) */
const travel = (v: number) => (v <= 0 ? 0 : faderPos(20 * Math.log10(v)))
const UNITY = travel(1)
const HOT = travel(Math.pow(10, -6 / 20))
const HOLD_MS = 1200

/**
 * Stereo peak / RMS meter of a track (its id) or the master ('master'), post-fader. It reads the
 * engine on the timeline's frame clock and moves its bars directly, without React renders; the
 * peak holds a moment, and the clip light stays on until clicked.
 */
export function Meter({ id, className, compact }: { id: string; className?: string; compact?: boolean }) {
  const t = useT()
  const { engine, frame } = useTimeline()
  const root = useRef<HTMLDivElement>(null)
  const peaks = useRef<(HTMLSpanElement | null)[]>([])
  const rms = useRef<(HTMLSpanElement | null)[]>([])
  const holds = useRef<(HTMLSpanElement | null)[]>([])

  useEffect(() => {
    const shown = [0, 0]
    const shownRms = [0, 0]
    const held = [0, 0]
    const heldAt = [0, 0]
    return frame.subscribe((now) => {
      const lv = engine.playing ? engine.levels(id) : null
      for (let ch = 0; ch < 2; ch++) {
        const p = travel(lv?.peak[ch] ?? 0)
        const r = travel(lv?.rms[ch] ?? 0)
        shown[ch] = p > shown[ch] ? p : Math.max(p, shown[ch] - 0.012)
        shownRms[ch] = r > shownRms[ch] ? r : Math.max(r, shownRms[ch] - 0.008)
        if (shown[ch] >= held[ch] || now - heldAt[ch] > HOLD_MS) {
          held[ch] = shown[ch]
          heldAt[ch] = now
        }
        const bar = peaks.current[ch]
        if (bar) {
          bar.style.transform = `scaleY(${shown[ch]})`
          bar.dataset.level = shown[ch] >= UNITY - 0.002 ? 'clip' : shown[ch] > HOT ? 'hot' : 'ok'
        }
        const rbar = rms.current[ch]
        if (rbar) rbar.style.transform = `scaleY(${shownRms[ch]})`
        const hold = holds.current[ch]
        if (hold) {
          hold.style.bottom = `calc(${held[ch] * 100}% - 1px)`
          hold.style.opacity = held[ch] > 0.01 ? '1' : '0'
        }
        if (!compact && (lv?.peak[ch] ?? 0) >= 0.999 && root.current) root.current.dataset.clipped = 'true'
      }
    })
  }, [engine, frame, id, compact])

  return (
    <div className={clsx('tl-meter', compact && 'compact', className)} ref={root} aria-hidden={compact || undefined}>
      {[0, 1].map((ch) => (
        <span key={ch} className="tl-meter-ch">
          <span className="tl-meter-peak" ref={(el) => void (peaks.current[ch] = el)} />
          <span className="tl-meter-rms" ref={(el) => void (rms.current[ch] = el)} />
          <span className="tl-meter-hold" ref={(el) => void (holds.current[ch] = el)} />
          <span className="tl-meter-unity" style={{ bottom: `${UNITY * 100}%` }} />
        </span>
      ))}
      {!compact && (
        <button className="tl-meter-clip" onClick={() => root.current && delete root.current.dataset.clipped} title={t('tl.clip_light')} aria-label={t('tl.clip_light')} />
      )}
    </div>
  )
}
