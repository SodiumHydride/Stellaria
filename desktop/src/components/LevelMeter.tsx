import { useEffect, useRef } from 'react'
import { readLevels, usePlayer } from '../state/player'

const FLOOR_DB = -48

/** Position of a linear level on a -48…0 dB scale. */
function scale(v: number): number {
  if (v <= 0) return 0
  return Math.min(1, Math.max(0, (20 * Math.log10(v) - FLOOR_DB) / -FLOOR_DB))
}

/**
 * Left/right peak meter of whatever the app is playing. Bars fall back smoothly and hold their
 * peak briefly; the colour warns near full scale.
 */
export function LevelMeter({ width = 72 }: { width?: number }) {
  const playing = usePlayer((s) => s.playing)
  const bars = useRef<(HTMLSpanElement | null)[]>([])
  const holds = useRef<(HTMLSpanElement | null)[]>([])

  useEffect(() => {
    let raf = 0
    const shown = [0, 0]
    const held = [0, 0]
    const heldAt = [0, 0]
    const frame = (now: number) => {
      const levels = playing ? readLevels() : null
      for (let ch = 0; ch < 2; ch++) {
        const target = scale(levels?.[ch] ?? 0)
        shown[ch] = target > shown[ch] ? target : shown[ch] * 0.9
        if (shown[ch] >= held[ch] || now - heldAt[ch] > 900) {
          held[ch] = shown[ch]
          heldAt[ch] = now
        }
        const bar = bars.current[ch]
        const hold = holds.current[ch]
        if (bar) {
          bar.style.transform = `scaleX(${shown[ch]})`
          bar.dataset.level = shown[ch] > 0.985 ? 'clip' : shown[ch] > 0.875 ? 'hot' : 'ok'
        }
        if (hold) hold.style.left = `calc(${held[ch] * 100}% - 1px)`
      }
      if (playing || shown[0] > 0.002 || shown[1] > 0.002) raf = requestAnimationFrame(frame)
    }
    raf = requestAnimationFrame(frame)
    return () => cancelAnimationFrame(raf)
  }, [playing])

  return (
    <span className="level-meter" style={{ width }} aria-hidden>
      {[0, 1].map((ch) => (
        <span key={ch} className="level-meter-ch">
          <span className="level-meter-bar" ref={(el) => void (bars.current[ch] = el)} />
          <span className="level-meter-hold" ref={(el) => void (holds.current[ch] = el)} />
        </span>
      ))}
    </span>
  )
}
