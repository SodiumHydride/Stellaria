import clsx from 'clsx'
import type { CSSProperties } from 'react'
import './logo.css'

// 旋律星: one phrase of melody whose highest note is a star. The same geometry as the app icon
// (src-tauri/icons/icon.svg); the viewBox is shifted so the mark sits optically centred.
const MELODY = 'M9 62C16 62 19 47 26 47S36 75 45 75S56 37 65 37S78 62 91 62'
const STAR = 'M0 -12C0.8 -3.13 3.13 -0.8 12 0C3.13 0.8 0.8 3.13 0 12C-0.8 3.13 -3.13 0.8 -12 0C-3.13 -0.8 -0.8 -3.13 0 -12Z'
// small sizes: a thicker line and a larger star, as in the tray icon
const STAR_SMALL = 'M0 -15C1 -3.91 3.91 -1 15 0C3.91 1 1 3.91 0 15C-1 3.91 -3.91 1 -15 0C-3.91 -1 -1 -3.91 0 -15Z'

/**
 * The brand mark. `running` means Stellaria is working on something: a light in the star's
 * colour travels along the melody and the star twinkles (still under reduced motion and low power).
 */
export function Logo({ size = 20, running = false, className }: { size?: number; running?: boolean; className?: string }) {
  const small = size <= 24
  const width = small ? 10 : 7.5
  return (
    <svg className={clsx('logo', running && 'running', className)} width={size} height={size} viewBox="0 -8 100 100" fill="none" aria-hidden>
      <path className="logo-line" d={MELODY} stroke="currentColor" strokeWidth={width} strokeLinecap="round" strokeLinejoin="round" />
      {running && <path className="logo-glint" d={MELODY} pathLength={100} strokeWidth={width} strokeLinecap="round" strokeLinejoin="round" />}
      <g transform={`translate(65 ${small ? 14 : 17})`}>
        <path className="logo-star" d={small ? STAR_SMALL : STAR} />
      </g>
    </svg>
  )
}

/** The app tile (paper, or ink in the dark theme) for the boot screen, the welcome and About. */
export function AppTile({ size = 84, running = false }: { size?: number; running?: boolean }) {
  return (
    <span className="app-tile" style={{ '--s': `${size}px` } as CSSProperties}>
      <Logo size={Math.round(size * 0.74)} running={running} />
    </span>
  )
}

// 声波星: level bars standing in a four-pointed star; while mixing or exporting they move like a
// level meter. Fewer, wider bars at icon sizes so each stays a pixel or more wide.
interface Bar {
  x: number
  h: number
  w: number
  style: CSSProperties
}

function bars(n: number, w: number, gap: number, floor: number, peak: number, curve: number): Bar[] {
  const half = (n - 1) / 2
  const x0 = (100 - (n * w + (n - 1) * gap)) / 2 + w / 2
  return Array.from({ length: n }, (_, i) => {
    const u = (i - half) / half
    const h = floor + peak * (1 - Math.abs(u)) ** curve
    // a fixed pseudo-random spread so the meter never moves in lockstep
    const r = (k: number) => (Math.sin((i + 1) * 12.9898 * k) * 43758.5453) % 1
    const near = 0.55 + 0.45 * (1 - Math.abs(u))
    const lo = 12 + 10 * Math.abs(r(1))
    const hi = (40 + 46 * Math.abs(r(2))) * near + 8
    return {
      x: x0 + i * (w + gap),
      h,
      w,
      style: {
        '--lo': (Math.min(lo, h) / h).toFixed(3),
        '--hi': (Math.max(hi, h) / h).toFixed(3),
        '--d': `${(0.42 + 0.36 * Math.abs(r(3))).toFixed(2)}s`,
        '--delay': `${(-Math.abs(r(4)) * 0.8).toFixed(2)}s`,
      } as CSSProperties,
    }
  })
}

const BARS_LARGE = bars(11, 5.4, 2.4, 6, 76, 2.6)
const BARS_SMALL = bars(7, 8.6, 4.6, 10, 76, 2.2)

export function LevelMark({ size = 16, running = false, className }: { size?: number; running?: boolean; className?: string }) {
  const list = size <= 24 ? BARS_SMALL : BARS_LARGE
  return (
    <svg className={clsx('level-mark', running && 'running', className)} width={size} height={size} viewBox="0 0 100 100" fill="none" stroke="currentColor" strokeLinecap="round" aria-hidden>
      {list.map((b) => (
        <line key={b.x} x1={b.x} x2={b.x} y1={50 - b.h / 2 + b.w / 2} y2={50 + b.h / 2 - b.w / 2} strokeWidth={b.w} style={b.style} />
      ))}
    </svg>
  )
}
