import clsx from 'clsx'
import { useRef, useState } from 'react'
import type { Track } from '../../../api/types'
import { Slider } from '../../../components/ui'
import { useT } from '../../../i18n'
import { clamp, faderDb, faderPos, fmtDb, MAX_DB, parseDb, SILENT_DB } from './ops'

/** "中" / "C", "左 30" / "L30", "右 30" / "R30" for a pan of −1…1 */
export function usePanText() {
  const t = useT()
  return (pan: number) => {
    const v = Math.round(pan * 100)
    return v === 0 ? t('tl.pan_center') : v < 0 ? t('tl.pan_left', { n: -v }) : t('tl.pan_right', { n: v })
  }
}

export function MuteSolo({ track, onPatch }: { track: Track; onPatch: (p: Partial<Track>) => void }) {
  const t = useT()
  return (
    <>
      <button className={clsx('tl-ms', track.mute && 'on')} onClick={() => onPatch({ mute: !track.mute })} title={`${t('tl.mute')} (${track.name})`} aria-label={`${t('tl.mute')} ${track.name}`} aria-pressed={track.mute}>
        M
      </button>
      <button className={clsx('tl-ms solo', track.solo && 'on')} onClick={() => onPatch({ solo: !track.solo })} title={`${t('tl.solo')} (${track.name})`} aria-label={`${t('tl.solo')} ${track.name}`} aria-pressed={track.solo}>
        S
      </button>
    </>
  )
}

/**
 * A slider that previews while dragged and commits once when let go (one undo step); double-click
 * puts it back to `reset`.
 */
export function LiveSlider({
  value,
  min,
  max,
  step,
  reset,
  label,
  centerMark,
  onLive,
  onCommit,
  onReset,
}: {
  value: number
  min: number
  max: number
  step: number
  reset: number
  label: string
  centerMark?: boolean
  onLive: (v: number) => void
  onCommit: () => void
  onReset: () => void
}) {
  return (
    <div
      className={clsx('tl-slider', centerMark && 'centered')}
      onDoubleClick={() => value !== reset && onReset()}
      onPointerUp={onCommit}
      onKeyUp={onCommit}
      onBlur={onCommit}
    >
      <Slider value={value} min={min} max={max} step={step} centerMark={centerMark} onChange={onLive} onCommit={onCommit} label={label} />
    </div>
  )
}

/** fader travel per pixel of drag (Shift: finer) */
const DRAG = 1 / 260

/**
 * A vertical fader on a −∞…+6 dB scale with a detent at 0 dB: drag the cap (or click the slot to
 * jump), double-click for 0 dB, arrows ±0.5 dB (Shift ±0.1), Page keys ±6 dB, Home / End to the ends.
 */
export function Fader({ db, label, onLive, onCommit }: { db: number; label: string; onLive: (db: number) => void; onCommit: (db: number) => void }) {
  const ref = useRef<HTMLDivElement>(null)
  const drag = useRef<{ y0: number; p0: number; last: number } | null>(null)
  const pos = faderPos(db)

  const posAt = (clientY: number) => {
    const r = ref.current!.getBoundingClientRect()
    return clamp(1 - (clientY - r.top - 6) / Math.max(1, r.height - 12), 0, 1)
  }
  const detent = (p: number) => (Math.abs(p - faderPos(0)) < 0.012 ? faderPos(0) : p)

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return
    e.preventDefault()
    ref.current?.focus()
    const onCap = (e.target as HTMLElement).closest('.tl-fader-cap')
    const p0 = onCap ? pos : posAt(e.clientY)
    if (!onCap) onLive(faderDb(p0))
    drag.current = { y0: e.clientY, p0, last: faderDb(p0) }
    ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
  }
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current
    if (!d) return
    const p = detent(clamp(d.p0 + (d.y0 - e.clientY) * DRAG * (e.shiftKey ? 0.2 : 1), 0, 1))
    d.last = faderDb(p)
    onLive(d.last)
  }
  const onPointerUp = () => {
    const d = drag.current
    drag.current = null
    if (d) onCommit(d.last)
  }
  const onKeyDown = (e: React.KeyboardEvent) => {
    const base = db <= SILENT_DB ? -48 : db
    let next: number | null = null
    if (e.key === 'ArrowUp' || e.key === 'ArrowRight') next = base + (e.shiftKey ? 0.1 : 0.5)
    else if (e.key === 'ArrowDown' || e.key === 'ArrowLeft') next = db <= SILENT_DB ? SILENT_DB : base - (e.shiftKey ? 0.1 : 0.5)
    else if (e.key === 'PageUp') next = base + 6
    else if (e.key === 'PageDown') next = base - 6
    else if (e.key === 'Home') next = SILENT_DB
    else if (e.key === 'End') next = MAX_DB
    if (next === null) return
    e.preventDefault()
    e.stopPropagation()
    onCommit(clamp(Math.round(next * 10) / 10, SILENT_DB, MAX_DB))
  }

  return (
    <div
      ref={ref}
      className="tl-fader"
      role="slider"
      tabIndex={0}
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={SILENT_DB}
      aria-valuemax={MAX_DB}
      aria-valuenow={db}
      aria-valuetext={`${fmtDb(db)} dB`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={() => onCommit(0)}
      onKeyDown={onKeyDown}
    >
      <span className="tl-fader-slot" />
      <span className="tl-fader-unity" style={{ bottom: `calc(6px + ${faderPos(0)} * (100% - 12px))` }} />
      <span className="tl-fader-cap" style={{ bottom: `calc(${pos} * (100% - 12px))` }} />
    </div>
  )
}

/** marks and how important they are: short faders drop the minor ones, then the middle ones */
const MARKS: [number, 'key' | 'mid' | 'minor'][] = [
  [6, 'mid'],
  [0, 'key'],
  [-6, 'minor'],
  [-12, 'mid'],
  [-24, 'minor'],
  [-36, 'minor'],
  [SILENT_DB, 'key'],
]

/** dB marks beside a fader (and its meter: they share the scale) */
export function DbScale() {
  return (
    <div className="tl-scale" aria-hidden>
      {MARKS.map(([m, rank]) => (
        <span key={m} className={rank} style={{ bottom: `calc(6px + ${faderPos(m)} * (100% - 12px))` }}>
          {m === SILENT_DB ? '−∞' : m > 0 ? `+${m}` : m < 0 ? `−${-m}` : '0'}
        </span>
      ))}
    </div>
  )
}

/** The fader's value; click to type one ("-3.5", "-inf"), Enter sets it, Esc gives up. */
export function DbInput({ db, label, onCommit }: { db: number; label: string; onCommit: (db: number) => void }) {
  const [text, setText] = useState<string | null>(null)
  const done = (keep: boolean) => {
    if (keep && text !== null) {
      const v = parseDb(text)
      if (v !== null) onCommit(Math.round(v * 10) / 10)
    }
    setText(null)
  }
  return text === null ? (
    <button className="tl-db tabular" onClick={() => setText(db <= SILENT_DB ? '-inf' : String(Math.round(db * 10) / 10))} aria-label={`${label} ${fmtDb(db)} dB`}>
      {fmtDb(db)}
    </button>
  ) : (
    <input
      autoFocus
      className="tl-db tabular editing"
      value={text}
      onFocus={(e) => e.currentTarget.select()}
      aria-label={label}
      onChange={(e) => setText(e.target.value)}
      onBlur={() => done(true)}
      onKeyDown={(e) => {
        if (e.key === 'Enter') done(true)
        else if (e.key === 'Escape') done(false)
        else return
        e.preventDefault()
        e.stopPropagation()
      }}
    />
  )
}
