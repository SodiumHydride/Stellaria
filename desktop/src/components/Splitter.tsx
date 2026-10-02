import clsx from 'clsx'
import { useRef } from 'react'
import './splitter.css'

/**
 * A drag handle between two panels that sets one panel's size (px). `grow` says which way the
 * pointer moves to make it bigger: a side panel on the right grows when dragged left, a bottom
 * panel grows when dragged up. Arrow keys step 16px (Shift 64px), Home/End go to the limits,
 * double-click restores the default.
 */
export function Splitter({
  axis,
  value,
  min,
  max,
  grow,
  onChange,
  onReset,
  label,
  className,
}: {
  axis: 'x' | 'y'
  value: number
  min: number
  max: number
  grow: -1 | 1
  onChange: (v: number) => void
  onReset: () => void
  label: string
  className?: string
}) {
  const drag = useRef<{ start: number; from: number } | null>(null)
  const clamp = (v: number) => Math.round(Math.min(max, Math.max(min, v)))
  const pos = (e: React.PointerEvent) => (axis === 'x' ? e.clientX : e.clientY)
  return (
    <div
      role="separator"
      aria-orientation={axis === 'x' ? 'vertical' : 'horizontal'}
      aria-label={label}
      aria-valuenow={value}
      aria-valuemin={min}
      aria-valuemax={max}
      tabIndex={0}
      title={label}
      className={clsx('splitter', axis, className)}
      onPointerDown={(e) => {
        if (e.button !== 0) return
        e.preventDefault()
        e.currentTarget.setPointerCapture(e.pointerId)
        drag.current = { start: pos(e), from: value }
      }}
      onPointerMove={(e) => {
        const d = drag.current
        if (d) onChange(clamp(d.from + grow * (pos(e) - d.start)))
      }}
      onPointerUp={() => (drag.current = null)}
      onPointerCancel={() => (drag.current = null)}
      onDoubleClick={onReset}
      onKeyDown={(e) => {
        const step = e.shiftKey ? 64 : 16
        const back = axis === 'x' ? 'ArrowLeft' : 'ArrowUp'
        const fwd = axis === 'x' ? 'ArrowRight' : 'ArrowDown'
        let next: number | null = null
        if (e.key === back) next = value - grow * step
        else if (e.key === fwd) next = value + grow * step
        else if (e.key === 'Home') next = min
        else if (e.key === 'End') next = max
        if (next === null) return
        e.preventDefault()
        e.stopPropagation()
        onChange(clamp(next))
      }}
    />
  )
}
