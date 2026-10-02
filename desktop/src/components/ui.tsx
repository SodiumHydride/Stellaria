import * as RDialog from '@radix-ui/react-dialog'
import * as RMenu from '@radix-ui/react-dropdown-menu'
import * as RSlider from '@radix-ui/react-slider'
import * as RSwitch from '@radix-ui/react-switch'
import * as RTooltip from '@radix-ui/react-tooltip'
import clsx from 'clsx'
import { X } from 'lucide-react'
import { useLayoutEffect, useRef, useState, type ReactNode } from 'react'

// ---------- Tooltip ----------
export function Tip({ label, children, side = 'top', delay = 350 }: { label: ReactNode; children: ReactNode; side?: 'top' | 'bottom' | 'left' | 'right'; delay?: number }) {
  if (!label) return <>{children}</>
  return (
    <RTooltip.Root delayDuration={delay}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content className="tooltip" side={side} sideOffset={6}>
          {label}
        </RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  )
}

export const TooltipProvider = RTooltip.Provider

// ---------- Switch ----------
export function Switch({ checked, onChange, disabled, label }: { checked: boolean; onChange: (v: boolean) => void; disabled?: boolean; label?: string }) {
  return (
    <RSwitch.Root className="switch" checked={checked} onCheckedChange={onChange} disabled={disabled} aria-label={label}>
      <RSwitch.Thumb className="switch-thumb" />
    </RSwitch.Root>
  )
}

// ---------- Slider ----------
export function Slider({
  value,
  onChange,
  onCommit,
  min,
  max,
  step = 1,
  centerMark,
  disabled,
  label,
}: {
  value: number
  onChange: (v: number) => void
  onCommit?: (v: number) => void
  min: number
  max: number
  step?: number
  centerMark?: boolean
  disabled?: boolean
  label?: string
}) {
  return (
    <RSlider.Root
      className="slider"
      value={[value]}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onValueChange={(v) => onChange(v[0])}
      onValueCommit={(v) => onCommit?.(v[0])}
      aria-label={label}
    >
      <RSlider.Track className="slider-track">
        <RSlider.Range className="slider-range" />
        {centerMark && <span className="slider-center-mark" />}
      </RSlider.Track>
      <RSlider.Thumb className="slider-thumb" aria-label={label} />
    </RSlider.Root>
  )
}

// ---------- Segmented ----------
/** `wrap`: options keep their labels on one line and flow onto more rows when the space is narrow. */
export function Segmented<T extends string>({
  value,
  onChange,
  options,
  size,
  wrap,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: ReactNode; icon?: ReactNode; title?: string }[]
  size?: 'sm'
  wrap?: boolean
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [thumb, setThumb] = useState<{ left: number; width: number; top: number; height: number } | null>(null)
  useLayoutEffect(() => {
    const root = ref.current
    if (!root) return
    const measure = () => {
      const el = root.querySelector<HTMLButtonElement>(`button[data-value="${CSS.escape(value)}"]`)
      setThumb(el ? { left: el.offsetLeft, width: el.offsetWidth, top: el.offsetTop, height: el.offsetHeight } : null)
    }
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(root)
    return () => ro.disconnect()
  }, [value, options.length])
  return (
    <div className={clsx('segmented', size, wrap && 'wrap')} ref={ref} role="tablist">
      {thumb && <span className="seg-thumb" style={wrap ? thumb : { left: thumb.left, width: thumb.width }} />}
      {options.map((o) => (
        <button
          key={o.value}
          role="tab"
          data-value={o.value}
          data-active={o.value === value}
          aria-selected={o.value === value}
          title={o.title}
          onClick={() => onChange(o.value)}
        >
          {o.icon}
          {o.label}
        </button>
      ))}
    </div>
  )
}

// ---------- Dialog ----------
export function Dialog({
  open,
  onOpenChange,
  title,
  description,
  children,
  wide,
  actions,
  dismissable = true,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: ReactNode
  description?: ReactNode
  children?: ReactNode
  wide?: boolean
  actions?: ReactNode
  /** false: only the actions close it (no ✕, Esc or click outside) */
  dismissable?: boolean
}) {
  const keep = dismissable ? undefined : (e: Event) => e.preventDefault()
  return (
    <RDialog.Root open={open} onOpenChange={onOpenChange}>
      <RDialog.Portal>
        <RDialog.Overlay className="dialog-overlay" />
        <RDialog.Content className={clsx('dialog', wide && 'wide')} aria-describedby={undefined} onEscapeKeyDown={keep} onInteractOutside={keep}>
          <div className="row" style={{ alignItems: 'flex-start' }}>
            <div style={{ flex: 1 }}>
              <RDialog.Title className="dialog-title">{title}</RDialog.Title>
              {description && <RDialog.Description className="dialog-desc">{description}</RDialog.Description>}
            </div>
            {dismissable && (
              <RDialog.Close className="icon-btn sm" aria-label="close">
                <X />
              </RDialog.Close>
            )}
          </div>
          {children && <div style={{ marginTop: 18 }}>{children}</div>}
          {actions && <div className="dialog-actions">{actions}</div>}
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  )
}

/** Promise-based confirm dialog rendered by <ConfirmHost/>. */
interface ConfirmRequest {
  title: string
  body?: string
  confirm: string
  cancel: string
  danger?: boolean
  resolve: (ok: boolean) => void
}

let pushConfirm: ((r: ConfirmRequest) => void) | null = null

export function confirmDialog(opts: Omit<ConfirmRequest, 'resolve'>): Promise<boolean> {
  return new Promise((resolve) => {
    if (!pushConfirm) resolve(window.confirm(opts.title))
    else pushConfirm({ ...opts, resolve })
  })
}

export function ConfirmHost() {
  const [req, setReq] = useState<ConfirmRequest | null>(null)
  pushConfirm = setReq
  const close = (ok: boolean) => {
    req?.resolve(ok)
    setReq(null)
  }
  return (
    <Dialog
      open={!!req}
      onOpenChange={(o) => !o && close(false)}
      title={req?.title ?? ''}
      description={req?.body}
      actions={
        <>
          <button className="btn btn-ghost" onClick={() => close(false)}>
            {req?.cancel}
          </button>
          <button className={clsx('btn', req?.danger ? 'btn-danger' : 'btn-primary')} onClick={() => close(true)} autoFocus>
            {req?.confirm}
          </button>
        </>
      }
    />
  )
}

// ---------- Menu ----------
export interface MenuItem {
  label: ReactNode
  icon?: ReactNode
  onSelect?: () => void
  danger?: boolean
  disabled?: boolean
  separator?: boolean
}

export function Menu({ trigger, items, align = 'end' }: { trigger: ReactNode; items: MenuItem[]; align?: 'start' | 'end' | 'center' }) {
  return (
    <RMenu.Root>
      <RMenu.Trigger asChild>{trigger}</RMenu.Trigger>
      <RMenu.Portal>
        <RMenu.Content className="menu" align={align} sideOffset={6}>
          {items.map((it, i) =>
            it.separator ? (
              <RMenu.Separator key={i} className="menu-sep" />
            ) : (
              <RMenu.Item key={i} className={clsx('menu-item', it.danger && 'danger')} onSelect={it.onSelect} disabled={it.disabled}>
                {it.icon}
                {it.label}
              </RMenu.Item>
            ),
          )}
        </RMenu.Content>
      </RMenu.Portal>
    </RMenu.Root>
  )
}

// ---------- Progress ----------
export function Progress({ value, className }: { value: number | null; className?: string }) {
  return (
    <div className={clsx('progress', value == null && 'indeterminate', className)}>
      <div className="bar" style={{ width: `${Math.round((value ?? 0.35) * 100)}%` }} />
    </div>
  )
}

/** Circular score, used by the health report and for compact progress. */
export function Ring({
  value,
  size = 64,
  stroke = 6,
  color,
  children,
  track = 'var(--fill-track)',
}: {
  value: number
  size?: number
  stroke?: number
  color?: string
  children?: ReactNode
  track?: string
}) {
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  return (
    <div className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" style={{ stroke: track }} strokeWidth={stroke} />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - Math.max(0, Math.min(1, value)))}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ stroke: color ?? 'var(--fill)', transition: 'stroke-dashoffset 700ms var(--ease)' }}
        />
      </svg>
      <div className="ring-center">{children}</div>
    </div>
  )
}

export function Empty({ icon, title, body, children }: { icon?: ReactNode; title: ReactNode; body?: ReactNode; children?: ReactNode }) {
  return (
    <div className="empty">
      {icon && <div className="empty-icon">{icon}</div>}
      <h3>{title}</h3>
      {body && <p>{body}</p>}
      {children && <div className="row" style={{ marginTop: 8, justifyContent: 'center' }}>{children}</div>}
    </div>
  )
}
