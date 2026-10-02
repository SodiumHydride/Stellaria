import clsx from 'clsx'
import { Search } from 'lucide-react'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties } from 'react'
import { createPortal } from 'react-dom'
import { Avatar } from '../../components/Avatar'
import { useT } from '../../i18n'
import { CATALOG, GROUP_ORDER, type CatalogEntry, type CatalogId } from './catalog'
import { useCanvas } from './context'
import type { Endpoint, FlowEdge } from './model'
import { inputArity } from './model'

/** A floating panel at a screen point that closes on outside click / Escape and stays on screen. */
export function Floating({ x, y, onClose, className, children, label }: { x: number; y: number; onClose: () => void; className?: string; children: React.ReactNode; label?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ left: x, top: y })

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const pad = 8
    setPos({ left: Math.max(pad, Math.min(x, window.innerWidth - r.width - pad)), top: Math.max(pad, Math.min(y, window.innerHeight - r.height - pad)) })
  }, [x, y])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && onClose()
    window.addEventListener('keydown', onKey)
    window.addEventListener('mousedown', onDown, true)
    window.addEventListener('blur', onClose)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('mousedown', onDown, true)
      window.removeEventListener('blur', onClose)
    }
  }, [onClose])

  return createPortal(
    <div ref={ref} className={clsx('floating glass-strong', className)} style={pos} role="menu" aria-label={label} onContextMenu={(e) => e.preventDefault()}>
      {children}
    </div>,
    document.body,
  )
}

export interface MenuItem {
  icon?: React.ReactNode
  label: string
  onClick?: () => void
  shortcut?: string
  danger?: boolean
  disabled?: boolean
  /** a divider before this item */
  divider?: boolean
  /** a row of colour swatches instead of a button (null = no colour) */
  swatches?: { value: string | null; active: boolean }[]
  onSwatch?: (value: string | null) => void
}

/** A plain context menu. */
export function ContextMenu({ x, y, items, title, onClose }: { x: number; y: number; items: MenuItem[]; title?: string; onClose: () => void }) {
  return (
    <Floating x={x} y={y} onClose={onClose} className="ctx-menu" label={title}>
      {title && <div className="ctx-title">{title}</div>}
      {items.map((it, i) => (
        <div key={i}>
          {it.divider && i > 0 && <div className="ctx-sep" />}
          {it.swatches ? (
            <div className="ctx-item ctx-swatches" role="group" aria-label={it.label}>
              <span className="ctx-icon">{it.icon}</span>
              <span className="ctx-label">{it.label}</span>
              <span className="ctx-swatch-row">
                {it.swatches.map((sw) => (
                  <button
                    key={sw.value ?? 'none'}
                    className={clsx('ctx-swatch', sw.active && 'active', !sw.value && 'none')}
                    style={sw.value ? ({ ['--sw' as string]: sw.value } as CSSProperties) : undefined}
                    aria-label={sw.value ?? 'none'}
                    aria-pressed={sw.active}
                    onClick={() => {
                      onClose()
                      it.onSwatch?.(sw.value)
                    }}
                  />
                ))}
              </span>
            </div>
          ) : (
          <button
            className={clsx('ctx-item', it.danger && 'danger')}
            disabled={it.disabled}
            role="menuitem"
            onClick={() => {
              onClose()
              it.onClick?.()
            }}
          >
            <span className="ctx-icon">{it.icon}</span>
            <span className="ctx-label">{it.label}</span>
            {it.shortcut && <span className="kbd ctx-kbd">{it.shortcut}</span>}
          </button>
          )}
        </div>
      ))}
    </Floating>
  )
}

export interface AddMenuState {
  x: number
  y: number
  flow: { x: number; y: number }
  from?: Endpoint
  edge?: FlowEdge
}

type Row = { kind: 'voice'; id: string } | { kind: 'entry'; entry: CatalogEntry }

/**
 * "Add anything here": a searchable list of everything that can go on the canvas. Opened by
 * double-click / right-click / Tab on the canvas, by dropping a connection on empty space
 * (then only things that take audio, already wired), or on an edge (then it's spliced in).
 */
export function AddMenu({ state, onClose }: { state: AddMenuState; onClose: () => void }) {
  const t = useT()
  const { voices, actions, project } = useCanvas()
  const [q, setQ] = useState('')
  const [active, setActive] = useState(0)
  const listRef = useRef<HTMLDivElement>(null)
  const fromNode = state.from ? project.canvas.nodes.find((n) => n.id === state.from!.nodeId) : undefined
  const wiring = !!state.from || !!state.edge

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase()
    const match = (text: string) => !needle || text.toLowerCase().includes(needle)
    const out: { group: string; rows: Row[] }[] = []
    if (state.from && fromNode && inputArity(fromNode) !== undefined) {
      const vs = voices.filter((c) => match(`${c.name} ${t('qa.sing', { name: c.name })} sing`))
      if (vs.length) out.push({ group: t('catg.voices'), rows: vs.map((c) => ({ kind: 'voice', id: c.id })) })
    }
    for (const g of GROUP_ORDER) {
      const entries = CATALOG.filter((e) => e.group === g)
        .filter((e) => (state.edge ? e.input === 1 && e.id !== 'export' : state.from ? e.input !== 0 : true))
        .filter((e) => match(`${t(`cat.${e.id}` as const)} ${e.keywords}`))
      if (entries.length) out.push({ group: t(`catg.${g}` as const), rows: entries.map((entry) => ({ kind: 'entry', entry })) })
    }
    return out
  }, [q, voices, state.from, state.edge, fromNode, t])

  const flat = rows.flatMap((g) => g.rows)
  useEffect(() => setActive(0), [q])
  useEffect(() => {
    listRef.current?.querySelector('.add-item.active')?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const pick = (r: Row) => {
    onClose()
    if (r.kind === 'voice' && state.from) void actions.sing({ nodeId: state.from.nodeId, port: state.from.handle ?? 'out' }, r.id, { position: state.flow })
    else if (r.kind === 'entry') void actions.addNode(r.entry.id as CatalogId, { position: state.flow, from: state.from, edge: state.edge })
  }

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActive((i) => Math.min(flat.length - 1, i + 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActive((i) => Math.max(0, i - 1))
    } else if (e.key === 'Enter' && flat[active]) {
      e.preventDefault()
      pick(flat[active])
    }
  }

  let index = -1
  return (
    <Floating x={state.x} y={state.y} onClose={onClose} className="add-menu" label={t('add.title')}>
      <div className="add-search">
        <Search />
        <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} onKeyDown={onKey} placeholder={state.edge ? t('add.insert_placeholder') : wiring ? t('add.wire_placeholder') : t('add.placeholder')} />
      </div>
      <div className="add-list" ref={listRef}>
        {flat.length === 0 && <div className="add-empty muted">{t('palette.empty')}</div>}
        {rows.map((g) => (
          <div key={g.group} className="add-group">
            <div className="add-group-title">{g.group}</div>
            {g.rows.map((r) => {
              index++
              const i = index
              if (r.kind === 'voice') {
                const c = voices.find((v) => v.id === r.id)!
                return (
                  <button key={`v:${r.id}`} className={clsx('add-item', i === active && 'active')} onMouseEnter={() => setActive(i)} onClick={() => pick(r)}>
                    <Avatar character={c} size={20} />
                    <span>{t('qa.sing', { name: c.name })}</span>
                  </button>
                )
              }
              const Icon = r.entry.icon
              return (
                <button key={r.entry.id} className={clsx('add-item', i === active && 'active')} onMouseEnter={() => setActive(i)} onClick={() => pick(r)}>
                  <span className={clsx('add-icon', `g-${r.entry.group}`)}>
                    <Icon />
                  </span>
                  <span>{t(`cat.${r.entry.id}` as const)}</span>
                  <span className="add-hint muted">{t(`cat.${r.entry.id}.hint` as const)}</span>
                </button>
              )
            })}
          </div>
        ))}
      </div>
    </Floating>
  )
}
