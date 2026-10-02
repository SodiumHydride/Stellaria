import clsx from 'clsx'
import { AlertTriangle, ArrowDown, ArrowUp, ChevronDown, Copy, GripVertical, MoreHorizontal, Power, RotateCcw, Trash2 } from 'lucide-react'
import { useEffect, useLayoutEffect, useMemo, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from 'react'
import { useFxEffects, useVstPlugins, useVstStatus } from '../../../api/hooks'
import type { EffectSpec, FxSlot, VstPlugin, VstStatus } from '../../../api/types'
import { confirmDialog, Menu, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { AddEffect, PresetsMenu } from './FxMenus'
import { BuiltinBody, SlotNotice, VstBody } from './FxSlotBody'
import { closePluginEditor, effectDefaults, lastKnownPlugin, moveItem, registerApplier, rememberPlugins, slotId, STARTERS, starterChain, useEditorJobs, type VstSlot } from './fxModel'
import './fx.css'

type Problem = { text: string; kind: 'missing' | 'host' | 'plugin' | 'effect' }

/**
 * An effects chain (fx node, canvas v3 §3): slots in processing order, built-in effects and
 * VST3/AU plugins, drag to reorder, per-slot bypass and dry/wet, local presets.
 * Built for the inspector column (≈ 300 px).
 */
export function FxChainEditor({ value, onChange, bpm }: { value: FxSlot[]; onChange: (next: FxSlot[]) => void; bpm?: number | null }) {
  const t = useT()
  const { data: effects, isFetched: effectsFetched } = useFxEffects()
  const { data: plugins, isFetched: pluginsFetched } = useVstPlugins()
  const { data: vst } = useVstStatus()
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set())
  const listRef = useRef<HTMLOListElement>(null)
  const refocus = useRef<string | null>(null)
  const latest = useRef({ value, onChange })
  useLayoutEffect(() => {
    latest.current = { value, onChange }
  })

  const specs = useMemo(() => new Map((effects ?? []).map((e) => [e.id, e])), [effects])
  const pluginMap = useMemo(() => new Map((plugins ?? []).map((p) => [p.id, p])), [plugins])

  useEffect(() => {
    if (plugins && plugins.length > 0) rememberPlugins(plugins)
  }, [plugins])

  // results of a plugin window opened from this chain (also after the inspector was closed and reopened)
  useEffect(
    () =>
      registerApplier((id, result) => {
        const { value: cur, onChange: emit } = latest.current
        if (!cur.some((s) => s.id === id)) return false
        emit(cur.map((s) => (s.id === id && s.kind === 'vst' ? { ...s, state: result.state, params: result.params } : s)))
        return true
      }),
    [],
  )

  // after each render: focus what a keyboard reorder asked for (set just before its update)
  useEffect(() => {
    if (!refocus.current) return
    listRef.current?.querySelector<HTMLElement>(`[data-grip="${CSS.escape(refocus.current)}"]`)?.focus()
    refocus.current = null
  })

  const patch = (id: string, update: (s: FxSlot) => FxSlot) => onChange(value.map((s) => (s.id === id ? update(s) : s)))
  const toggle = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  const add = (slot: FxSlot) => {
    onChange([...value, slot])
    setExpanded((prev) => new Set(prev).add(slot.id))
  }
  const move = (from: number, to: number, keepFocus = false) => {
    if (to < 0 || to >= value.length || from === to) return
    if (keepFocus) refocus.current = value[from].id
    onChange(moveItem(value, from, to))
  }
  const duplicate = (index: number) => {
    const copy = { ...value[index], id: slotId(), params: { ...value[index].params } }
    onChange([...value.slice(0, index + 1), copy, ...value.slice(index + 1)])
  }
  const reset = (slot: FxSlot) =>
    patch(slot.id, (s) => {
      if (s.kind === 'vst') return { ...s, state: null, params: {} }
      const spec = specs.get(s.effect)
      return { ...s, params: spec ? effectDefaults(spec) : {} }
    })
  /** Writes a chain; a removed slot whose plugin window is still open gets that window closed. */
  const commit = (next: FxSlot[]) => {
    const { value: cur, onChange: emit } = latest.current
    const kept = new Set(next.map((s) => s.id))
    const open = useEditorJobs.getState().jobs
    for (const s of cur) if (!kept.has(s.id) && open[s.id]) void closePluginEditor(open[s.id])
    emit(next)
  }
  const load = async (chain: FxSlot[], name: string) => {
    if (chain.length === 0) return
    if (latest.current.value.length > 0) {
      const ok = await confirmDialog({ title: t('fx.presets.replace', { name }), confirm: t('fx.presets.replace_ok'), cancel: t('common.cancel') })
      if (!ok) return
    }
    commit(chain)
    setExpanded(new Set())
  }

  const starters = STARTERS.filter((s) => s.slots.some((x) => specs.has(x.effect)))

  const drag = useSortable(listRef, value, (from, to) => latest.current.onChange(moveItem(latest.current.value, from, to)))

  const describe = (slot: FxSlot): { name: string; sub: string | null; problem: Problem | null; plugin: VstPlugin | null; spec: EffectSpec | null } => {
    if (slot.kind === 'builtin') {
      const spec = specs.get(slot.effect) ?? null
      const name = spec ? t.label(spec.name) : t.msg({ code: `fx.effect.${slot.effect}` }, slot.effect)
      const problem: Problem | null = !spec && effectsFetched ? { text: t('fx.unknown_effect', { id: slot.effect }), kind: 'effect' } : null
      return { name, sub: null, problem, plugin: null, spec }
    }
    const plugin = pluginMap.get(slot.pluginId) ?? null
    // without the host add-on nothing can load, whether or not the plugin is still on disk
    const hostMissing = vst !== undefined && !vst?.hostInstalled
    if (!plugin) {
      const known = lastKnownPlugin(slot.pluginId)
      return {
        name: known?.name ?? t('fx.missing_plugin'),
        sub: known?.vendor ?? null,
        problem: hostMissing ? { text: t.msg({ code: 'vst.host_missing' }), kind: 'host' } : pluginsFetched ? { text: t('fx.missing_title'), kind: 'missing' } : null,
        plugin: null,
        spec: null,
      }
    }
    const sub = [plugin.vendor, plugin.format === 'au' ? 'AU' : 'VST3'].filter(Boolean).join(' · ')
    let problem: Problem | null = null
    if (hostMissing) problem = { text: t.msg({ code: 'vst.host_missing' }), kind: 'host' }
    else if (plugin.status === 'failed') problem = { text: t.msg(plugin.error, t.msg({ code: 'vst.load_failed', params: { name: plugin.name, reason: '' } })), kind: 'plugin' }
    else if (plugin.category === 'instrument') problem = { text: t.msg({ code: 'vst.instrument', params: { name: plugin.name } }), kind: 'plugin' }
    return { name: plugin.name, sub, problem, plugin, spec: null }
  }

  return (
    <div className="fx-chain">
      <div className="fx-head">
        <span className="eyebrow">{t('fx.chain')}</span>
        {value.length > 0 && <span className="fx-count tabular">{value.length}</span>}
        <span className="spacer" />
        <PresetsMenu chain={value} specs={specs} onLoad={load} />
      </div>

      {value.length > 0 ? (
        <ol className={clsx('fx-slots', drag.state && 'sorting')} ref={listRef}>
          {value.map((slot, i) => {
            const info = describe(slot)
            return (
              <SlotRow
                key={slot.id}
                slot={slot}
                index={i}
                count={value.length}
                info={info}
                vst={vst}
                bpm={bpm ?? null}
                expanded={expanded.has(slot.id)}
                lifted={drag.state?.id === slot.id}
                style={drag.style(i, slot.id)}
                onToggle={() => toggle(slot.id)}
                onPatch={(update) => patch(slot.id, update)}
                onRemove={() => commit(value.filter((s) => s.id !== slot.id))}
                onDuplicate={() => duplicate(i)}
                onReset={() => reset(slot)}
                onMove={(to, keepFocus) => move(i, to, keepFocus)}
                onGripDown={(e) => drag.start(e, i)}
              />
            )
          })}
        </ol>
      ) : (
        starters.length > 0 && (
          <div className="fx-empty">
            <span className="fx-empty-label">{t('fx.empty')}</span>
            <div className="fx-starters">
              {starters.map((s) => (
                <button key={s.id} className="chip button" onClick={() => void load(starterChain(s, specs), t(s.name))}>
                  {t(s.name)}
                </button>
              ))}
            </div>
          </div>
        )
      )}

      <AddEffect effects={effects ?? []} plugins={plugins ?? []} vst={vst} onAdd={add} />
    </div>
  )
}

function SlotRow({
  slot,
  index,
  count,
  info,
  vst,
  bpm,
  expanded,
  lifted,
  style,
  onToggle,
  onPatch,
  onRemove,
  onDuplicate,
  onReset,
  onMove,
  onGripDown,
}: {
  slot: FxSlot
  index: number
  count: number
  info: { name: string; sub: string | null; problem: Problem | null; plugin: VstPlugin | null; spec: EffectSpec | null }
  vst: VstStatus | null | undefined
  bpm: number | null
  expanded: boolean
  lifted: boolean
  style: CSSProperties | undefined
  onToggle: () => void
  onPatch: (update: (s: FxSlot) => FxSlot) => void
  onRemove: () => void
  onDuplicate: () => void
  onReset: () => void
  onMove: (to: number, keepFocus?: boolean) => void
  onGripDown: (e: ReactPointerEvent<HTMLButtonElement>) => void
}) {
  const t = useT()
  const { name, sub, problem, plugin, spec } = info
  const bodyId = `fx-body-${slot.id}`

  let body: React.ReactNode = null
  if (expanded) {
    if (problem && problem.kind !== 'effect') body = <SlotNotice problem={problem.kind} text={problem.text} onRemove={onRemove} />
    else if (slot.kind === 'builtin' && spec) body = <BuiltinBody spec={spec} slot={slot} bpm={bpm} onParams={(params) => onPatch((s) => ({ ...s, params }))} />
    else if (slot.kind === 'vst' && plugin) body = <VstBody slot={slot} plugin={plugin} vst={vst} onSlot={(next: VstSlot) => onPatch(() => next)} />
    else if (problem) body = <SlotNotice problem={problem.kind} text={problem.text} onRemove={onRemove} />
  }

  return (
    <li className={clsx('fx-slot', slot.bypass && 'bypassed', expanded && 'open', lifted && 'lifted', problem && 'problem')} style={style}>
      <div className="fx-slot-head">
        <button
          className="fx-grip"
          data-grip={slot.id}
          aria-label={t('fx.reorder')}
          onPointerDown={onGripDown}
          onKeyDown={(e) => {
            if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
              e.preventDefault()
              onMove(index + (e.key === 'ArrowUp' ? -1 : 1), true)
            }
          }}
        >
          <GripVertical />
        </button>
        <Tip label={slot.bypass ? t('fx.enable') : t('fx.bypass')}>
          <button className={clsx('fx-power', !slot.bypass && 'on')} aria-pressed={!slot.bypass} aria-label={t('fx.bypass')} onClick={() => onPatch((s) => ({ ...s, bypass: !s.bypass }))}>
            <Power />
          </button>
        </Tip>
        <button className="fx-slot-title" onClick={onToggle} aria-expanded={expanded} aria-controls={bodyId}>
          <span className="fx-slot-names">
            <span className="fx-slot-name">{name}</span>
            {sub && <span className="fx-slot-sub">{sub}</span>}
          </span>
          {problem ? (
            <Tip label={problem.text}>
              <AlertTriangle className="fx-slot-warn" aria-label={problem.text} />
            </Tip>
          ) : (
            <ChevronDown className="fx-slot-chevron" />
          )}
        </button>
        <MiniFader value={slot.mix} disabled={slot.bypass} onChange={(mix) => onPatch((s) => ({ ...s, mix }))} label={t('fx.wet')} hint={t('fx.wet_hint')} />
        <Menu
          trigger={
            <button className="icon-btn sm fx-more" aria-label={t('common.more')}>
              <MoreHorizontal />
            </button>
          }
          items={[
            { label: t('fx.move_up'), icon: <ArrowUp />, onSelect: () => onMove(index - 1), disabled: index === 0 },
            { label: t('fx.move_down'), icon: <ArrowDown />, onSelect: () => onMove(index + 1), disabled: index === count - 1 },
            { label: t('fx.duplicate'), icon: <Copy />, onSelect: onDuplicate },
            { label: t('common.reset'), icon: <RotateCcw />, onSelect: onReset, disabled: slot.kind === 'vst' ? !slot.state && Object.keys(slot.params).length === 0 : !spec },
            { label: '', separator: true },
            { label: t('fx.remove'), icon: <Trash2 />, onSelect: onRemove, danger: true },
          ]}
        />
      </div>
      {body && (
        <div className="fx-slot-body" id={bodyId}>
          {body}
        </div>
      )}
    </li>
  )
}

/** Dry/wet as a compact bar: drag or click to set, arrows to nudge (Shift coarse, Alt fine), double-click for 100 %. */
function MiniFader({ value, onChange, label, hint, disabled }: { value: number; onChange: (v: number) => void; label: string; hint: string; disabled?: boolean }) {
  const ref = useRef<HTMLDivElement>(null)
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100)
  const setFrom = (clientX: number) => {
    const r = ref.current?.getBoundingClientRect()
    if (!r || r.width === 0) return
    onChange(Math.round(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * 100) / 100)
  }
  const nudge = (d: number) => onChange(Math.round(Math.max(0, Math.min(100, pct + d))) / 100)
  return (
    <Tip label={hint} delay={600}>
      <div
        ref={ref}
        className={clsx('fx-fader', disabled && 'dim', pct === 100 && 'full')}
        role="slider"
        tabIndex={0}
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        aria-valuetext={`${pct}%`}
        onPointerDown={(e) => {
          if (e.button !== 0) return
          e.currentTarget.setPointerCapture(e.pointerId)
          setFrom(e.clientX)
        }}
        onPointerMove={(e) => {
          if (e.currentTarget.hasPointerCapture(e.pointerId)) setFrom(e.clientX)
        }}
        onDoubleClick={() => onChange(1)}
        onKeyDown={(e) => {
          const step = e.shiftKey ? 10 : e.altKey ? 1 : 5
          if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') nudge(-step)
          else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') nudge(step)
          else if (e.key === 'PageDown') nudge(-10)
          else if (e.key === 'PageUp') nudge(10)
          else if (e.key === 'Home') onChange(0)
          else if (e.key === 'End') onChange(1)
          else return
          e.preventDefault()
        }}
      >
        {/* the bar is 22 px; the slider around it is a 24 px target */}
        <span className="fx-fader-box">
          <span className="fx-fader-fill" style={{ width: `${pct}%` }} />
          <span className="fx-fader-text tabular">{pct}%</span>
        </span>
      </div>
    </Tip>
  )
}

/** Nearest scrollable ancestor (the inspector), for auto-scroll while dragging. */
function scrollParent(el: HTMLElement | null): HTMLElement | null {
  for (let node = el?.parentElement ?? null; node; node = node.parentElement) {
    const { overflowY } = getComputedStyle(node)
    if ((overflowY === 'auto' || overflowY === 'scroll') && node.scrollHeight > node.clientHeight) return node
  }
  return null
}

/**
 * Pointer-driven reordering. HTML5 drag-and-drop is off limits: the desktop shell takes OS file
 * drops (dragDropEnabled), which disables in-page HTML5 drag on Windows.
 */
function useSortable(listRef: React.RefObject<HTMLOListElement | null>, items: FxSlot[], onMove: (from: number, to: number) => void) {
  const [state, setState] = useState<{ id: string; from: number; to: number; dy: number; shift: number } | null>(null)

  const start = (e: ReactPointerEvent<HTMLButtonElement>, index: number) => {
    if (e.button !== 0 || items.length < 2 || !listRef.current) return
    e.preventDefault()
    const handle = e.currentTarget
    const rects = Array.from(listRef.current.children).map((el) => el.getBoundingClientRect())
    const mids = rects.map((r) => r.top + r.height / 2)
    const gap = rects.length > 1 ? Math.max(0, rects[1].top - rects[0].bottom) : 0
    const shift = rects[index].height + gap
    const scroller = scrollParent(listRef.current)
    const startY = e.clientY
    const startScroll = scroller?.scrollTop ?? 0
    const id = items[index].id
    let lastY = startY
    let to = index
    let started = false
    let raf = 0

    const compute = () => {
      const dy = lastY - startY + ((scroller?.scrollTop ?? 0) - startScroll)
      if (!started && Math.abs(dy) < 4) return
      started = true
      const center = mids[index] + dy
      to = index
      for (let i = index + 1; i < mids.length; i++) if (center > mids[i]) to = i
      for (let i = index - 1; i >= 0; i--) if (center < mids[i]) to = i
      setState({ id, from: index, to, dy, shift })
    }
    const tick = () => {
      if (scroller && started) {
        const r = scroller.getBoundingClientRect()
        const edge = 44
        const v = lastY < r.top + edge ? -Math.ceil((r.top + edge - lastY) / 5) : lastY > r.bottom - edge ? Math.ceil((lastY - r.bottom + edge) / 5) : 0
        if (v) {
          scroller.scrollTop += v
          compute()
        }
      }
      raf = requestAnimationFrame(tick)
    }
    const onPointerMove = (ev: PointerEvent) => {
      lastY = ev.clientY
      compute()
    }
    const finish = (commit: boolean) => {
      cancelAnimationFrame(raf)
      handle.removeEventListener('pointermove', onPointerMove)
      handle.removeEventListener('pointerup', onUp)
      handle.removeEventListener('pointercancel', onCancel)
      window.removeEventListener('keydown', onKey, true)
      setState(null)
      if (commit && started && to !== index) onMove(index, to)
    }
    const onUp = () => finish(true)
    const onCancel = () => finish(false)
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key !== 'Escape') return
      ev.preventDefault()
      ev.stopPropagation()
      finish(false)
    }
    handle.setPointerCapture(e.pointerId)
    handle.addEventListener('pointermove', onPointerMove)
    handle.addEventListener('pointerup', onUp)
    handle.addEventListener('pointercancel', onCancel)
    window.addEventListener('keydown', onKey, true)
    raf = requestAnimationFrame(tick)
  }

  const style = (i: number, id: string): CSSProperties | undefined => {
    if (!state) return undefined
    if (id === state.id) return { transform: `translateY(${state.dy}px)` }
    if (state.from < state.to && i > state.from && i <= state.to) return { transform: `translateY(${-state.shift}px)` }
    if (state.to < state.from && i >= state.to && i < state.from) return { transform: `translateY(${state.shift}px)` }
    return undefined
  }

  return { state, start, style }
}
