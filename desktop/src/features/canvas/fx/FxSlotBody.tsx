import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import clsx from 'clsx'
import { AlertTriangle, AppWindow, Download, Loader2, RefreshCw, Search, Settings2, Trash2, X } from 'lucide-react'
import { Fragment, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../../../api/client'
import { keys } from '../../../api/keys'
import type { EffectSpec, Job, ParamSpec, VstParam, VstPlugin, VstStatus } from '../../../api/types'
import { ParamControl } from '../../../components/ParamControls'
import { Slider, Switch, Tip } from '../../../components/ui'
import { useT } from '../../../i18n'
import { attempt } from '../../../lib/actions'
import { waitForJob } from '../../../lib/jobs'
import {
  closePluginEditor,
  fmtVst,
  NOTE_VALUES,
  noteMs,
  openPluginEditor,
  parseNote,
  sameValue,
  stateKey,
  useEditorJobs,
  vstRange,
  type BuiltinSlot,
  type FxParamValues,
  type FxValue,
  type VstSlot,
} from './fxModel'

// ---------- built-in effects ----------

export function BuiltinBody({ spec, slot, bpm, onParams }: { spec: EffectSpec; slot: BuiltinSlot; bpm: number | null; onParams: (params: FxParamValues) => void }) {
  const set = (id: string, v: FxValue | null) => {
    const next = { ...slot.params }
    if (v === null) delete next[id]
    else next[id] = v
    onParams(next)
  }
  if (spec.params.length === 0) return null
  return (
    <div className="fx-params">
      {spec.params.map((p) => {
        const v = slot.params[p.id] ?? p.default
        return (
          <Fragment key={p.id}>
            <ParamControl spec={p} value={v} onChange={(x) => set(p.id, x)} />
            {bpm && bpm > 0 && isTimeParam(spec, p) && typeof v === 'number' && <NoteSync value={v} bpm={bpm} spec={p} onPick={(ms) => set(p.id, ms)} />}
            {bpm && bpm > 0 && p.type === 'choice' && typeof v === 'string' && parseNote(v) !== null && <NoteLength beats={parseNote(v)!} bpm={bpm} />}
          </Fragment>
        )
      })}
    </div>
  )
}

function isTimeParam(spec: EffectSpec, p: ParamSpec): boolean {
  return spec.id === 'delay' && p.type === 'number' && p.unit === 'ms' && /time/i.test(p.id)
}

/** Delay time in ms: one click to the note value at the song's tempo. */
function NoteSync({ value, bpm, spec, onPick }: { value: number; bpm: number; spec: ParamSpec; onPick: (ms: number) => void }) {
  const t = useT()
  const step = spec.step && spec.step > 0 ? spec.step : 1
  const options = NOTE_VALUES.map((n) => ({ ...n, ms: Math.round(noteMs(n.beats, bpm) / step) * step })).filter(
    (n) => n.ms >= (spec.min ?? 0) && n.ms <= (spec.max ?? Infinity),
  )
  if (options.length === 0) return null
  return (
    <div className="fx-notes">
      <Tip label={t('fx.bpm_hint', { bpm: Math.round(bpm) })}>
        <span className="fx-notes-bpm tabular">♩ {Math.round(bpm)}</span>
      </Tip>
      {options.map((n) => (
        <Tip key={n.id} label={`${Math.round(n.ms)} ms`} delay={500}>
          <button className={clsx('fx-note', Math.abs(value - n.ms) <= Math.max(0.5, step / 2) && 'on')} onClick={() => onPick(n.ms)}>
            {n.id}
          </button>
        </Tip>
      ))}
    </div>
  )
}

function NoteLength({ beats, bpm }: { beats: number; bpm: number }) {
  return (
    <div className="fx-note-ms tabular">
      ♩ {Math.round(bpm)} · {Math.round(noteMs(beats, bpm))} ms
    </div>
  )
}

// ---------- problems ----------

export function SlotNotice({ problem, text, onRemove }: { problem: 'missing' | 'host' | 'plugin' | 'effect'; text: string; onRemove: () => void }) {
  const t = useT()
  const nav = useNavigate()
  const qc = useQueryClient()
  const [scanning, setScanning] = useState(false)
  const [installing, setInstalling] = useState(false)
  const rescan = async () => {
    setScanning(true)
    const job = await attempt(() => api.post<Job>('/vst/scan', { full: false }))
    if (job) await attempt(() => waitForJob(job.id))
    await qc.invalidateQueries({ queryKey: keys.vstPlugins })
    setScanning(false)
  }
  // the plugin host is an add-on: install it right here, then look for plugins
  const installHost = async () => {
    setInstalling(true)
    const job = await attempt(() => api.post<Job>('/resources/vsthost/install', {}))
    const done = job ? await attempt(() => waitForJob(job.id)) : undefined
    await qc.invalidateQueries({ queryKey: keys.vstStatus })
    setInstalling(false)
    if (done?.status === 'succeeded') await rescan()
  }
  return (
    <div className="fx-notice">
      <div className="fx-notice-text">
        <AlertTriangle />
        <span>{problem === 'missing' ? t('fx.missing_body') : text}</span>
      </div>
      <div className="fx-notice-actions">
        {problem === 'missing' && (
          <button className="btn btn-sm" onClick={rescan} disabled={scanning}>
            {scanning ? <Loader2 className="spin" /> : <RefreshCw />}
            {t('fx.rescan')}
          </button>
        )}
        {problem === 'host' && (
          <button className="btn btn-sm btn-primary" onClick={() => void installHost()} disabled={installing || scanning}>
            {installing || scanning ? <Loader2 className="spin" /> : <Download />}
            {installing ? t('fx.host_installing') : t('fx.host_install')}
          </button>
        )}
        {problem !== 'effect' && (
          <button className="btn btn-sm" onClick={() => nav('/settings#vst')}>
            <Settings2 />
            {problem === 'host' ? t('fx.host_setup') : t('fx.open_settings')}
          </button>
        )}
        <span className="spacer" />
        <button className="btn btn-sm btn-ghost" onClick={onRemove}>
          <Trash2 />
          {t('fx.remove')}
        </button>
      </div>
    </div>
  )
}

// ---------- VST3 / AU ----------

export function VstBody({ slot, plugin, vst, onSlot }: { slot: VstSlot; plugin: VstPlugin; vst: VstStatus | null | undefined; onSlot: (next: VstSlot) => void }) {
  const t = useT()
  const jobId = useEditorJobs((s) => s.jobs[slot.id])
  const hostOk = !!vst?.hostInstalled
  const editorOk = hostOk && !!vst?.editorSupported
  const needsInspect = hostOk && !!slot.state && Object.keys(slot.params).length < plugin.params.length
  const { data: inspected, isFetching } = useQuery({
    queryKey: ['vst-inspect', plugin.id, stateKey(slot.state)],
    queryFn: () =>
      api
        .post<{ params?: FxParamValues }>(`/vst/plugins/${plugin.id}/inspect`, { state: slot.state })
        .then((r) => r.params ?? null)
        .catch(() => null),
    enabled: needsInspect,
    staleTime: Infinity,
    retry: false,
  })

  const open = () => void openPluginEditor(slot, plugin.name, (r) => onSlot({ ...slot, state: r.state, params: r.params }))

  return (
    <div className="fx-vst">
      {jobId ? (
        <div className="fx-editor-open" role="status">
          <span className="fx-editor-dot" />
          <span className="fx-editor-text">{t('fx.editor_open')}</span>
          <Tip label={t('fx.editor_cancel_hint')}>
            <button className="btn btn-sm btn-ghost" onClick={() => void closePluginEditor(jobId)}>
              {t('common.cancel')}
            </button>
          </Tip>
        </div>
      ) : (
        <div className="fx-vst-actions">
          <Tip label={editorOk ? t('fx.editor_hint') : hostOk ? t.msg({ code: 'vst.editor_unsupported' }) : t.msg({ code: 'vst.host_missing' })}>
            <span className="fx-tip-wrap">
              <button className="btn btn-sm" disabled={!editorOk} onClick={open}>
                <AppWindow />
                {t('fx.open_editor')}
              </button>
            </span>
          </Tip>
          {slot.state && (
            <Tip label={t('fx.state_saved_hint')}>
              <span className="chip">{t('fx.state_saved')}</span>
            </Tip>
          )}
        </div>
      )}
      {plugin.params.length > 0 ? (
        <VstParamList plugin={plugin} slot={slot} inspected={inspected ?? null} loading={needsInspect && isFetching} onParams={(params) => onSlot({ ...slot, params })} />
      ) : (
        <div className="fx-vp-empty">{t('fx.no_params')}</div>
      )}
    </div>
  )
}

const ROW_HEIGHT: Record<VstParam['kind'], number> = { float: 50, bool: 34, choice: 38 }
const LIST_MAX = 312

/** Hundreds of parameters: searchable, virtualised, overrides marked and resettable. */
function VstParamList({
  plugin,
  slot,
  inspected,
  loading,
  onParams,
}: {
  plugin: VstPlugin
  slot: VstSlot
  inspected: FxParamValues | null
  loading: boolean
  onParams: (params: FxParamValues) => void
}) {
  const t = useT()
  const [query, setQuery] = useState('')
  const [changedOnly, setChangedOnly] = useState(false)
  const [atEnd, setAtEnd] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)

  const valueOf = (p: VstParam): FxValue | null => slot.params[p.id] ?? inspected?.[p.id] ?? p.default
  const isChanged = (p: VstParam) => (p.default === null ? p.id in slot.params : !sameValue(valueOf(p), p.default, p))
  const changedCount = plugin.params.reduce((n, p) => n + (isChanged(p) ? 1 : 0), 0)
  const needle = query.trim().toLowerCase()
  const list = plugin.params.filter((p) => (!needle || p.name.toLowerCase().includes(needle) || p.id.toLowerCase().includes(needle)) && (!changedOnly || isChanged(p)))

  const virtual = useVirtualizer({
    count: list.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: (i) => ROW_HEIGHT[list[i].kind],
    getItemKey: (i) => list[i].id,
    overscan: 8,
  })
  const total = virtual.getTotalSize()

  const set = (p: VstParam, v: FxValue) => onParams({ ...slot.params, [p.id]: v })
  const reset = (p: VstParam) => {
    if (p.default === null) {
      const next = { ...slot.params }
      delete next[p.id]
      onParams(next)
    } else set(p, p.default)
  }

  return (
    <div className="fx-vp">
      <div className="fx-vp-tools">
        <div className="input-wrap fx-vp-search">
          {loading ? <Loader2 className="spin" /> : <Search />}
          <input className="input" value={query} placeholder={t('fx.search_params', { n: plugin.params.length })} onChange={(e) => setQuery(e.target.value)} aria-label={t('common.search')} />
          {query && (
            <button className="icon-btn sm fx-vp-clear" onClick={() => setQuery('')} aria-label={t('common.close')}>
              <X />
            </button>
          )}
        </div>
        {(changedCount > 0 || changedOnly) && (
          <Tip label={t('fx.changed_hint')}>
            <button className={clsx('chip button fx-vp-changed', changedOnly && 'accent')} aria-pressed={changedOnly} onClick={() => setChangedOnly(!changedOnly)}>
              {t('fx.changed', { n: changedCount })}
            </button>
          </Tip>
        )}
      </div>
      {list.length === 0 ? (
        <div className="fx-vp-empty">{t('fx.no_params_match')}</div>
      ) : (
        <div
          ref={scrollRef}
          className={clsx('fx-vp-scroll', total > LIST_MAX && !atEnd && 'more')}
          style={{ height: Math.min(total, LIST_MAX) }}
          onScroll={(e) => setAtEnd(e.currentTarget.scrollTop + e.currentTarget.clientHeight >= e.currentTarget.scrollHeight - 4)}
        >
          <div style={{ height: total, position: 'relative' }}>
            {virtual.getVirtualItems().map((item) => {
              const p = list[item.index]
              return (
                <div key={item.key} className={clsx('fx-vp-row', p.kind)} style={{ height: item.size, transform: `translateY(${item.start}px)` }}>
                  <VstParamRow param={p} value={valueOf(p)} changed={isChanged(p)} onChange={(v) => set(p, v)} onReset={() => reset(p)} />
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

function VstParamRow({ param: p, value, changed, onChange, onReset }: { param: VstParam; value: FxValue | null; changed: boolean; onChange: (v: FxValue) => void; onReset: () => void }) {
  const t = useT()
  const mark = changed ? (
    <Tip label={t('fx.reset_param')} delay={400}>
      <button className="fx-vp-dot" onClick={onReset} aria-label={t('fx.reset_param')} />
    </Tip>
  ) : null
  const name = (
    <span className="fx-vp-name" title={p.name}>
      {p.name}
    </span>
  )

  if (p.kind === 'bool') {
    return (
      <div className="fx-vp-line">
        {mark}
        {name}
        <Switch checked={value === true || value === 1 || value === 'true'} onChange={onChange} label={p.name} />
      </div>
    )
  }
  if (p.kind === 'choice') {
    const choices = p.choices ?? []
    const current = value === null ? '' : String(value)
    return (
      <div className="fx-vp-line">
        {mark}
        {name}
        <select className="input fx-vp-select" value={current} onChange={(e) => onChange(e.target.value)} aria-label={p.name}>
          {!choices.includes(current) && current !== '' && <option value={current}>{current}</option>}
          {choices.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>
    )
  }
  const [min, max] = vstRange(p)
  const v = typeof value === 'number' ? value : Number(value ?? min)
  const num = Number.isFinite(v) ? Math.max(min, Math.min(max, v)) : min
  return (
    <>
      <div className="fx-vp-line">
        {mark}
        {name}
        <span className="fx-vp-val tabular">{fmtVst(num, p)}</span>
      </div>
      <div className="fx-vp-slider">
        <Slider value={num} min={min} max={max} step={(max - min) / 1000} onChange={onChange} label={p.name} />
      </div>
    </>
  )
}
