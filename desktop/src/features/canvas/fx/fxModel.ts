// Effects chain helpers: slots, presets (local), editor jobs, plugin name memory, formatting.

import { create } from 'zustand'
import { api, ApiError } from '../../../api/client'
import type { EffectSpec, FxSlot, Job, VstParam, VstPlugin } from '../../../api/types'
import { translateMsg, useLang, type StringKey } from '../../../i18n'
import { errorText } from '../../../lib/actions'
import { waitForJob } from '../../../lib/jobs'
import { toast } from '../../../state/ui'

export type FxValue = number | boolean | string
export type FxParamValues = Record<string, FxValue>
export type VstSlot = Extract<FxSlot, { kind: 'vst' }>
export type BuiltinSlot = Extract<FxSlot, { kind: 'builtin' }>

export const CATEGORY_ORDER: EffectSpec['category'][] = ['eq', 'dynamics', 'space', 'color', 'utility']

export function slotId(): string {
  try {
    return crypto.randomUUID()
  } catch {
    return `fx_${Math.random().toString(36).slice(2, 10)}${Date.now().toString(36)}`
  }
}

/** Every default written out, so a chain keeps its sound when the app's defaults change later. */
export function effectDefaults(spec: EffectSpec): FxParamValues {
  const out: FxParamValues = {}
  for (const p of spec.params) if (p.default !== null && p.default !== undefined) out[p.id] = p.default
  return out
}

export function newBuiltinSlot(spec: EffectSpec, mix = 1): BuiltinSlot {
  return { id: slotId(), bypass: false, mix, kind: 'builtin', effect: spec.id, params: effectDefaults(spec) }
}

export function newVstSlot(plugin: VstPlugin): VstSlot {
  return { id: slotId(), bypass: false, mix: 1, kind: 'vst', pluginId: plugin.id, state: null, params: {} }
}

/** A copy of a chain with fresh slot ids (loading a preset, duplicating). */
export function withFreshIds(chain: FxSlot[]): FxSlot[] {
  return chain.map((s) => ({ ...s, id: slotId(), params: { ...s.params } }))
}

export function moveItem<T>(list: T[], from: number, to: number): T[] {
  const next = list.slice()
  const [item] = next.splice(from, 1)
  next.splice(to, 0, item)
  return next
}

// ---------- presets ----------

export interface FxPreset {
  id: string
  name: string
  chain: FxSlot[]
  createdAt: string
}

const PRESETS_KEY = 'stellaria.fxPresets'

function readPresets(): FxPreset[] {
  try {
    const raw = localStorage.getItem(PRESETS_KEY)
    const list: unknown = raw ? JSON.parse(raw) : []
    if (!Array.isArray(list)) return []
    return list.filter((p): p is FxPreset => !!p && typeof p.name === 'string' && Array.isArray(p.chain))
  } catch {
    return []
  }
}

function writePresets(presets: FxPreset[]): boolean {
  try {
    localStorage.setItem(PRESETS_KEY, JSON.stringify(presets))
    return true
  } catch {
    return false
  }
}

export const usePresets = create<{ presets: FxPreset[]; save: (name: string, chain: FxSlot[]) => boolean; remove: (id: string) => void }>((set, get) => ({
  presets: readPresets(),
  save: (name, chain) => {
    const key = name.trim().toLowerCase()
    const rest = get().presets.filter((p) => p.name.trim().toLowerCase() !== key)
    const presets = [...rest, { id: slotId(), name: name.trim(), chain: chain.map((s) => ({ ...s })), createdAt: new Date().toISOString() }]
    set({ presets })
    return writePresets(presets)
  },
  remove: (id) => {
    const presets = get().presets.filter((p) => p.id !== id)
    set({ presets })
    writePresets(presets)
  },
}))

/** Starter chains; effects the backend doesn't have are skipped. `mix` is the slot's dry/wet. */
export const STARTERS: { id: string; name: StringKey; slots: { effect: string; mix?: number; params?: FxParamValues }[] }[] = [
  {
    id: 'vocal',
    name: 'fx.starter.vocal',
    slots: [{ effect: 'eq' }, { effect: 'compressor' }, { effect: 'deesser' }, { effect: 'smooth', params: { amount: 0.4 } }],
  },
  {
    // a classic hand-built vocal chain: firm de-essing, a broad low cut, presence and air, 5:1 leveling
    id: 'classic',
    name: 'fx.starter.classic',
    slots: [
      { effect: 'deesser', params: { frequency: 6550, range: 12, sensitivity: 0.5 } },
      {
        effect: 'eq',
        params: {
          highPass: true, highPassFreq: 90, lowShelfFreq: 230, lowShelfGain: -3,
          band1Freq: 2700, band1Gain: -2, band1Q: 1, band2Freq: 1400, band2Gain: 3, band2Q: 1.15,
          band3Freq: 8500, band3Gain: 2.5, band3Q: 1, highShelfFreq: 20000, highShelfGain: -2,
        },
      },
      { effect: 'compressor', params: { autoThreshold: true, ratio: 5, attack: 19, release: 400, knee: 4.5, autoMakeup: true } },
      { effect: 'reverb', mix: 0.3, params: { type: 'plate', decay: 1.5, preDelay: 15, lowCut: 200, level: -16 } },
    ],
  },
  { id: 'space', name: 'fx.starter.space', slots: [{ effect: 'reverb', mix: 0.25 }, { effect: 'delay', mix: 0.15 }] },
  { id: 'warm', name: 'fx.starter.warm', slots: [{ effect: 'saturation', mix: 0.4 }, { effect: 'doubler', mix: 0.35 }] },
  { id: 'master', name: 'fx.starter.master', slots: [{ effect: 'eq' }, { effect: 'compressor' }, { effect: 'limiter' }] },
]

export function starterChain(starter: (typeof STARTERS)[number], specs: Map<string, EffectSpec>): FxSlot[] {
  return starter.slots.flatMap((s) => {
    const spec = specs.get(s.effect)
    if (!spec) return []
    const slot = newBuiltinSlot(spec, s.mix ?? 1)
    // only params this backend knows
    for (const [k, v] of Object.entries(s.params ?? {})) if (spec.params.some((p) => p.id === k)) slot.params[k] = v
    return [slot]
  })
}

// ---------- plugin names (so a missing plugin can still be named) ----------

const NAMES_KEY = 'stellaria.vstNames'

export function rememberPlugins(plugins: VstPlugin[]) {
  try {
    const known = JSON.parse(localStorage.getItem(NAMES_KEY) ?? '{}') as Record<string, { name: string; vendor: string | null }>
    let changed = false
    for (const p of plugins) {
      if (known[p.id]?.name !== p.name) {
        known[p.id] = { name: p.name, vendor: p.vendor }
        changed = true
      }
    }
    if (changed) localStorage.setItem(NAMES_KEY, JSON.stringify(known))
  } catch {
    /* storage unavailable: missing plugins just stay unnamed */
  }
}

export function lastKnownPlugin(id: string): { name: string; vendor: string | null } | null {
  try {
    const known = JSON.parse(localStorage.getItem(NAMES_KEY) ?? '{}') as Record<string, { name: string; vendor: string | null }>
    return known[id] ?? null
  } catch {
    return null
  }
}

// ---------- native plugin windows ----------

/** slot id → running `vst_editor` job, shared so a reopened inspector still shows the open window. */
export const useEditorJobs = create<{ jobs: Record<string, string> }>(() => ({ jobs: {} }))

type EditorResult = { state: string | null; params: FxParamValues }
type Applier = (slotId: string, result: EditorResult) => boolean
const appliers = new Set<Applier>()

/** A mounted chain editor writes editor results into its current chain (returns false if the slot isn't its). */
export function registerApplier(fn: Applier): () => void {
  appliers.add(fn)
  return () => {
    appliers.delete(fn)
  }
}

function dropEditorJob(slotKey: string) {
  useEditorJobs.setState((s) => {
    const jobs = { ...s.jobs }
    delete jobs[slotKey]
    return { jobs }
  })
}

/** Localised error with the plugin's name filled in (backend details may carry more params as JSON). */
export function showFxError(code: string, message: string, detail: string | undefined, params: Record<string, string | number>) {
  let extra: Record<string, string | number> = {}
  if (detail?.startsWith('{')) {
    try {
      const parsed = JSON.parse(detail) as { params?: Record<string, string | number> }
      extra = parsed.params ?? {}
    } catch {
      /* plain text detail */
    }
  }
  const lang = useLang.getState().lang
  const title = translateMsg(lang, { code, params: { seconds: 30, reason: detail ?? '', ...params, ...extra } }, message)
  toast({ kind: 'error', title, detail: detail && !detail.startsWith('{') ? detail : undefined })
}

/**
 * Opens the plugin's own window. Resolves when the window is closed; the new state and parameter
 * values go into whichever editor shows the slot then, else into `fallback` (the chain it was opened from).
 */
export async function openPluginEditor(slot: VstSlot, pluginName: string, fallback: (result: EditorResult) => void) {
  let job: Job
  try {
    job = await api.post<Job>(`/vst/plugins/${slot.pluginId}/editor`, { state: slot.state })
  } catch (e) {
    if (e instanceof ApiError) showFxError(e.code, e.message, e.detail, { name: pluginName })
    else toast({ kind: 'error', ...errorText(e) })
    return
  }
  useEditorJobs.setState((s) => ({ jobs: { ...s.jobs, [slot.id]: job.id } }))
  let done: Job
  try {
    done = await waitForJob(job.id)
  } catch (e) {
    toast({ kind: 'error', ...errorText(e) })
    return
  } finally {
    dropEditorJob(slot.id)
  }
  if (done.status === 'succeeded') {
    const r = (done.result ?? {}) as Partial<EditorResult>
    const result: EditorResult = { state: r.state ?? slot.state, params: r.params ?? slot.params }
    if (![...appliers].some((apply) => apply(slot.id, result))) fallback(result)
  } else if (done.status === 'failed' && done.error) {
    showFxError(done.error.code, done.error.message, done.error.detail, { name: pluginName })
  }
}

export function closePluginEditor(jobId: string) {
  return api.post(`/jobs/${jobId}/cancel`).catch(() => undefined)
}

// ---------- tempo ----------

export const NOTE_VALUES: { id: string; beats: number }[] = [
  { id: '1/4', beats: 1 },
  { id: '1/8.', beats: 0.75 },
  { id: '1/8', beats: 0.5 },
  { id: '1/4T', beats: 2 / 3 },
  { id: '1/16', beats: 0.25 },
]

export function noteMs(beats: number, bpm: number): number {
  return (60000 / bpm) * beats
}

/** "1/8", "1/8d", "1/8.", "1/4t", "1/4T" → length in beats (quarter notes); null when it isn't a note value. */
export function parseNote(value: string): number | null {
  const m = /^1\/(\d{1,2})([.dDtT]?)$/.exec(value.trim())
  if (!m) return null
  const base = 4 / Number(m[1])
  if (m[2] === '.' || m[2].toLowerCase() === 'd') return base * 1.5
  if (m[2].toLowerCase() === 't') return (base * 2) / 3
  return base
}

// ---------- VST parameters ----------

export function vstRange(p: VstParam): [number, number] {
  const min = p.min ?? 0
  const max = p.max ?? 1
  return max > min ? [min, max] : [min, min + 1]
}

export function fmtVst(v: number, p: VstParam): string {
  const [min, max] = vstRange(p)
  const unit = p.unit?.trim() ?? ''
  if (/^hz$/i.test(unit) && Math.abs(v) >= 1000) return `${(v / 1000).toFixed(Math.abs(v) >= 10000 ? 1 : 2)} kHz`
  const span = max - min
  const digits = span >= 100 ? 0 : span >= 10 ? 1 : 2
  const text = v.toFixed(digits)
  return unit ? `${text} ${unit}` : text
}

export function sameValue(a: FxValue | null | undefined, b: FxValue | null | undefined, p?: VstParam): boolean {
  if (typeof a === 'number' && typeof b === 'number') {
    const [min, max] = p ? vstRange(p) : [0, 1]
    return Math.abs(a - b) <= Math.max(1e-6, (max - min) * 1e-4)
  }
  return a === b
}

/** Short stable key for a (possibly large) base64 plugin state. */
export function stateKey(state: string | null): string {
  if (!state) return ''
  let h = 0
  for (let i = 0; i < state.length; i++) h = (Math.imul(31, h) + state.charCodeAt(i)) | 0
  return `${state.length}:${(h >>> 0).toString(36)}`
}
