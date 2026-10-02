// The user's own presets: mix settings (processor + preset + params) kept like the effect-chain
// presets in this app profile, and both kinds saved to / loaded from `.stellapreset` files.

import { create } from 'zustand'
import { api } from '../../api/client'
import type { FxSlot } from '../../api/types'
import type { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { isTauri, pickFiles, pickSavePath } from '../../shell/bridge'
import { toast } from '../../state/ui'
import { slotId, usePresets as useFxPresets } from './fx/fxModel'

type ParamValue = number | string | boolean

export interface MixPreset {
  id: string
  name: string
  /** 'builtin' or a plugin id */
  processor: string
  /** the processor's preset the params apply on top of */
  preset: string | null
  params: Record<string, ParamValue>
  createdAt: string
}

const MIX_KEY = 'stellaria.mixPresets'

function readMix(): MixPreset[] {
  try {
    const raw = localStorage.getItem(MIX_KEY)
    const list: unknown = raw ? JSON.parse(raw) : []
    return Array.isArray(list) ? list.filter(isMixPreset) : []
  } catch {
    return []
  }
}

function writeMix(presets: MixPreset[]): boolean {
  try {
    localStorage.setItem(MIX_KEY, JSON.stringify(presets))
    return true
  } catch {
    return false
  }
}

function isMixPreset(p: unknown): p is MixPreset {
  const x = p as MixPreset
  return !!x && typeof x.name === 'string' && !!x.name.trim() && typeof x.processor === 'string' && !!x.params && typeof x.params === 'object'
}

const sameName = (a: string, b: string) => a.trim().toLowerCase() === b.trim().toLowerCase()

export const useMixPresets = create<{
  presets: MixPreset[]
  /** saves (replacing one with the same name); false when storage is unavailable */
  save: (entry: Omit<MixPreset, 'id' | 'createdAt'>) => boolean
  remove: (id: string) => void
}>((set, get) => ({
  presets: readMix(),
  save: (entry) => {
    const rest = get().presets.filter((p) => !sameName(p.name, entry.name))
    const presets = [...rest, { ...entry, name: entry.name.trim(), params: { ...entry.params }, id: slotId(), createdAt: new Date().toISOString() }]
    set({ presets })
    return writeMix(presets)
  },
  remove: (id) => {
    const presets = get().presets.filter((p) => p.id !== id)
    set({ presets })
    writeMix(presets)
  },
}))

// ---------- files ----------

export const PRESET_EXT = 'stellapreset'
const FORMAT = 'stellaria-presets'

export interface PresetFile {
  format: typeof FORMAT
  version: 1
  mix: Pick<MixPreset, 'name' | 'processor' | 'preset' | 'params'>[]
  fx: { name: string; chain: FxSlot[] }[]
}

function fileName(name: string) {
  return (name.replace(/[\\/:*?"<>|]+/g, ' ').trim() || 'presets').slice(0, 80)
}

/** The user's presets of one kind as a file. */
export function presetFile(kind: 'mix' | 'fx'): PresetFile {
  return {
    format: FORMAT,
    version: 1,
    mix: kind === 'mix' ? useMixPresets.getState().presets.map(({ name, processor, preset, params }) => ({ name, processor, preset, params })) : [],
    fx: kind === 'fx' ? useFxPresets.getState().presets.map(({ name, chain }) => ({ name, chain })) : [],
  }
}

/** Saves `file`: a save dialog in the app, a download in the browser. */
export async function savePresetFile(file: PresetFile, name: string, t: ReturnType<typeof useT>): Promise<void> {
  const count = file.mix.length + file.fx.length
  if (count === 0) return
  if (isTauri) {
    const path = await pickSavePath(`${fileName(name)}.${PRESET_EXT}`, [PRESET_EXT])
    if (!path) return
    const res = await attempt(() => api.post<{ path: string }>('/presets/export', { path, file }))
    if (res) toast({ kind: 'success', title: t('presets.exported', { n: count }), body: res.path })
    return
  }
  const url = URL.createObjectURL(new Blob([JSON.stringify(file, null, 2)], { type: 'application/json' }))
  const a = document.createElement('a')
  a.href = url
  a.download = `${fileName(name)}.${PRESET_EXT}`
  a.click()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
  toast({ kind: 'success', title: t('presets.exported', { n: count }) })
}

/** The well-formed presets of a file (the backend checks the same in the app). */
function readFile(data: unknown): PresetFile | null {
  const d = data as Partial<PresetFile> | null
  if (!d || d.format !== FORMAT || typeof d.version !== 'number' || d.version > 1) return null
  return {
    format: FORMAT,
    version: 1,
    mix: (Array.isArray(d.mix) ? d.mix : []).filter(isMixPreset),
    fx: (Array.isArray(d.fx) ? d.fx : []).filter((p) => !!p && typeof p.name === 'string' && !!p.name.trim() && Array.isArray(p.chain)),
  }
}

/** Picks a `.stellapreset` file and adds its presets (both kinds; same names are replaced). */
export async function importPresetFile(t: ReturnType<typeof useT>): Promise<void> {
  const picked = await pickFiles({ multiple: false, extensions: [PRESET_EXT], filterName: t('presets.file_kind') })
  const first = picked?.[0]
  if (!first) return
  let file: PresetFile | null
  if (typeof first === 'string') {
    const res = await attempt(() => api.post<unknown>('/presets/import', { path: first }))
    if (res === undefined) return
    file = readFile(res)
  } else {
    try {
      file = readFile(JSON.parse(await first.text()))
    } catch {
      file = null
    }
  }
  if (!file) {
    toast({ kind: 'error', title: t('presets.invalid') })
    return
  }
  const mix = useMixPresets.getState()
  for (const p of file.mix) mix.save(p)
  const fx = useFxPresets.getState()
  for (const p of file.fx) fx.save(p.name, p.chain.map((s) => ({ ...s, id: slotId(), params: { ...s.params } })))
  const n = file.mix.length + file.fx.length
  toast(n ? { kind: 'success', title: t('presets.imported', { n }) } : { kind: 'info', title: t('presets.empty_file') })
}
