import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { api } from './client'
import { keys } from './keys'
import { onServerEvent } from './live'
import type {
  BaseModel,
  CleanupModels,
  EffectSpec,
  MediaSections,
  VstPlugin,
  VstStatus,
  MixProcessor,
  Plugin,
  Character,
  Clip,
  DiskUsage,
  HealthReport,
  ID,
  Job,
  Media,
  Model,
  OrganizeOptions,
  Preferences,
  ProjectSummary,
  RenderStatus,
  RawImport,
  Resource,
  ScalarSeries,
  SeparationModels,
  SystemInfo,
  TrainPreset,
  ValidationSnapshot,
} from './types'
import { isActiveJob } from './types'

export function useSystem() {
  return useQuery({ queryKey: keys.system, queryFn: () => api.get<SystemInfo>('/system'), staleTime: 60_000 })
}

export function useRenderStatus() {
  return useQuery({ queryKey: keys.renderStatus, queryFn: () => api.get<RenderStatus>('/system/render'), refetchInterval: 30_000 })
}

export function usePreferences() {
  return useQuery({ queryKey: keys.preferences, queryFn: () => api.get<Preferences>('/preferences'), staleTime: Infinity })
}

export function useDisk(enabled = true) {
  return useQuery({ queryKey: keys.disk, queryFn: () => api.get<DiskUsage>('/disk'), enabled, staleTime: 30_000 })
}

export function useCharacters(archived = false) {
  return useQuery({
    queryKey: keys.characters(archived),
    queryFn: () => api.get<Character[]>('/characters', { archived: archived ? 1 : 0 }),
    staleTime: 30_000,
  })
}

export function useCharacter(id: ID | undefined) {
  const qc = useQueryClient()
  return useQuery({
    queryKey: keys.character(id ?? ''),
    queryFn: () => api.get<Character>(`/characters/${id}`),
    enabled: !!id,
    initialData: () => {
      for (const archived of [false, true]) {
        const hit = qc.getQueryData<Character[]>(keys.characters(archived))?.find((c) => c.id === id)
        if (hit) return hit
      }
      return undefined
    },
    staleTime: 10_000,
  })
}

export function useRaw(id: ID) {
  return useQuery({ queryKey: keys.raw(id), queryFn: () => api.get<RawImport[]>(`/characters/${id}/raw`) })
}

export function useClips(id: ID, status: 'active' | 'trashed' | 'all' = 'all') {
  return useQuery({ queryKey: keys.clips(id, status), queryFn: () => api.get<Clip[]>(`/characters/${id}/clips`, { status }) })
}

export function useHealth(id: ID) {
  return useQuery({ queryKey: keys.health(id), queryFn: () => api.get<HealthReport | null>(`/characters/${id}/health`) })
}

export function useOrganizeDefaults(id: ID) {
  return useQuery({ queryKey: keys.organizeDefaults(id), queryFn: () => api.get<OrganizeOptions>(`/characters/${id}/organize/defaults`), staleTime: Infinity })
}

export function useTrainPresets() {
  return useQuery({ queryKey: keys.presets, queryFn: () => api.get<TrainPreset[]>('/train/presets'), staleTime: Infinity })
}

export function useModels(characterId?: ID) {
  return useQuery({
    queryKey: keys.models(characterId),
    queryFn: () => api.get<Model[]>('/models', characterId ? { characterId } : undefined),
  })
}

export function useModel(id: ID | null | undefined) {
  return useQuery({ queryKey: keys.model(id ?? ''), queryFn: () => api.get<Model>(`/models/${id}`), enabled: !!id })
}

export function useValidationSteps(modelId: ID | null | undefined) {
  return useQuery({
    queryKey: keys.validationSteps(modelId ?? ''),
    queryFn: () => api.get<number[]>(`/models/${modelId}/validation/steps`),
    enabled: !!modelId,
  })
}

export function useValidation(modelId: ID | null | undefined, step: number | 'latest') {
  return useQuery({
    queryKey: keys.validation(modelId ?? '', step),
    queryFn: () => api.get<ValidationSnapshot | null>(`/models/${modelId}/validation`, step === 'latest' ? undefined : { step }),
    enabled: !!modelId,
  })
}

export function useProjects() {
  return useQuery({ queryKey: keys.projects, queryFn: () => api.get<ProjectSummary[]>('/projects') })
}

export function useMedia(projectId: ID | null | undefined) {
  return useQuery({
    queryKey: keys.media(projectId ?? ''),
    queryFn: () => api.get<Media[]>(`/projects/${projectId}/media`),
    enabled: !!projectId,
  })
}

export function useJobs() {
  return useQuery({ queryKey: keys.jobs, queryFn: () => api.get<Job[]>('/jobs'), staleTime: Infinity })
}

export function useActiveJobs(filter?: (j: Job) => boolean): Job[] {
  const { data } = useJobs()
  return (data ?? []).filter((j) => isActiveJob(j) && (!filter || filter(j)))
}

export function useResources() {
  return useQuery({ queryKey: keys.resources, queryFn: () => api.get<Resource[]>('/resources'), staleTime: Infinity })
}

/** Vocal separation models (Demucs, Roformer checkpoints, imported ones) and the default. */
export function useSeparationModels() {
  return useQuery({ queryKey: keys.separationModels, queryFn: () => api.get<SeparationModels>('/resources/separation'), staleTime: Infinity })
}

/** Vocal cleanup (dereverb) models and the default (null while none is ready). */
export function useCleanupModels() {
  return useQuery({ queryKey: keys.cleanupModels, queryFn: () => api.get<CleanupModels>('/resources/cleanup'), staleTime: Infinity })
}

/**
 * Training curves: one fetch of the full (downsampled) history, then live appends from
 * `metrics` events — the backend reads tfevents, nothing touches the training loop.
 */
export function useMetrics(modelId: ID | null | undefined) {
  const [series, setSeries] = useState<Record<string, ScalarSeries['points']>>({})
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    if (!modelId) return
    let cancelled = false
    setLoaded(false)
    setSeries({})
    api
      .get<ScalarSeries[]>(`/models/${modelId}/metrics`, { maxPoints: 1500 })
      .then((list) => {
        if (cancelled) return
        const map: Record<string, ScalarSeries['points']> = {}
        for (const s of list) map[s.tag] = s.points
        setSeries(map)
        setLoaded(true)
      })
      .catch(() => !cancelled && setLoaded(true))
    const off = onServerEvent((ev) => {
      if (ev.type !== 'metrics' || ev.data.modelId !== modelId) return
      const now = Date.now() / 1000
      setSeries((prev) => {
        const next = { ...prev }
        for (const [tag, value] of Object.entries(ev.data.scalars)) {
          const pts = next[tag] ? next[tag].slice() : []
          const last = pts[pts.length - 1]
          if (!last || last[0] < ev.data.step) pts.push([ev.data.step, value, now])
          next[tag] = pts.length > 4000 ? pts.filter((_, i) => i % 2 === 0 || i === pts.length - 1) : pts
        }
        return next
      })
    })
    return () => {
      cancelled = true
      off()
    }
  }, [modelId])

  return { series, loaded }
}

/** Fallback while the backend predates /mix/processors: the built-in mixer with its defaults. */
const BUILTIN_FALLBACK: MixProcessor = {
  id: 'builtin',
  name: { code: 'mix.builtin' },
  kind: 'builtin',
  presets: ['natural', 'pop', 'airy', 'dry'].map((id) => ({ id, name: { code: `mix.preset.${id}` } })),
  params: [],
  maxVocals: 4,
  available: true,
  error: null,
}

export function useMixProcessors() {
  return useQuery({
    queryKey: keys.mixProcessors,
    queryFn: () => api.get<MixProcessor[]>('/mix/processors').catch(() => [BUILTIN_FALLBACK]),
    staleTime: 60_000,
  })
}

export function usePlugins() {
  return useQuery({ queryKey: keys.plugins, queryFn: () => api.get<Plugin[]>('/plugins').catch(() => [] as Plugin[]) })
}

export function useBaseModels() {
  return useQuery({ queryKey: keys.baseModels, queryFn: () => api.get<BaseModel[]>('/base-models').catch(() => [] as BaseModel[]) })
}

export function useFxEffects() {
  return useQuery({ queryKey: keys.fxEffects, queryFn: () => api.get<EffectSpec[]>('/fx/effects').catch(() => [] as EffectSpec[]), staleTime: Infinity })
}

export function useVstStatus() {
  return useQuery({ queryKey: keys.vstStatus, queryFn: () => api.get<VstStatus>('/vst/status').catch(() => null) })
}

export function useVstPlugins() {
  return useQuery({ queryKey: keys.vstPlugins, queryFn: () => api.get<VstPlugin[]>('/vst/plugins').catch(() => [] as VstPlugin[]) })
}

/** Beats, phrases and sections of a song (for snapping cuts); cached by the backend. */
export function useSections(projectId: ID | null | undefined, mediaId: ID | null | undefined) {
  return useQuery({
    queryKey: keys.sections(projectId ?? '', mediaId ?? ''),
    queryFn: () => api.get<MediaSections>(`/projects/${projectId}/media/${mediaId}/sections`),
    enabled: !!projectId && !!mediaId,
    staleTime: Infinity,
    retry: false,
  })
}
