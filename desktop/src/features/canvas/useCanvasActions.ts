import { useReactFlow } from '@xyflow/react'
import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type RefObject, type SetStateAction } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, mediaUrl, sendFiles } from '../../api/client'
import { onServerEvent } from '../../api/live'
import type { CanvasNode, Character, Job, KeySuggestion, Media, Model, ProcessOp, Project, RenderData, SourceData, StemPort, Track } from '../../api/types'
import { useT } from '../../i18n'
import { attempt } from '../../lib/actions'
import { VOICE_COLORS } from '../../lib/color'
import { waitForJob } from '../../lib/jobs'
import { rememberedParams, rememberParams } from '../../lib/voiceParams'
import { isTauri, pickFiles, pickSavePath } from '../../shell/bridge'
import { usePlayer } from '../../state/player'
import { toast } from '../../state/ui'
import type { CatalogId } from './catalog'
import { copyNodes, duplicateNodes, pasteNodes } from './clipboard'
import { mediaAudioPath, type AddOptions, type CanvasActions, type SingOptions } from './context'
import { expectImport, takeImport, type ImportIntent } from './imports'
import {
  byIdOf,
  inputArity,
  inputsOf,
  insertOnEdge,
  isProcess,
  isSource,
  isTake,
  makeRoom,
  newAudio,
  newMix,
  newMixInput,
  newProcess,
  newSource,
  newTake,
  nextSourceSlot,
  playableMedia,
  removeNodes as removeFromNodes,
  resolvePort,
  rootSourceId,
  slotRightOf,
  stemMedia,
} from './model'
import { postRun, startRun } from './runner'
import { newId, type UpdateOptions } from './useProject'
import { latestRelease, voiceModelFor } from './voices'

export const FIT = { padding: 0.2, maxZoom: 1, duration: 400 }

export type Point = { x: number; y: number }

export interface CanvasActionArgs {
  projectId: string
  /** the project as rendered */
  project: Project | null
  /** the project as it is now, edits since the last render included (useProject) */
  getProject: () => Project | null
  update: (fn: (p: Project) => Project, opts?: UpdateOptions) => void
  flush: () => Promise<void>
  /** the project's media, loaded (first open fills an empty canvas with its songs) */
  mediaList: Media[] | undefined
  media: Map<string, Media>
  characters: Map<string, Character>
  modelList: Model[] | undefined
  selection: string[]
  setSelection: Dispatch<SetStateAction<string[]>>
  setRenaming: (id: string | null) => void
  /** open 多人试唱 for these songs */
  setBatchFor: (ids: string[]) => void
  showTimeline: () => void
  /** last pointer position on screen */
  mouse: RefObject<Point>
}

/**
 * Everything the canvas can do to its project (the CanvasActions its nodes and panels call),
 * plus the helpers the workspace's menus, keys and flow handlers share.
 */
export function useCanvasActions({
  projectId,
  project,
  getProject,
  update,
  flush,
  mediaList,
  media,
  characters,
  modelList,
  selection,
  setSelection,
  setRenaming,
  setBatchFor,
  showTimeline,
  mouse,
}: CanvasActionArgs) {
  const t = useT()
  const nav = useNavigate()
  const rf = useReactFlow()
  const [suggestions, setSuggestions] = useState<Record<string, KeySuggestion>>({})

  const nodesNow = useCallback(() => getProject()?.canvas.nodes ?? [], [getProject])
  const findNode = useCallback((id: string) => nodesNow().find((n) => n.id === id), [nodesNow])
  const measure = useCallback(
    (id: string) => {
      const n = rf.getNode(id)
      return n?.measured?.width && n.measured.height ? { w: n.measured.width, h: n.measured.height } : undefined
    },
    [rf],
  )
  const flowAtMouse = useCallback(() => rf.screenToFlowPosition(mouse.current), [rf, mouse])

  const setNodes = useCallback(
    (fn: (nodes: CanvasNode[]) => CanvasNode[], opts?: UpdateOptions) =>
      update((p) => {
        const nodes = fn(p.canvas.nodes)
        return nodes === p.canvas.nodes ? p : { ...p, canvas: { ...p.canvas, nodes, edges: [] } }
      }, opts),
    [update],
  )

  const reveal = useCallback(
    (ids: string[]) => {
      window.setTimeout(() => {
        const nodes = ids.map((id) => rf.getNode(id)).filter((n): n is NonNullable<typeof n> => !!n)
        if (nodes.length) rf.fitView({ nodes, ...FIT, padding: 0.6 })
      }, 60)
    },
    [rf],
  )

  // ---------- imports ----------
  const addImported = useCallback(
    (mediaIds: string[], intent: ImportIntent) => {
      const nodes = nodesNow()
      const present = new Set(nodes.flatMap((n) => (n.type === 'source' || n.type === 'audio' ? [(n.data as SourceData).mediaId] : [])))
      const fresh = mediaIds.filter((id) => !present.has(id) && media.get(id)?.kind !== 'stem')
      if (fresh.length === 0) return
      let slot = intent.position ?? nextSourceSlot(nodes, measure)
      const added: CanvasNode[] = []
      for (const mid of fresh) {
        const name = media.get(mid)?.name ?? 'Audio'
        added.push(intent.kind === 'audio' ? newAudio(newId('nd'), slot, mid, name) : newSource(newId('nd'), slot, mid, name))
        slot = { x: slot.x, y: slot.y + (intent.kind === 'audio' ? 140 : 250) }
      }
      setNodes((ns) => [...ns, ...added])
      // a new song is selected, so the side panel offers what to do with it (一键出歌 first)
      if (intent.kind === 'song') {
        setSelection([added[0].id])
        reveal(added.map((n) => n.id))
      }
    },
    [nodesNow, setNodes, media, measure, setSelection, reveal],
  )

  // first open: put every imported song on an empty canvas
  const populated = useRef(false)
  useEffect(() => {
    if (!project || !mediaList || populated.current) return
    populated.current = true
    if (project.canvas.nodes.length === 0) {
      const imports = mediaList.filter((m) => m.kind === 'import').map((m) => m.id)
      if (imports.length > 0) addImported(imports, { kind: 'song' })
    }
  }, [project, mediaList, addImported])

  // imports finishing while the canvas is open become nodes
  useEffect(
    () =>
      onServerEvent((ev) => {
        if (ev.type !== 'job' || ev.data.type !== 'import' || ev.data.target.projectId !== projectId || ev.data.status !== 'succeeded') return
        const ids = (ev.data.result as { mediaIds?: string[] } | null)?.mediaIds ?? []
        const intent = takeImport(ev.data.id)
        window.setTimeout(() => addImported(ids, intent), 300)
      }),
    [projectId, addImported],
  )

  const importFiles = useCallback(
    async (kind: 'song' | 'audio', position?: Point) => {
      const picked = await pickFiles({ multiple: true })
      if (!picked || picked.length === 0) return
      const job = await attempt(() => sendFiles<Job>(`/projects/${projectId}/media`, picked))
      if (job) expectImport(job.id, { kind, position })
    },
    [projectId],
  )

  // ---------- actions ----------
  const patchNode: CanvasActions['patchNode'] = useCallback((nodeId, fn, opts) => setNodes((nodes) => nodes.map((n) => (n.id === nodeId ? fn(n) : n)), opts), [setNodes])

  const removeNodes: CanvasActions['removeNodes'] = useCallback(
    (ids) => {
      setNodes((nodes) => removeFromNodes(nodes, ids))
      setSelection((sel) => sel.filter((x) => !ids.includes(x)))
    },
    [setNodes, setSelection],
  )

  const run: CanvasActions['run'] = useCallback(
    async (targets) => {
      await flush()
      await startRun(projectId, targets, (r) => {
        if (r.failed || r.skipped) {
          const title = r.skipped ? t('run.done_failed_skipped', { done: r.done, failed: r.failed, skipped: r.skipped }) : t('run.done_failed', { done: r.done, failed: r.failed })
          const body = r.error ? t.msg({ code: r.error.code, params: r.error.params }, r.error.message) : undefined
          const action = r.error && ['addon.missing', 'addon.outdated', 'separation.model_missing'].includes(r.error.code) ? { label: t('run.install_addon'), run: () => nav('/resources') } : undefined
          toast({ kind: 'error', title, body, action })
        } else if (r.done > 1 || !targets || targets.length > 1) toast({ kind: 'success', title: t('run.done', { n: r.done }) })
      })
    },
    [flush, projectId, t, nav],
  )

  const runNode: CanvasActions['runNode'] = useCallback(
    async (nodeId) => {
      const n = findNode(nodeId)
      if (isTake(n)) rememberParams(n.data.characterId, n.data.params)
      // a single node whose inputs exist: start it directly (it may be up to date, re-run anyway)
      const byId = byIdOf(nodesNow())
      if (n && inputsOf(n).every((r) => resolvePort(byId, r)) && inputArity(n) !== 0) {
        await flush()
        const job = await attempt(() => postRun(projectId, nodeId))
        if (job) patchNode(nodeId, (x) => ({ ...x, data: { ...x.data, status: 'queued', jobId: job.id, error: null } as CanvasNode['data'] }), { history: false })
        return
      }
      await run([nodeId])
    },
    [findNode, nodesNow, flush, projectId, patchNode, run],
  )

  const separate: CanvasActions['separate'] = useCallback(
    async (sourceId, stems = 2) => {
      const before = findNode(sourceId)
      if (!isSource(before)) return
      patchNode(sourceId, (n) => ({ ...n, data: { ...(n.data as SourceData), stems: { ...(n.data as SourceData).stems, status: 'queued', error: null } } }), { history: false })
      await flush()
      const job = await attempt(() => postRun(projectId, sourceId, 'separate', stems))
      if (!job) patchNode(sourceId, (n) => ({ ...n, data: { ...(n.data as SourceData), stems: before.data.stems } }), { history: false })
    },
    [findNode, patchNode, flush, projectId],
  )

  const sing: CanvasActions['sing'] = useCallback(
    async (input, characterId, opts: SingOptions = {}) => {
      const c = characters.get(characterId)
      if (!c || !findNode(input.nodeId)) return null
      const model = voiceModelFor(c, modelList ?? [])
      if (!model) {
        toast({ kind: 'info', title: t('create.no_voices') })
        return null
      }
      const id = newId('nd')
      const position = opts.position ?? slotRightOf(nodesNow(), input.nodeId, measure)
      const releaseId = opts.releaseId !== undefined ? opts.releaseId : (latestRelease(model)?.id ?? null)
      const take = newTake(id, input, characterId, model.id, releaseId, { ...rememberedParams(characterId), ...opts.params }, position, opts.batchId ?? null)
      setNodes((nodes) => [...nodes, take])
      if (opts.focus !== false) {
        setSelection([id])
        reveal([input.nodeId, id])
      }
      const mid = resolvePort(byIdOf(nodesNow()), input)
      if (mid && opts.params?.key === undefined) {
        const s = await api.get<KeySuggestion>(`/projects/${projectId}/suggest-key`, { mediaId: mid, characterId }).catch(() => null)
        if (s) {
          setSuggestions((prev) => ({ ...prev, [id]: s }))
          if (s.confidence === 'high' && s.key !== 0) patchNode(id, (n) => ({ ...n, data: { ...(n.data as RenderData), params: { ...(n.data as RenderData).params, key: s.key } } }), { history: false })
        }
      }
      return id
    },
    [characters, findNode, modelList, t, nodesNow, measure, setNodes, setSelection, reveal, projectId, patchNode],
  )

  const variant: CanvasActions['variant'] = useCallback(
    (nodeId) => {
      const n = findNode(nodeId)
      if (!isTake(n)) return
      const id = newId('nd')
      const position = slotRightOf(nodesNow(), n.data.input.nodeId, measure)
      const copy: CanvasNode = { ...n, id, position, data: { ...n.data, mediaId: null, status: 'draft', jobId: null, error: null, rating: 0, favorite: false, label: '', stale: false } }
      setNodes((nodes) => [...nodes, copy])
      setSelection([id])
      reveal([nodeId, id])
      if (suggestions[nodeId]) setSuggestions((prev) => ({ ...prev, [id]: prev[nodeId] }))
    },
    [findNode, nodesNow, measure, setNodes, setSelection, reveal, suggestions],
  )

  const createMix: CanvasActions['createMix'] = useCallback(
    (refs, position, processor = 'builtin') => {
      const nodes = nodesNow()
      const byId = byIdOf(nodes)
      const inputs = refs.reduce<ReturnType<typeof newMixInput>[]>((acc, ref) => [...acc, newMixInput(byId, ref, acc, newId('mi'))], [])
      // an auto mix of voices gets the song's accompaniment when there is one
      if (processor === 'builtin' && inputs.length && inputs.every((i) => i.role === 'lead' || i.role === 'harmony')) {
        const root = rootSourceId(byId, refs[0].nodeId)
        const src = root ? byId.get(root) : undefined
        if (isSource(src) && stemMedia(src.data, 'inst')) inputs.push(newMixInput(byId, { nodeId: src.id, port: 'inst' }, inputs, newId('mi'), 'backing'))
      }
      const id = newId('nd')
      const anchor = refs[0]?.nodeId
      const pos = position ?? (anchor ? slotRightOf(nodes, anchor, measure) : flowAtMouse())
      setNodes((ns) => [...ns, newMix(id, pos, inputs, processor)])
      setSelection([id])
      reveal([id, ...refs.map((r) => r.nodeId)])
      return id
    },
    [nodesNow, measure, flowAtMouse, setNodes, setSelection, reveal],
  )

  const addNode: CanvasActions['addNode'] = useCallback(
    async (id: CatalogId, { position, from, edge }: AddOptions) => {
      if (id === 'song' || id === 'audio') {
        await importFiles(id, position)
        return null
      }
      const nid = newId('nd')
      let node: CanvasNode
      if (id === 'note') node = { id: nid, type: 'note', position, width: 220, height: 150, data: { text: '', color: VOICE_COLORS[3] } }
      else if (id === 'group') node = { id: nid, type: 'group', position, width: 520, height: 340, data: { label: '', color: VOICE_COLORS[0] } }
      else if (id === 'mix' || id === 'bus') {
        const byId = byIdOf(nodesNow())
        const ref = from?.handle ? { nodeId: from.nodeId, port: from.handle } : null
        node = newMix(nid, position, ref ? [newMixInput(byId, ref, [], newId('mi'))] : [], id === 'bus' ? 'bus' : 'builtin')
      } else {
        const ref = from?.handle ? { nodeId: from.nodeId, port: from.handle } : edge ? { nodeId: edge.source, port: edge.sourceHandle } : null
        const mid = ref ? resolvePort(byIdOf(nodesNow()), ref) : null
        node = newProcess(id as ProcessOp, nid, position, ref && !edge ? [ref] : [], undefined, mid ? (media.get(mid)?.duration ?? 0) : 0)
      }
      if (edge) {
        // take the target's place; the target and what follows move right
        const target = findNode(edge.target)
        if (target) node = { ...node, position: { ...target.position } }
        const next = insertOnEdge(nodesNow(), edge, node)
        if (!next) return null
        setNodes(() => makeRoom(next, nid, edge.target, measure))
      } else setNodes((nodes) => [...nodes, node])
      setSelection([nid])
      return nid
    },
    [importFiles, nodesNow, media, findNode, setNodes, measure, setSelection],
  )

  const importStem: CanvasActions['importStem'] = useCallback(
    async (sourceId: string, port: StemPort) => {
      const picked = await pickFiles({ multiple: false })
      if (!picked || picked.length === 0) return
      await flush()
      await attempt(() => sendFiles(`/projects/${projectId}/nodes/${sourceId}/stems`, picked, { port }))
    },
    [flush, projectId],
  )

  const exportMedia: CanvasActions['exportMedia'] = useCallback(
    async (mediaId, name) => {
      let path: string | undefined
      if (isTauri) {
        const picked = await pickSavePath(`${name}.wav`, ['wav', 'flac', 'mp3'])
        if (!picked) return
        path = picked
      }
      const format = (path?.split('.').pop()?.toLowerCase() ?? 'wav') as 'wav' | 'flac' | 'mp3'
      const job = await attempt(() => api.post<Job>(`/projects/${projectId}/media/${mediaId}/export`, { path, format: ['wav', 'flac', 'mp3'].includes(format) ? format : 'wav' }))
      if (!job) return
      const done = await attempt(() => waitForJob(job.id))
      if (!done) return
      if (done.status !== 'succeeded') {
        toast({ kind: 'error', title: t.msg(done.error ? { code: done.error.code, params: done.error.params } : null, t('common.error')), detail: done.error?.detail ?? undefined })
        return
      }
      const result = done.result as { path?: string; downloadUrl?: string } | null
      if (result?.downloadUrl) window.open(mediaUrl(result.downloadUrl))
      else toast({ kind: 'success', title: t('canvas.exported'), body: result?.path })
    },
    [projectId, t],
  )

  const addToTimeline: CanvasActions['addToTimeline'] = useCallback(
    (nodeId) => {
      const n = findNode(nodeId)
      if (!n) return
      const byId = byIdOf(nodesNow())
      let mid = playableMedia(n, byId)
      if (isSource(n)) mid = stemMedia(n.data, 'inst') ?? n.data.mediaId
      if (!mid) return
      const m = media.get(mid)
      const kind: Track['kind'] = isTake(n) || (isProcess(n) && !!inputsOf(n)[0] && isTake(byId.get(inputsOf(n)[0].nodeId))) ? 'vocal' : isSource(n) || n.type === 'audio' ? 'instrumental' : 'other'
      const start = m?.offset ?? (isTake(n) ? (n.data.params.region?.start ?? 0) : 0)
      const trackName = t(kind === 'vocal' ? 'tl.track.vocal' : kind === 'instrumental' ? 'tl.track.instrumental' : 'tl.track.mix')
      update((p) => {
        const tracks = p.timeline.tracks.slice()
        let ti = tracks.findIndex((tr) => tr.kind === kind)
        if (ti === -1) {
          tracks.push({ id: newId('trk'), name: trackName, kind, gainDb: 0, pan: 0, mute: false, solo: false, clips: [] })
          ti = tracks.length - 1
        }
        const clip = { id: newId('tc'), mediaId: mid!, nodeId, start, offset: 0, duration: m?.duration ?? 10, gainDb: 0, fadeIn: 0.01, fadeOut: 0.05 }
        tracks[ti] = { ...tracks[ti], clips: [...tracks[ti].clips, clip] }
        const length = Math.max(p.timeline.length, ...tracks.flatMap((tr) => tr.clips.map((c) => c.start + c.duration)))
        return { ...p, timeline: { tracks, length } }
      })
      showTimeline()
    },
    [findNode, nodesNow, media, t, update, showTimeline],
  )

  /** 多人试唱 for one song, or for every selected song when the clicked one is among them */
  const openBatch = useCallback(
    (sourceId: string) => {
      const selectedSources = selection.filter((id) => isSource(findNode(id)))
      setBatchFor(selectedSources.includes(sourceId) && selectedSources.length > 1 ? selectedSources : [sourceId])
    },
    [selection, findNode, setBatchFor],
  )

  const copy: CanvasActions['copy'] = useCallback((ids) => copyNodes(projectId, nodesNow(), ids), [projectId, nodesNow])
  const paste: CanvasActions['paste'] = useCallback(
    (at) => {
      const added = pasteNodes(projectId, nodesNow(), at ?? flowAtMouse(), () => newId('nd'))
      if (!added) return
      setNodes((nodes) => [...nodes, ...added])
      setSelection(added.map((n) => n.id))
    },
    [projectId, nodesNow, flowAtMouse, setNodes, setSelection],
  )
  const duplicate: CanvasActions['duplicate'] = useCallback(
    (ids) => {
      const added = duplicateNodes(projectId, nodesNow(), ids, () => newId('nd'))
      if (!added) return
      setNodes((nodes) => [...nodes, ...added])
      setSelection(added.map((n) => n.id))
    },
    [projectId, nodesNow, setNodes, setSelection],
  )

  const quickSong: CanvasActions['quickSong'] = useCallback(
    async (sourceId, characterId) => {
      const take = await sing({ nodeId: sourceId, port: 'vocals' }, characterId, { focus: false })
      if (!take) return
      const mixId = createMix([
        { nodeId: take, port: 'out' },
        { nodeId: sourceId, port: 'inst' },
      ])
      toast({ kind: 'info', title: t('quick.started') })
      await flush()
      await startRun(projectId, [mixId], (r) => {
        if (r.failed || r.skipped) {
          const body = r.error ? t.msg({ code: r.error.code, params: r.error.params }, r.error.message) : undefined
          toast({ kind: 'error', title: t('quick.failed'), body })
          return
        }
        // the finished mix reaches this window by event: wait for it before placing it
        let tries = 0
        const place = () => {
          const n = findNode(mixId)
          const mid = n ? playableMedia(n, byIdOf(nodesNow())) : null
          if (!mid) {
            if (++tries < 20) window.setTimeout(place, 250)
            return
          }
          addToTimeline(mixId)
          toast({
            kind: 'success',
            title: t('quick.done'),
            action: { label: t('common.play'), run: () => usePlayer.getState().toggle(`node:${mixId}`, mediaUrl(mediaAudioPath(projectId, mid))) },
          })
        }
        place()
      })
    },
    [sing, createMix, flush, projectId, t, findNode, nodesNow, addToTimeline],
  )

  const actions: CanvasActions = useMemo(
    () => ({ runNode, run, separate, sing, variant, openBatch, addNode, createMix, importStem, exportMedia, addToTimeline, quickSong, patchNode, removeNodes, duplicate, copy, paste, select: setSelection, reveal, rename: setRenaming }),
    [runNode, run, separate, sing, variant, openBatch, addNode, createMix, importStem, exportMedia, addToTimeline, quickSong, patchNode, removeNodes, duplicate, copy, paste, setSelection, reveal, setRenaming],
  )

  return { actions, suggestions, nodesNow, findNode, setNodes, measure, reveal, flowAtMouse, importFiles }
}

export type CanvasEdit = ReturnType<typeof useCanvasActions>
