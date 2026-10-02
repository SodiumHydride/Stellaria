import { useQueryClient } from '@tanstack/react-query'
import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../../api/client'
import { keys } from '../../api/keys'
import { onServerEvent } from '../../api/live'
import type { CanvasNode, MixData, ProcessData, Project, RenderData, SourceData } from '../../api/types'
import { onBeforeQuit } from '../../app/beforeQuit'
import { errorText } from '../../lib/actions'
import { toast } from '../../state/ui'
import { markStale, migrateNodes } from './model'

/** Fields of a node the backend owns (job results); everything else is the user's. */
function serverFields(n: CanvasNode): Record<string, unknown> {
  const d = n.data as unknown as Record<string, unknown>
  switch (n.type) {
    case 'render':
      return { status: d.status, mediaId: d.mediaId, jobId: d.jobId, error: d.error }
    case 'source':
      return { stems: d.stems, bpm: d.bpm, key: d.key }
    case 'mix':
      return { status: d.status, outputs: d.outputs, jobId: d.jobId, error: d.error }
    case 'process':
      return { status: d.status, outputs: d.outputs, jobId: d.jobId, error: d.error, result: d.result }
    default:
      return {}
  }
}

/** Merge a node the backend sent into the local copy without losing the user's edits. */
function mergeServerNode(local: CanvasNode | undefined, server: CanvasNode): CanvasNode {
  if (!local) return server
  if (server.type === 'source') {
    const s = server.data as SourceData
    return { ...local, data: { ...(local.data as SourceData), stems: s.stems, bpm: s.bpm ?? null, key: s.key ?? null } }
  }
  if (server.type === 'render') {
    const s = server.data as RenderData
    const l = local.data as RenderData
    const finished = s.status === 'done' && (l.status !== 'done' || s.mediaId !== l.mediaId)
    return { ...local, data: { ...l, status: s.status, mediaId: s.mediaId, jobId: s.jobId, error: s.error, stale: finished ? !!s.stale : l.stale || !!s.stale } }
  }
  if (server.type === 'mix') {
    const s = server.data as MixData
    const l = local.data as MixData
    const finished = s.status === 'done' && (l.status !== 'done' || s.outputs[0]?.mediaId !== l.outputs[0]?.mediaId)
    return {
      ...local,
      data: {
        ...l,
        status: s.status,
        outputs: s.outputs,
        jobId: s.jobId,
        error: s.error,
        selected: finished ? 0 : Math.min(l.selected, Math.max(0, s.outputs.length - 1)),
        stale: finished ? !!s.stale : l.stale || !!s.stale,
      },
    }
  }
  if (server.type === 'process') {
    const s = server.data as ProcessData
    const l = local.data as ProcessData
    const finished = s.status === 'done' && (l.status !== 'done' || JSON.stringify(s.outputs) !== JSON.stringify(l.outputs))
    return { ...local, data: { ...l, status: s.status, outputs: s.outputs ?? {}, jobId: s.jobId, error: s.error, result: s.result ?? null, stale: finished ? !!s.stale : l.stale || !!s.stale } as ProcessData }
  }
  return local
}

/**
 * Undo/redo restores what the user made (structure, wiring, parameters) but keeps the job
 * results that exist now; a node whose settings changed back becomes out of date.
 */
function restore(target: CanvasNode[], current: CanvasNode[]): CanvasNode[] {
  const now = new Map(current.map((n) => [n.id, n]))
  return target.map((n) => {
    const cur = now.get(n.id)
    if (!cur || cur.type !== n.type) return n
    const fields = serverFields(cur)
    if (Object.keys(fields).length === 0) return n
    const merged = { ...n, data: { ...n.data, ...fields } } as CanvasNode
    const strip = (x: CanvasNode) => JSON.stringify({ ...x.data, ...Object.fromEntries(Object.keys(fields).map((k) => [k, null])), stale: null, label: null, rating: null, favorite: null, selected: null })
    return strip(merged) === strip(cur) ? { ...merged, data: { ...merged.data, stale: (cur.data as { stale?: boolean }).stale } as CanvasNode['data'] } : markStale(merged)
  })
}

export type SaveState = 'idle' | 'dirty' | 'saving' | 'saved'

export interface UpdateOptions {
  /**
   * false: not an undoable edit (server merges, viewport). A string coalesces consecutive edits
   * with the same key (typing in a field, dragging a slider, moving nodes) into one undo step.
   */
  history?: false | string
}

const HISTORY_LIMIT = 100
const COALESCE_MS = 1200

/** What undo restores: the canvas's nodes and the arrangement. */
type Snapshot = { nodes: CanvasNode[]; timeline: Project['timeline'] }

interface SaveQueue {
  /** settles once every save sent so far has been answered */
  tail: Promise<void>
  /** a save waiting for the one in flight; newer states replace what it will send */
  waiting: { project: Project; sent: Promise<void> } | null
}

/**
 * Saves per project, one at a time and in order. Module-wide, so they outlive the canvas that
 * made them: leaving a project and opening it again waits for its last edits to be written.
 */
const saveQueues = new Map<string, SaveQueue>()

/** Write a state of a project; resolves once the backend has it (or a newer one). A failure is toasted here, once. */
function putProject(p: Project): Promise<void> {
  let q = saveQueues.get(p.id)
  if (q?.waiting) {
    q.waiting.project = p
    return q.waiting.sent
  }
  if (!q) {
    q = { tail: Promise.resolve(), waiting: null }
    saveQueues.set(p.id, q)
  }
  const queue = q
  const waiting: NonNullable<SaveQueue['waiting']> = { project: p, sent: Promise.resolve() }
  queue.waiting = waiting
  waiting.sent = queue.tail.then(async () => {
    if (queue.waiting === waiting) queue.waiting = null
    const x = waiting.project
    try {
      await api.put<Project>(`/projects/${x.id}`, { name: x.name, canvas: { ...x.canvas, edges: [] }, timeline: x.timeline, coverColor: x.coverColor })
    } catch (e) {
      const { title, detail } = errorText(e)
      toast({ kind: 'error', title, detail })
      throw e
    }
  })
  const tail = waiting.sent.catch(() => undefined)
  queue.tail = tail
  void tail.then(() => {
    if (queue.tail === tail && saveQueues.get(p.id) === queue) saveQueues.delete(p.id)
  })
  return waiting.sent
}

/** Resolves once every save of this project sent so far has been answered. */
function savesSettled(id: string): Promise<void> {
  return saveQueues.get(id)?.tail ?? Promise.resolve()
}

/**
 * Local-first project state: edits apply instantly and are saved debounced; backend node
 * updates (render finished, stems ready) are merged without overwriting the user's edits.
 * Canvas edits are undoable.
 */
export function useProject(id: string) {
  const qc = useQueryClient()
  const [project, setProject] = useState<Project | null>(null)
  const [saveState, setSaveState] = useState<SaveState>('idle')
  const [historyState, setHistoryState] = useState({ canUndo: false, canRedo: false })
  /** the project on screen (null while it loads) */
  const latest = useRef<Project | null>(null)
  /** the newest state not sent yet; it carries its own id, so it is saved where it belongs even after a switch */
  const unsaved = useRef<Project | null>(null)
  const timer = useRef<number | undefined>(undefined)
  const past = useRef<Snapshot[]>([])
  const future = useRef<Snapshot[]>([])
  const lastKey = useRef<{ key: string; at: number } | null>(null)

  const syncHistory = useCallback(() => setHistoryState({ canUndo: past.current.length > 0, canRedo: future.current.length > 0 }), [])

  /** Send one state; the save indicator follows only the project on screen. */
  const send = useCallback(
    (p: Project) => {
      const onScreen = () => latest.current?.id === p.id
      if (onScreen()) setSaveState('saving')
      return putProject(p).then(
        () => {
          if (onScreen()) setSaveState(unsaved.current ? 'dirty' : saveQueues.get(p.id)?.waiting ? 'saving' : 'saved')
          void qc.invalidateQueries({ queryKey: keys.projects })
        },
        () => {
          // the next save tries again with everything
          if (onScreen()) {
            unsaved.current ??= latest.current
            setSaveState('dirty')
          }
        },
      )
    },
    [qc],
  )

  /** Send what hasn't been sent yet; resolves when the backend has it. */
  const save = useCallback(() => {
    window.clearTimeout(timer.current)
    const p = unsaved.current
    unsaved.current = null
    return p ? send(p) : Promise.resolve()
  }, [send])

  const commit = useCallback(
    (next: Project) => {
      latest.current = next
      unsaved.current = next
      setProject(next)
      setSaveState('dirty')
      window.clearTimeout(timer.current)
      timer.current = window.setTimeout(() => void save(), 700)
    },
    [save],
  )

  const update = useCallback(
    (fn: (p: Project) => Project, opts: UpdateOptions = {}) => {
      const cur = latest.current
      if (!cur) return
      const next = fn(cur)
      if (next === cur) return
      if (opts.history !== false && (next.canvas.nodes !== cur.canvas.nodes || next.timeline !== cur.timeline)) {
        const now = Date.now()
        const key = opts.history
        const coalesce = key && lastKey.current?.key === key && now - lastKey.current.at < COALESCE_MS
        if (!coalesce) {
          past.current.push({ nodes: cur.canvas.nodes, timeline: cur.timeline })
          if (past.current.length > HISTORY_LIMIT) past.current.shift()
        }
        lastKey.current = key ? { key, at: now } : null
        future.current = []
        syncHistory()
      }
      commit(next)
    },
    [commit, syncHistory],
  )

  const step = useCallback(
    (from: React.MutableRefObject<Snapshot[]>, to: React.MutableRefObject<Snapshot[]>) => {
      const cur = latest.current
      const target = from.current.pop()
      if (!cur || !target) return
      to.current.push({ nodes: cur.canvas.nodes, timeline: cur.timeline })
      lastKey.current = null
      syncHistory()
      const nodes = target.nodes === cur.canvas.nodes ? cur.canvas.nodes : restore(target.nodes, cur.canvas.nodes)
      commit({ ...cur, canvas: { ...cur.canvas, nodes, edges: [] }, timeline: target.timeline })
    },
    [commit, syncHistory],
  )
  const undo = useCallback(() => step(past, future), [step])
  const redo = useCallback(() => step(future, past), [step])

  /** Persist immediately and wait for every pending save (needed before asking the backend to act on a node). */
  const flush = useCallback(async () => {
    await save()
    if (latest.current) await savesSettled(latest.current.id)
  }, [save])

  /** The current state, including edits made since the last render. */
  const getProject = useCallback(() => latest.current, [])

  // quitting sends what the debounce still holds
  useEffect(() => onBeforeQuit(flush), [flush])

  useEffect(() => {
    let cancelled = false
    latest.current = null
    past.current = []
    future.current = []
    lastKey.current = null
    syncHistory()
    setProject(null)
    // a save of this project may still be on its way (it was just left): load what it wrote
    savesSettled(id)
      .then(() => (cancelled ? null : api.get<Project>(`/projects/${id}`)))
      .then((loaded) => {
        if (!loaded || cancelled) return
        const nodes = migrateNodes(loaded.canvas.nodes)
        const p = nodes === loaded.canvas.nodes ? loaded : { ...loaded, canvas: { ...loaded.canvas, nodes, edges: [] } }
        latest.current = p
        unsaved.current = p === loaded ? null : p
        setProject(p)
        setSaveState(p === loaded ? 'saved' : 'dirty')
      })
      .catch((e) => {
        if (cancelled) return
        const { title, detail } = errorText(e)
        toast({ kind: 'error', title, detail })
      })
    const off = onServerEvent((ev) => {
      if (ev.type !== 'project.nodes' || ev.data.projectId !== id) return
      update(
        (p) => {
          const byId = new Map(p.canvas.nodes.map((n) => [n.id, n]))
          let changed = false
          const incoming = migrateNodes(ev.data.nodes)
          for (const sn of incoming) {
            const merged = mergeServerNode(byId.get(sn.id), sn)
            if (merged !== byId.get(sn.id)) {
              byId.set(sn.id, merged)
              changed = true
            }
          }
          if (!changed) return p
          const order = [...p.canvas.nodes.map((n) => n.id), ...incoming.map((n) => n.id).filter((nid) => !p.canvas.nodes.some((n) => n.id === nid))]
          return { ...p, canvas: { ...p.canvas, nodes: order.map((nid) => byId.get(nid)!) } }
        },
        { history: false },
      )
    })
    return () => {
      cancelled = true
      off()
      // the project being left: its edits go out now, under its own id, whatever loads next
      void save()
    }
  }, [id, update, save, syncHistory])

  return { project, getProject, update, flush, saveState, undo, redo, ...historyState }
}

export function newId(prefix: string): string {
  return `${prefix}_${Math.random().toString(36).slice(2, 10)}`
}
