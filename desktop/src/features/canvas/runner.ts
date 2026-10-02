// "Run": execute a canvas graph in dependency order. Songs are split when something uses a stem
// that doesn't exist yet; every other node runs once its inputs exist and nothing upstream is
// still pending. A node re-runs when it is new, failed or out of date, or when something it
// reads from re-runs in the same go.
//
// The runner lives outside the canvas component and talks to the backend directly, so a run
// keeps going when the user leaves the canvas. Progress is in a small store the canvas shows.

import { create } from 'zustand'
import { ApiError, api } from '../../api/client'
import { onServerEvent } from '../../api/live'
import type { CanvasNode, Job, PortRef, Project } from '../../api/types'
import { translate, useLang } from '../../i18n'
import { byIdOf, EXTRA_STEMS, inputsOf, isContent, isMix, isProcess, isSource, isTake, migrateNodes, opPorts, refsAlong, resolvePort, type ById } from './model'

type Kind = 'separate' | 'run'
interface Task {
  id: string
  kind: Kind
  /** separation: how many stems are needed */
  stems?: 2 | 4
  jobId: string | null
}

export interface RunState {
  total: number
  done: number
  failed: number
  /** not started because something they read from failed */
  skipped: number
  running: boolean
}

/** Why the first failure in a run failed (an API rejection or a failed job's error). */
export interface RunError {
  code: string
  message?: string
  params?: Record<string, string | number>
}

export interface RunSummary extends RunState {
  error: RunError | null
}

interface RunnerStore {
  runs: Record<string, RunState>
  set: (projectId: string, s: RunState | null) => void
}

export const useRunner = create<RunnerStore>((set) => ({
  runs: {},
  set: (projectId, s) =>
    set((st) => {
      const runs = { ...st.runs }
      if (s) runs[projectId] = s
      else delete runs[projectId]
      return { runs }
    }),
}))

const active = new Map<string, { cancel: () => void }>()

const unfinished = (status: string, stale: boolean) => status === 'draft' || status === 'failed' || stale

/** Whether a node can run at all (has what it needs to be wired). */
export function runnable(n: CanvasNode): boolean {
  if (isTake(n)) return !!n.data.input
  if (isMix(n)) return n.data.inputs.length > 0
  if (isProcess(n)) {
    if (n.data.bypass && (n.data.op === 'fx' || n.data.op === 'pitch')) return false
    const min = n.data.op === 'comp' || n.data.op === 'concat' ? 2 : 1
    return n.data.inputs.length >= min
  }
  return false
}

/** Whether a node has to run to be finished, looking at itself only. */
export function needsRun(n: CanvasNode): boolean {
  if (!runnable(n)) return false
  if (isTake(n)) return unfinished(n.data.status, n.data.stale) || !n.data.mediaId
  if (isMix(n)) return unfinished(n.data.status, n.data.stale) || n.data.outputs.length === 0
  if (isProcess(n)) {
    if (unfinished(n.data.status, n.data.stale)) return true
    if (n.data.op === 'export') return n.data.status !== 'done'
    return opPorts(n.data).some((p) => !n.data.outputs?.[p])
  }
  return false
}

/**
 * Whether reading `ref` depends on a node `has` picks, including nodes it reads through bypassed
 * effects. A song only counts for its stems: separating it doesn't change its full mix.
 */
function readsFrom(byId: ById, ref: PortRef, has: (id: string) => boolean, self?: string): boolean {
  return refsAlong(byId, ref).some((r) => r.nodeId !== self && has(r.nodeId) && !(isSource(byId.get(r.nodeId)) && r.port === 'mix'))
}

/** Topological order of the given ids (inputs first). */
function topo(byId: ById, ids: Set<string>): string[] {
  const order: string[] = []
  const seen = new Set<string>()
  const visit = (id: string, stack: Set<string>) => {
    if (seen.has(id) || stack.has(id) || !ids.has(id)) return
    stack.add(id)
    for (const r of inputsOf(byId.get(id))) visit(r.nodeId, stack)
    stack.delete(id)
    seen.add(id)
    order.push(id)
  }
  for (const id of ids) visit(id, new Set())
  return order
}

/** The nodes `targets` depend on, plus the targets themselves (every node when no targets). */
function closure(byId: ById, nodes: CanvasNode[], targets: string[] | null): Set<string> {
  const out = new Set<string>()
  const visit = (id: string) => {
    if (out.has(id) || !byId.has(id)) return
    out.add(id)
    for (const r of inputsOf(byId.get(id))) visit(r.nodeId)
  }
  for (const n of nodes) if (isContent(n) && (!targets || targets.includes(n.id))) visit(n.id)
  return out
}

/** What has to happen for the chosen part of the graph to be finished, in order. */
function plan(nodes: CanvasNode[], targets: string[] | null): Task[] {
  const byId = byIdOf(nodes)
  const scope = closure(byId, nodes, targets)
  const order = topo(byId, scope)
  const tasks = new Map<string, Task>()
  // stems somebody in scope reads that don't exist yet
  const wanted = new Map<string, Set<string>>()
  for (const id of order) {
    for (const r of inputsOf(byId.get(id))) {
      const src = byId.get(r.nodeId)
      if (isSource(src) && r.port !== 'mix' && !resolvePort(byId, r)) wanted.set(src.id, (wanted.get(src.id) ?? new Set()).add(r.port))
    }
  }
  for (const id of order) {
    const n = byId.get(id)!
    if (isSource(n)) {
      const ports = wanted.get(id)
      if (ports?.size && n.data.stems.status !== 'queued' && n.data.stems.status !== 'running') {
        tasks.set(id, { id, kind: 'separate', stems: EXTRA_STEMS.some((p) => ports.has(p)) ? 4 : 2, jobId: null })
      }
      continue
    }
    if (!runnable(n)) continue
    const upstreamChanges = inputsOf(n).some((r) => readsFrom(byId, r, (x) => tasks.has(x)))
    if (needsRun(n) || upstreamChanges) tasks.set(id, { id, kind: 'run', jobId: null })
  }
  return Array.from(tasks.values())
}

/** Count of jobs a run would start, for the button label. */
export function pendingWork(nodes: CanvasNode[], targets: string[] | null): number {
  return plan(nodes, targets).length
}

export function cancelRun(projectId: string) {
  active.get(projectId)?.cancel()
}

/** Start one node's job (the backend picks separate / render / mix / process by node type). */
export async function postRun(projectId: string, nodeId: string, kind: Kind = 'run', stems?: 2 | 4): Promise<Job> {
  try {
    return await api.post<Job>(`/projects/${projectId}/nodes/${nodeId}/run`, stems ? { stems } : undefined)
  } catch (e) {
    // an older backend without /run: the v2 endpoints
    if (!(e instanceof ApiError) || (e.code !== 'not_found' && e.code !== 'http.404')) throw e
    const node = (await api.get<Project>(`/projects/${projectId}`)).canvas.nodes.find((n) => n.id === nodeId)
    const legacy = kind === 'separate' || node?.type === 'source' ? 'separate' : node?.type === 'mix' ? 'mix' : 'render'
    return api.post<Job>(`/projects/${projectId}/nodes/${nodeId}/${legacy}`)
  }
}

/**
 * Start a run. `flush` must have saved the canvas first: the backend reads node data from the
 * saved project.
 */
export async function startRun(projectId: string, targets: string[] | null, onFinish?: (s: RunSummary) => void) {
  if (active.has(projectId)) return
  const fetchNodes = async () => migrateNodes((await api.get<Project>(`/projects/${projectId}`)).canvas.nodes)
  let nodes = await fetchNodes()
  const tasks = plan(nodes, targets)
  if (tasks.length === 0) return
  const total = tasks.length
  const store = useRunner.getState()
  let done = 0
  let failed = 0
  let skipped = 0
  let firstError: RunError | null = null
  let stopped = false
  const publish = (running: boolean) => store.set(projectId, { total, done, failed, skipped, running })
  publish(true)

  const failedIds = new Set<string>()
  const pending = () => new Set(tasks.map((t) => t.id))
  const jobs = new Map<string, Job>()
  const FINAL_OK = 'succeeded'
  const FINAL_BAD = new Set(['failed', 'cancelled'])

  /** inputs exist and nothing they pass through is still waiting in this run */
  const ready = (t: Task, byId: ById, waiting: Set<string>): boolean => {
    if (t.kind === 'separate') return true
    const n = byId.get(t.id)
    if (!n) return false
    return inputsOf(n).every((r) => !readsFrom(byId, r, (x) => waiting.has(x), t.id) && !!resolvePort(byId, r))
  }

  /** something this task reads from failed (or was skipped) in this run */
  const blocked = (t: Task, byId: ById): boolean => inputsOf(byId.get(t.id)).some((r) => readsFrom(byId, r, (x) => failedIds.has(x)))

  const settle = (t: Task, outcome: 'done' | 'failed' | 'skipped', error?: RunError | null) => {
    tasks.splice(tasks.indexOf(t), 1)
    if (outcome === 'done') done++
    else {
      if (outcome === 'failed') failed++
      else skipped++
      failedIds.add(t.id)
      if (outcome === 'failed' && error && !firstError) firstError = error
    }
  }
  /** null: not reachable right now; 'lost': the backend restarted and forgot it, it will never finish */
  const fetchJob = (id: string): Promise<Job | 'lost' | null> =>
    api.get<Job>(`/jobs/${id}`).catch((e) => (e instanceof ApiError && e.status === 404 ? 'lost' : null))
  const rejection = (e: unknown): RunError =>
    e instanceof ApiError ? { code: e.code, message: e.message, params: e.params as RunError['params'] } : { code: 'common.error', message: e instanceof Error ? e.message : String(e) }

  let ticking = false
  let again = false
  const tick = async () => {
    if (stopped) return
    if (ticking) {
      again = true
      return
    }
    ticking = true
    try {
      nodes = await fetchNodes()
      const byId = byIdOf(nodes)
      for (const t of [...tasks]) {
        if (!byId.has(t.id)) {
          settle(t, 'skipped') // deleted while running
          continue
        }
        if (t.jobId) {
          const job = jobs.get(t.jobId) ?? (await fetchJob(t.jobId))
          if (job === 'lost') settle(t, 'failed', { code: 'job.lost', message: translate(useLang.getState().lang, 'jobs.lost') })
          else if (job?.status === FINAL_OK) settle(t, 'done')
          else if (job && FINAL_BAD.has(job.status)) settle(t, job.status === 'cancelled' ? 'skipped' : 'failed', job.error ? { code: job.error.code, message: job.error.message, params: job.error.params as RunError['params'] } : null)
          continue
        }
        if (blocked(t, byId)) {
          settle(t, 'skipped')
          continue
        }
        if (ready(t, byId, pending())) {
          try {
            t.jobId = (await postRun(projectId, t.id, t.kind, t.stems)).id
          } catch (e) {
            settle(t, 'failed', rejection(e))
          }
        }
      }
    } catch {
      /* transient (backend restarting): the next event or poll retries */
    } finally {
      ticking = false
    }
    publish(true)
    if (tasks.length === 0) finish()
    else if (again) {
      again = false
      void tick()
    }
  }

  let debounce: number | undefined
  const off = onServerEvent((ev) => {
    if (ev.type === 'job') jobs.set(ev.data.id, ev.data)
    const relevant = (ev.type === 'project.nodes' && ev.data.projectId === projectId) || (ev.type === 'job' && ev.data.target.projectId === projectId)
    if (!relevant) return
    window.clearTimeout(debounce)
    debounce = window.setTimeout(() => void tick(), 250)
  })
  const poll = window.setInterval(() => {
    // events are lost while the connection is down: ask about unfinished jobs again instead of
    // trusting the last state an event reported
    for (const [id, job] of jobs) if (job.status !== FINAL_OK && !FINAL_BAD.has(job.status)) jobs.delete(id)
    void tick()
  }, 4000)

  const finish = () => {
    if (stopped) return
    stopped = true
    off()
    window.clearInterval(poll)
    window.clearTimeout(debounce)
    active.delete(projectId)
    store.set(projectId, null)
    onFinish?.({ total: done + failed + skipped, done, failed, skipped, running: false, error: firstError })
  }

  active.set(projectId, {
    cancel: () => {
      // stop scheduling, and stop what was started and hasn't finished
      for (const t of tasks) if (t.jobId) void api.post(`/jobs/${t.jobId}/cancel`).catch(() => {})
      tasks.length = 0
      finish()
    },
  })
  void tick()
}
