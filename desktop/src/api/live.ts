import type { QueryClient } from '@tanstack/react-query'
import { create } from 'zustand'
import { eventsUrl } from './client'
import type { Character, GpuInfo, Job, Model, Preferences, RenderStatus, Resource, ServerEvent, VstPlugin, VstStatus } from './types'
import { keys } from './keys'

/** Live connection state, for the UI's connection indicator. */
export const useLive = create<{ connected: boolean; gpus: GpuInfo[] }>(() => ({ connected: false, gpus: [] }))

type Listener = (event: ServerEvent) => void
const listeners = new Set<Listener>()

/** Subscribe to raw server events (charts and logs append in place instead of refetching). */
export function onServerEvent(listener: Listener): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

let socket: WebSocket | null = null
let wantedTopics = new Map<string, number>()
let stopped = false

/** Ref-counted topic subscriptions ('gpu', 'job.log:<id>'). */
export function subscribeTopic(topic: string): () => void {
  const n = wantedTopics.get(topic) ?? 0
  wantedTopics.set(topic, n + 1)
  if (n === 0) send({ type: 'subscribe', topics: [topic] })
  return () => {
    const left = (wantedTopics.get(topic) ?? 1) - 1
    if (left <= 0) {
      wantedTopics.delete(topic)
      send({ type: 'unsubscribe', topics: [topic] })
    } else {
      wantedTopics.set(topic, left)
    }
  }
}

function send(msg: unknown) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(msg))
}

function upsert<T extends { id: string }>(list: T[] | undefined, item: T, prepend = false): T[] | undefined {
  if (!list) return list
  const i = list.findIndex((x) => x.id === item.id)
  if (i === -1) return prepend ? [item, ...list] : [...list, item]
  const next = list.slice()
  next[i] = item
  return next
}

function apply(qc: QueryClient, ev: ServerEvent) {
  switch (ev.type) {
    case 'job': {
      qc.setQueryData<Job[]>(keys.jobs, (list) => upsert(list, ev.data, true))
      break
    }
    case 'character': {
      const c: Character = ev.data
      qc.setQueryData(keys.character(c.id), c)
      qc.setQueryData<Character[]>(keys.characters(c.archived), (list) => upsert(list, c))
      qc.setQueryData<Character[]>(keys.characters(!c.archived), (list) => list?.filter((x) => x.id !== c.id))
      break
    }
    case 'character.deleted': {
      qc.setQueriesData<Character[]>({ queryKey: ['characters'] }, (list) => list?.filter((x) => x.id !== ev.data.id))
      qc.removeQueries({ queryKey: keys.character(ev.data.id) })
      break
    }
    case 'clips': {
      const id = ev.data.characterId
      qc.invalidateQueries({ queryKey: ['clips', id] })
      qc.invalidateQueries({ queryKey: keys.raw(id) })
      break
    }
    case 'health': {
      qc.setQueryData(keys.health(ev.data.characterId), ev.data.report)
      break
    }
    case 'model': {
      const m: Model = ev.data
      qc.setQueryData(keys.model(m.id), m)
      qc.setQueriesData<Model[]>({ queryKey: ['models'] }, (list) => upsert(list, m))
      break
    }
    case 'model.deleted': {
      qc.setQueriesData<Model[]>({ queryKey: ['models'] }, (list) => list?.filter((x) => x.id !== ev.data.id))
      qc.removeQueries({ queryKey: keys.model(ev.data.id) })
      break
    }
    case 'validation': {
      qc.invalidateQueries({ queryKey: ['validation', ev.data.modelId] })
      break
    }
    case 'project.media': {
      qc.invalidateQueries({ queryKey: keys.media(ev.data.projectId) })
      break
    }
    case 'resource': {
      const r: Resource = ev.data
      qc.setQueryData<Resource[]>(keys.resources, (list) => {
        if (!list) return list
        const i = list.findIndex((x) => x.id === r.id)
        if (i === -1) return [...list, r]
        const next = list.slice()
        next[i] = r
        return next
      })
      break
    }
    case 'separation': {
      qc.setQueryData(keys.separationModels, ev.data)
      break
    }
    case 'cleanup': {
      qc.setQueryData(keys.cleanupModels, ev.data)
      break
    }
    case 'plugin':
    case 'plugin.deleted': {
      qc.invalidateQueries({ queryKey: keys.plugins })
      qc.invalidateQueries({ queryKey: keys.mixProcessors })
      break
    }
    case 'base_model':
    case 'base_model.deleted': {
      qc.invalidateQueries({ queryKey: keys.baseModels })
      break
    }
    case 'vst.status': {
      qc.setQueryData<VstStatus>(keys.vstStatus, ev.data)
      break
    }
    case 'vst.plugins': {
      qc.setQueryData<VstPlugin[]>(keys.vstPlugins, ev.data)
      break
    }
    case 'gpu': {
      useLive.setState({ gpus: ev.data })
      break
    }
    case 'preferences': {
      qc.setQueryData<Preferences>(keys.preferences, ev.data)
      break
    }
    case 'render.status': {
      qc.setQueryData<RenderStatus>(keys.renderStatus, ev.data)
      break
    }
    default:
      break
  }
}

/** Keeps one WebSocket open with exponential back-off; on reconnect everything is refetched. */
export function startLive(qc: QueryClient): () => void {
  stopped = false
  let attempt = 0
  let timer: number | undefined

  const connect = () => {
    if (stopped) return
    let ws: WebSocket
    try {
      ws = new WebSocket(eventsUrl())
    } catch {
      schedule()
      return
    }
    socket = ws
    ws.onopen = () => {
      const wasReconnect = attempt > 0
      attempt = 0
      useLive.setState({ connected: true })
      if (wantedTopics.size > 0) send({ type: 'subscribe', topics: Array.from(wantedTopics.keys()) })
      if (wasReconnect) qc.invalidateQueries()
    }
    ws.onmessage = (msg) => {
      let ev: ServerEvent
      try {
        ev = JSON.parse(msg.data)
      } catch {
        return
      }
      apply(qc, ev)
      for (const l of listeners) l(ev)
    }
    ws.onclose = () => {
      if (socket === ws) socket = null
      useLive.setState({ connected: false })
      schedule()
    }
    ws.onerror = () => ws.close()
  }

  const schedule = () => {
    if (stopped) return
    attempt += 1
    const delay = Math.min(8000, 400 * 2 ** Math.min(attempt, 5))
    timer = window.setTimeout(connect, delay)
  }

  connect()
  return () => {
    stopped = true
    window.clearTimeout(timer)
    socket?.close()
    socket = null
    wantedTopics = new Map()
  }
}
