// Canvas v3 model: pure functions over project nodes.
//
// Every edge is audio. Edges are never stored; they are derived from node data (a take's
// `input`, a mix's `inputs[].ref`, a process node's `inputs`), so every line on the canvas is a
// real input. Any output port can feed any input — the graph just must not loop.

import type {
  AudioData,
  CanvasNode,
  MixData,
  MixInput,
  MixRole,
  PortRef,
  ProcessData,
  ProcessOp,
  ProcessParamsByOp,
  RenderData,
  RenderParams,
  SourceData,
  SourcePort,
  StemPort,
} from '../../api/types'

export type SourceNode = CanvasNode & { type: 'source'; data: SourceData }
export type AudioNode = CanvasNode & { type: 'audio'; data: AudioData }
export type TakeNode = CanvasNode & { type: 'render'; data: RenderData }
export type MixNode = CanvasNode & { type: 'mix'; data: MixData }
export type ProcessNode<Op extends ProcessOp = ProcessOp> = Omit<CanvasNode, 'type' | 'data'> & { type: 'process'; data: ProcessData<Op> }

export const isSource = (n: CanvasNode | undefined): n is SourceNode => n?.type === 'source'
export const isAudio = (n: CanvasNode | undefined): n is AudioNode => n?.type === 'audio'
export const isTake = (n: CanvasNode | undefined): n is TakeNode => n?.type === 'render'
export const isMix = (n: CanvasNode | undefined): n is MixNode => n?.type === 'mix'
export const isProcess = (n: CanvasNode | undefined): n is CanvasNode & ProcessNode => n?.type === 'process'
export const isOp = <Op extends ProcessOp>(n: CanvasNode | undefined, op: Op): n is CanvasNode & ProcessNode<Op> => isProcess(n) && n.data.op === op
/** Nodes that carry audio (everything but notes and groups). */
export const isContent = (n: CanvasNode | undefined): boolean => !!n && n.type !== 'note' && n.type !== 'group'

export type ById = Map<string, CanvasNode>
export const byIdOf = (nodes: CanvasNode[]): ById => new Map(nodes.map((n) => [n.id, n]))

export const STEM_PORTS: StemPort[] = ['vocals', 'inst', 'drums', 'bass', 'other']
export const SOURCE_PORTS: SourcePort[] = ['mix', ...STEM_PORTS]
export const EXTRA_STEMS: StemPort[] = ['drums', 'bass', 'other']

export const DEFAULT_PARAMS: RenderParams = {
  key: 0,
  formant: 0,
  register: 0,
  f0Extractor: 'rmvpe',
  steps: 50,
  method: 'euler',
  tStart: 0,
  threshold: -60,
  spkMix: null,
  region: null,
}

export const EMPTY_STEMS: SourceData['stems'] = { status: 'none', vocals: null, inst: null, drums: null, bass: null, other: null, provided: [], model: null, jobId: null, error: null }

// ---------------------------------------------------------------------------------------------
// Ports

/** The media a song port carries, or null while it doesn't exist yet. */
export function stemMedia(src: SourceData, port: SourcePort | string): string | null {
  if (port === 'mix') return src.mediaId
  return (src.stems?.[port as StemPort] as string | null | undefined) ?? null
}

/** Output ports of a process node, in display order. */
export function opPorts(d: ProcessData): string[] {
  switch (d.op) {
    case 'split':
      return Array.from({ length: ((d as ProcessData<'split'>).params.cuts?.length ?? 0) + 1 }, (_, i) => `part${i + 1}`)
    case 'separate':
      return (d as ProcessData<'separate'>).params.stems === 4 ? STEM_PORTS.slice() : ['vocals', 'inst']
    case 'dereverb':
      return ['dry', 'reverb']
    case 'export':
      return []
    default:
      return ['out']
  }
}

/** Every output port a node offers. */
export function outputPorts(n: CanvasNode): string[] {
  if (isSource(n)) return SOURCE_PORTS.slice()
  if (isAudio(n) || isTake(n) || isMix(n)) return ['out']
  if (isProcess(n)) return opPorts(n.data)
  return []
}

export const MANY_INPUTS: Partial<Record<ProcessOp, number>> = { comp: 8, concat: 32 }
export const MAX_MIX_INPUTS = 16

/** How many inputs a node takes: 0, exactly one, or a list. */
export function inputArity(n: CanvasNode | undefined): 0 | 1 | 'many' {
  if (isTake(n)) return 1
  if (isMix(n)) return 'many'
  if (isProcess(n)) return MANY_INPUTS[n.data.op] ? 'many' : 1
  return 0
}

export function maxInputs(n: CanvasNode): number {
  if (isMix(n)) return MAX_MIX_INPUTS
  if (isProcess(n)) return MANY_INPUTS[n.data.op] ?? 1
  return inputArity(n) === 1 ? 1 : 0
}

export function inputsOf(n: CanvasNode | undefined): PortRef[] {
  if (isTake(n)) return n.data.input ? [n.data.input] : []
  if (isMix(n)) return n.data.inputs.map((i) => i.ref)
  if (isProcess(n)) return n.data.inputs
  return []
}

const sameRef = (a: PortRef, b: PortRef) => a.nodeId === b.nodeId && a.port === b.port

/** The media a port carries now, or null (following bypassed effect/pitch nodes). */
export function resolvePort(byId: ById, ref: PortRef, depth = 0): string | null {
  const n = byId.get(ref.nodeId)
  if (!n || depth > 64) return null
  if (isSource(n)) return stemMedia(n.data, ref.port)
  if (isAudio(n)) return n.data.mediaId
  if (isTake(n)) return n.data.status === 'done' || n.data.mediaId ? n.data.mediaId : null
  if (isMix(n)) return n.data.outputs[n.data.selected]?.mediaId ?? n.data.outputs[0]?.mediaId ?? null
  if (isProcess(n)) {
    if (n.data.bypass && (n.data.op === 'fx' || n.data.op === 'pitch')) return n.data.inputs[0] ? resolvePort(byId, n.data.inputs[0], depth + 1) : null
    return n.data.outputs?.[ref.port] ?? null
  }
  return null
}

/** The references a port's audio passes through on its way here: the port itself, plus the inputs of anything bypassed. */
export function refsAlong(byId: ById, ref: PortRef): PortRef[] {
  const out = [ref]
  let n = byId.get(ref.nodeId)
  let guard = 0
  while (isProcess(n) && n.data.bypass && (n.data.op === 'fx' || n.data.op === 'pitch') && n.data.inputs[0] && guard++ < 64) {
    out.push(n.data.inputs[0])
    n = byId.get(n.data.inputs[0].nodeId)
  }
  return out
}

/** Nodes a port's audio passes through on its way here (the port's node, plus anything it bypasses to). */
export function chainOf(byId: ById, ref: PortRef): string[] {
  return refsAlong(byId, ref).map((r) => r.nodeId)
}

/** The media a node plays: its main output. */
export function playableMedia(n: CanvasNode, byId?: ById): string | null {
  if (isSource(n)) return n.data.mediaId
  if (isAudio(n)) return n.data.mediaId
  if (isTake(n)) return n.data.status === 'done' ? n.data.mediaId : null
  if (isMix(n)) return n.data.outputs[n.data.selected]?.mediaId ?? n.data.outputs[0]?.mediaId ?? null
  if (isProcess(n)) {
    if (n.data.bypass && byId && n.data.inputs[0]) return resolvePort(byId, n.data.inputs[0])
    for (const p of opPorts(n.data)) if (n.data.outputs?.[p]) return n.data.outputs[p]
  }
  return null
}

/** Where a new take should listen on a song: the separated vocals when they exist. */
export function preferredPort(src: SourceData): SourcePort {
  return stemMedia(src, 'vocals') ? 'vocals' : 'mix'
}

// ---------------------------------------------------------------------------------------------
// Relations

export function ancestors(byId: ById, id: string): Set<string> {
  const out = new Set<string>()
  const stack = inputsOf(byId.get(id)).map((r) => r.nodeId)
  while (stack.length) {
    const x = stack.pop()!
    if (out.has(x)) continue
    out.add(x)
    for (const r of inputsOf(byId.get(x))) stack.push(r.nodeId)
  }
  return out
}

/** Nodes that read from `id` directly. */
export function dependents(nodes: CanvasNode[], id: string): CanvasNode[] {
  return nodes.filter((n) => inputsOf(n).some((r) => r.nodeId === id))
}

export function descendants(nodes: CanvasNode[], id: string): Set<string> {
  const out = new Set<string>()
  const stack = [id]
  while (stack.length) {
    const x = stack.pop()!
    for (const d of dependents(nodes, x)) {
      if (out.has(d.id)) continue
      out.add(d.id)
      stack.push(d.id)
    }
  }
  return out
}

/** The song a node derives from (first song found walking up its inputs). */
export function rootSourceId(byId: ById, id: string): string | null {
  const seen = new Set<string>()
  const queue = [id]
  while (queue.length) {
    const x = queue.shift()!
    if (seen.has(x)) continue
    seen.add(x)
    const n = byId.get(x)
    if (isSource(n)) return n.id
    for (const r of inputsOf(n)) queue.push(r.nodeId)
  }
  return null
}

/** The voice heard through a port (for edge colours): the nearest take upstream along single inputs. */
export function voiceOf(byId: ById, ref: PortRef, depth = 0): string | null {
  const n = byId.get(ref.nodeId)
  if (!n || depth > 64) return null
  if (isTake(n)) return n.data.characterId
  if (isProcess(n) && inputArity(n) === 1 && n.data.inputs[0]) return voiceOf(byId, n.data.inputs[0], depth + 1)
  return null
}

/** How a mix should treat an input by default. */
export function defaultRole(byId: ById, ref: PortRef, depth = 0): MixRole {
  const n = byId.get(ref.nodeId)
  if (!n || depth > 64) return 'other'
  if (isSource(n)) return ref.port === 'vocals' ? 'lead' : 'backing'
  if (isAudio(n)) return n.data.role === 'backing' ? 'backing' : n.data.role === 'vocal' ? 'lead' : 'other'
  if (isTake(n)) return 'lead'
  if (isMix(n)) return 'backing'
  if (isProcess(n)) {
    if (n.data.op === 'separate') return ref.port === 'vocals' ? 'lead' : 'backing'
    return n.data.inputs[0] ? defaultRole(byId, n.data.inputs[0], depth + 1) : 'other'
  }
  return 'other'
}

// ---------------------------------------------------------------------------------------------
// Compatibility: v1 (stem nodes, inputNodeId) and v2 (mix vocals/backing) projects. The backend
// migrates on load too; this keeps the canvas usable against an older backend.

type LegacyStem = { mediaId: string | null; sourceNodeId: string; stem: 'vocals' | 'instrumental'; status: string; jobId: string | null }
type LegacyMix = { vocals?: { nodeId: string; gainDb: number; pan: number }[]; backing?: { nodeId: string; port: string; gainDb: number } | null }

export function migrateNodes(nodes: CanvasNode[]): CanvasNode[] {
  const stems = nodes.filter((n) => (n.type as string) === 'stem')
  const stemOwner = new Map(stems.map((s) => [s.id, s.data as unknown as LegacyStem]))
  const types = new Map(nodes.map((n) => [n.id, n.type]))
  let changed = stems.length > 0
  const out: CanvasNode[] = []
  for (const n of nodes) {
    if ((n.type as string) === 'stem') continue
    if (n.type === 'source') {
      const d = n.data as SourceData
      const mine = stems.map((s) => s.data as unknown as LegacyStem).filter((s) => s.sourceNodeId === n.id)
      if (!d.stems || mine.length || d.stems.provided === undefined) {
        const vocals = mine.find((s) => s.stem === 'vocals')?.mediaId ?? d.stems?.vocals ?? null
        const inst = mine.find((s) => s.stem === 'instrumental')?.mediaId ?? d.stems?.inst ?? null
        const status = vocals && inst ? 'done' : (d.stems?.status ?? 'none')
        out.push({ ...n, data: { ...d, stems: { ...EMPTY_STEMS, ...d.stems, vocals, inst, status, provided: d.stems?.provided ?? [] } } })
        changed = true
        continue
      }
    }
    if (n.type === 'render') {
      const d = n.data as RenderData & { inputNodeId?: string }
      if (!d.input && d.inputNodeId) {
        const stem = stemOwner.get(d.inputNodeId)
        const input = stem ? { nodeId: stem.sourceNodeId, port: stem.stem === 'vocals' ? 'vocals' : 'inst' } : { nodeId: d.inputNodeId, port: 'mix' }
        out.push({ ...n, data: { ...d, input, stale: d.stale ?? false, batchId: d.batchId ?? null } })
        changed = true
        continue
      }
      // v2: 'mix' of a non-source node meant its own media
      if (d.input && d.input.port === 'mix' && types.get(d.input.nodeId) !== 'source' && types.has(d.input.nodeId)) {
        out.push({ ...n, data: { ...d, input: { ...d.input, port: 'out' } } })
        changed = true
        continue
      }
    }
    if (n.type === 'mix') {
      const d = n.data as MixData & LegacyMix
      if (!d.inputs) {
        const inputs: MixInput[] = []
        for (const v of d.vocals ?? []) inputs.push({ id: `${v.nodeId}`, ref: { nodeId: v.nodeId, port: 'out' }, role: 'lead', gainDb: v.gainDb, pan: v.pan, mute: false, solo: false })
        if (d.backing) inputs.push({ id: `${d.backing.nodeId}:${d.backing.port}`, ref: { nodeId: d.backing.nodeId, port: d.backing.port }, role: 'backing', gainDb: d.backing.gainDb, pan: 0, mute: false, solo: false })
        const rest = { ...d }
        delete rest.vocals
        delete rest.backing
        out.push({ ...n, data: { ...rest, inputs } })
        changed = true
        continue
      }
    }
    out.push(n)
  }
  return changed ? out : nodes
}

// ---------------------------------------------------------------------------------------------
// Edges

/** voice = a sung line (coloured by the singer); backing = accompaniment (dashed); audio = anything else */
export type EdgeKind = 'voice' | 'backing' | 'audio'

export interface FlowEdge {
  id: string
  source: string
  sourceHandle: string
  target: string
  targetHandle: 'in'
  /** where this edge sits in the target's input list */
  index: number
  kind: EdgeKind
  characterId: string | null
}

export function deriveEdges(nodes: CanvasNode[]): FlowEdge[] {
  const byId = byIdOf(nodes)
  const edges: FlowEdge[] = []
  for (const n of nodes) {
    const refs = inputsOf(n)
    refs.forEach((ref, index) => {
      const src = byId.get(ref.nodeId)
      if (!src || !outputPorts(src).includes(ref.port)) return
      const characterId = voiceOf(byId, ref)
      let kind: EdgeKind = characterId ? 'voice' : 'audio'
      if (isMix(n)) {
        const role = n.data.inputs[index].role
        kind = role === 'backing' || role === 'other' ? 'backing' : characterId ? 'voice' : 'audio'
      } else if (!characterId && defaultRole(byId, ref) === 'backing') kind = 'backing'
      const id = isMix(n) ? `${n.id}:in:${n.data.inputs[index].id}` : `${n.id}:in:${index}`
      edges.push({ id, source: ref.nodeId, sourceHandle: ref.port, target: n.id, targetHandle: 'in', index, kind, characterId })
    })
  }
  return edges
}

export interface Endpoint {
  nodeId: string
  handle: string | null
}

/** Whether connecting one output to a node's input means something (and doesn't loop). */
export function canConnect(nodes: CanvasNode[], from: Endpoint, to: Endpoint): boolean {
  const byId = byIdOf(nodes)
  const a = byId.get(from.nodeId)
  const b = byId.get(to.nodeId)
  if (!a || !b || a.id === b.id || !from.handle) return false
  if (!outputPorts(a).includes(from.handle)) return false
  const arity = inputArity(b)
  if (arity === 0) return false
  if (ancestors(byId, a.id).has(b.id)) return false
  const ref = { nodeId: a.id, port: from.handle }
  if (arity === 'many') {
    const refs = inputsOf(b)
    if (refs.some((r) => sameRef(r, ref))) return false
    if (refs.length >= maxInputs(b)) return false
  }
  return true
}

/** Spread voices a little in a mix: centre, then alternating sides. */
function autoPan(existing: MixInput[], role: MixRole): number {
  if (role === 'backing' || role === 'other') return 0
  const voices = existing.filter((i) => i.role === 'lead' || i.role === 'harmony').length
  if (voices === 0) return 0
  return (voices % 2 === 1 ? -1 : 1) * Math.min(0.5, 0.2 * Math.ceil(voices / 2))
}

export function newMixInput(byId: ById, ref: PortRef, existing: MixInput[], id: string, role?: MixRole): MixInput {
  const r = role ?? defaultRole(byId, ref)
  return { id, ref, role: r, gainDb: 0, pan: autoPan(existing, r), mute: false, solo: false }
}

const wasProduced = (n: CanvasNode): boolean => {
  if (isTake(n)) return n.data.status === 'done' || n.data.stale
  if (isMix(n)) return n.data.outputs.length > 0 || n.data.stale
  if (isProcess(n)) return n.data.status === 'done' || n.data.stale
  return false
}

/** Mark a node out of date when its sound changes after it produced something. */
export function markStale<T extends CanvasNode>(n: T): T {
  if (!wasProduced(n)) return n
  return { ...n, data: { ...n.data, stale: true } }
}

/** Apply a connection to node data. Returns the new node list, or null when it means nothing. */
export function connect(nodes: CanvasNode[], from: Endpoint, to: Endpoint, makeId: () => string): CanvasNode[] | null {
  if (!canConnect(nodes, from, to)) return null
  const byId = byIdOf(nodes)
  const ref: PortRef = { nodeId: from.nodeId, port: from.handle! }
  return nodes.map((n) => {
    if (n.id !== to.nodeId) return n
    if (isTake(n)) return sameRef(n.data.input, ref) ? n : markStale({ ...n, data: { ...n.data, input: ref } })
    if (isMix(n)) return markStale({ ...n, data: { ...n.data, inputs: [...n.data.inputs, newMixInput(byId, ref, n.data.inputs, makeId())] } })
    if (isProcess(n)) {
      const inputs = inputArity(n) === 'many' ? [...n.data.inputs, ref] : [ref]
      return markStale({ ...n, data: { ...n.data, inputs } as ProcessData })
    }
    return n
  })
}

/** Remove an edge by removing that input. A take keeps its input (a take needs something to sing). */
export function disconnect(nodes: CanvasNode[], edge: FlowEdge): CanvasNode[] | null {
  const target = nodes.find((n) => n.id === edge.target)
  if (!target || isTake(target)) return null
  return nodes.map((n) => {
    if (n.id !== edge.target) return n
    if (isMix(n)) return markStale({ ...n, data: { ...n.data, inputs: n.data.inputs.filter((_, i) => i !== edge.index) } })
    if (isProcess(n)) return markStale({ ...n, data: { ...n.data, inputs: n.data.inputs.filter((_, i) => i !== edge.index) } as ProcessData })
    return n
  })
}

/** Splice a single-input node into an edge: source → inserted → target. */
export function insertOnEdge(nodes: CanvasNode[], edge: FlowEdge, inserted: CanvasNode): CanvasNode[] | null {
  if (inputArity(inserted) !== 1 || outputPorts(inserted).length === 0) return null
  const ref: PortRef = { nodeId: edge.source, port: edge.sourceHandle }
  const outRef: PortRef = { nodeId: inserted.id, port: outputPorts(inserted)[0] }
  const wired = setInputs(inserted, [ref])
  const next = nodes.map((n) => {
    if (n.id !== edge.target) return n
    if (isTake(n)) return markStale({ ...n, data: { ...n.data, input: outRef } })
    if (isMix(n)) return markStale({ ...n, data: { ...n.data, inputs: n.data.inputs.map((x, i) => (i === edge.index ? { ...x, ref: outRef } : x)) } })
    if (isProcess(n)) return markStale({ ...n, data: { ...n.data, inputs: n.data.inputs.map((x, i) => (i === edge.index ? outRef : x)) } as ProcessData })
    return n
  })
  return [...next, wired]
}

/**
 * After splicing `insertedId` in front of `targetId`, push the target and everything after it
 * to the right so the new node has room.
 */
export function makeRoom(nodes: CanvasNode[], insertedId: string, targetId: string, measure?: Measure): CanvasNode[] {
  const ins = nodes.find((n) => n.id === insertedId)
  const target = nodes.find((n) => n.id === targetId)
  if (!ins || !target) return nodes
  const w = measure?.(ins.id)?.w ?? DEFAULT_SIZE[ins.type]?.w ?? 300
  const need = ins.position.x + w + 60 - target.position.x
  if (need <= 0) return nodes
  const moving = descendants(nodes, targetId)
  moving.add(targetId)
  return nodes.map((n) => (moving.has(n.id) && n.position.x >= target.position.x - 1 ? { ...n, position: { x: n.position.x + need, y: n.position.y } } : n))
}

export function setInputs(n: CanvasNode, refs: PortRef[]): CanvasNode {
  if (isTake(n)) return refs[0] ? { ...n, data: { ...n.data, input: refs[0] } } : n
  if (isProcess(n)) return { ...n, data: { ...n.data, inputs: refs } as ProcessData }
  return n
}

/** Deleting nodes also removes them from every input list. Takes whose input is deleted go too. */
export function removeNodes(nodes: CanvasNode[], ids: string[]): CanvasNode[] {
  const gone = new Set(ids)
  let grew = true
  while (grew) {
    grew = false
    for (const n of nodes) if (isTake(n) && !gone.has(n.id) && gone.has(n.data.input?.nodeId)) (gone.add(n.id), (grew = true))
  }
  return nodes
    .filter((n) => !gone.has(n.id))
    .map((n) => {
      if (isMix(n)) {
        const inputs = n.data.inputs.filter((i) => !gone.has(i.ref.nodeId))
        return inputs.length === n.data.inputs.length ? n : markStale({ ...n, data: { ...n.data, inputs } })
      }
      if (isProcess(n)) {
        const inputs = n.data.inputs.filter((r) => !gone.has(r.nodeId))
        return inputs.length === n.data.inputs.length ? n : markStale({ ...n, data: { ...n.data, inputs } as ProcessData })
      }
      if (n.parentId && gone.has(n.parentId)) {
        const rest = { ...n }
        delete rest.parentId
        return rest
      }
      return n
    })
}

// ---------------------------------------------------------------------------------------------
// Layout: one block per song, columns by depth (song | takes & cuts | effects | mixes | export).

export const COLUMN_W = 380
export const DEFAULT_SIZE: Record<string, { w: number; h: number }> = {
  source: { w: 310, h: 240 },
  audio: { w: 280, h: 100 },
  render: { w: 300, h: 170 },
  mix: { w: 310, h: 160 },
  process: { w: 280, h: 155 },
}
const GAP = 18
const BLOCK_GAP = 80

export type Measure = (id: string) => { w: number; h: number } | undefined

function heightOf(n: CanvasNode, measure?: Measure) {
  return measure?.(n.id)?.h ?? DEFAULT_SIZE[n.type]?.h ?? 150
}

/** Longest path from a root to each node. */
export function depths(nodes: CanvasNode[]): Map<string, number> {
  const byId = byIdOf(nodes)
  const memo = new Map<string, number>()
  const visit = (id: string, stack: Set<string>): number => {
    if (memo.has(id)) return memo.get(id)!
    if (stack.has(id)) return 0
    stack.add(id)
    const refs = inputsOf(byId.get(id)).filter((r) => byId.has(r.nodeId))
    const d = refs.length ? 1 + Math.max(...refs.map((r) => visit(r.nodeId, stack))) : 0
    stack.delete(id)
    memo.set(id, d)
    return d
  }
  for (const n of nodes) if (isContent(n)) visit(n.id, new Set())
  return memo
}

export function layoutCanvas(nodes: CanvasNode[], measure?: Measure): Map<string, { x: number; y: number }> {
  const pos = new Map<string, { x: number; y: number }>()
  const content = nodes.filter((n) => isContent(n) && !n.parentId)
  if (content.length === 0) return pos
  const byId = byIdOf(nodes)
  const depth = depths(nodes)
  const roots = content.filter((n) => inputsOf(n).length === 0).sort((a, b) => a.position.y - b.position.y)
  const blockOf = new Map<string, string>()
  for (const n of content) {
    const root = rootSourceId(byId, n.id) ?? (() => {
      // no song upstream: the first root found walking up
      let cur = n
      for (let i = 0; i < 64; i++) {
        const up = inputsOf(cur)[0]
        const next = up ? byId.get(up.nodeId) : undefined
        if (!next) break
        cur = next
      }
      return cur.id
    })()
    blockOf.set(n.id, root)
  }
  const x0 = Math.min(...roots.map((r) => r.position.x), 0)
  let top = roots[0]?.position.y ?? 0
  for (const root of roots) {
    const members = content.filter((n) => blockOf.get(n.id) === root.id)
    if (members.length === 0) continue
    const columns = new Map<number, CanvasNode[]>()
    for (const n of members) {
      const d = depth.get(n.id) ?? 0
      columns.set(d, [...(columns.get(d) ?? []), n])
    }
    let bottom = top
    for (const [d, col] of columns) {
      col.sort((a, b) => a.position.y - b.position.y)
      let y = top
      for (const n of col) {
        pos.set(n.id, { x: x0 + d * COLUMN_W, y })
        y += heightOf(n, measure) + GAP
      }
      bottom = Math.max(bottom, y - GAP)
    }
    top = bottom + BLOCK_GAP
  }
  return pos
}

/** Whether `n`'s top-left corner sits inside the frame (group) `frame`: what a frame carries and runs. */
export function insideFrame(frame: CanvasNode, n: CanvasNode): boolean {
  if (n.id === frame.id || n.type === 'group') return false
  const w = frame.width ?? 520
  const h = frame.height ?? 340
  return n.position.x >= frame.position.x && n.position.y >= frame.position.y && n.position.x < frame.position.x + w && n.position.y < frame.position.y + h
}

/**
 * Cards that overlap the one above them in the same column (they grew since the canvas was
 * laid out, e.g. after a redesign) move down just enough; everything else stays put.
 * Returns only the moved positions.
 */
export function separateOverlaps(nodes: CanvasNode[], measure?: Measure): Map<string, { x: number; y: number }> {
  const moved = new Map<string, { x: number; y: number }>()
  const content = nodes.filter((n) => isContent(n) && !n.parentId).sort((a, b) => a.position.x - b.position.x)
  const columns: CanvasNode[][] = []
  for (const n of content) {
    const col = columns.find((c) => Math.abs(c[0].position.x - n.position.x) < 40)
    if (col) col.push(n)
    else columns.push([n])
  }
  for (const col of columns) {
    col.sort((a, b) => a.position.y - b.position.y)
    let bottom = -Infinity
    for (const n of col) {
      let y = n.position.y
      if (y < bottom - 2) {
        y = bottom + GAP
        moved.set(n.id, { x: n.position.x, y })
      }
      bottom = Math.max(bottom, y + heightOf(n, measure))
    }
  }
  return moved
}

/** A free spot to the right of `anchorId`, below what already hangs off it. */
export function slotRightOf(nodes: CanvasNode[], anchorId: string, measure?: Measure): { x: number; y: number } {
  const anchor = nodes.find((n) => n.id === anchorId)
  const base = anchor?.position ?? { x: 0, y: 0 }
  const x = base.x + COLUMN_W
  const peers = nodes.filter((n) => isContent(n) && Math.abs(n.position.x - x) < COLUMN_W / 2 && n.position.y >= base.y - 4 && inputsOf(n).some((r) => r.nodeId === anchorId))
  if (peers.length === 0) return { x, y: base.y }
  const bottom = Math.max(...peers.map((n) => n.position.y + heightOf(n, measure)))
  return { x, y: bottom + GAP }
}

/** Below everything: where a newly imported song goes. */
export function nextSourceSlot(nodes: CanvasNode[], measure?: Measure): { x: number; y: number } {
  const blocks = nodes.filter((n) => isContent(n) && !n.parentId)
  if (blocks.length === 0) return { x: 0, y: 0 }
  const bottom = Math.max(...blocks.map((n) => n.position.y + heightOf(n, measure)))
  const x = Math.min(...nodes.filter((n) => inputsOf(n).length === 0 && isContent(n)).map((n) => n.position.x), 0)
  return { x, y: bottom + BLOCK_GAP }
}

// ---------------------------------------------------------------------------------------------
// Factories

export function newSource(id: string, position: { x: number; y: number }, mediaId: string, name: string): SourceNode {
  return { id, type: 'source', position, data: { mediaId, name, stems: { ...EMPTY_STEMS }, bpm: null, key: null } }
}

export function newAudio(id: string, position: { x: number; y: number }, mediaId: string, name: string, role: AudioData['role'] = 'backing'): AudioNode {
  return { id, type: 'audio', position, data: { mediaId, name, role } }
}

export function newTake(id: string, input: PortRef, characterId: string, modelId: string, releaseId: string | null, params: Partial<RenderParams>, position: { x: number; y: number }, batchId: string | null = null): TakeNode {
  return {
    id,
    type: 'render',
    position,
    data: {
      input,
      characterId,
      modelId,
      releaseId,
      params: { ...DEFAULT_PARAMS, ...params },
      mediaId: null,
      status: 'draft',
      jobId: null,
      error: null,
      rating: 0,
      favorite: false,
      label: '',
      stale: false,
      batchId,
    },
  }
}

export function newMix(id: string, position: { x: number; y: number }, inputs: MixInput[], processor = 'builtin'): MixNode {
  return {
    id,
    type: 'mix',
    position,
    data: {
      inputs,
      processor,
      preset: processor === 'builtin' ? 'natural' : null,
      params: {},
      variants: 1,
      outputs: [],
      selected: 0,
      status: 'draft',
      jobId: null,
      error: null,
      stale: false,
      label: '',
    },
  }
}

export function defaultProcessParams<Op extends ProcessOp>(op: Op, duration = 0): ProcessParamsByOp[Op] {
  const d = Math.max(1, duration)
  const table: ProcessParamsByOp = {
    trim: { start: 0, end: Math.min(d, 30), fadeIn: 0.01, fadeOut: 0.05 },
    split: { cuts: [], fade: 0.01, labels: [] },
    pitch: { semitones: 0, cents: 0, tempo: 1, preserveFormant: true, quality: 'high' },
    fx: { chain: [] },
    separate: { model: 'auto', stems: 4 },
    dereverb: { model: 'auto' },
    comp: { segments: [], crossfade: 0.03 },
    concat: { gap: 0, crossfade: 0 },
    export: { format: 'flac', sampleRate: 44100, bitDepth: 24, mp3Bitrate: 320, loudness: null, folder: null, fileName: '{project} - {label}', keepPosition: false, tags: null },
  }
  return table[op]
}

export function newProcess<Op extends ProcessOp>(op: Op, id: string, position: { x: number; y: number }, inputs: PortRef[], params?: Partial<ProcessParamsByOp[Op]>, duration = 0): ProcessNode<Op> {
  return {
    id,
    type: 'process',
    position,
    data: {
      op,
      label: '',
      inputs,
      params: { ...defaultProcessParams(op, duration), ...params } as ProcessParamsByOp[Op],
      outputs: {},
      status: 'draft',
      jobId: null,
      error: null,
      stale: false,
      bypass: false,
      result: null,
    },
  }
}

// ---------------------------------------------------------------------------------------------
// Render quality presets

export type Quality = 'fast' | 'standard' | 'fine'

export function qualityOf(p: RenderParams): Quality {
  if (p.method === 'rk4') return 'fine'
  return p.steps <= 25 ? 'fast' : 'standard'
}

export const QUALITY: Record<Quality, Pick<RenderParams, 'steps' | 'method'>> = {
  fast: { steps: 20, method: 'euler' },
  standard: { steps: 50, method: 'euler' },
  fine: { steps: 30, method: 'rk4' },
}
