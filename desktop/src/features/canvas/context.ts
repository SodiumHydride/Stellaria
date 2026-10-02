import { createContext, useContext } from 'react'
import type { CanvasNode, Character, Media, MixProcessor, Model, PortRef, Project, RenderParams, StemPort } from '../../api/types'
import type { CatalogId } from './catalog'
import type { Endpoint, FlowEdge } from './model'
import type { UpdateOptions } from './useProject'

export interface SingOptions {
  position?: { x: number; y: number }
  params?: Partial<RenderParams>
  batchId?: string | null
  /** a specific published version; null = latest checkpoint; omitted = newest release */
  releaseId?: string | null
  /** select and frame the new take (single adds); batches select all at the end instead */
  focus?: boolean
}

export interface AddOptions {
  position: { x: number; y: number }
  /** wire the new node's input from this output */
  from?: Endpoint
  /** splice the new node into this edge */
  edge?: FlowEdge
}

export interface CanvasActions {
  /** run one node (and whatever it still needs upstream) */
  runNode: (nodeId: string) => Promise<void>
  /** run the graph (or what the given nodes need) in dependency order */
  run: (targets: string[] | null) => Promise<void>
  separate: (sourceId: string, stems?: 2 | 4) => Promise<void>
  /** A voice sings an output port. Returns the new take's id. */
  sing: (input: PortRef, characterId: string, opts?: SingOptions) => Promise<string | null>
  /** same settings, new take next to it */
  variant: (nodeId: string) => void
  openBatch: (sourceId: string) => void
  addNode: (id: CatalogId, opts: AddOptions) => Promise<string | null>
  createMix: (inputs: PortRef[], position?: { x: number; y: number }, processor?: string) => string
  importStem: (sourceId: string, port: StemPort) => Promise<void>
  exportMedia: (mediaId: string, name: string) => Promise<void>
  addToTimeline: (nodeId: string) => void
  /** 一键出歌: a voice sings the song's vocals, mixed with its accompaniment, run, and put in the arrangement */
  quickSong: (sourceId: string, characterId: string) => Promise<void>
  patchNode: (nodeId: string, fn: (n: CanvasNode) => CanvasNode, opts?: UpdateOptions) => void
  removeNodes: (ids: string[]) => void
  duplicate: (ids: string[]) => void
  copy: (ids: string[]) => void
  paste: (at?: { x: number; y: number }) => void
  select: (ids: string[]) => void
  reveal: (ids: string[]) => void
  /** start (id) or stop (null) renaming a node in place */
  rename: (id: string | null) => void
}

export interface CanvasCtx {
  project: Project
  media: Map<string, Media>
  characters: Map<string, Character>
  models: Map<string, Model>
  processors: MixProcessor[]
  /** voices that have a usable model */
  voices: Character[]
  actions: CanvasActions
  /** the node whose title is being edited */
  renaming: string | null
}

export const CanvasContext = createContext<CanvasCtx | null>(null)

export function useCanvas(): CanvasCtx {
  const ctx = useContext(CanvasContext)
  if (!ctx) throw new Error('CanvasContext missing')
  return ctx
}

export function mediaAudioPath(projectId: string, mediaId: string) {
  return `/projects/${projectId}/media/${mediaId}/audio`
}

export function mediaPeaksPath(projectId: string, mediaId: string) {
  return `/projects/${projectId}/media/${mediaId}/peaks`
}

/** Drag data for audio dropped into the arrangement: JSON `{mediaId, nodeId, name}`. */
export const CLIP_DRAG_TYPE = 'application/x-stellaria-clip'
