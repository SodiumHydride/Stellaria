// Copy / paste / duplicate canvas nodes. Wiring between copied nodes is kept; wiring to nodes
// that exist where you paste is kept too (a copied take still sings the same song); anything
// else is dropped. Pasted nodes start fresh (no results) so they can be re-run with changes.

import type { CanvasNode, MixData, PortRef, ProcessData, RenderData } from '../../api/types'
import { DEFAULT_SIZE, inputsOf, isMix, isProcess, isTake } from './model'

let clip: { projectId: string; nodes: CanvasNode[] } | null = null

export function hasClip() {
  return !!clip && clip.nodes.length > 0
}

export function copyNodes(projectId: string, nodes: CanvasNode[], ids: string[]) {
  const set = new Set(ids)
  const picked = nodes.filter((n) => set.has(n.id))
  if (picked.length) clip = { projectId, nodes: structuredClone(picked) }
}

function fresh(n: CanvasNode): CanvasNode {
  if (isTake(n)) return { ...n, data: { ...n.data, mediaId: null, status: 'draft', jobId: null, error: null, stale: false, rating: 0, favorite: false } as RenderData }
  if (isMix(n)) return { ...n, data: { ...n.data, outputs: [], selected: 0, status: 'draft', jobId: null, error: null, stale: false } as MixData }
  if (isProcess(n)) return { ...n, data: { ...n.data, outputs: {}, status: 'draft', jobId: null, error: null, stale: false, result: null } as ProcessData }
  return n
}

/**
 * Nodes to add for a paste of `source` into a canvas holding `existing`, placed so their
 * top-left corner lands at `at`. Returns null when there is nothing to paste.
 */
export function pasteNodes(projectId: string, existing: CanvasNode[], at: { x: number; y: number }, makeId: () => string, source = clip): CanvasNode[] | null {
  if (!source || source.nodes.length === 0) return null
  const sameProject = source.projectId === projectId
  // songs and audio carry media of their own project
  const nodes = source.nodes.filter((n) => sameProject || (n.type !== 'source' && n.type !== 'audio'))
  if (nodes.length === 0) return null
  const ids = new Map(nodes.map((n) => [n.id, makeId()]))
  const present = new Set(existing.map((n) => n.id))
  const remap = (r: PortRef): PortRef | null => (ids.has(r.nodeId) ? { ...r, nodeId: ids.get(r.nodeId)! } : present.has(r.nodeId) && sameProject ? r : null)
  const minX = Math.min(...nodes.map((n) => n.position.x))
  const minY = Math.min(...nodes.map((n) => n.position.y))
  const out: CanvasNode[] = []
  for (const orig of nodes) {
    let n: CanvasNode = { ...fresh(structuredClone(orig)), id: ids.get(orig.id)!, position: { x: at.x + orig.position.x - minX, y: at.y + orig.position.y - minY } }
    if (n.parentId) n = { ...n, parentId: ids.get(n.parentId) }
    if (isTake(n)) {
      const input = remap(n.data.input)
      if (!input) continue // a take without anything to sing makes no sense
      n = { ...n, data: { ...n.data, input } }
    } else if (isMix(n)) {
      const inputs = n.data.inputs.flatMap((i) => {
        const ref = remap(i.ref)
        return ref ? [{ ...i, id: makeId(), ref }] : []
      })
      n = { ...n, data: { ...n.data, inputs } }
    } else if (isProcess(n)) {
      n = { ...n, data: { ...n.data, inputs: inputsOf(n).flatMap((r) => remap(r) ?? []) } as ProcessData }
    }
    out.push(n)
  }
  return out.length ? out : null
}

/** Duplicate in place (doesn't touch the clipboard). */
export function duplicateNodes(projectId: string, nodes: CanvasNode[], ids: string[], makeId: () => string): CanvasNode[] | null {
  const set = new Set(ids)
  const picked = nodes.filter((n) => set.has(n.id))
  if (picked.length === 0) return null
  const minX = Math.min(...picked.map((n) => n.position.x))
  const bottom = Math.max(...picked.map((n) => n.position.y + (n.height ?? DEFAULT_SIZE[n.type]?.h ?? 150)))
  return pasteNodes(projectId, nodes, { x: minX, y: bottom + 24 }, makeId, { projectId, nodes: structuredClone(picked) })
}
