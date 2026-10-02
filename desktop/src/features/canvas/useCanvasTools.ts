import { useReactFlow } from '@xyflow/react'
import type { Dispatch, SetStateAction } from 'react'
import { mediaUrl } from '../../api/client'
import type { CanvasNode, NodeLook, ProcessData } from '../../api/types'
import { VOICE_COLORS } from '../../lib/color'
import { usePlayer } from '../../state/player'
import { mediaAudioPath } from './context'
import { byIdOf, isContent, isProcess, layoutCanvas, playableMedia } from './model'
import { FIT, type CanvasEdit } from './useCanvasActions'
import { newId } from './useProject'

/** Layout and look of the selection: tidy up, align, distribute, frame, colour, bypass, play. */
export function useCanvasTools(projectId: string, edit: CanvasEdit, selection: string[], setSelection: Dispatch<SetStateAction<string[]>>) {
  const rf = useReactFlow()
  const { setNodes, nodesNow, findNode, measure } = edit

  const arrange = () => {
    setNodes((nodes) => {
      const pos = layoutCanvas(nodes, measure)
      return nodes.map((n) => (pos.has(n.id) ? { ...n, position: pos.get(n.id)! } : n))
    })
    window.setTimeout(() => rf.fitView(FIT), 60)
  }

  const align = (axis: 'x' | 'y') => {
    const ids = new Set(selection)
    const sel = nodesNow().filter((n) => ids.has(n.id))
    if (sel.length < 2) return
    const v = Math.min(...sel.map((n) => n.position[axis]))
    setNodes((nodes) => nodes.map((n) => (ids.has(n.id) ? { ...n, position: { ...n.position, [axis]: v } } : n)))
  }

  /** Even gaps between the selected nodes along one axis (the outer two stay put). */
  const distribute = (axis: 'x' | 'y') => {
    const sel = selection.map((id) => rf.getNode(id)).filter((n): n is NonNullable<typeof n> => !!n && n.type !== 'group')
    if (sel.length < 3) return
    const size = (n: (typeof sel)[number]) => (axis === 'x' ? (n.measured?.width ?? 300) : (n.measured?.height ?? 150))
    const sorted = [...sel].sort((a, b) => a.position[axis] - b.position[axis])
    const first = sorted[0].position[axis]
    const last = sorted[sorted.length - 1]
    const span = last.position[axis] + size(last) - first
    const gap = (span - sorted.reduce((sum, n) => sum + size(n), 0)) / (sorted.length - 1)
    const next = new Map<string, number>()
    let at = first
    for (const n of sorted) {
      next.set(n.id, Math.round(at))
      at += size(n) + gap
    }
    setNodes((nodes) => nodes.map((n) => (next.has(n.id) ? { ...n, position: { ...n.position, [axis]: next.get(n.id)! } } : n)))
  }

  /** Colour tag / collapsed state of content nodes. */
  const setLook = (ids: string[], patch: NodeLook) => {
    const set = new Set(ids)
    setNodes((nodes) => nodes.map((n) => (set.has(n.id) && isContent(n) ? ({ ...n, data: { ...n.data, ...patch } } as CanvasNode) : n)))
  }

  /** Put a frame around the selection. */
  const frameSelection = () => {
    const sel = selection.map((id) => rf.getNode(id)).filter((n): n is NonNullable<typeof n> => !!n && n.type !== 'group')
    if (sel.length === 0) return
    const x0 = Math.min(...sel.map((n) => n.position.x)) - 24
    const y0 = Math.min(...sel.map((n) => n.position.y)) - 52
    const x1 = Math.max(...sel.map((n) => n.position.x + (n.measured?.width ?? 300))) + 24
    const y1 = Math.max(...sel.map((n) => n.position.y + (n.measured?.height ?? 150))) + 24
    const id = newId('nd')
    setNodes((nodes) => [{ id, type: 'group', position: { x: x0, y: y0 }, width: x1 - x0, height: y1 - y0, data: { label: '', color: VOICE_COLORS[0] } }, ...nodes])
    setSelection([id])
  }

  const toggleBypass = (ids: string[]) => {
    const targets = ids.map(findNode).filter((n): n is CanvasNode => isProcess(n) && (n.data.op === 'fx' || n.data.op === 'pitch'))
    if (targets.length === 0) return
    const on = !targets.every((n) => (n.data as ProcessData).bypass)
    const set = new Set(targets.map((n) => n.id))
    setNodes((nodes) => nodes.map((n) => (set.has(n.id) ? { ...n, data: { ...(n.data as ProcessData), bypass: on } } : n)))
  }

  const playNode = (n: CanvasNode) => {
    const mid = playableMedia(n, byIdOf(nodesNow()))
    if (mid) usePlayer.getState().toggle(`node:${n.id}`, mediaUrl(mediaAudioPath(projectId, mid)))
  }

  return { arrange, align, distribute, setLook, frameSelection, toggleBypass, playNode }
}

export type CanvasTools = ReturnType<typeof useCanvasTools>
