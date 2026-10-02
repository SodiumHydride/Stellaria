import { useReactFlow, type Connection, type Edge, type EdgeChange, type FinalConnectionState, type Node, type NodeChange } from '@xyflow/react'
import clsx from 'clsx'
import { useCallback, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react'
import type { CanvasNode, Character, Project } from '../../api/types'
import { useT } from '../../i18n'
import { cssVar, VOICE_COLORS, waveColor } from '../../lib/color'
import { useTheme } from '../../state/theme'
import { toast } from '../../state/ui'
import type { CanvasActions } from './context'
import type { AddMenuState } from './menus'
import { byIdOf, canConnect, connect, deriveEdges, disconnect, inputArity, inputsOf, insertOnEdge, insideFrame, makeRoom, outputPorts, type Endpoint } from './model'
import type { CanvasEdit } from './useCanvasActions'
import { newId } from './useProject'

// same CanvasNode object → same data object, so memoized node components skip re-rendering
const nodeData = new WeakMap<CanvasNode, { node: CanvasNode }>()
function dataFor(n: CanvasNode) {
  let d = nodeData.get(n)
  if (!d) {
    d = { node: n }
    nodeData.set(n, d)
  }
  return d
}

/** Edge colours come from CSS variables, so they are read again for each theme. */
function edgeInk(theme: string) {
  return { theme, plain: cssVar('--text-4'), voice: (hex: string | undefined) => waveColor(hex ?? VOICE_COLORS[4]) }
}

export interface FlowBridgeArgs {
  project: Project | null
  characters: Map<string, Character>
  selection: string[]
  setSelection: Dispatch<SetStateAction<string[]>>
  edit: CanvasEdit
  removeNodes: CanvasActions['removeNodes']
  /** a connection dropped on empty canvas opens the add menu there */
  openAddMenu: (state: AddMenuState) => void
}

/** The controlled React Flow view of the project: nodes and edges to draw, and what its changes do. */
export function useFlowBridge({ project, characters, selection, setSelection, edit, removeNodes, openAddMenu }: FlowBridgeArgs) {
  const t = useT()
  const rf = useReactFlow()
  const theme = useTheme((s) => s.resolved)
  const { setNodes, nodesNow, findNode, measure } = edit
  const [selectedEdges, setSelectedEdges] = useState<string[]>([])

  // React Flow (controlled) takes a node's size from the objects we pass it: hand back what it
  // measured, or every node is hidden again after each server update until it re-measures
  const measured = useRef(new Map<string, { width: number; height: number }>())

  const rfNodes: Node[] = useMemo(() => {
    if (!project) return []
    const sel = new Set(selection)
    return project.canvas.nodes.map((n) => {
      const sized = n.type === 'note' || n.type === 'group'
      const w = n.width ?? (n.type === 'note' ? 220 : 520)
      const h = n.height ?? (n.type === 'note' ? 150 : 340)
      return {
        id: n.id,
        type: n.type,
        position: n.position,
        data: dataFor(n),
        selected: sel.has(n.id),
        zIndex: n.type === 'group' ? -1 : undefined,
        style: sized ? { width: w, height: h } : undefined,
        width: sized ? w : undefined,
        height: sized ? h : undefined,
        measured: measured.current.get(n.id),
      }
    })
  }, [project, selection])

  const flowEdges = useMemo(() => (project ? deriveEdges(project.canvas.nodes) : []), [project])
  const ink = useMemo(() => edgeInk(theme), [theme])

  const rfEdges: Edge[] = useMemo(() => {
    if (!project) return []
    const byId = byIdOf(project.canvas.nodes)
    const sel = new Set(selectedEdges)
    return flowEdges.map((e) => {
      const target = byId.get(e.target)
      const status = target && 'status' in target.data ? (target.data as { status: string }).status : 'draft'
      const color = e.characterId ? ink.voice(characters.get(e.characterId)?.color) : ink.plain
      return {
        id: e.id,
        source: e.source,
        sourceHandle: e.sourceHandle,
        target: e.target,
        targetHandle: e.targetHandle,
        animated: status === 'running' || status === 'queued',
        selected: sel.has(e.id),
        className: clsx('cn-edge', `kind-${e.kind}`),
        style: { stroke: color, strokeWidth: sel.has(e.id) ? 2.6 : e.kind === 'backing' ? 1.4 : 1.8, strokeDasharray: e.kind === 'backing' ? '5 5' : undefined },
      }
    })
  }, [project, flowEdges, characters, selectedEdges, ink])

  const onNodesChange = useCallback(
    (changes: NodeChange[]) => {
      const moves = new Map<string, { x: number; y: number }>()
      const sizes = new Map<string, { width: number; height: number }>()
      let sel: string[] | null = null
      const removed: string[] = []
      for (const ch of changes) {
        if (ch.type === 'position' && ch.position) moves.set(ch.id, ch.position)
        else if (ch.type === 'dimensions' && ch.dimensions) {
          measured.current.set(ch.id, ch.dimensions)
          if (ch.resizing) sizes.set(ch.id, ch.dimensions)
        }
        else if (ch.type === 'select') {
          sel = sel ?? selection.slice()
          if (ch.selected && !sel.includes(ch.id)) sel.push(ch.id)
          if (!ch.selected) sel = sel.filter((x) => x !== ch.id)
        } else if (ch.type === 'remove') removed.push(ch.id)
      }
      if (sel) setSelection(sel)
      if (removed.length) removeNodes(removed)
      if (moves.size || sizes.size) {
        setNodes(
          (nodes) => {
            // a frame carries what sits inside it
            const carried = new Map<string, { dx: number; dy: number }>()
            for (const n of nodes) {
              const to = moves.get(n.id)
              if (n.type !== 'group' || !to) continue
              const dx = to.x - n.position.x
              const dy = to.y - n.position.y
              for (const m of nodes) if (!moves.has(m.id) && insideFrame(n, m)) carried.set(m.id, { dx, dy })
            }
            return nodes.map((n) => {
              const pos = moves.get(n.id)
              const carry = carried.get(n.id)
              const size = sizes.get(n.id)
              if (!pos && !size && !carry) return n
              const position = pos ?? (carry ? { x: n.position.x + carry.dx, y: n.position.y + carry.dy } : n.position)
              return { ...n, position, ...(size ? { width: size.width, height: size.height } : {}) }
            })
          },
          { history: sizes.size ? 'resize' : 'move' },
        )
      }
    },
    [selection, setSelection, setNodes, removeNodes],
  )

  const onEdgesChange = useCallback(
    (changes: EdgeChange[]) => {
      let sel: string[] | null = null
      for (const ch of changes) {
        if (ch.type === 'select') {
          sel = sel ?? selectedEdges.slice()
          if (ch.selected && !sel.includes(ch.id)) sel.push(ch.id)
          if (!ch.selected) sel = sel.filter((x) => x !== ch.id)
        } else if (ch.type === 'remove') {
          const edge = flowEdges.find((e) => e.id === ch.id)
          if (!edge) continue
          const next = disconnect(nodesNow(), edge)
          if (next) setNodes(() => next)
          else toast({ kind: 'info', title: t('canvas.take_needs_input') })
        }
      }
      if (sel) setSelectedEdges(sel)
    },
    [selectedEdges, flowEdges, nodesNow, setNodes, t],
  )

  const isValidConnection = useCallback(
    (c: Connection | Edge) => canConnect(nodesNow(), { nodeId: c.source, handle: c.sourceHandle ?? null }, { nodeId: c.target, handle: c.targetHandle ?? null }),
    [nodesNow],
  )

  const onConnect = useCallback(
    (c: Connection) => {
      const next = connect(nodesNow(), { nodeId: c.source, handle: c.sourceHandle ?? null }, { nodeId: c.target, handle: c.targetHandle ?? null }, () => newId('mi'))
      if (next) setNodes(() => next)
    },
    [nodesNow, setNodes],
  )

  // a connection dropped on empty canvas: offer what that output can become
  const onConnectEnd = useCallback(
    (event: MouseEvent | TouchEvent, state: FinalConnectionState) => {
      if (state.isValid || state.toNode || !state.fromNode || !state.fromHandle || state.fromHandle.type !== 'source') return
      const point = 'changedTouches' in event ? event.changedTouches[0] : event
      const from: Endpoint = { nodeId: state.fromNode.id, handle: state.fromHandle.id ?? null }
      openAddMenu({ x: point.clientX, y: point.clientY, flow: rf.screenToFlowPosition({ x: point.clientX, y: point.clientY }), from })
    },
    [rf, openAddMenu],
  )

  /** A freshly added, unwired node dropped onto a line is spliced into it. */
  const onNodeDragStop = useCallback(
    (event: MouseEvent | TouchEvent, node: Node) => {
      const point = 'changedTouches' in event ? event.changedTouches[0] : event
      const n = findNode(node.id)
      if (!n || inputArity(n) !== 1 || inputsOf(n).length > 0 || outputPorts(n).length === 0) return
      if (nodesNow().some((x) => inputsOf(x).some((r) => r.nodeId === n.id))) return
      const hit = document
        .elementsFromPoint(point.clientX, point.clientY)
        .map((el) => el.closest<SVGGElement>('.react-flow__edge'))
        .find((el) => el?.dataset.id)
      const edge = hit ? flowEdges.find((e) => e.id === hit.dataset.id) : undefined
      if (!edge || edge.source === n.id || edge.target === n.id) return
      const rest = nodesNow().filter((x) => x.id !== n.id)
      const next = insertOnEdge(rest, edge, n)
      if (next) setNodes(() => makeRoom(next, n.id, edge.target, measure))
    },
    [findNode, nodesNow, flowEdges, setNodes, measure],
  )

  return { rfNodes, rfEdges, flowEdges, onNodesChange, onEdgesChange, isValidConnection, onConnect, onConnectEnd, onNodeDragStop }
}
