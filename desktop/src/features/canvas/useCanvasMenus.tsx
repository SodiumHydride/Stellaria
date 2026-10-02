import { useReactFlow, type Edge, type Node } from '@xyflow/react'
import {
  AlignHorizontalSpaceAround,
  AlignStartHorizontal,
  AlignStartVertical,
  AlignVerticalSpaceAround,
  BoxSelect,
  ChevronsDownUp,
  ChevronsUpDown,
  CircleSlash,
  ClipboardPaste,
  Copy,
  CopyPlus,
  Download,
  Ear,
  FastForward,
  FileAudio,
  Frame,
  LayoutDashboard,
  ListPlus,
  Music,
  Palette,
  Pencil,
  Play,
  Plus,
  Scissors,
  StickyNote,
  Trash2,
  Upload,
  Users,
  Workflow,
} from 'lucide-react'
import { useCallback, useState } from 'react'
import { mediaUrl } from '../../api/client'
import { useResources } from '../../api/hooks'
import type { CanvasNode, Character, NodeLook } from '../../api/types'
import { useT } from '../../i18n'
import { VOICE_COLORS } from '../../lib/color'
import { usePlayer } from '../../state/player'
import { catalogIdOf } from './catalog'
import { hasClip } from './clipboard'
import { mediaAudioPath, type CanvasActions } from './context'
import type { AddMenuState, MenuItem } from './menus'
import { byIdOf, disconnect, inputArity, insideFrame, isContent, isProcess, isSource, isTake, outputPorts, playableMedia, resolvePort, type FlowEdge } from './model'
import { nodeDisplayName } from './nodes'
import { ensureSeparator } from './separator'
import type { CanvasEdit } from './useCanvasActions'
import type { CanvasTools } from './useCanvasTools'
import { singablePort } from './voices'
import { modKey, RUN_KEYS } from '../../shell/bridge'

type CtxMenuState = { x: number; y: number; items: MenuItem[]; title?: string }

/** The "add anything" menu: where it is open, and opening it at a screen point. */
export function useAddMenu() {
  const rf = useReactFlow()
  const [addMenu, setAddMenu] = useState<AddMenuState | null>(null)
  const openAddAt = useCallback((x: number, y: number, extra?: Partial<AddMenuState>) => setAddMenu({ x, y, flow: rf.screenToFlowPosition({ x, y }), ...extra }), [rf])
  return { addMenu, setAddMenu, openAddAt }
}

export interface CanvasMenuArgs {
  projectId: string
  projectName: string
  characters: Map<string, Character>
  selection: string[]
  setSelection: (ids: string[]) => void
  actions: CanvasActions
  edit: CanvasEdit
  tools: CanvasTools
  flowEdges: FlowEdge[]
  openAddAt: ReturnType<typeof useAddMenu>['openAddAt']
  openTemplate: (sourceId: string) => void
}

/** Right-click menus of nodes, edges and the empty canvas. */
export function useCanvasMenus({ projectId, projectName, characters, selection, setSelection, actions, edit, tools, flowEdges, openAddAt, openTemplate }: CanvasMenuArgs) {
  const t = useT()
  const rf = useReactFlow()
  const [ctxMenu, setCtxMenu] = useState<CtxMenuState | null>(null)
  const { findNode, nodesNow, importFiles } = edit
  const { run, runNode, importStem, openBatch, createMix, addToTimeline, exportMedia, copy, duplicate, removeNodes, paste, addNode, rename } = actions
  const { data: resources } = useResources()
  const separate = async (id: string, stems: 2 | 4) => {
    if (await ensureSeparator(resources, t)) await actions.separate(id, stems)
  }
  const { frameSelection, align, distribute, toggleBypass, playNode, setLook, arrange } = tools

  const nodeMenu = (e: React.MouseEvent, node: Node) => {
    e.preventDefault()
    const ids = selection.includes(node.id) ? selection : [node.id]
    if (!selection.includes(node.id)) setSelection([node.id])
    const nodes = ids.map(findNode).filter((n): n is CanvasNode => !!n)
    const n = findNode(node.id)
    if (!n) return
    const items: MenuItem[] = []
    if (nodes.length > 1) {
      items.push({ icon: <Play />, label: t('ctx.run_selected', { n: nodes.length }), shortcut: RUN_KEYS, onClick: () => void run(ids) })
      const voiced = nodes.filter((x) => isTake(x) || (isProcess(x) && outputPorts(x).length > 0))
      if (voiced.length) items.push({ icon: <Music />, label: t('qa.mix_take'), shortcut: 'M', onClick: () => createMix(voiced.map((x) => ({ nodeId: x.id, port: outputPorts(x)[0] }))) })
      items.push({ icon: <Frame />, label: t('ctx.frame'), shortcut: 'G', divider: true, onClick: frameSelection })
      items.push({ icon: <AlignStartVertical />, label: t('ctx.align_left'), onClick: () => align('x') })
      items.push({ icon: <AlignStartHorizontal />, label: t('ctx.align_top'), onClick: () => align('y') })
      if (nodes.length > 2) {
        items.push({ icon: <AlignHorizontalSpaceAround />, label: t('ctx.distribute_x'), onClick: () => distribute('x') })
        items.push({ icon: <AlignVerticalSpaceAround />, label: t('ctx.distribute_y'), onClick: () => distribute('y') })
      }
    } else {
      const playable = playableMedia(n, byIdOf(nodesNow()))
      if (inputArity(n) !== 0) {
        items.push({ icon: <Play />, label: t('ctx.run'), shortcut: RUN_KEYS, onClick: () => void runNode(n.id) })
        items.push({ icon: <FastForward />, label: t('ctx.run_to_here'), onClick: () => void run([n.id]) })
      }
      if (n.type === 'group') {
        const inside = nodesNow().filter((x) => isContent(x) && insideFrame(n, x))
        const runnable = inside.filter((x) => inputArity(x) !== 0).map((x) => x.id)
        if (runnable.length) items.push({ icon: <Play />, label: t('ctx.run_frame', { n: runnable.length }), onClick: () => void run(runnable) })
        if (inside.length) items.push({ icon: <BoxSelect />, label: t('ctx.select_frame', { n: inside.length }), onClick: () => setSelection(inside.map((x) => x.id)) })
      }
      if (isSource(n)) {
        items.push({ icon: <Scissors />, label: t('ctx.separate2'), onClick: () => void separate(n.id, 2) })
        items.push({ icon: <Scissors />, label: t('ctx.separate4'), onClick: () => void separate(n.id, 4) })
        items.push({ icon: <Upload />, label: t('ctx.import_inst'), onClick: () => void importStem(n.id, 'inst') })
        items.push({ icon: <Workflow />, label: t('ctx.template'), onClick: () => openTemplate(n.id) })
        items.push({ icon: <Users />, label: t('batch.open'), onClick: () => openBatch(n.id) })
      }
      if (playable) items.push({ icon: <Ear />, label: t('ctx.play'), shortcut: 'Space', onClick: () => playNode(n) })
      if (outputPorts(n).length) {
        const rect = (e.target as HTMLElement).closest('.react-flow__node')?.getBoundingClientRect()
        const ref = singablePort(n)
        items.push({
          icon: <Plus />,
          label: t('ctx.add_after'),
          shortcut: 'Tab',
          onClick: () => openAddAt(rect ? rect.right + 12 : e.clientX, rect ? rect.top : e.clientY, { from: ref && !isTake(n) ? { nodeId: ref.nodeId, handle: ref.port } : { nodeId: n.id, handle: outputPorts(n)[0] } }),
        })
      }
      if (isProcess(n) && (n.data.op === 'fx' || n.data.op === 'pitch')) items.push({ icon: <CircleSlash />, label: n.data.bypass ? t('ctx.unbypass') : t('ctx.bypass'), shortcut: 'B', onClick: () => toggleBypass([n.id]) })
      if (playable) {
        items.push({ icon: <ListPlus />, label: t('insp.add_to_timeline'), divider: true, onClick: () => addToTimeline(n.id) })
        items.push({ icon: <Download />, label: t('ctx.export_audio'), onClick: () => void exportMedia(playable, nodeDisplayName(n, characters, t) || projectName || 'audio') })
      }
    }
    const content = nodes.filter(isContent)
    if (content.length) {
      if (nodes.length === 1) items.push({ icon: <Pencil />, label: t('ctx.rename'), shortcut: 'F2', divider: true, onClick: () => rename(n.id) })
      const tags = new Set(content.map((x) => (x.data as NodeLook).tag ?? null))
      items.push({
        icon: <Palette />,
        label: t('ctx.color'),
        divider: nodes.length > 1,
        swatches: [null, ...VOICE_COLORS.slice(0, 7)].map((value) => ({ value, active: tags.size === 1 && tags.has(value) })),
        onSwatch: (tag) => setLook(content.map((x) => x.id), { tag }),
      })
      const collapsed = content.every((x) => (x.data as NodeLook).collapsed)
      items.push({ icon: collapsed ? <ChevronsUpDown /> : <ChevronsDownUp />, label: collapsed ? t('ctx.expand') : t('ctx.collapse'), onClick: () => setLook(content.map((x) => x.id), { collapsed: !collapsed }) })
    }
    items.push({ icon: <Copy />, label: t('ctx.copy'), shortcut: modKey('C'), divider: true, onClick: () => copy(ids) })
    items.push({ icon: <CopyPlus />, label: t('ctx.duplicate'), shortcut: modKey('D'), onClick: () => duplicate(ids) })
    items.push({ icon: <Trash2 />, label: t('insp.delete'), shortcut: 'Del', danger: true, divider: true, onClick: () => removeNodes(ids) })
    const entry = catalogIdOf(n)
    setCtxMenu({ x: e.clientX, y: e.clientY, items, title: nodes.length > 1 ? t('common.items', { n: nodes.length }) : entry ? t(`cat.${entry}` as const) : undefined })
  }

  const edgeMenu = (e: React.MouseEvent, edge: Edge) => {
    e.preventDefault()
    const fe = flowEdges.find((x) => x.id === edge.id)
    if (!fe) return
    const mid = resolvePort(byIdOf(nodesNow()), { nodeId: fe.source, port: fe.sourceHandle })
    const target = findNode(fe.target)
    const items: MenuItem[] = [
      { icon: <Plus />, label: t('ctx.insert'), onClick: () => openAddAt(e.clientX, e.clientY, { edge: fe }) },
      { icon: <Ear />, label: t('ctx.listen_here'), disabled: !mid, onClick: () => mid && usePlayer.getState().toggle(`edge:${fe.id}`, mediaUrl(mediaAudioPath(projectId, mid))) },
      {
        icon: <Trash2 />,
        label: t('ctx.disconnect'),
        danger: true,
        divider: true,
        disabled: isTake(target),
        onClick: () => {
          const next = disconnect(nodesNow(), fe)
          if (next) edit.setNodes(() => next)
        },
      },
    ]
    setCtxMenu({ x: e.clientX, y: e.clientY, items, title: t('ctx.edge') })
  }

  const paneMenu = (e: React.MouseEvent | MouseEvent) => {
    e.preventDefault()
    const at = rf.screenToFlowPosition({ x: e.clientX, y: e.clientY })
    const items: MenuItem[] = [
      { icon: <Plus />, label: t('ctx.add_node'), shortcut: 'Tab', onClick: () => openAddAt(e.clientX, e.clientY) },
      { icon: <ClipboardPaste />, label: t('ctx.paste'), shortcut: modKey('V'), disabled: !hasClip(), onClick: () => paste(at) },
      { icon: <Upload />, label: t('canvas.import'), divider: true, onClick: () => void importFiles('song', at) },
      { icon: <FileAudio />, label: t('canvas.import_audio'), onClick: () => void importFiles('audio', at) },
      { icon: <StickyNote />, label: t('canvas.note'), divider: true, onClick: () => void addNode('note', { position: at }) },
      { icon: <LayoutDashboard />, label: t('canvas.arrange'), shortcut: 'L', onClick: arrange },
    ]
    setCtxMenu({ x: e.clientX, y: e.clientY, items })
  }

  return { ctxMenu, closeCtxMenu: () => setCtxMenu(null), nodeMenu, edgeMenu, paneMenu }
}
